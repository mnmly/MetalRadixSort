// SPDX-License-Identifier: AGPL-3.0-only
// MSD scaffolding ported from forge-sort-0.2.3 (AGPL-3.0). The inner LSD
// passes use a custom stable bit-split kernel (sort_inner_stable_64) that
// replaces forge-sort's buggy sort_inner_fused; see Shaders/sort.metal.

import Foundation
import Metal

/// GPU radix sort for `(UInt64 key, UInt32 value)` pairs.
///
/// `MetalRadixSortU64Pairs` sorts up to a pre-declared maximum number of
/// paired keys and values on a Metal device, with no intermediate CPU
/// readback. A single instance owns all scratch buffers and pipeline state
/// objects; construct it once per `MTLDevice` and reuse across frames.
///
/// ## Algorithm
///
/// One MSD scatter on byte 7 distributes elements into 256 buckets via the
/// `sort_msd_stable_scatter` kernel — a deterministic split-and-flag
/// bit-sort within each tile plus a pre-scanned per-(tile, digit) offset
/// table from `sort_msd_prep`. Seven LSD passes on bytes 0..6 then finish
/// each bucket in place via `sort_inner_stable_64`, using the same
/// bit-split technique. The whole pipeline (init, MSD trio, 7 inner
/// passes, value gather) runs in a single compute encoder.
///
/// Value pairing follows the "argsort + gather" idiom: an internal index
/// buffer is permuted alongside the keys, then used at the end to gather
/// the caller's values into the caller's values buffer.
///
/// ## Stability
///
/// The sort is **stable**: equal-key elements come out in their original
/// input order. Every rank computation along the pipeline is
/// deterministic — the MSD scatter uses pre-scanned offsets per
/// (tile, digit) plus an in-tile bit-split sort, and each inner LSD pass
/// uses the same bit-split. No racing atomics on element ranks anywhere.
public final class MetalRadixSortU64Pairs {

    // MARK: - Public API

    /// Errors thrown by ``MetalRadixSortU64Pairs/init(device:maxElements:)``
    /// when the Metal device or its shader library cannot satisfy the
    /// requirements of the sort pipeline.
    public enum Error: Swift.Error {
        /// The package's default Metal library could not be loaded from
        /// `Bundle.module`. Usually means the SwiftPM resource pipeline
        /// did not compile the `.metal` files into a metallib — verify
        /// the package is built via `xcodebuild` (or an Xcode target),
        /// not bare `swift build`.
        case shaderLibraryUnavailable

        /// A specific Metal compute pipeline state object failed to compile.
        /// `name` is the kernel name; `underlying` is the Metal error.
        case pipelineCreationFailed(name: String, underlying: Swift.Error)

        /// A kernel function listed in the bundled metallib was not found
        /// by name. Indicates a build-pipeline mismatch between the Swift
        /// wrapper and the compiled shaders.
        case functionNotFound(name: String)

        /// `MTLDevice.makeBuffer(length:options:)` returned `nil` for a
        /// scratch buffer of the given size. Typically out-of-memory.
        case bufferAllocationFailed(bytes: Int)

        /// `count` passed to ``MetalRadixSortU64Pairs/encode(onto:keys:values:count:beginBit:endBit:)``
        /// exceeded `maxElements` set at construction. The sorter cannot
        /// grow its scratch buffers after init.
        case countExceedsCapacity(count: Int, capacity: Int)
    }

    /// Pre-allocates scratch buffers and compiles pipeline state objects.
    ///
    /// All scratch allocations are sized for `maxElements`, so a subsequent
    /// ``encode(onto:keys:values:count:beginBit:endBit:)`` call with `count > maxElements`
    /// will fail a precondition. Call this once per device and reuse the
    /// instance.
    ///
    /// - Parameters:
    ///   - device: The Metal device the sort will run on. Scratch buffers
    ///     and pipeline state objects are bound to this device.
    ///   - maxElements: Upper bound on the number of pairs any future
    ///     `encode` call will sort. Determines scratch buffer sizes.
    /// - Throws: ``Error`` if the shader library cannot be loaded, a
    ///   pipeline state object fails to compile, or scratch buffers cannot
    ///   be allocated.
    public init(device: MTLDevice, maxElements: Int) throws {
        self.device = device
        self.maxElements = max(maxElements, 1)

        // 1. Load the SPM-processed default library (the .metal file under
        //    Sources/MetalRadixSort/Shaders becomes the bundle's default
        //    Metal library when packaged with `.process("Shaders")`).
        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            throw Error.shaderLibraryUnavailable
        }

        // 2. Compile the 4 specialized PSOs with HAS_VALUES=true, IS_64BIT=true,
        //    TRANSFORM_MODE=0. `sort_msd_prep` is not specialized (it never
        //    references the function constants).
        let fc = MTLFunctionConstantValues()
        var t = true
        var zero: UInt32 = 0
        fc.setConstantValue(&t, type: .bool, index: 0)  // HAS_VALUES
        fc.setConstantValue(&t, type: .bool, index: 1)  // IS_64BIT
        fc.setConstantValue(&zero, type: .uint, index: 2) // TRANSFORM_MODE

        psoMsdHist     = try Self.makePSO(device: device, library: library, name: "sort_msd_histogram", fc: fc)
        psoMsdPrep     = try Self.makePSO(device: device, library: library, name: "sort_msd_prep",      fc: nil)
        psoMsdStable   = try Self.makePSO(device: device, library: library, name: "sort_msd_stable_scatter", fc: fc)
        psoInnerStable = try Self.makePSO(device: device, library: library, name: "sort_inner_stable_64", fc: fc)
        psoCopyKeys    = try Self.makePSO(device: device, library: library, name: "sort_copy_keys_64",   fc: nil)
        psoInitIndices = try Self.makePSO(device: device, library: library, name: "sort_init_indices",  fc: nil)
        psoGather      = try Self.makePSO(device: device, library: library, name: "sort_gather_values", fc: nil)

        // 3. Allocate scratch buffers. `bufTileHists` and `bufTileOffsets`
        // hold the per-(tile,digit) data the stable MSD scatter consumes;
        // they're sized for the worst-case tile count.
        let keyBytes      = self.maxElements * MemoryLayout<UInt64>.stride
        let valBytes      = self.maxElements * MemoryLayout<UInt32>.stride
        let maxNumTiles   = (self.maxElements + Self.tileSize64 - 1) / Self.tileSize64
        self.maxNumTiles  = maxNumTiles
        let perTileBytes  = max(maxNumTiles, 1) * 256 * MemoryLayout<UInt32>.stride

        bufKeyScratch    = try Self.makeBuffer(device: device, length: keyBytes)
        bufValsA         = try Self.makeBuffer(device: device, length: valBytes)
        bufValsB         = try Self.makeBuffer(device: device, length: valBytes)
        bufValsOrig      = try Self.makeBuffer(device: device, length: valBytes)
        bufMsdHist       = try Self.makeBuffer(device: device, length: 256 * MemoryLayout<UInt32>.stride)
        bufTileHists     = try Self.makeBuffer(device: device, length: perTileBytes)
        bufTileOffsets   = try Self.makeBuffer(device: device, length: perTileBytes)
        bufBucketDescs   = try Self.makeBuffer(device: device, length: 256 * MemoryLayout<BucketDesc>.stride)
    }

    /// Encodes a sort of `count` `(key, value)` pairs onto the supplied
    /// compute command encoder.
    ///
    /// The caller owns the command buffer and is responsible for ending the
    /// encoder and committing. After the command buffer completes,
    /// `keys[i]` is the i-th smallest key and `values[i]` is the value
    /// originally paired with that key.
    ///
    /// Both buffers must be at least `count` elements long
    /// (`count * MemoryLayout<UInt64>.stride` for `keys`,
    /// `count * MemoryLayout<UInt32>.stride` for `values`). `count` must
    /// not exceed `maxElements` from
    /// ``init(device:maxElements:)``. `count` of 0 or 1 returns
    /// without encoding any work.
    ///
    /// ## Bit-range hint
    ///
    /// `beginBit` and `endBit` mirror `cub::DeviceRadixSort::SortPairs`'s
    /// hint: only bits `[beginBit, endBit)` of the key are considered
    /// significant. Bits outside this range are treated as "don't care" —
    /// the sort processes them anyway (to a byte boundary, see below), but
    /// the caller takes responsibility for any ordering they produce. For
    /// the gsplat `(cam_id | tile_id | depth)` use case, passing
    /// `endBit = 32 + tile_n_bits + cam_n_bits` skips the unused high
    /// bytes and gives a ~12-50% speedup depending on key density.
    ///
    /// Resolution is **byte-precise**, not bit-precise: the wrapper sorts
    /// whole bytes covering `[beginBit, endBit)`, so e.g. `endBit = 50`
    /// and `endBit = 56` dispatch identically (both touch byte 6 as the
    /// MSD byte). Sorting a few extra "don't care" bits is free
    /// correctness-wise.
    ///
    /// - Parameters:
    ///   - encoder: A live `MTLComputeCommandEncoder`. The sort encodes
    ///     up to ~13 dispatches; `endEncoding` is the caller's
    ///     responsibility.
    ///   - keys: Input keys, sorted in place. The final byte pass writes
    ///     here.
    ///   - values: Input values, paired with `keys` by index. Sorted in
    ///     place to match the final key permutation.
    ///   - count: Number of valid pairs at the start of each buffer.
    ///   - beginBit: Lowest bit considered significant. Default `0`.
    ///     Rounded down to the nearest byte boundary.
    ///   - endBit: One past the highest bit considered significant.
    ///     Default `64`. Rounded up to the nearest byte boundary. Must
    ///     satisfy `0 <= beginBit < endBit <= 64`.
    public func encode(
        onto encoder: MTLComputeCommandEncoder,
        keys: MTLBuffer,
        values: MTLBuffer,
        count: Int,
        beginBit: Int = 0,
        endBit: Int = 64
    ) {
        guard count > 1 else { return }
        precondition(count <= maxElements,
                     "count (\(count)) exceeds maxElements (\(maxElements))")
        precondition(beginBit >= 0 && endBit <= 64 && beginBit < endBit,
                     "invalid bit range: beginBit=\(beginBit), endBit=\(endBit) (require 0 <= beginBit < endBit <= 64)")

        // Resolve [beginBit, endBit) to byte boundaries. msdByte is the
        // highest byte that contains a relevant bit (used by the MSD
        // scatter); firstByte is the lowest (used by the first inner LSD
        // pass).
        let msdByte    = (endBit - 1) / 8        // 0...7
        let firstByte  = beginBit / 8            // 0...7, <= msdByte
        let innerBytes = msdByte - firstByte     // number of LSD passes

        let n = UInt32(count)
        let numTiles = (count + TILE_SIZE_64 - 1) / TILE_SIZE_64

        // Treat caller's `keys` as buf_a. Internal `bufKeyScratch` is buf_b.
        let bufA = keys
        let bufB = bufKeyScratch
        let valsA = bufValsA
        let valsB = bufValsB
        let valsOrig = bufValsOrig

        // ──────────────────────────────────────────────────────────────
        // Pre-step: zero the MSD histogram buffer via CPU contents() write.
        // Buffer is .storageModeShared so CPU writes are visible to GPU at
        // commit time. This matches forge-sort's reference pattern.
        // ──────────────────────────────────────────────────────────────
        let histPtr = bufMsdHist.contents().assumingMemoryBound(to: UInt32.self)
        histPtr.update(repeating: 0, count: 256)

        // Dispatch 0: init_indices — vals_a[i] = i
        encodeInitIndices(encoder, indices: valsA, count: n)

        // Dispatch 1: "copy-in" — gather identity copies caller.values → valsOrig
        // gathered[i] = original[indices[i]] = values[i]  (indices is identity here)
        encodeGather(encoder, sortedIndices: valsA, original: values, gathered: valsOrig, count: n)

        // Build SortParams for MSD pass on byte `msdByte`.
        var params = SortParams(element_count: n,
                                num_tiles: UInt32(numTiles),
                                shift: UInt32(msdByte * 8),
                                pass: 0)
        var tileSizeU32: UInt32 = UInt32(TILE_SIZE_64)

        let tgSize  = MTLSize(width: THREADS_PER_TG, height: 1, depth: 1)
        let histGrid = MTLSize(width: numTiles, height: 1, depth: 1)
        let oneGrid  = MTLSize(width: 1, height: 1, depth: 1)
        let fusedGrid = MTLSize(width: 256, height: 1, depth: 1)

        // Dispatch 2: sort_msd_histogram — emits both the global digit
        // histogram (atomic-summed across tiles) and the per-tile
        // histograms (direct writes) that the next two dispatches
        // consume.
        encoder.setComputePipelineState(psoMsdHist)
        encoder.setBuffer(bufA,         offset: 0, index: 0)
        encoder.setBuffer(bufMsdHist,   offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<SortParams>.size, index: 2)
        encoder.setBuffer(bufTileHists, offset: 0, index: 3)
        encoder.dispatchThreadgroups(histGrid, threadsPerThreadgroup: tgSize)

        // Dispatch 3: sort_msd_prep (1 TG) — derives bucket_descs and
        // tile_offsets[num_tiles * 256] from the histograms. Each
        // thread runs the per-digit cross-tile prefix scan, so the
        // stable scatter has deterministic start slots per (tile, digit).
        var numTilesU32: UInt32 = UInt32(numTiles)
        encoder.setComputePipelineState(psoMsdPrep)
        encoder.setBuffer(bufMsdHist,     offset: 0, index: 0)
        encoder.setBuffer(bufTileHists,   offset: 0, index: 1)
        encoder.setBuffer(bufBucketDescs, offset: 0, index: 2)
        encoder.setBytes(&tileSizeU32, length: 4, index: 3)
        encoder.setBytes(&numTilesU32, length: 4, index: 4)
        encoder.setBuffer(bufTileOffsets, offset: 0, index: 5)
        encoder.dispatchThreadgroups(oneGrid, threadsPerThreadgroup: tgSize)

        // Dispatch 4: sort_msd_stable_scatter (bufA → bufB, valsA → valsB).
        // Replaces the racing-atomic scatter — uses pre-scanned
        // tile_offsets for cross-tile placement and a bit-split sort for
        // within-tile rank, so equal-key ordering is deterministic.
        encoder.setComputePipelineState(psoMsdStable)
        encoder.setBuffer(bufA,           offset: 0, index: 0)
        encoder.setBuffer(bufB,           offset: 0, index: 1)
        encoder.setBuffer(bufTileOffsets, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<SortParams>.size, index: 3)
        encoder.setBuffer(valsA, offset: 0, index: 4)
        encoder.setBuffer(valsB, offset: 0, index: 5)
        encoder.dispatchThreadgroups(histGrid, threadsPerThreadgroup: tgSize)

        // Inner LSD passes over bytes [firstByte, msdByte). After MSD,
        // keys are in bufB and values in valsB; the stable kernel reads
        // buffer(0) and writes buffer(1). Pass index 0 reads bufB; each
        // subsequent pass swaps. After N inner passes the data lives in
        // bufA iff N is odd, bufB iff N is even.
        for k in 0..<innerBytes {
            let byte = UInt32(firstByte + k)
            let readsFromB = (k % 2 == 0)
            let srcK  = readsFromB ? bufB  : bufA
            let dstK  = readsFromB ? bufA  : bufB
            let srcV  = readsFromB ? valsB : valsA
            let dstV  = readsFromB ? valsA : valsB
            var ip = InnerParams(start_shift: byte, pass_count: 1, batch_start: 0)
            encoder.setComputePipelineState(psoInnerStable)
            encoder.setBuffer(srcK,           offset: 0, index: 0)
            encoder.setBuffer(dstK,           offset: 0, index: 1)
            encoder.setBuffer(bufBucketDescs, offset: 0, index: 2)
            encoder.setBytes(&ip, length: MemoryLayout<InnerParams>.size, index: 3)
            encoder.setBuffer(srcV, offset: 0, index: 4)
            encoder.setBuffer(dstV, offset: 0, index: 5)
            encoder.dispatchThreadgroups(fusedGrid, threadsPerThreadgroup: tgSize)
        }

        // Parity reconciliation. After MSD + innerBytes inner passes:
        // - keys are in bufA iff innerBytes is odd, else bufB.
        // - values are in valsA iff innerBytes is odd, else valsB.
        // Caller expects keys in bufA; copy if needed. For values, point
        // the final gather at the buffer that actually holds the sorted
        // permutation.
        let keysInB = (innerBytes % 2 == 0)
        let valsLive = keysInB ? valsB : valsA
        if keysInB {
            encodeCopyKeys(encoder, src: bufB, dst: bufA, count: n)
        }

        // Final gather: caller.values[i] = valsOrig[sortedIndices[i]].
        encodeGather(encoder, sortedIndices: valsLive, original: valsOrig, gathered: values, count: n)
    }

    // MARK: - Private

    private let device: MTLDevice
    private let maxElements: Int

    private let psoMsdHist: MTLComputePipelineState
    private let psoMsdPrep: MTLComputePipelineState
    private let psoMsdStable: MTLComputePipelineState
    private let psoInnerStable: MTLComputePipelineState
    private let psoCopyKeys: MTLComputePipelineState
    private let psoInitIndices: MTLComputePipelineState
    private let psoGather: MTLComputePipelineState

    private let bufKeyScratch: MTLBuffer
    private let bufValsA: MTLBuffer
    private let bufValsB: MTLBuffer
    private let bufValsOrig: MTLBuffer
    private let bufMsdHist: MTLBuffer
    private let bufTileHists: MTLBuffer
    private let bufTileOffsets: MTLBuffer
    private let bufBucketDescs: MTLBuffer
    private let maxNumTiles: Int

    // Constants mirrored from sort.metal:
    private static let tileSize64 = 2048
    private let TILE_SIZE_64 = 2048
    private let THREADS_PER_TG = 256

    // MUST match the MSL struct layouts (both are 16 bytes, 4×u32 / 4×u32).
    private struct SortParams {
        var element_count: UInt32
        var num_tiles: UInt32
        var shift: UInt32
        var pass: UInt32
    }
    private struct InnerParams {
        var start_shift: UInt32
        var pass_count: UInt32
        var batch_start: UInt32
    }
    private struct BucketDesc {
        var offset: UInt32
        var count: UInt32
        var tile_count: UInt32
        var tile_base: UInt32
    }

    private func encodeInitIndices(
        _ encoder: MTLComputeCommandEncoder,
        indices: MTLBuffer,
        count: UInt32
    ) {
        var c = count
        encoder.setComputePipelineState(psoInitIndices)
        encoder.setBuffer(indices, offset: 0, index: 0)
        encoder.setBytes(&c, length: 4, index: 1)
        let n = Int(count)
        let tpg = THREADS_PER_TG
        let grid = MTLSize(width: ((n + tpg - 1) / tpg) * tpg, height: 1, depth: 1)
        let tgs  = MTLSize(width: tpg, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tgs)
    }

    private func encodeCopyKeys(
        _ encoder: MTLComputeCommandEncoder,
        src: MTLBuffer,
        dst: MTLBuffer,
        count: UInt32
    ) {
        var c = count
        encoder.setComputePipelineState(psoCopyKeys)
        encoder.setBuffer(src, offset: 0, index: 0)
        encoder.setBuffer(dst, offset: 0, index: 1)
        encoder.setBytes(&c, length: 4, index: 2)
        let n = Int(count)
        let tpg = THREADS_PER_TG
        let grid = MTLSize(width: ((n + tpg - 1) / tpg) * tpg, height: 1, depth: 1)
        let tgs  = MTLSize(width: tpg, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tgs)
    }

    private func encodeGather(
        _ encoder: MTLComputeCommandEncoder,
        sortedIndices: MTLBuffer,
        original: MTLBuffer,
        gathered: MTLBuffer,
        count: UInt32
    ) {
        var c = count
        encoder.setComputePipelineState(psoGather)
        encoder.setBuffer(sortedIndices, offset: 0, index: 0)
        encoder.setBuffer(original,      offset: 0, index: 1)
        encoder.setBuffer(gathered,      offset: 0, index: 2)
        encoder.setBytes(&c, length: 4, index: 3)
        let n = Int(count)
        let tpg = THREADS_PER_TG
        let grid = MTLSize(width: ((n + tpg - 1) / tpg) * tpg, height: 1, depth: 1)
        let tgs  = MTLSize(width: tpg, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tgs)
    }

    // MARK: - Static helpers

    private static func makePSO(
        device: MTLDevice,
        library: MTLLibrary,
        name: String,
        fc: MTLFunctionConstantValues?
    ) throws -> MTLComputePipelineState {
        let function: MTLFunction
        do {
            if let fc = fc {
                function = try library.makeFunction(name: name, constantValues: fc)
            } else {
                guard let f = library.makeFunction(name: name) else {
                    throw Error.functionNotFound(name: name)
                }
                function = f
            }
        } catch let e as Error {
            throw e
        } catch {
            throw Error.pipelineCreationFailed(name: name, underlying: error)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw Error.pipelineCreationFailed(name: name, underlying: error)
        }
    }

    private static func makeBuffer(device: MTLDevice, length: Int) throws -> MTLBuffer {
        guard let buf = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw Error.bufferAllocationFailed(bytes: length)
        }
        return buf
    }
}
