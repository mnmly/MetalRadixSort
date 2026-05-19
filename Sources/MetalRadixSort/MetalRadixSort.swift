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
/// One MSD scatter on byte 7 distributes elements into 256 buckets, then
/// seven LSD passes on bytes 0..6 (LSB first) sort each bucket in place via
/// the `sort_inner_stable_64` kernel — a deterministic split-and-flag
/// bit-sort in threadgroup memory. The whole pipeline (init, MSD trio,
/// 7 inner passes, value gather) runs in a single compute encoder.
///
/// Value pairing follows the "argsort + gather" idiom: an internal index
/// buffer is permuted alongside the keys, then used at the end to gather
/// the caller's values into the caller's values buffer.
///
/// ## Stability
///
/// Keys come out in monotonic non-decreasing order, but equal-key
/// sub-orderings are **not** stable — the MSD scatter uses racing atomics
/// for within-bucket rank assignment. Callers who need a stable result can
/// pack the original input index into spare low bits of the key to break
/// ties deterministically; see the `STATUS.md` "Future work" section.
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

        /// `count` passed to ``MetalRadixSortU64Pairs/encode(onto:keys:values:count:)``
        /// exceeded `maxElements` set at construction. The sorter cannot
        /// grow its scratch buffers after init.
        case countExceedsCapacity(count: Int, capacity: Int)
    }

    /// Pre-allocates scratch buffers and compiles pipeline state objects.
    ///
    /// All scratch allocations are sized for `maxElements`, so a subsequent
    /// ``encode(onto:keys:values:count:)`` call with `count > maxElements`
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
        psoMsdScatter  = try Self.makePSO(device: device, library: library, name: "sort_msd_atomic_scatter", fc: fc)
        psoInnerStable = try Self.makePSO(device: device, library: library, name: "sort_inner_stable_64", fc: fc)
        psoInitIndices = try Self.makePSO(device: device, library: library, name: "sort_init_indices",  fc: nil)
        psoGather      = try Self.makePSO(device: device, library: library, name: "sort_gather_values", fc: nil)

        // 3. Allocate scratch buffers.
        let keyBytes  = self.maxElements * MemoryLayout<UInt64>.stride
        let valBytes  = self.maxElements * MemoryLayout<UInt32>.stride

        bufKeyScratch    = try Self.makeBuffer(device: device, length: keyBytes)
        bufValsA         = try Self.makeBuffer(device: device, length: valBytes)
        bufValsB         = try Self.makeBuffer(device: device, length: valBytes)
        bufValsOrig      = try Self.makeBuffer(device: device, length: valBytes)
        bufMsdHist       = try Self.makeBuffer(device: device, length: 256 * MemoryLayout<UInt32>.stride)
        bufCounters      = try Self.makeBuffer(device: device, length: 256 * MemoryLayout<UInt32>.stride)
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
    /// - Parameters:
    ///   - encoder: A live `MTLComputeCommandEncoder`. The sort encodes
    ///     ~13 dispatches; `endEncoding` is the caller's responsibility.
    ///   - keys: Input keys, sorted in place. The final byte pass writes
    ///     here.
    ///   - values: Input values, paired with `keys` by index. Sorted in
    ///     place to match the final key permutation.
    ///   - count: Number of valid pairs at the start of each buffer.
    public func encode(
        onto encoder: MTLComputeCommandEncoder,
        keys: MTLBuffer,
        values: MTLBuffer,
        count: Int
    ) {
        guard count > 1 else { return }
        precondition(count <= maxElements,
                     "count (\(count)) exceeds maxElements (\(maxElements))")

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

        // Build SortParams for MSD pass (shift=56 = bits 56..63).
        var params = SortParams(element_count: n,
                                num_tiles: UInt32(numTiles),
                                shift: 56,
                                pass: 0)
        var tileSizeU32: UInt32 = UInt32(TILE_SIZE_64)

        let tgSize  = MTLSize(width: THREADS_PER_TG, height: 1, depth: 1)
        let histGrid = MTLSize(width: numTiles, height: 1, depth: 1)
        let oneGrid  = MTLSize(width: 1, height: 1, depth: 1)
        let fusedGrid = MTLSize(width: 256, height: 1, depth: 1)

        // Dispatch 2: sort_msd_histogram
        encoder.setComputePipelineState(psoMsdHist)
        encoder.setBuffer(bufA,        offset: 0, index: 0)
        encoder.setBuffer(bufMsdHist,  offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<SortParams>.size, index: 2)
        encoder.dispatchThreadgroups(histGrid, threadsPerThreadgroup: tgSize)

        // Dispatch 3: sort_msd_prep (1 TG)
        encoder.setComputePipelineState(psoMsdPrep)
        encoder.setBuffer(bufMsdHist,     offset: 0, index: 0)
        encoder.setBuffer(bufCounters,    offset: 0, index: 1)
        encoder.setBuffer(bufBucketDescs, offset: 0, index: 2)
        encoder.setBytes(&tileSizeU32, length: 4, index: 3)
        encoder.dispatchThreadgroups(oneGrid, threadsPerThreadgroup: tgSize)

        // Dispatch 4: sort_msd_atomic_scatter (bufA → bufB, valsA → valsB)
        encoder.setComputePipelineState(psoMsdScatter)
        encoder.setBuffer(bufA,         offset: 0, index: 0)
        encoder.setBuffer(bufB,         offset: 0, index: 1)
        encoder.setBuffer(bufCounters,  offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<SortParams>.size, index: 3)
        encoder.setBuffer(valsA, offset: 0, index: 4)
        encoder.setBuffer(valsB, offset: 0, index: 5)
        encoder.dispatchThreadgroups(histGrid, threadsPerThreadgroup: tgSize)

        // Inner LSD passes (LSB → MSB within bucket): 7 single-byte dispatches
        // of sort_inner_stable_64, one per byte 0..6. The lower 7 bytes are
        // sorted here; byte 7 was the MSD pass.
        //
        // Ping-pong: after MSD scatter, keys are in bufB and values in valsB.
        // The stable kernel reads from buffer(0) and writes to buffer(1).
        // Byte 0: src=bufB, dst=bufA. Byte 1: src=bufA, dst=bufB. …
        // Byte 6 (even-indexed pass, starting from 0) ends with data in bufA,
        // which is the caller's keys buffer. ✓
        let innerConfigs: [(UInt32, MTLBuffer, MTLBuffer, MTLBuffer, MTLBuffer)] = [
            (0, bufB, bufA, valsB, valsA), // src,dst,src_vals,dst_vals
            (1, bufA, bufB, valsA, valsB),
            (2, bufB, bufA, valsB, valsA),
            (3, bufA, bufB, valsA, valsB),
            (4, bufB, bufA, valsB, valsA),
            (5, bufA, bufB, valsA, valsB),
            (6, bufB, bufA, valsB, valsA),
        ]

        for cfg in innerConfigs {
            var ip = InnerParams(start_shift: cfg.0, pass_count: 1, batch_start: 0)
            encoder.setComputePipelineState(psoInnerStable)
            encoder.setBuffer(cfg.1,             offset: 0, index: 0) // src_keys
            encoder.setBuffer(cfg.2,             offset: 0, index: 1) // dst_keys
            encoder.setBuffer(bufBucketDescs,    offset: 0, index: 2)
            encoder.setBytes(&ip, length: MemoryLayout<InnerParams>.size, index: 3)
            encoder.setBuffer(cfg.3, offset: 0, index: 4) // src_vals
            encoder.setBuffer(cfg.4, offset: 0, index: 5) // dst_vals
            encoder.dispatchThreadgroups(fusedGrid, threadsPerThreadgroup: tgSize)
        }

        // Dispatch 8: final gather — caller.values[i] = valsOrig[valsA[i]]
        encodeGather(encoder, sortedIndices: valsA, original: valsOrig, gathered: values, count: n)
    }

    // MARK: - Private

    private let device: MTLDevice
    private let maxElements: Int

    private let psoMsdHist: MTLComputePipelineState
    private let psoMsdPrep: MTLComputePipelineState
    private let psoMsdScatter: MTLComputePipelineState
    private let psoInnerStable: MTLComputePipelineState
    private let psoInitIndices: MTLComputePipelineState
    private let psoGather: MTLComputePipelineState

    private let bufKeyScratch: MTLBuffer
    private let bufValsA: MTLBuffer
    private let bufValsB: MTLBuffer
    private let bufValsOrig: MTLBuffer
    private let bufMsdHist: MTLBuffer
    private let bufCounters: MTLBuffer
    private let bufBucketDescs: MTLBuffer

    // Constants mirrored from sort.metal:
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
