// SPDX-License-Identifier: AGPL-3.0-only

import XCTest
import Metal
@testable import MetalRadixSort

final class MetalRadixSortU64PairsTests: XCTestCase {

    // MARK: - Helpers

    private func makeDevice() throws -> MTLDevice {
        try XCTUnwrap(MTLCreateSystemDefaultDevice(), "No Metal device on this host")
    }

    /// Runs the GPU sort end-to-end and returns sorted keys+values.
    /// `values` for input are seeded to `[0, 1, ..., count-1]` so that
    /// `values[i]` after sort reveals the original index of `keys[i]`.
    private func runSort(
        keys inputKeys: [UInt64],
        device: MTLDevice,
        sorter: MetalRadixSortU64Pairs
    ) throws -> (keys: [UInt64], values: [UInt32]) {
        let count = inputKeys.count
        if count == 0 {
            return ([], [])
        }
        let inputVals: [UInt32] = (0..<UInt32(count)).map { $0 }

        let keyBytes = max(count, 1) * MemoryLayout<UInt64>.stride
        let valBytes = max(count, 1) * MemoryLayout<UInt32>.stride
        let keysBuf = try XCTUnwrap(device.makeBuffer(length: keyBytes, options: .storageModeShared))
        let valsBuf = try XCTUnwrap(device.makeBuffer(length: valBytes, options: .storageModeShared))

        keysBuf.contents().withMemoryRebound(to: UInt64.self, capacity: count) { ptr in
            for i in 0..<count { ptr[i] = inputKeys[i] }
        }
        valsBuf.contents().withMemoryRebound(to: UInt32.self, capacity: count) { ptr in
            for i in 0..<count { ptr[i] = inputVals[i] }
        }

        let queue = try XCTUnwrap(device.makeCommandQueue())
        let cmd = try XCTUnwrap(queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cmd.makeComputeCommandEncoder())
        sorter.encode(onto: enc, keys: keysBuf, values: valsBuf, count: count)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            XCTFail("Command buffer error: \(err)")
        }

        var outKeys = [UInt64](repeating: 0, count: count)
        var outVals = [UInt32](repeating: 0, count: count)
        keysBuf.contents().withMemoryRebound(to: UInt64.self, capacity: count) { ptr in
            for i in 0..<count { outKeys[i] = ptr[i] }
        }
        valsBuf.contents().withMemoryRebound(to: UInt32.self, capacity: count) { ptr in
            for i in 0..<count { outVals[i] = ptr[i] }
        }
        return (outKeys, outVals)
    }

    /// Asserts: keys are monotonically non-decreasing AND for every i,
    /// `inputKeys[outVals[i]] == outKeys[i]` (values track the permutation).
    private func assertSortedPaired(
        inputKeys: [UInt64],
        outKeys: [UInt64],
        outVals: [UInt32],
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(outKeys.count, inputKeys.count, file: file, line: line)
        XCTAssertEqual(outVals.count, inputKeys.count, file: file, line: line)

        // CPU reference: stable sort indices by key.
        let n = inputKeys.count
        let refIndices: [UInt32] = (0..<UInt32(n)).sorted { a, b in
            inputKeys[Int(a)] < inputKeys[Int(b)]
        }
        let refKeys: [UInt64] = refIndices.map { inputKeys[Int($0)] }

        // Keys must match the reference key sequence exactly.
        XCTAssertEqual(outKeys, refKeys, "GPU sorted keys differ from CPU reference",
                       file: file, line: line)

        // Each output value must point back to a key matching the output key.
        // (We can't require strict equality with refIndices because the GPU
        // sort is not guaranteed stable across equal keys.)
        for i in 0..<n {
            let v = outVals[i]
            XCTAssertLessThan(Int(v), n, "value out of range at i=\(i)", file: file, line: line)
            XCTAssertEqual(inputKeys[Int(v)], outKeys[i],
                           "value at i=\(i) (\(v)) points to key \(inputKeys[Int(v)]) but sorted key is \(outKeys[i])",
                           file: file, line: line)
        }

        // Monotonic.
        for i in 1..<n {
            XCTAssertLessThanOrEqual(outKeys[i - 1], outKeys[i],
                                     "non-monotonic at i=\(i)", file: file, line: line)
        }
    }

    // MARK: - Tests

    func testRandom1024() throws {
        let device = try makeDevice()
        let sorter = try MetalRadixSortU64Pairs(device: device, maxElements: 1024)
        var rng = SystemRandomNumberGenerator()
        let keys: [UInt64] = (0..<1024).map { _ in UInt64.random(in: 0...UInt64.max, using: &rng) }
        let out = try runSort(keys: keys, device: device, sorter: sorter)
        assertSortedPaired(inputKeys: keys, outKeys: out.keys, outVals: out.values)
    }

    func testRandom1M() throws {
        let device = try makeDevice()
        let n = 1_000_000
        let sorter = try MetalRadixSortU64Pairs(device: device, maxElements: n)
        var rng = SystemRandomNumberGenerator()
        let keys: [UInt64] = (0..<n).map { _ in UInt64.random(in: 0...UInt64.max, using: &rng) }

        // Correctness first.
        let out = try runSort(keys: keys, device: device, sorter: sorter)
        assertSortedPaired(inputKeys: keys, outKeys: out.keys, outVals: out.values)

        // Perf measurement: GPU-only wall time for a single sort.
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let keysBuf = try XCTUnwrap(device.makeBuffer(length: n * 8, options: .storageModeShared))
        let valsBuf = try XCTUnwrap(device.makeBuffer(length: n * 4, options: .storageModeShared))
        // Re-seed buffers for each iteration to avoid timing "already sorted".
        let iterations = 5
        var totalMs: Double = 0
        for _ in 0..<iterations {
            keysBuf.contents().withMemoryRebound(to: UInt64.self, capacity: n) { p in
                for i in 0..<n { p[i] = keys[i] }
            }
            valsBuf.contents().withMemoryRebound(to: UInt32.self, capacity: n) { p in
                for i in 0..<n { p[i] = UInt32(i) }
            }
            let cmd = try XCTUnwrap(queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cmd.makeComputeCommandEncoder())
            sorter.encode(onto: enc, keys: keysBuf, values: valsBuf, count: n)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            let ms = (cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0
            totalMs += ms
        }
        let avgMs = totalMs / Double(iterations)
        print("[MetalRadixSort] 1M u64 pairs avg sort time: \(String(format: "%.3f", avgMs)) ms (n=\(iterations) iters)")
    }

    /// Probe which kernel stage breaks: keys with only the top byte varying.
    /// If this passes → MSD scatter is correct, bug is in inner passes.
    /// If this fails → MSD scatter is broken.
    func testProbeTopByteOnly() throws {
        let device = try makeDevice()
        let n = 50_000
        let sorter = try MetalRadixSortU64Pairs(device: device, maxElements: n)
        // Only top byte varies; lower 56 bits all zero.
        let keys: [UInt64] = (0..<n).map { _ in (UInt64.random(in: 0...255) << 56) }
        let out = try runSort(keys: keys, device: device, sorter: sorter)
        var failures = 0
        for i in 1..<n where out.keys[i-1] > out.keys[i] { failures += 1 }
        print("[PROBE top-byte-only] n=\(n) failures=\(failures)")
    }

    /// Probe: only lower byte varies; top 56 bits all zero. All elements end
    /// up in MSD bucket 0; inner passes do all the work.
    /// If this passes → inner passes are correct, bug is in MSD scatter.
    /// If this fails → inner passes are broken.
    func testProbeLowerByteOnly() throws {
        let device = try makeDevice()
        let n = 50_000
        let sorter = try MetalRadixSortU64Pairs(device: device, maxElements: n)
        let keys: [UInt64] = (0..<n).map { _ in UInt64.random(in: 0...255) }
        let out = try runSort(keys: keys, device: device, sorter: sorter)
        var failures = 0
        for i in 1..<n where out.keys[i-1] > out.keys[i] { failures += 1 }
        print("[PROBE lower-byte-only] n=\(n) failures=\(failures)")
    }

    /// Probe: bytes 1..3 vary, byte 0 and bytes 4..7 zero. Exercises Inner #2 only.
    func testProbeMidBytesOnly() throws {
        let device = try makeDevice()
        let n = 50_000
        let sorter = try MetalRadixSortU64Pairs(device: device, maxElements: n)
        let keys: [UInt64] = (0..<n).map { _ in
            (UInt64.random(in: 0...0xFFFFFF) << 8)
        }
        let out = try runSort(keys: keys, device: device, sorter: sorter)
        var failures = 0
        for i in 1..<n where out.keys[i-1] > out.keys[i] { failures += 1 }
        print("[PROBE bytes-1..3-only] n=\(n) failures=\(failures)")
    }

    /// Probe: bytes 4..6 vary. Exercises Inner #3 only.
    func testProbeUpperMidBytesOnly() throws {
        let device = try makeDevice()
        let n = 50_000
        let sorter = try MetalRadixSortU64Pairs(device: device, maxElements: n)
        let keys: [UInt64] = (0..<n).map { _ in
            (UInt64.random(in: 0...0xFFFFFF) << 32)
        }
        let out = try runSort(keys: keys, device: device, sorter: sorter)
        var failures = 0
        for i in 1..<n where out.keys[i-1] > out.keys[i] { failures += 1 }
        print("[PROBE bytes-4..6-only] n=\(n) failures=\(failures)")
    }

    func testProbeScales() throws {
        let device = try makeDevice()
        for n in [10_000, 50_000, 100_000, 300_000, 1_000_000] {
            let sorter = try MetalRadixSortU64Pairs(device: device, maxElements: n)
            let keys: [UInt64] = (0..<n).map { _ in UInt64.random(in: 0...UInt64.max) }
            let out = try runSort(keys: keys, device: device, sorter: sorter)
            var firstBad = -1
            var failCount = 0
            for i in 1..<n where out.keys[i-1] > out.keys[i] {
                if firstBad < 0 { firstBad = i }
                failCount += 1
            }
            let frac = firstBad < 0 ? "—" : String(format: "%.4f", Double(firstBad)/Double(n))
            print("[PROBE] n=\(n) failures=\(failCount) firstBadIdx=\(firstBad) frac=\(frac)")
        }
    }

    func testEdgeCases() throws {
        let device = try makeDevice()
        let sorter = try MetalRadixSortU64Pairs(device: device, maxElements: 4096)

        // 0 elements — should not crash.
        do {
            let out = try runSort(keys: [], device: device, sorter: sorter)
            XCTAssertTrue(out.keys.isEmpty)
            XCTAssertTrue(out.values.isEmpty)
        }

        // 1 element — early return, but we still allocate the buffers.
        do {
            let keys: [UInt64] = [42]
            let out = try runSort(keys: keys, device: device, sorter: sorter)
            XCTAssertEqual(out.keys, [42])
            XCTAssertEqual(out.values, [0])
        }

        // All-equal keys.
        do {
            let keys: [UInt64] = Array(repeating: 7, count: 3000)
            let out = try runSort(keys: keys, device: device, sorter: sorter)
            assertSortedPaired(inputKeys: keys, outKeys: out.keys, outVals: out.values)
        }

        // Already sorted.
        do {
            let keys: [UInt64] = (0..<3000).map { UInt64($0) * 100 }
            let out = try runSort(keys: keys, device: device, sorter: sorter)
            assertSortedPaired(inputKeys: keys, outKeys: out.keys, outVals: out.values)
        }

        // Reverse sorted.
        do {
            let keys: [UInt64] = (0..<3000).map { UInt64(2_999 - $0) * 100 }
            let out = try runSort(keys: keys, device: device, sorter: sorter)
            assertSortedPaired(inputKeys: keys, outKeys: out.keys, outVals: out.values)
        }
    }
}
