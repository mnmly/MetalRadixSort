# MetalRadixSort

Stable GPU radix sort for `(UInt64 key, UInt32 value)` pairs on Metal.
Single command encoder, no CPU readback. ~4 ms to sort 1M pairs on an
Apple M5 Max.

Built for the [gsplat](https://github.com/nerfstudio-project/gsplat) tile-key
sort path (the macOS equivalent of `cub::DeviceRadixSort::SortPairs`), but
not gsplat-specific — it sorts any `(u64, u32)` pair workload.

## Use

```swift
// Package.swift
.package(url: "https://github.com/mnmly/MetalRadixSort", from: "0.3.0"),

// then:
.target(name: "MyApp", dependencies: ["MetalRadixSort"]),
```

```swift
import Metal
import MetalRadixSort

let device = MTLCreateSystemDefaultDevice()!
let queue  = device.makeCommandQueue()!

// Construct once per device. Sized at max element count.
let sorter = try MetalRadixSortU64Pairs(device: device, maxElements: 1_000_000)

// `keys` and `values` are MTLBuffers the caller already owns.
let cmd = queue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
sorter.encode(onto: enc, keys: keys, values: values, count: n)
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

// keys[i] is now the i-th smallest key; values[i] is the value
// originally paired with that key.
```

## Guarantees

- **Stable across equal keys** (since 0.3.0). Equal-key elements come out
  in their original input order. No racing atomics on element ranks
  anywhere in the pipeline.
- **Single compute encoder.** Up to ~13 dispatches, no encoder boundaries,
  no CPU readback. Schedule alongside the rest of your frame work.
- **In-place output.** Sorted keys and values land back in the caller's
  buffers.
- **Bit-range hint** (since 0.2.0). Pass `beginBit` / `endBit` to skip
  known-zero bytes for ~12–50% fewer dispatches, mirroring
  `cub::DeviceRadixSort::SortPairs`.

```swift
// gsplat-style tile-key sort: cam=2 bits, tile=16 bits, depth=32 bits.
// Skip the unused upper byte.
sorter.encode(
    onto: enc, keys: tileKeys, values: instanceIdx,
    count: nInstances,
    beginBit: 0,
    endBit: 32 + tileNBits + camNBits   // typically ~50
)
```

## Performance

On Apple M5 Max, 1M random `(UInt64, UInt32)` pairs, single sort:

| Configuration               | Dispatches | Time    |
|-----------------------------|------------|---------|
| Default (`endBit=64`)       | 8 + gather | ~4.1 ms |
| Tile-key (`endBit=50`)      | 7 + gather | ~3.6 ms |
| Depth-only (`endBit=32`)    | 4 + gather | ~2.1 ms |

(Earlier 0.1.0 / 0.2.0 numbers were ~3.5 ms for full 64-bit before the
stability fix. The ~17% delta is the cost of replacing the racing
atomic in the MSD scatter; see `STATUS.md`.)

## Build and test

`swift test` does **not** work — SwiftPM does not compile `.metal`
resources into a metallib. Use `xcodebuild`:

```bash
xcodebuild test \
  -scheme MetalRadixSort \
  -destination 'platform=macOS' \
  -derivedDataPath ./.xcdd
```

## Documentation

DocC reference docs build via `Scripts/build_docs.sh`. Output lands in
`docs/MetalRadixSort/` (GitHub Pages-ready). Set `EMIT_LLMS_TXT=1` for an
`llms.txt`-style flat Markdown export under `docs/llms.txt`.

See [STATUS.md](STATUS.md) for current state, kernel architecture notes,
and the change history (0.1.0 → 0.2.0 → 0.3.0). See
[CLAUDE.md](CLAUDE.md) for the invariants that future contributors —
human or agent — should preserve when changing the API surface.

## Compatibility

- macOS 14+ / iOS 17+ (Metal 3, function constants, simd_prefix_*).
- Apple Silicon (`MTLGPUFamily.apple7+`) for best performance.
- Built with Swift 6.0 tools. The library itself doesn't use Swift
  Concurrency or actors — the encoder pattern is thread-affined to the
  caller's command buffer.

## License

[AGPL-3.0-only](LICENSE). The MSD scaffolding ports from
[forge-sort 0.2.3](https://crates.io/crates/forge-sort) (also AGPL-3.0).
The stable inner kernel (`sort_inner_stable_64`) and stable MSD scatter
(`sort_msd_stable_scatter`) are this project's contribution.

If your consuming project can't accept AGPL-3.0's source-sharing
obligation transitively, vendor the kernel ideas separately rather than
linking this package.
