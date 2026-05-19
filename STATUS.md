# MetalRadixSort — Status

**As of 2026-05-18: working.** 1M u64-pair sort in ~3.7 ms on Apple M5 Max.
All probe and edge-case tests pass with zero monotonicity failures.

## What works

Pure Swift + Metal hybrid MSD+LSD radix sort for `(UInt64 key, UInt32 value)`
pairs. One MSD scatter on byte 7 distributes into 256 buckets; seven LSD
passes (bytes 0..6) finish each bucket in-place. All in a single command
buffer, no CPU readback.

```sh
cd /Users/mnmly/Development-local/GitHub/swift/MetalRadixSort
xcodebuild test \
  -scheme MetalRadixSort \
  -destination 'platform=macOS' \
  -derivedDataPath ./.xcdd
```

`swift test` doesn't work (SwiftPM does not compile `.metal` resources into
a metallib); use `xcodebuild` so the bundle resource pipeline runs.

## Architecture

- `sort_msd_histogram` + `sort_msd_prep` + `sort_msd_atomic_scatter` — byte 7
  ports verbatim from forge-sort 0.2.3. The MSD scatter's racing-atomic
  ranking is fine here because within-bucket order is later re-sorted by LSD.
- `sort_inner_stable_64` — replaces forge-sort's buggy `sort_inner_fused`.
  Runs one byte pass per dispatch, one threadgroup per MSD bucket.

The inner kernel sorts a bucket via a deterministic split-and-flag bit-sort
in threadgroup memory. Each 2048-element sub-tile gets nine bit passes:
bits 0..7 are the digit bits; bit 8 is a synthetic "is padding" sentinel
that pushes unused slots in the tail sub-tile to the end without polluting
digit 0xFF. Each pass is a two-level exclusive prefix scan
(SIMD-group prefix + serial cross-SG combine) so no atomics race during
rank computation, giving a stable per-sub-tile order. The per-bucket
cross-sub-tile concatenation uses a per-digit running counter (`bkt_run`),
so the overall bucket ordering is stable across sub-tiles too.

Data layout inside a sub-tile is block-major: thread `lid` holds the eight
elements at sub-tile-local positions `[lid*8, lid*8+8)`. This makes the
per-thread internal prefix sum directly correspond to its segment of the
2048-element scan, so the cross-SG combine is a single 8-entry scan rather
than a full 256-wide one.

## History

The first port of forge-sort 0.2.3 was mechanically correct but produced
wrong orderings on real (non-toy) data because forge-sort's
`sort_inner_fused` Phase 4 used a racing `atomic_fetch_add` to compute the
per-element within-SG rank. The atomic *count* was correct, but *which*
element got which rank was non-deterministic, so each per-byte LSD pass was
not stable. LSD requires pass-to-pass stability — the first multi-byte input
broke it (~1% wrong on random u64 keys, more on adversarial probes).
forge-sort's own tests only exercised a 3-element input with trivially-zero
upper bytes and so never hit it.

While writing the replacement, the first cut had a different correctness
bug worth recording: the bit-split's `is_padding` flag was computed from the
*current slot* `i >= tile_valid` rather than the *element's original
position* `cur >= tile_valid`. That works in pass 0 (identity permutation)
but in later passes real elements that moved to slots `i >= tile_valid` were
treated as padding, corrupting all tail sub-tiles. The fix is one line; the
test signal was `testEdgeCases`' "3000 equal keys" case producing zeros in
the first ~2048 positions (sub-tile 1 wrote padding zeros over sub-tile 0's
real writes).

## Future work (queued)

Both are deliberate omissions, not bugs. Document the workaround on the
caller side until/unless we invest in the kernel work.

### Stability across equal keys

The current sort is monotonic on keys but not stable across equal keys.
The MSD scatter (`sort_msd_atomic_scatter`) uses a racing
`atomic_fetch_add` for both cross-TG `tile_base[d]` assignment and within-SG
`within_sg` rank assignment, so two elements with the same MSD digit can
swap positions non-deterministically.

**Caller workaround**: pack the original input index into the low bits of
the key so no two elements share a key. For gsplat the natural key
`(cam_id | tile_id | depth)` typically uses ~50 of 64 bits — the remaining
~14-20 bits cheaply hold `log2(n)` of tiebreaker. Same trick CUB users use
when they want determinism without depending on CUB's stability contract.

**Kernel-side fix (when it's worth it)**: replace the MSD scatter with
deterministic cross-TG and within-TG ordering:
1. Per-TG histogram → global prefix scan across tiles (one extra dispatch)
   gives each TG a stable `tile_base[d]` instead of an atomic-counter slot.
2. Replace Phase 4's racing atomic with the same `sort_inner_stable_64`
   bit-split technique (one TG, 8 bit passes on the digit).

Estimated ~1-2 days, ~10-15% perf cost. The same architecture pattern as
NVIDIA Onesweep, minus the chained-lookback.

### Bit-range hint (`beginBit` / `endBit`)

`cub::DeviceRadixSort::SortPairs` takes `(begin_bit, end_bit)` to skip
known-zero high/low bits. The gsplat call passes `0, 32 + tile_n_bits +
cam_n_bits` — typically ~50 — so CUB skips ~14 bits of work. Our wrapper
always processes all 8 bytes.

The change is byte-aligned (cheap):
- Add `beginBit: Int = 0, endBit: Int = 64` to
  `MetalRadixSortU64Pairs.encode(...)`.
- Derive `msdByte = (endBit - 1) / 8`, `firstByte = beginBit / 8` inside the
  wrapper; thread `msdByte * 8` into `SortParams.shift` and dispatch the
  inner kernel for bytes `[firstByte, msdByte)`. No kernel changes.
- Add a `sort_copy_keys_64` kernel for the parity-mismatch case (when the
  number of inner passes is even, the final result lands in scratch
  instead of the caller's buffer).

For typical gsplat (`endBit=50`) this is 7 dispatches instead of 8, ~12%
faster. For depth-only sort (`endBit=32`), 4 instead of 8, ~50% faster.
Semantics are byte-precise, not bit-precise — `endBit=50` and `endBit=56`
dispatch identically. Documented as a known difference from CUB.

## Useful artifacts

- `Sources/MetalRadixSort/Shaders/sort.metal` — the MSD kernels are AGPL-3.0
  forge-sort code; the `sort_inner_stable_64` kernel is the replacement.
- `Sources/MetalRadixSort/MetalRadixSort.swift` — single-encoder dispatch
  pattern with scratch-buffer allocation and the "argsort + gather" value
  pairing strategy from forge-sort.
- `Tests/MetalRadixSortTests/MetalRadixSortU64PairsTests.swift` — probe
  tests (`testProbeTopByteOnly`, `testProbeLowerByteOnly`,
  `testProbeMidBytesOnly`, `testProbeUpperMidBytesOnly`, `testProbeScales`)
  isolate which stage is responsible for any future regression.
