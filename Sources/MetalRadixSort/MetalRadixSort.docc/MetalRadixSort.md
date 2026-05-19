# ``MetalRadixSort``

A stable, GPU-resident radix sort for `(UInt64 key, UInt32 value)` pairs on Metal.

## Overview

`MetalRadixSort` sorts up to ~1M paired `UInt64` keys with `UInt32`
payload values directly on a Metal device — no CPU readback, no
intermediate copies, one command buffer. On Apple Silicon (M-series),
1M pairs sort in roughly 3–4 ms.

The library exposes a single type: ``MetalRadixSortU64Pairs``. You
construct it once per device with an upper-bound element count
(scratch buffers and pipeline state objects are sized at construction
time), then call ``MetalRadixSortU64Pairs/encode(onto:keys:values:count:beginBit:endBit:)``
on each sort. The caller owns the command buffer and decides when to
commit.

```swift
import Metal
import MetalRadixSort

let device = MTLCreateSystemDefaultDevice()!
let queue  = device.makeCommandQueue()!
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

### Algorithm

The sort is a hybrid MSD+LSD radix sort over the eight bytes of the
`UInt64` key:

- **MSD pass on byte 7** distributes elements into 256 buckets by
  exclusive-prefix-sum offsets, using a per-tile atomic scatter
  (`sort_msd_histogram` → `sort_msd_prep` → `sort_msd_atomic_scatter`).
- **Seven LSD passes on bytes 0..6** sort each bucket in place via the
  custom `sort_inner_stable_64` kernel — one dispatch per byte, one
  threadgroup per bucket. Within a threadgroup, each 2048-element
  sub-tile is sorted by a deterministic split-and-flag bit-sort
  (no racing atomics for rank computation), keeping each pass stable
  so LSD chaining is correct.

Value pairing uses the "argsort + gather" strategy: an internal index
buffer is permuted alongside the keys, then used at the end to gather
the caller's values into the caller's values buffer.

### Limits and assumptions

- Keys must be `UInt64`, values `UInt32`. The Metal shaders are
  specialised via function constants at PSO compile time.
- Sort is **ascending by key**. There is no descending mode.
- Equal keys are not guaranteed to preserve their *input* order — the
  inner kernel is stable per pass but the MSD scatter is not stable
  within a bucket. Use the values to recover original indices if the
  caller needs an explicit permutation.
- The maximum element count is fixed at construction time; exceeding
  it triggers a `precondition` in
  ``MetalRadixSortU64Pairs/encode(onto:keys:values:count:beginBit:endBit:)``.

## Topics

### Sorting key-value pairs

- ``MetalRadixSortU64Pairs``
- ``MetalRadixSortU64Pairs/init(device:maxElements:)``
- ``MetalRadixSortU64Pairs/encode(onto:keys:values:count:beginBit:endBit:)``

### Errors

- ``MetalRadixSortU64Pairs/Error``
