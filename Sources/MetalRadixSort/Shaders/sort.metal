// SPDX-License-Identifier: AGPL-3.0-only
// Ported verbatim from forge-sort-0.2.3 (https://crates.io/crates/forge-sort),
// shaders/sort.metal. AGPL-3.0-only. No algorithm changes.

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════
// forge-sort: MSD+fused-inner radix sort (4 dispatches)
//
// Extracted from exp17 Investigation W.
// Architecture: 1 MSD scatter (bits 24:31) → 256 buckets
// → 3 inner LSD passes per bucket at SLC speed.
// Single encoder, 4 dispatches, zero CPU readback.
// ═══════════════════════════════════════════════════════════════════

#define SORT_NUM_BINS  256u
#define SORT_TILE_SIZE 4096u
#define SORT_ELEMS     16u
#define SORT_THREADS   256u
#define SORT_NUM_SGS   8u
#define SORT_MAX_TPB   17u

// 64-bit: half the elements per thread (registers are 2x wider)
#define SORT_ELEMS_64     8u
#define SORT_TILE_64   2048u

struct SortParams {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

struct BucketDesc {
    uint offset;
    uint count;
    uint tile_count;
    uint tile_base;
};

struct InnerParams {
    uint start_shift;
    uint pass_count;
    uint batch_start;
};

// ═══════════════════════════════════════════════════════════════════
// Function constants — specialized at PSO creation time
// Defaults ensure existing kernels are unaffected when unset.
// ═══════════════════════════════════════════════════════════════════

constant bool HAS_VALUES [[function_constant(0)]];
constant bool IS_64BIT   [[function_constant(1)]];
constant uint TRANSFORM_MODE [[function_constant(2)]];
constant bool has_values = is_function_constant_defined(HAS_VALUES) ? HAS_VALUES : false;
constant bool is_64bit   = is_function_constant_defined(IS_64BIT)   ? IS_64BIT   : false;
constant uint transform_mode = is_function_constant_defined(TRANSFORM_MODE) ? TRANSFORM_MODE : 0u;

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: MSD Histogram — single-pass, bits[24:31]
//
// Reads ALL data once, computes 256-bin histogram for MSD byte only.
// Uses per-SG atomic histogram on TG memory.
// Output: global_hist[digit] = total count for digit 0..255
// ═══════════════════════════════════════════════════════════════════

kernel void sort_msd_histogram(
    device const uint*     src          [[buffer(0)]],
    device atomic_uint*    global_hist  [[buffer(1)]],
    constant SortParams&   params       [[buffer(2)]],
    device uint*           tile_hists   [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    uint n = params.element_count;
    uint shift = params.shift;

    // Per-SG accumulator in shared memory
    threadgroup atomic_uint sg_counts[SORT_NUM_SGS * SORT_NUM_BINS]; // 8 KB

    // Zero sg_counts (all 256 threads cooperate)
    for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
        atomic_store_explicit(&sg_counts[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_64bit) {
        // ── 64-bit path: 8 elements/thread, 2048-element tiles ──
        uint base = gid * SORT_TILE_64;
        device const ulong* src64 = reinterpret_cast<device const ulong*>(src);

        ulong keys64[SORT_ELEMS_64];
        bool valid[SORT_ELEMS_64];
        for (uint e = 0u; e < SORT_ELEMS_64; e++) {
            uint idx = base + e * SORT_THREADS + lid;
            valid[e] = idx < n;
            keys64[e] = valid[e] ? src64[idx] : 0ul;
        }

        for (uint e = 0u; e < SORT_ELEMS_64; e++) {
            if (valid[e]) {
                uint digit = (uint)(keys64[e] >> shift) & 0xFFu;
                atomic_fetch_add_explicit(
                    &sg_counts[simd_id * SORT_NUM_BINS + digit],
                    1u, memory_order_relaxed);
            }
        }
    } else {
        // ── 32-bit path: 16 elements/thread, 4096-element tiles ──
        uint base = gid * SORT_TILE_SIZE;

        uint keys[SORT_ELEMS];
        bool valid[SORT_ELEMS];
        for (uint e = 0u; e < SORT_ELEMS; e++) {
            uint idx = base + e * SORT_THREADS + lid;
            valid[e] = idx < n;
            keys[e] = valid[e] ? src[idx] : 0u;
        }

        for (uint e = 0u; e < SORT_ELEMS; e++) {
            if (valid[e]) {
                uint digit = (keys[e] >> shift) & 0xFFu;
                atomic_fetch_add_explicit(
                    &sg_counts[simd_id * SORT_NUM_BINS + digit],
                    1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SGs: 256 threads handle one bin each
    {
        uint total = 0u;
        for (uint sg = 0u; sg < SORT_NUM_SGS; sg++) {
            total += atomic_load_explicit(
                &sg_counts[sg * SORT_NUM_BINS + lid],
                memory_order_relaxed);
        }
        // Emit the per-tile histogram so a downstream prep kernel can
        // compute deterministic cross-tile offsets without atomics.
        tile_hists[gid * SORT_NUM_BINS + lid] = total;
        if (total > 0u) {
            atomic_fetch_add_explicit(&global_hist[lid],
                                      total, memory_order_relaxed);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: MSD Prep — combined prefix sum + bucket descs
//
// Single dispatch, 1 TG, 256 threads.
// Input:  global_hist[256] = per-digit counts
// Output: bucket_descs[256] = offset/count/tile_count for inner sort
//         tile_offsets[num_tiles * 256] = per-(tile,digit) absolute output
//             position for the stable scatter. tile_offsets[t*256+d] =
//             bkt_pfx[d] + sum_of_tile_hists[0..t-1][d].
// ═══════════════════════════════════════════════════════════════════

kernel void sort_msd_prep(
    device const uint*     global_hist  [[buffer(0)]],
    device const uint*     tile_hists   [[buffer(1)]],
    device BucketDesc*     bucket_descs [[buffer(2)]],
    constant uint&         tile_size    [[buffer(3)]],
    constant uint&         num_tiles    [[buffer(4)]],
    device uint*           tile_offsets [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]])
{
    // Thread 0 does serial prefix sum across the 256 digit bins.
    threadgroup uint prefix[SORT_NUM_BINS];

    if (lid == 0u) {
        uint sum = 0u;
        for (uint i = 0u; i < SORT_NUM_BINS; i++) {
            prefix[i] = sum;
            sum += global_hist[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread emits one bucket_desc and runs the cross-tile scan for
    // its assigned digit. The scan gives every tile its exact start slot
    // for digit `lid`, no atomics required during scatter.
    uint count  = global_hist[lid];
    uint offset = prefix[lid];

    uint tc = (count + tile_size - 1u) / tile_size;
    bucket_descs[lid] = BucketDesc{offset, count, tc, 0u};

    // tile_offsets[t * 256 + lid] = offset + sum_{t' < t} tile_hists[t'][lid]
    uint running = 0u;
    for (uint t = 0u; t < num_tiles; t++) {
        tile_offsets[t * SORT_NUM_BINS + lid] = offset + running;
        running += tile_hists[t * SORT_NUM_BINS + lid];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Stable MSD Scatter — bit-split per tile + pre-scanned offsets
//
// Replaces the legacy `sort_msd_atomic_scatter`. The old kernel was
// unstable on two axes: tiles raced for `counters[d]` slots
// (cross-tile non-determinism) and lanes within an SG raced for
// `sg_hist_or_rank[sg*256+d]` slots (within-tile non-determinism). This
// kernel removes both races:
//
//   - Cross-tile: `tile_offsets[gid * 256 + d]` is precomputed by
//     `sort_msd_prep` (deterministic per-digit prefix scan across
//     tiles). Every TG knows its exact start slot for each digit before
//     scatter — no atomic on a global counter.
//
//   - Within-tile: the per-element rank within a digit is computed by
//     the same split-and-flag bit-sort the inner kernel uses (9 bit
//     passes: bits 0..7 of the digit + bit 8 = padding sentinel for the
//     partial last tile). The post-sort position minus the per-tile
//     digit prefix is the deterministic rank.
//
// Output for an element at tile-local position `orig` with digit `d`:
//   dst[tile_offsets[gid * 256 + d] + (sorted_pos - tile_pfx[d])]
//     = src[base + orig]
//
// TG memory: ~20 KB (tg_dig 2KB + tg_idx_a 8KB + tg_idx_b 8KB
//                    + tile_hist_at 1KB + tile_pfx 1KB + scratch).
// ═══════════════════════════════════════════════════════════════════

kernel void sort_msd_stable_scatter(
    device const ulong*       src_keys     [[buffer(0)]],
    device ulong*             dst_keys     [[buffer(1)]],
    device const uint*        tile_offsets [[buffer(2)]],
    constant SortParams&      params       [[buffer(3)]],
    device const uint*        src_vals     [[buffer(4)]],
    device uint*              dst_vals     [[buffer(5)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    constexpr uint TILE     = SORT_TILE_64;
    constexpr uint ELEMS    = TILE / SORT_THREADS;       // 8
    constexpr uint SG_SIZE  = 32u;
    constexpr uint NUM_SGS  = SORT_THREADS / SG_SIZE;    // 8

    uint n          = params.element_count;
    uint shift      = params.shift;
    uint base       = gid * TILE;
    uint tile_valid = (base >= n) ? 0u : min(TILE, n - base);

    if (tile_valid == 0u) return;

    threadgroup uchar       tg_dig[TILE];
    threadgroup uint        tg_idx_a[TILE];
    threadgroup uint        tg_idx_b[TILE];
    threadgroup atomic_uint tile_hist_at[SORT_NUM_BINS];
    threadgroup uint        tile_pfx[SORT_NUM_BINS];
    threadgroup uint        sg_partials[NUM_SGS];
    threadgroup uint        total_ones_tg;

    // ── Phase 1: cache digits, init identity permutation ──
    for (uint e = 0u; e < ELEMS; e++) {
        uint i = lid * ELEMS + e;
        uchar d = 0;
        if (i < tile_valid) {
            ulong k = src_keys[base + i];
            d = (uchar)((k >> shift) & 0xFFu);
        }
        tg_dig[i] = d;
        tg_idx_a[i] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: 9 bit passes — split-and-flag stable sort ──
    // Bits 0..7 are the digit, bit 8 is the padding sentinel. Real
    // elements have key9 = digit (bit 8 = 0); padding has key9 = 0x100
    // (bit 8 = 1). After 9 passes, real elements are sorted by digit
    // and padding has been pushed to the upper [tile_valid, TILE) slots.
    for (uint b = 0u; b < 9u; b++) {
        threadgroup uint* read_idx  = ((b & 1u) == 0u) ? tg_idx_a : tg_idx_b;
        threadgroup uint* write_idx = ((b & 1u) == 0u) ? tg_idx_b : tg_idx_a;

        uint flags[ELEMS];
        uint thread_total = 0u;
        for (uint e = 0u; e < ELEMS; e++) {
            uint i   = lid * ELEMS + e;
            uint cur = read_idx[i];
            uint dig = (uint)tg_dig[cur];
            // is_padding follows the element, not the slot — same fix
            // as in sort_inner_stable_64 (real elements moved into slots
            // i >= tile_valid in later passes would otherwise look like
            // padding).
            bool is_padding = cur >= tile_valid;
            uint key9 = is_padding ? 0x100u : dig;
            uint bit  = (key9 >> b) & 1u;
            flags[e] = bit;
            thread_total += bit;
        }

        // Two-level exclusive prefix scan over TILE flags.
        uint sg_excl = simd_prefix_exclusive_sum(thread_total);
        if (simd_lane == SG_SIZE - 1u) {
            sg_partials[simd_id] = sg_excl + thread_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0u) {
            uint running = 0u;
            for (uint c = 0u; c < NUM_SGS; c++) {
                uint cv = sg_partials[c];
                sg_partials[c] = running;
                running += cv;
            }
            total_ones_tg = running;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint thread_base = sg_partials[simd_id] + sg_excl;
        uint total_ones  = total_ones_tg;
        uint total_zeros = TILE - total_ones;

        uint running = 0u;
        for (uint e = 0u; e < ELEMS; e++) {
            uint i = lid * ELEMS + e;
            uint elem_pfx = thread_base + running;
            uint new_pos = (flags[e] != 0u)
                ? (total_zeros + elem_pfx)
                : (i - elem_pfx);
            running += flags[e];
            write_idx[new_pos] = read_idx[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // After 9 passes (b=8 even), final permutation lives in tg_idx_b.

    // ── Phase 3: per-tile histogram + exclusive prefix ──
    atomic_store_explicit(&tile_hist_at[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = 0u; e < ELEMS; e++) {
        uint i = lid * ELEMS + e;
        if (i < tile_valid) {
            uint orig = tg_idx_b[i];
            uint d = (uint)tg_dig[orig];
            atomic_fetch_add_explicit(&tile_hist_at[d], 1u,
                                      memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint v = atomic_load_explicit(&tile_hist_at[lid], memory_order_relaxed);
        uint sg_excl = simd_prefix_exclusive_sum(v);
        if (simd_lane == SG_SIZE - 1u) sg_partials[simd_id] = sg_excl + v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0u) {
            uint running = 0u;
            for (uint c = 0u; c < NUM_SGS; c++) {
                uint cv = sg_partials[c];
                sg_partials[c] = running;
                running += cv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_pfx[lid] = sg_excl + sg_partials[simd_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: scatter to global dst using pre-scanned offsets ──
    for (uint e = 0u; e < ELEMS; e++) {
        uint i = lid * ELEMS + e;
        if (i < tile_valid) {
            uint orig = tg_idx_b[i];
            uint d = (uint)tg_dig[orig];
            uint dst_idx = tile_offsets[gid * SORT_NUM_BINS + d]
                         + (i - tile_pfx[d]);
            dst_keys[dst_idx] = src_keys[base + orig];
            if (has_values) {
                dst_vals[dst_idx] = src_vals[base + orig];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Fused Inner Sort — 3-pass LSD per bucket
//
// Self-contained: computes own histograms for all 3 inner passes
// during the first scan. No separate precompute dispatch needed.
// 4 dispatches total instead of 5.
//
// Extra TG memory: 2 × 256 atomic_uint = 2KB (pass 1 & 2 histograms)
// Total TG memory: ~22KB (within 32KB limit)
// ═══════════════════════════════════════════════════════════════════

kernel void sort_inner_fused(
    device uint*                buf_a         [[buffer(0)]],
    device uint*                buf_b         [[buffer(1)]],
    device const BucketDesc*    bucket_descs  [[buffer(2)]],
    constant InnerParams&       inner_params  [[buffer(3)]],
    device uint*                vals_a        [[buffer(4)]],
    device uint*                vals_b        [[buffer(5)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    BucketDesc desc = bucket_descs[gid + inner_params.batch_start];
    if (desc.count == 0u) return;

    uint tile_count = desc.tile_count;

    // ═══ Shared memory (22 KB total) ═══
    threadgroup atomic_uint sg_ctr[SORT_NUM_SGS * SORT_NUM_BINS]; // 8 KB (scatter ranking)
    threadgroup uint sg_pfx[SORT_NUM_SGS * SORT_NUM_BINS];        // 8 KB (cross-SG prefix)
    threadgroup uint bkt_hist[SORT_NUM_BINS];                       // 1 KB (current pass histogram)
    threadgroup uint glb_pfx[SORT_NUM_BINS];                        // 1 KB (exclusive prefix sum)
    threadgroup uint run_pfx[SORT_NUM_BINS];                        // 1 KB (running tile prefix)
    threadgroup uint chk_tot[8];                                      // 32 B (prefix sum helper)
    // Self-computed histograms for passes 1 and 2 (accumulated during pass 0)
    threadgroup atomic_uint hist_p1[SORT_NUM_BINS];                  // 1 KB
    threadgroup atomic_uint hist_p2[SORT_NUM_BINS];                  // 1 KB

    // ═══ Zero all histogram accumulators ═══
    atomic_store_explicit(&hist_p1[lid], 0u, memory_order_relaxed);
    atomic_store_explicit(&hist_p2[lid], 0u, memory_order_relaxed);
    // Zero per-SG counters (used for pass 0 histogram accumulation)
    for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
        atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_64bit) {
        // ═══════════════════════════════════════════════════════════
        // 64-bit path: ulong keys, 8 elements/thread, 2048-element tiles
        // Uses inner_params.start_shift and pass_count for variable passes
        // ═══════════════════════════════════════════════════════════
        device ulong* buf_a_64 = reinterpret_cast<device ulong*>(buf_a);
        device ulong* buf_b_64 = reinterpret_cast<device ulong*>(buf_b);

        for (uint pass = 0u; pass < inner_params.pass_count; pass++) {
            uint shift = (inner_params.start_shift + pass) * 8u;

            // Alternate buffers: pass 0: b->a, pass 1: a->b, pass 2: b->a
            device ulong* src = (pass % 2u == 0u) ? buf_b_64 : buf_a_64;
            device ulong* dst = (pass % 2u == 0u) ? buf_a_64 : buf_b_64;

            // Values follow same ping-pong as keys (values are always uint)
            device uint* src_vals = (pass % 2u == 0u) ? vals_b : vals_a;
            device uint* dst_vals = (pass % 2u == 0u) ? vals_a : vals_b;

            // ═══ Load histogram ═══
            if (pass == 0u) {
                bkt_hist[lid] = 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // First pass: read all tiles, compute histograms for up to 3 passes
                for (uint t = 0u; t < tile_count; t++) {
                    for (uint e = 0u; e < SORT_ELEMS_64; e++) {
                        uint local_idx = t * SORT_TILE_64
                                       + simd_id * (SORT_ELEMS_64 * 32u) + e * 32u + simd_lane;
                        if (local_idx < desc.count) {
                            ulong val = src[desc.offset + local_idx];
                            uint d0 = (uint)(val >> shift) & 0xFFu;
                            // Accumulate pass 0 histogram using per-SG accumulation
                            atomic_fetch_add_explicit(&sg_ctr[simd_id * SORT_NUM_BINS + d0],
                                                      1u, memory_order_relaxed);
                            // Pre-compute pass 1 & 2 histograms if they exist
                            if (inner_params.pass_count > 1u) {
                                uint d1 = (uint)(val >> (shift + 8u)) & 0xFFu;
                                atomic_fetch_add_explicit(&hist_p1[d1], 1u, memory_order_relaxed);
                            }
                            if (inner_params.pass_count > 2u) {
                                uint d2 = (uint)(val >> (shift + 16u)) & 0xFFu;
                                atomic_fetch_add_explicit(&hist_p2[d2], 1u, memory_order_relaxed);
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Reduce pass 0 per-SG histogram to per-bucket histogram
                {
                    uint total = 0u;
                    for (uint sg = 0u; sg < SORT_NUM_SGS; sg++) {
                        total += atomic_load_explicit(
                            &sg_ctr[sg * SORT_NUM_BINS + lid],
                            memory_order_relaxed);
                    }
                    bkt_hist[lid] = total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else if (pass == 1u) {
                bkt_hist[lid] = atomic_load_explicit(&hist_p1[lid], memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                bkt_hist[lid] = atomic_load_explicit(&hist_p2[lid], memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // ═══ 256-bin exclusive prefix sum ═══
            {
                uint chunk = lid / 32u;
                uint lane = lid % 32u;
                uint val = bkt_hist[lid];
                uint prefix = simd_prefix_exclusive_sum(val);

                if (lane == 31u) chk_tot[chunk] = prefix + val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lid == 0u) {
                    uint running = 0u;
                    for (uint c = 0u; c < 8u; c++) {
                        uint ct = chk_tot[c];
                        chk_tot[c] = running;
                        running += ct;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                glb_pfx[lid] = prefix + chk_tot[chunk];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ═══ Serial scatter with running cross-tile prefix ═══
            run_pfx[lid] = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Zero sg_ctr before scatter
            for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
                atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint t = 0u; t < tile_count; t++) {
                uint tile_base_off = desc.offset + t * SORT_TILE_64;

                ulong keys64[SORT_ELEMS_64];
                uint v_vals[SORT_ELEMS_64];
                uint digits[SORT_ELEMS_64];
                bool valid[SORT_ELEMS_64];
                for (uint e = 0u; e < SORT_ELEMS_64; e++) {
                    uint local_idx = t * SORT_TILE_64
                                   + simd_id * (SORT_ELEMS_64 * 32u) + e * 32u + simd_lane;
                    valid[e] = local_idx < desc.count;
                    uint idx = tile_base_off + simd_id * (SORT_ELEMS_64 * 32u) + e * 32u + simd_lane;
                    keys64[e] = valid[e] ? src[idx] : 0xFFFFFFFFFFFFFFFFul;
                    digits[e] = valid[e] ? (uint)((keys64[e] >> shift) & 0xFFu) : 0xFFu;
                    if (has_values) {
                        v_vals[e] = valid[e] ? src_vals[idx] : 0u;
                    }
                }

                for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
                    atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint e = 0u; e < SORT_ELEMS_64; e++) {
                    if (valid[e]) {
                        atomic_fetch_add_explicit(
                            &sg_ctr[simd_id * SORT_NUM_BINS + digits[e]],
                            1u, memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                uint tile_digit_count;
                {
                    uint total = 0u;
                    for (uint sg = 0u; sg < SORT_NUM_SGS; sg++) {
                        uint c = atomic_load_explicit(
                            &sg_ctr[sg * SORT_NUM_BINS + lid],
                            memory_order_relaxed);
                        sg_pfx[sg * SORT_NUM_BINS + lid] = total;
                        total += c;
                    }
                    tile_digit_count = total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
                    atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint e = 0u; e < SORT_ELEMS_64; e++) {
                    if (valid[e]) {
                        uint d = digits[e];
                        uint within_sg = atomic_fetch_add_explicit(
                            &sg_ctr[simd_id * SORT_NUM_BINS + d],
                            1u, memory_order_relaxed);
                        uint dst_idx = desc.offset
                                     + glb_pfx[d]
                                     + run_pfx[d]
                                     + sg_pfx[simd_id * SORT_NUM_BINS + d]
                                     + within_sg;
                        dst[dst_idx] = keys64[e];
                        if (has_values) {
                            dst_vals[dst_idx] = v_vals[e];
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                run_pfx[lid] += tile_digit_count;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Device memory barrier: ensure scatter writes visible for next pass reads
            threadgroup_barrier(mem_flags::mem_device);
        }
    } else {
        // ═══════════════════════════════════════════════════════════
        // 32-bit path: uint keys, 16 elements/thread, 4096-element tiles
        // ═══════════════════════════════════════════════════════════
        for (uint pass = 0u; pass < inner_params.pass_count; pass++) {
            uint shift = (inner_params.start_shift + pass) * 8u;

            // Alternate buffers: pass 0: b->a, pass 1: a->b, pass 2: b->a
            device uint* src = (pass % 2u == 0u) ? buf_b : buf_a;
            device uint* dst = (pass % 2u == 0u) ? buf_a : buf_b;

            // Values follow same ping-pong as keys
            device uint* src_vals = (pass % 2u == 0u) ? vals_b : vals_a;
            device uint* dst_vals = (pass % 2u == 0u) ? vals_a : vals_b;

            // ═══ Load histogram ═══
            if (pass == 0u) {
                // Pass 0: compute histogram via first scan through data
                bkt_hist[lid] = 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // First pass: read all tiles, compute histograms for ALL 3 passes
                for (uint t = 0u; t < tile_count; t++) {
                    for (uint e = 0u; e < SORT_ELEMS; e++) {
                        uint local_idx = t * SORT_TILE_SIZE
                                       + simd_id * (SORT_ELEMS * 32u) + e * 32u + simd_lane;
                        if (local_idx < desc.count) {
                            uint val = src[desc.offset + local_idx];
                            uint d0 = val & 0xFFu;
                            uint d1 = (val >> 8u) & 0xFFu;
                            uint d2 = (val >> 16u) & 0xFFu;
                            // Accumulate pass 0 histogram using per-SG accumulation
                            atomic_fetch_add_explicit(&sg_ctr[simd_id * SORT_NUM_BINS + d0],
                                                      1u, memory_order_relaxed);
                            // Pass 1 & 2: direct TG atomic (256 bins, avg 1 collision — fast)
                            atomic_fetch_add_explicit(&hist_p1[d1], 1u, memory_order_relaxed);
                            atomic_fetch_add_explicit(&hist_p2[d2], 1u, memory_order_relaxed);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Reduce pass 0 per-SG histogram to per-bucket histogram
                {
                    uint total = 0u;
                    for (uint sg = 0u; sg < SORT_NUM_SGS; sg++) {
                        total += atomic_load_explicit(
                            &sg_ctr[sg * SORT_NUM_BINS + lid],
                            memory_order_relaxed);
                    }
                    bkt_hist[lid] = total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else if (pass == 1u) {
                bkt_hist[lid] = atomic_load_explicit(&hist_p1[lid], memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                bkt_hist[lid] = atomic_load_explicit(&hist_p2[lid], memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // ═══ 256-bin exclusive prefix sum ═══
            {
                uint chunk = lid / 32u;
                uint lane = lid % 32u;
                uint val = bkt_hist[lid];
                uint prefix = simd_prefix_exclusive_sum(val);

                if (lane == 31u) chk_tot[chunk] = prefix + val;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lid == 0u) {
                    uint running = 0u;
                    for (uint c = 0u; c < 8u; c++) {
                        uint ct = chk_tot[c];
                        chk_tot[c] = running;
                        running += ct;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                glb_pfx[lid] = prefix + chk_tot[chunk];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ═══ Serial scatter with running cross-tile prefix ═══
            run_pfx[lid] = 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Zero sg_ctr before scatter
            for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
                atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint t = 0u; t < tile_count; t++) {
                uint tile_base = desc.offset + t * SORT_TILE_SIZE;

                uint keys[SORT_ELEMS];
                uint vals[SORT_ELEMS];
                uint digits[SORT_ELEMS];
                bool valid[SORT_ELEMS];
                for (uint e = 0u; e < SORT_ELEMS; e++) {
                    uint local_idx = t * SORT_TILE_SIZE
                                   + simd_id * (SORT_ELEMS * 32u) + e * 32u + simd_lane;
                    valid[e] = local_idx < desc.count;
                    uint idx = tile_base + simd_id * (SORT_ELEMS * 32u) + e * 32u + simd_lane;
                    keys[e] = valid[e] ? src[idx] : 0xFFFFFFFFu;
                    digits[e] = valid[e] ? ((keys[e] >> shift) & 0xFFu) : 0xFFu;
                    if (has_values) {
                        vals[e] = valid[e] ? src_vals[idx] : 0u;
                    }
                }

                for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
                    atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint e = 0u; e < SORT_ELEMS; e++) {
                    if (valid[e]) {
                        atomic_fetch_add_explicit(
                            &sg_ctr[simd_id * SORT_NUM_BINS + digits[e]],
                            1u, memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                uint tile_digit_count;
                {
                    uint total = 0u;
                    for (uint sg = 0u; sg < SORT_NUM_SGS; sg++) {
                        uint c = atomic_load_explicit(
                            &sg_ctr[sg * SORT_NUM_BINS + lid],
                            memory_order_relaxed);
                        sg_pfx[sg * SORT_NUM_BINS + lid] = total;
                        total += c;
                    }
                    tile_digit_count = total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = lid; i < SORT_NUM_SGS * SORT_NUM_BINS; i += SORT_THREADS) {
                    atomic_store_explicit(&sg_ctr[i], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint e = 0u; e < SORT_ELEMS; e++) {
                    if (valid[e]) {
                        uint d = digits[e];
                        uint within_sg = atomic_fetch_add_explicit(
                            &sg_ctr[simd_id * SORT_NUM_BINS + d],
                            1u, memory_order_relaxed);
                        uint dst_idx = desc.offset
                                     + glb_pfx[d]
                                     + run_pfx[d]
                                     + sg_pfx[simd_id * SORT_NUM_BINS + d]
                                     + within_sg;
                        dst[dst_idx] = keys[e];
                        if (has_values) {
                            dst_vals[dst_idx] = vals[e];
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                run_pfx[lid] += tile_digit_count;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Device memory barrier: ensure scatter writes visible for next pass reads
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4b: Stable Inner Sort (64-bit) — single-byte LSD per bucket
//
// Replaces sort_inner_fused for correctness.
//
// The original kernel's Phase 4 used racing `atomic_fetch_add` on TG
// memory to compute per-element `within_sg` ranks during scatter. The
// atomic ordering is non-deterministic across lanes, which made each
// pass non-stable. LSD radix sort requires pass-to-pass stability, so
// across multiple byte passes the kernel produced ~1% wrong orderings
// on random data. (Diagnosis & verification: see git log + STATUS.md.)
//
// This kernel replaces the per-tile rank with a deterministic bit-split
// (a.k.a. split-and-flag) sort in threadgroup memory. For each
// 2048-element sub-tile, 9 sequential bit-passes partition elements
// with bit=0 to the lower half and bit=1 to the upper half via a
// 2048-wide exclusive prefix scan over flag bits — fully deterministic.
// The 9th bit is a synthetic "is padding" sentinel that sorts unused
// slots in the tail sub-tile to the end without polluting digit 0xFF.
//
// One dispatch per inner byte pass; one TG per MSD bucket. Each TG
// loops over its bucket's sub-tiles in order, holding a per-digit
// running counter so that within-bucket cross-sub-tile concatenation
// remains stable.
//
// Data layout: block-major within sub-tile. Thread `lid` holds the
// 8 elements at sub-tile local positions [lid*8, lid*8+8). This makes
// the per-thread internal prefix correctly correspond to the per-thread
// segment of a 2048-element stable order, and a single 2-level scan
// (SIMD-group prefix + cross-SG combine) gives the global tile prefix.
//
// TG memory:
//   tg_dig[2048]         uchar  2KB  (cached digits for current byte)
//   tg_idx_a[2048]       uint   8KB  (permutation buffer A)
//   tg_idx_b[2048]       uint   8KB  (permutation buffer B)
//   bkt_hist_at[256]     atomic 1KB  (bucket-wide histogram for byte)
//   bkt_pfx[256]         uint   1KB  (bucket-wide exclusive prefix)
//   bkt_run[256]         uint   1KB  (cross-sub-tile running counter)
//   tile_hist_at[256]    atomic 1KB  (per-sub-tile hist on sorted)
//   tile_pfx[256]        uint   1KB  (per-sub-tile exclusive prefix)
//   sg_partials[8]       uint    32B
//   total_ones_tg        uint     4B
//   Total: ~23 KB (within 32 KB)
// ═══════════════════════════════════════════════════════════════════

kernel void sort_inner_stable_64(
    device const ulong*         src_keys      [[buffer(0)]],
    device ulong*               dst_keys      [[buffer(1)]],
    device const BucketDesc*    bucket_descs  [[buffer(2)]],
    constant InnerParams&       inner_params  [[buffer(3)]],
    device const uint*          src_vals      [[buffer(4)]],
    device uint*                dst_vals      [[buffer(5)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    BucketDesc desc = bucket_descs[gid + inner_params.batch_start];
    if (desc.count == 0u) return;

    constexpr uint SUB_TILE = 2048u;
    constexpr uint ELEMS    = SUB_TILE / SORT_THREADS;   // 8
    constexpr uint SG_SIZE  = 32u;
    constexpr uint NUM_SGS  = SORT_THREADS / SG_SIZE;    // 8

    uint shift        = inner_params.start_shift * 8u;
    uint base_offset  = desc.offset;
    uint count        = desc.count;
    uint num_sub_tiles = (count + SUB_TILE - 1u) / SUB_TILE;

    threadgroup uchar      tg_dig[SUB_TILE];
    threadgroup uint       tg_idx_a[SUB_TILE];
    threadgroup uint       tg_idx_b[SUB_TILE];
    threadgroup atomic_uint bkt_hist_at[SORT_NUM_BINS];
    threadgroup uint       bkt_pfx[SORT_NUM_BINS];
    threadgroup uint       bkt_run[SORT_NUM_BINS];
    threadgroup atomic_uint tile_hist_at[SORT_NUM_BINS];
    threadgroup uint       tile_pfx[SORT_NUM_BINS];
    threadgroup uint       sg_partials[NUM_SGS];
    threadgroup uint       total_ones_tg;

    // ── Phase 1: bucket-wide histogram for current byte ──
    atomic_store_explicit(&bkt_hist_at[lid], 0u, memory_order_relaxed);
    bkt_run[lid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0u; t < num_sub_tiles; t++) {
        uint sub_tile_off = t * SUB_TILE;
        for (uint e = 0u; e < ELEMS; e++) {
            uint local_idx = sub_tile_off + lid * ELEMS + e;
            if (local_idx < count) {
                ulong k = src_keys[base_offset + local_idx];
                uint d = (uint)((k >> shift) & 0xFFu);
                atomic_fetch_add_explicit(&bkt_hist_at[d], 1u,
                                          memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: 256-bin exclusive prefix of bkt_hist → bkt_pfx ──
    {
        uint v = atomic_load_explicit(&bkt_hist_at[lid], memory_order_relaxed);
        uint sg_excl = simd_prefix_exclusive_sum(v);
        if (simd_lane == SG_SIZE - 1u) sg_partials[simd_id] = sg_excl + v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0u) {
            uint running = 0u;
            for (uint c = 0u; c < NUM_SGS; c++) {
                uint cv = sg_partials[c];
                sg_partials[c] = running;
                running += cv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bkt_pfx[lid] = sg_excl + sg_partials[simd_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: serial sub-tile loop ──
    for (uint t = 0u; t < num_sub_tiles; t++) {
        uint sub_tile_off = t * SUB_TILE;
        uint tile_valid = min(SUB_TILE, count - sub_tile_off);

        // 3a. Cache digits in TG memory; init permutation buffer A as identity
        for (uint e = 0u; e < ELEMS; e++) {
            uint i = lid * ELEMS + e;
            uint local_idx = sub_tile_off + i;
            uchar d = 0;
            if (local_idx < count) {
                ulong k = src_keys[base_offset + local_idx];
                d = (uchar)((k >> shift) & 0xFFu);
            }
            tg_dig[i] = d;
            tg_idx_a[i] = i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3b. 9 bit-passes: bits 0..7 are digit bits; bit 8 is a synthetic
        // "is padding" sentinel that sorts unused slots to the end.
        for (uint b = 0u; b < 9u; b++) {
            threadgroup uint* read_idx  = ((b & 1u) == 0u) ? tg_idx_a : tg_idx_b;
            threadgroup uint* write_idx = ((b & 1u) == 0u) ? tg_idx_b : tg_idx_a;

            uint flags[ELEMS];
            uint thread_total = 0u;
            for (uint e = 0u; e < ELEMS; e++) {
                uint i = lid * ELEMS + e;
                uint cur = read_idx[i];
                uint dig = (uint)tg_dig[cur];
                // is_padding follows the element, not the slot — the element
                // that came from sub-tile-local position `cur >= tile_valid`
                // is a placeholder. Checking slot `i` would only work in pass 0
                // (identity permutation) and would treat real elements that
                // moved to slots i >= tile_valid as padding in later passes.
                bool is_padding = cur >= tile_valid;
                uint key9 = is_padding ? 0x100u : dig;
                uint bit = (key9 >> b) & 1u;
                flags[e] = bit;
                thread_total += bit;
            }

            // 2-level exclusive prefix scan over 2048 flags.
            // Stage 1: SG-local exclusive prefix of per-thread totals.
            uint sg_excl = simd_prefix_exclusive_sum(thread_total);
            // Lane 31 of each SG writes the SG's total.
            if (simd_lane == SG_SIZE - 1u) {
                sg_partials[simd_id] = sg_excl + thread_total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Stage 2: thread 0 computes the exclusive prefix across SGs.
            if (lid == 0u) {
                uint running = 0u;
                for (uint c = 0u; c < NUM_SGS; c++) {
                    uint cv = sg_partials[c];
                    sg_partials[c] = running;
                    running += cv;
                }
                total_ones_tg = running;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint thread_base = sg_partials[simd_id] + sg_excl;
            uint total_ones  = total_ones_tg;
            uint total_zeros = SUB_TILE - total_ones;

            uint running = 0u;
            for (uint e = 0u; e < ELEMS; e++) {
                uint i = lid * ELEMS + e;
                uint elem_pfx = thread_base + running;
                uint new_pos = (flags[e] != 0u)
                    ? (total_zeros + elem_pfx)
                    : (i - elem_pfx);
                running += flags[e];
                write_idx[new_pos] = read_idx[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // After 9 passes (b=8 even), final permutation is in tg_idx_b.

        // 3c. Per-sub-tile histogram on sorted-valid range → tile_pfx
        atomic_store_explicit(&tile_hist_at[lid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint e = 0u; e < ELEMS; e++) {
            uint i = lid * ELEMS + e;
            if (i < tile_valid) {
                uint orig = tg_idx_b[i];
                uint d = (uint)tg_dig[orig];
                atomic_fetch_add_explicit(&tile_hist_at[d], 1u,
                                          memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            uint v = atomic_load_explicit(&tile_hist_at[lid], memory_order_relaxed);
            uint sg_excl = simd_prefix_exclusive_sum(v);
            if (simd_lane == SG_SIZE - 1u) sg_partials[simd_id] = sg_excl + v;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0u) {
                uint running = 0u;
                for (uint c = 0u; c < NUM_SGS; c++) {
                    uint cv = sg_partials[c];
                    sg_partials[c] = running;
                    running += cv;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tile_pfx[lid] = sg_excl + sg_partials[simd_id];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3d. Scatter sorted elements to global dst.
        // Valids occupy sorted positions [0, tile_valid). For each such i,
        // the element's source local index is tg_idx_b[i], its digit is
        // tg_dig[tg_idx_b[i]], and the destination is
        //   base + bkt_pfx[d] + bkt_run[d] + (i - tile_pfx[d]).
        for (uint e = 0u; e < ELEMS; e++) {
            uint i = lid * ELEMS + e;
            if (i < tile_valid) {
                uint orig = tg_idx_b[i];
                uint d = (uint)tg_dig[orig];
                uint dst_idx = base_offset + bkt_pfx[d] + bkt_run[d]
                             + (i - tile_pfx[d]);
                uint src_local = sub_tile_off + orig;
                dst_keys[dst_idx] = src_keys[base_offset + src_local];
                if (has_values) {
                    dst_vals[dst_idx] = src_vals[base_offset + src_local];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3e. Advance per-digit running counter by this sub-tile's count.
        bkt_run[lid] += atomic_load_explicit(&tile_hist_at[lid],
                                             memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Ensure scatter writes are visible to subsequent passes.
    threadgroup_barrier(mem_flags::mem_device);
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4c: ulong key copy — dst[i] = src[i] for parity-driven landing
//
// The Swift wrapper alternates src/dst between bufA (caller's keys)
// and bufB (scratch) for each byte pass. With seven inner passes the
// final result lands back in bufA; with any other inner-pass count the
// result ends up in scratch and needs to be copied. This kernel is
// dispatched only when that parity mismatch occurs.
//
// Simple 1D dispatch: 1 thread per element.
// ═══════════════════════════════════════════════════════════════════

kernel void sort_copy_keys_64(
    device const ulong* src   [[buffer(0)]],
    device ulong*       dst   [[buffer(1)]],
    constant uint&      count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) dst[gid] = src[gid];
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 5: Index Initializer — fill indices[i] = i for argsort
//
// Simple 1D dispatch: 1 thread per element.
// ═══════════════════════════════════════════════════════════════════

kernel void sort_init_indices(
    device uint*   indices [[buffer(0)]],
    constant uint& count   [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) indices[gid] = gid;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 6: Gather Values — rearrange values by sorted index permutation
//
// gathered_vals[i] = original_vals[sorted_indices[i]]
// Simple 1D dispatch: 1 thread per element.
// ═══════════════════════════════════════════════════════════════════

kernel void sort_gather_values(
    device const uint* sorted_indices [[buffer(0)]],
    device const uint* original_vals  [[buffer(1)]],
    device uint*       gathered_vals  [[buffer(2)]],
    constant uint&     count          [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) gathered_vals[gid] = original_vals[sorted_indices[gid]];
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 7: 32-bit Key Transform — pre/post sort bit manipulation
//
// mode 0: XOR 0x80000000 (i32 sign flip, self-inverse)
// mode 1: FloatFlip forward (map float order → unsigned order)
// mode 2: IFloatFlip inverse (map unsigned order → float order)
//
// Simple 1D dispatch: 1 thread per element.
// ═══════════════════════════════════════════════════════════════════

kernel void sort_transform_32(
    device uint*       data  [[buffer(0)]],
    constant uint&     count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint v = data[gid];

    if (transform_mode == 0u) {
        // i32: flip sign bit (self-inverse)
        v ^= 0x80000000u;
    } else if (transform_mode == 1u) {
        // FloatFlip forward: negative (sign set) → flip all; positive → flip sign only
        v = (v & 0x80000000u) ? ~v : (v ^ 0x80000000u);
    } else if (transform_mode == 2u) {
        // IFloatFlip inverse: sign set (was positive) → flip sign; sign clear (was negative) → flip all
        v = (v & 0x80000000u) ? (v ^ 0x80000000u) : ~v;
    }

    data[gid] = v;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 8: 64-bit Key Transform — pre/post sort bit manipulation
//
// mode 1: FloatFlip64 forward (also works for i64 sign-XOR)
//         if sign bit set: flip all 64 bits (negative float/i64)
//         else: flip only sign bit (positive float/i64)
//         Mathematically: FloatFlip for f64, XOR 0x8000000000000000 for i64
// mode 2: IFloatFlip64 inverse
//         if sign bit set (was positive): flip only sign bit
//         else (was negative): flip all 64 bits
//
// Simple 1D dispatch: 1 thread per element.
// ═══════════════════════════════════════════════════════════════════

kernel void sort_transform_64(
    device ulong*      data  [[buffer(0)]],
    constant uint&     count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ulong v = data[gid];

    if (transform_mode == 0u) {
        // i64: flip sign bit (self-inverse)
        v ^= 0x8000000000000000UL;
    } else if (transform_mode == 1u) {
        // FloatFlip64 forward: negative (sign set) → flip all; positive → flip sign only
        v = (v & 0x8000000000000000UL) ? ~v : (v ^ 0x8000000000000000UL);
    } else if (transform_mode == 2u) {
        // IFloatFlip64 inverse: sign set (was positive) → flip sign; sign clear (was negative) → flip all
        v = (v & 0x8000000000000000UL) ? (v ^ 0x8000000000000000UL) : ~v;
    }

    data[gid] = v;
}
