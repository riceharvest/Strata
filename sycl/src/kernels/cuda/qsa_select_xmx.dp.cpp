// SYCL port: the prompt path's QSA block scores on XMX (ESIMD DPAS, Battlemage) - see qsa_select_xmx.hpp.
//
// The same GEMM as prefill.cpp's oneMKL tiles - rows (block) x columns (query, indexer head), K = 128 - with the relu-sum
// over the four heads in the epilogue, so the [blocks x 4 * queries] product never goes through memory (the FP32 path
// writes and re-reads it: 121 MB a 256-query batch at 118K). FP32-level as the CUDA build's 3xTF32 scorer: every key
// row and query row is scaled by a power of two (its largest value just under 2^14) and split into FP16 hi + lo halves
// (x = hi + lo to 2^-23 relative), the products hi * hi + hi * lo + lo * hi run on DPAS with FP32 accumulation (exact
// 22-bit products; the dropped lo * lo is 2^-22 of the product) and the scales are undone exactly after the sum. Another
// summation order than the FP32 GEMM, so not bitwise: a near-tie can select differently (qsa_select_bench counts them).
//
// The keys are split once per layer and chunk (they are final after the chunk's indexer appends), the batch's queries
// once per batch by the kernel that also scores each query's tail block n_bid (`dead`, +1e9). The scorer reads both
// with 2D block loads: the keys as the DPAS A operand (8 blocks x 16 k a tile, row-major as stored), the queries as B
// through a transposed dword load (16 rows x 8 dwords -> the VNNI layout). A thread owns 32 blocks x 64 rows (16 queries).
#include "strata/kernels/qsa_select_xmx.hpp"
#include "strata/core/geometry.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
static_assert(IDX_DIM == strata::core::kCompiledGeometry.idx_key_dim && IDX_HEADS == strata::core::kCompiledGeometry.idx_q_heads &&
                  R == strata::core::kCompiledGeometry.idx_block,
              "qsa_select_xmx is compiled for core/geometry.hpp");
constexpr int ROW_BYTES = 4 * IDX_DIM;   // a split row: 128 hi halves, then 128 lo halves
constexpr int TM = 32, TN = 64;          // the padding of the scratch: a thread tile is at most 32 blocks x 64 query rows,
constexpr int WM = 4;                    // a work-group 128 blocks

constexpr uint64_t align64(uint64_t b) { return (b + 63) & ~(uint64_t) 63; }
constexpr int64_t round_up(int64_t n, int64_t m) { return (n + m - 1) / m * m; }
// scratch: [query rows: max_q * 4 x 512 B][their inverse scales][key rows: key_blocks x 512 B][their inverse scales]
struct Offsets {
    uint64_t qrows = 0, qinv = 0, keys = 0, kinv = 0, end = 0;
};
Offsets offsets(int64_t key_blocks, int64_t max_q) {
    Offsets o;
    const int64_t rows = round_up(max_q * IDX_HEADS, TN), blocks = round_up(key_blocks, TM * WM);
    o.qinv = align64((uint64_t) rows * ROW_BYTES);
    o.keys = align64(o.qinv + (uint64_t) rows * 4);
    o.kinv = align64(o.keys + (uint64_t) blocks * ROW_BYTES);
    o.end = align64(o.kinv + (uint64_t) blocks * 4);
    return o;
}

bool xmx_device(sycl::queue* q) {
    static int ok = -1;   // the prompt path runs on one device; a layer split's second card is the same model
    if (ok < 0) {
        namespace ex = sycl::ext::oneapi::experimental;
        const sycl::device d = q->get_device();
        bool bmg = false;
        if (d.is_gpu()) {
            const auto a = d.get_info<ex::info::device::architecture>();
            bmg = a == ex::architecture::intel_gpu_bmg_g21 || a == ex::architecture::intel_gpu_bmg_g31;
        }
        ok = bmg ? 1 : 0;
    }
    return ok == 1;
}

// one sub-group of 16 per row: x * 2^(14 - e) with |x| < 2^e for the row's largest, split into hi + lo halves
inline void split_row(sycl::sub_group sg, const float* v8, sycl::half* dst, float* inv, int lane) {
    float m = 0.0f;
    for (int i = 0; i < 8; ++i) m = sycl::fmax(m, sycl::fabs(v8[i]));
    m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
    int e = 0;   // frexp's exponent from the bits (m < 2^e), clamped so the scale stays a finite power of two
    if (m > 0.0f) e = sycl::clamp((int) ((sycl::bit_cast<uint32_t>(m) >> 23) & 0xFFu) - 126, -100, 120);
    const float up = sycl::ldexp(1.0f, 14 - e);
    for (int i = 0; i < 8; ++i) {
        const float y = v8[i] * up;
        const sycl::half hi = sycl::half(y);
        dst[lane * 8 + i] = hi;
        dst[IDX_DIM + lane * 8 + i] = sycl::half(y - (float) hi);
    }
    if (lane == 0) *inv = sycl::ldexp(1.0f, e - 14);
}

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using half = sycl::half;

template <int TM, int TN>
SYCL_ESIMD_FUNCTION inline void score_tile(const half* keys, const float* kinv, const half* qrows, const float* qinv,
                                           const int32_t* steps, int nq, int active, int64_t max_blocks, float* scores,
                                           int bt, int rt) {
    constexpr int MI = TM / 8, NJ = TN / 16;
    const int b0 = bt * TM, r0 = rt * TN;
    if (b0 >= active || r0 >= nq * IDX_HEADS) return;
    // surfaces (bytes - 1): rows past the last block / query row read as zeros
    const unsigned w = ROW_BYTES - 1, kh = (unsigned) active - 1, qh = (unsigned) (nq * IDX_HEADS) - 1;
    const uint32_t* qd = reinterpret_cast<const uint32_t*>(qrows);
    simd<float, TM * TN> acc = 0.0f;
#pragma unroll
    for (int kk = 0; kk < IDX_DIM / 16; ++kk) {
        simd<half, TM * 16> ah, al;
#pragma unroll
        for (int m = 0; m < TM; m += 32) {
            ah.template select<32 * 16, 1>(m * 16) = load_2d<half, 16, 32>(keys, w, kh, w, 16 * kk, b0 + m);
            al.template select<32 * 16, 1>(m * 16) = load_2d<half, 16, 32>(keys, w, kh, w, IDX_DIM + 16 * kk, b0 + m);
        }
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            simd<uint32_t, 128> bh = load_2d<uint32_t, 8, 16, 1, true>(qd, w, qh, w, 8 * kk, r0 + 16 * j);
            simd<uint32_t, 128> bl = load_2d<uint32_t, 8, 16, 1, true>(qd, w, qh, w, IDX_DIM / 2 + 8 * kk, r0 + 16 * j);
            const simd<half, 256> bhh = bh.bit_cast_view<half>(), blh = bl.bit_cast_view<half>();
            // the three products in rounds over the MI accumulators: no DPAS waits on the one before it
#pragma unroll
            for (int p = 0; p < 3; ++p) {
#pragma unroll
                for (int i = 0; i < MI; ++i) {
                    const simd<half, 128> a = p < 2 ? ah.template select<128, 1>(128 * i) : al.template select<128, 1>(128 * i);
                    simd<float, 128> c = acc.template select<128, 1>((i * NJ + j) * 128);
                    c = xmx::dpas<8, 8, float, float, half, half>(c, p == 1 ? blh : bhh, a);
                    acc.template select<128, 1>((i * NJ + j) * 128) = c;
                }
            }
        }
    }
    // epilogue: undo the scales, relu per head, the heads summed in order; blocks below each query's n_bid stored
    simd<float, TM> ks = block_load<float, TM>(kinv + b0);
    const simd<uint32_t, TM> lane_b(0, 1);
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        simd<float, 16> qs = block_load<float, 16>(qinv + r0 + 16 * j);
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            const int q = (r0 + 16 * j) / IDX_HEADS + qq;
            if (q >= nq) break;
            const simd<int32_t, kStepCount> st = block_load<int32_t, kStepCount>(steps + (int64_t) q * kStepCount);
            const int n_bid = st[kStepNBid];
            if (b0 >= n_bid) continue;
            simd<float, TM> v;
#pragma unroll
            for (int i = 0; i < MI; ++i) {
                simd<float, 128> c = acc.template select<128, 1>((i * NJ + j) * 128);
                const simd<float, 8> kb = ks.template select<8, 1>(8 * i);
                simd<float, 8> sum = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const int col = 4 * qq + h;
                    simd<float, 8> d = c.template select<8, 16>(col) * kb * (float) qs[col];
                    d.merge(simd<float, 8>(0.0f), !(d > 0.0f));
                    sum += d;
                }
                v.template select<8, 1>(8 * i) = sum;
            }
            const simd_mask<TM> keep = (lane_b + (uint32_t) b0) < (uint32_t) n_bid;
            scatter<float, TM>(scores + (int64_t) q * max_blocks + b0, lane_b * 4u, v, keep);
        }
    }
}

template <int, int> class ScoreTileTag;
}  // namespace

uint64_t qsa_select_xmx_bytes(int64_t key_blocks, int64_t max_q) { return offsets(key_blocks, max_q).end; }

bool qsa_select_xmx_keys(const QsaSelectXmx& x, const float* pooled, const QsaShapes& s, void* stream) {
    sycl::queue* q = strata::q_of(stream);
    // the 2D block loads need a 64-byte aligned base (prefill.cpp's arena hands out 256-byte multiples)
    if (!x.scratch || ((uintptr_t) x.scratch & 63) != 0 || x.key_blocks <= 0 || s.idx_dim != IDX_DIM ||
        s.idx_n_head != IDX_HEADS || s.idx_block != R || !xmx_device(q))
        return false;
    const Offsets o = offsets(x.key_blocks, x.max_q);
    uint8_t* base = (uint8_t*) x.scratch;
    half* keys = (half*) (base + o.keys);
    float* kinv = (float*) (base + o.kinv);
    const int64_t n = x.key_blocks;
    q->parallel_for(sycl::nd_range<1>((size_t) round_up(n, 8) * 16, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t row = (int64_t) (it.get_global_id(0) / 16);
        if (row >= n) return;   // the whole sub-group: one row
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_id()[0];
        float v[8];
        const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(pooled + row * IDX_DIM + lane * 8);
        const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(pooled + row * IDX_DIM + lane * 8 + 4);
        v[0] = a.x(); v[1] = a.y(); v[2] = a.z(); v[3] = a.w(); v[4] = b.x(); v[5] = b.y(); v[6] = b.z(); v[7] = b.w();
        split_row(sg, v, keys + row * (ROW_BYTES / 2), kinv + row, lane);
    });
    return true;
}

bool qsa_block_scores_xmx(const QsaSelectXmx& x, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                          int64_t max_blocks, int64_t active_blocks, float* scores, void* stream) {
    sycl::queue* q = strata::q_of(stream);
    if (nq <= 0) return true;
    if (!x.scratch || ((uintptr_t) x.scratch & 63) != 0 || ((uintptr_t) steps & 15) != 0 || nq > x.max_q ||
        active_blocks <= 0 || active_blocks > x.key_blocks || active_blocks > max_blocks || !xmx_device(q))
        return false;
    const Offsets o = offsets(x.key_blocks, x.max_q);
    uint8_t* base = (uint8_t*) x.scratch;
    half* qrows = (half*) (base + o.qrows);
    float* qinv = (float*) (base + o.qinv);
    const half* keys = (const half*) (base + o.keys);
    const float* kinv = (const float*) (base + o.kinv);
    // the batch's query rows split, and each query's tail block: block_scores_kernel's rule (`dead`, +1e9 with cells);
    // its value never decides a selection (+1e9 always selects it, a block without cells is never selected)
    q->parallel_for(sycl::nd_range<1>((size_t) round_up(nq, 8) * 16, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t qi = (int64_t) (it.get_global_id(0) / 16);
        if (qi >= nq) return;
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_id()[0];
        float dv[8], v[8];
        for (int i = 0; i < 8; ++i) dv[i] = dead[lane * 8 + i];
        float score = 0.0f;
        for (int h = 0; h < IDX_HEADS; ++h) {
            const float* src = q_idx + (qi * IDX_HEADS + h) * IDX_DIM + lane * 8;
            float p = 0.0f;
            for (int i = 0; i < 8; ++i) { v[i] = src[i]; p += v[i] * dv[i]; }
            split_row(sg, v, qrows + (qi * IDX_HEADS + h) * (ROW_BYTES / 2), qinv + qi * IDX_HEADS + h, lane);
            const float d = sycl::reduce_over_group(sg, p, sycl::plus<float>());
            score += d > 0.0f ? d : 0.0f;
        }
        if (lane == 0) {
            const int64_t n_kv = steps[qi * kStepCount + kStepNKv], n_bid = steps[qi * kStepCount + kStepNBid];
            if (n_bid < max_blocks) {
                if (n_kv % R != 0) score += 1e9f;
                scores[qi * max_blocks + n_bid] = score;
            }
        }
    });
    // a thread's 32 blocks x 64 query rows in 256 registers, work-groups of 128 blocks x 256 rows; B70, 118K, 16,384
    // queries: 19.7 ms, against 20.6 with the next k step prefetched to L1 and 24.6 for 32 x 32 tiles in 128 registers
    constexpr int TMv = 32, TNv = 64, WMv = 128 / TMv, WNv = 256 / TNv;
    const int nqi = (int) nq, act = (int) active_blocks;
    const int ntm = (int) ((active_blocks + TMv - 1) / TMv), ntn = (int) ((nq * IDX_HEADS + TNv - 1) / TNv);
    const size_t gm = (size_t) round_up(ntm, WMv), gn = (size_t) round_up(ntn, WNv);
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q->submit([&](sycl::handler& h) {
        h.parallel_for<ScoreTileTag<TMv, TNv>>(sycl::nd_range<2>(sycl::range<2>(gm, gn), sycl::range<2>(WMv, WNv)), props,
                                               [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
                                                   score_tile<TMv, TNv>(keys, kinv, qrows, qinv, steps, nqi, act, max_blocks,
                                                                        scores, (int) it.get_global_id(0), (int) it.get_global_id(1));
                                               });
    });
    return true;
}
}  // namespace strata::kernels
