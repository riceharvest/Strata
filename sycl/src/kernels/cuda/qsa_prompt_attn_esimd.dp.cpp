// SYCL port: the prompt attention on XMX, v3 (ESIMD, Battlemage, INT8 KV) - qsa_prompt_attn_xmx.dp.cpp's v2 ran 4.7x slower
// than the per-cell FP32 fallback (its local-memory staging and work-group barriers, ~120 KB of local memory: one work-group a
// Xe core). Here K and V go from global memory straight into DPAS operands, 16 selected cells a step.
//
// A work-group of 4 hardware threads per (query position, KV head). Thread t computes the scores and the online softmax of
// 3 of the 12 query heads, and O += P V of all 12 heads over 64 of the 256 dims (V scale group t); P (and the softmax
// rescales) pass through 6 KB of local memory, one barrier a step (two buffers):
//  * Scores on INT8 DPAS, exact: S = Q K^T with A = Q as 3 int8 limbs per 64 dims (q ~ s (q0 + q1 / 254 + q2 / 254^2),
//    |qi| <= 127, s = the group's largest / 127: 2^-24 of it, FP32 level; 3 heads x 3 limbs = 9 rows, kept in registers) and
//    B = K^T, gathered in VNNI order by one 8-dword-per-cell gather a 32 dims (a K row's dword i holds dims 4i..4i+3: B's
//    packed (dims, cell) element). The int32 sums of a 64-dim scale group are exact; they are weighted in FP32 by the limb,
//    the Q scale and the softmax scale (folded together) and by the cell's K scale.
//  * Online softmax (exp2) over the 16 cells of a step.
//  * O += P V on FP16 DPAS, as the CUDA kernel: P times the cell's V scale per 64 dims, relative to a running power of 2 per
//    group (2^14 / that bound: every rescale exact), split into hi + lo FP16 halves, both into one FP32 accumulator; V's int8
//    codes are exact in FP16, converted and VNNI-packed (two cells a dword) in registers.
// v3's first form (one thread per 4 heads, no local memory, 2048 queries at 118K: 6.0 ms) spent its time on the instructions
// around the DPAS, not on the loads or the DPAS (bring-up probes: 5.1 ms without the K / V loads, ~6 without the DPAS; ~1,900
// instructions a step, ~640 of them the V conversion, which all three threads of a (query, KV head) repeated).
// FP32-level accuracy, another summation order than the per-cell kernel (qsa_prompt_attn_parity: vs FP64 and vs the warp
// kernel). Masked cells (a negative page, KV streaming) weigh 0, as in the per-cell kernel.
#include "strata/kernels/qsa_prompt_attn_xmx.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#ifndef STRATA_I8_NO_ESIMD
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#endif
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace strata::kernels {
#ifndef STRATA_I8_NO_ESIMD
namespace {
using namespace sycl::ext::intel::esimd;
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using half = sycl::half;
constexpr int HD = 256, G = 12, NT = 4, HT = G / NT, QS = 320, CC = 16;
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_prompt_attn_esimd is compiled for core/geometry.hpp");
static_assert(KV_Q8_GROUP == 64, "four K / V scales a row");

struct Args {
    const float* q;
    const uint32_t* kq;     // int8 K codes, read as dwords
    const int8_t* vq;
    const uint32_t* ks;     // fp16 scales, 4 a row (two dwords)
    const uint32_t* vs;
    const int32_t* table;
    const int32_t* ids;
    const int32_t* steps;
    float* attn;
    int nkv, ps_shift, cap;
    float scale_log2;
    int pf;                 // prefetch the K / V rows two steps ahead into L1
};
constexpr auto kPfL1 = sycl::ext::oneapi::experimental::properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};

// 2^k as a float, k in [-126, 127]
SYCL_ESIMD_FUNCTION inline float pow2i(int k) { return sycl::bit_cast<float>((uint32_t) (127 + k) << 23); }

// the pool rows of the 16 selected cells at c0; a cell past the width or on a negative page: row 0, not live
SYCL_ESIMD_FUNCTION inline void rows_at(const Args& a, const int32_t* ids, int n, int c0, int kvh, simd<int, CC>& row,
                                        simd_mask<CC>& live) {
    const simd<int, CC> lane(0, 1);
    live = (lane + c0) < n;
    simd<int, CC> cell = gather<int, CC>(ids, simd<uint32_t, CC>((lane + c0) * 4), live);
    cell.merge(simd<int, CC>(0), !live);
    simd<int, CC> page = gather<int, CC>(a.table, simd<uint32_t, CC>((cell >> a.ps_shift) * 4), live);
    page.merge(simd<int, CC>(-1), !live);
    live = page >= 0;
    row = ((page * a.nkv + kvh) << a.ps_shift) | (cell & ((1 << a.ps_shift) - 1));
    row.merge(simd<int, CC>(0), !live);
}

// the K and V rows and the scales of 16 cells into L1: thread t of the work-group the t-th 64-byte line of each row
SYCL_ESIMD_FUNCTION inline void prefetch_rows(const Args& a, simd<int, CC> row, int t) {
    const simd<uint32_t, CC> rowu = row;
    const simd<uint32_t, CC> roff = rowu * (uint32_t) HD + (uint32_t) (64 * t);
    prefetch<uint32_t, CC>(a.kq, roff, kPfL1);
    prefetch<uint32_t, CC>(reinterpret_cast<const uint32_t*>(a.vq), roff, kPfL1);
    if (t == 0) {
        prefetch<uint32_t, CC>(a.ks, rowu * 8u, kPfL1);
        prefetch<uint32_t, CC>(a.vs, rowu * 8u, kPfL1);
    }
}

// the local memory of a work-group: two buffers (steps alternate) of P' [group][hi, lo][12 heads][16 cells] halves + alpha [thread][4],
// then the final l [thread][4]
constexpr int SLM_P = 4 * 2 * G * CC * 2, SLM_BUF = SLM_P + NT * 4 * 4, SLM_L = 2 * SLM_BUF, SLM_BYTES = SLM_L + NT * 4 * 4;

SYCL_ESIMD_FUNCTION inline void attn_thread(const Args& a, int qi, int kvh, int t) {
    const int n_head = a.nkv * G;
    const size_t head0 = (size_t) qi * n_head + (size_t) kvh * G;   // the KV head's first query head
    // Q of this thread's 3 heads: three int8 limbs per (head, 64 dims), the A operands of the 8 K steps of 32 dims:
    // [step][row = limb * 3 + head][32] (a step padded to 320 bytes: the operands start on a register)
    simd<int8_t, 8 * QS> qa;
    simd<float, 4 * HT * 3> wq;   // [group][head][limb]: s * scale_log2 / 254^limb
    {
        const float* qp = a.q + (head0 + (size_t) t * HT) * HD;
#pragma unroll
        for (int h = 0; h < HT; ++h) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const simd<float, 64> x = block_load<float, 64>(qp + h * HD + g * 64);
                const float amax = es::hmax<float>(es::abs(x));
                const float inv = amax > 1e-30f ? 127.0f / amax : 0.0f;   // (a group below that: zero)
                simd<float, 64> u = x * inv;
                const simd<float, 64> l0 = es::rnde<float>(u);
                u = (u - l0) * 254.0f;
                const simd<float, 64> l1 = es::rnde<float>(u);
                u = (u - l1) * 254.0f;
                const simd<float, 64> l2 = es::rnde<float>(u);
                const float s = amax * (1.0f / 127.0f) * a.scale_log2;
                wq.template select<1, 1>((g * HT + h) * 3 + 0) = s;
                wq.template select<1, 1>((g * HT + h) * 3 + 1) = s * (1.0f / 254.0f);
                wq.template select<1, 1>((g * HT + h) * 3 + 2) = s * (1.0f / 64516.0f);
                simd<int8_t, 64> c0 = l0, c1 = l1, c2 = l2;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int kk = 2 * g + j;
                    qa.template select<32, 1>(kk * QS + (0 * HT + h) * 32) = c0.template select<32, 1>(32 * j);
                    qa.template select<32, 1>(kk * QS + (1 * HT + h) * 32) = c1.template select<32, 1>(32 * j);
                    qa.template select<32, 1>(kk * QS + (2 * HT + h) * 32) = c2.template select<32, 1>(32 * j);
                }
            }
        }
    }
    const int n = a.steps[(size_t) qi * kStepCount + kStepWidth];
    const int32_t* ids = a.ids + (size_t) qi * a.cap;
    // O of all 12 heads over this thread's 64 dims (group t): [dim tile][12 rows x 16 dims: 8 + 4 rows], in units of
    // 2^(ev[t] - 14) of the V scales
    simd<float, 4 * G * CC> O = 0.0f;
    simd<float, 4> m = -INFINITY, l = 0.0f;   // this thread's 3 heads (and a pad)
    simd<int, 4> ev = -1000;                    // per V scale group: every V scale seen so far is below 2^ev (-1000: none yet)
    // the rows one and two steps ahead are looked up a step early (their loads under this step's work), and the second is
    // prefetched at the end of the step
    simd<int, CC> row, row1, row2;
    simd_mask<CC> live, live1, live2;
    rows_at(a, ids, n, 0, kvh, row, live);
    rows_at(a, ids, n, CC, kvh, row1, live1);
    if (a.pf) prefetch_rows(a, row1, t);
    int buf = 0;
    for (int c0 = 0; c0 < n; c0 += CC) {
        rows_at(a, ids, n, c0 + 2 * CC, kvh, row2, live2);
        const simd<uint32_t, CC> rowu = row;
        const simd<uint32_t, CC> roff = rowu * (uint32_t) HD;   // byte offsets (a pool below 4 GB)
        // this thread's V dims of the 16 rows (their loads under the scores)
        simd<uint8_t, CC * 64> vr;
#pragma unroll
        for (int c = 0; c < CC; ++c) {
            const int rc = (int) row[c];
            vr.template select<64, 1>(c * 64).template bit_cast_view<uint32_t>() =
                block_load<uint32_t, 16>(reinterpret_cast<const uint32_t*>(a.vq + (size_t) rc * HD + t * 64));
        }
        // the K / V scales: dword e of a row holds groups 2e, 2e + 1: [e][cell][2] halves
        simd<uint32_t, 2 * CC> ksw = gather<uint32_t, 2 * CC, 2>(a.ks, rowu * 8u);
        simd<uint32_t, 2 * CC> vsw = gather<uint32_t, 2 * CC, 2>(a.vs, rowu * 8u);
        simd<half, 4 * CC> ksh = ksw.template bit_cast_view<half>();
        simd<half, 4 * CC> vsh = vsw.template bit_cast_view<half>();
        // ---- scores of this thread's 3 heads: S[h][cell], log2 units
        simd<float, CC> sc[HT];
#pragma unroll
        for (int h = 0; h < HT; ++h) sc[h] = 0.0f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            simd<int, 8 * CC> c8;
            simd<int, CC> c1;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int kk = 2 * g + j;
                simd<uint32_t, 8 * CC> bk = gather<uint32_t, 8 * CC, 8>(a.kq, roff + (uint32_t) (kk * 32));
                const simd<int8_t, 4 * 8 * CC> b = bk.template bit_cast_view<int8_t>();
                const simd<int8_t, 256> a8 = qa.template select<256, 1>(kk * QS);
                const simd<int8_t, 32> a1 = qa.template select<32, 1>(kk * QS + 256);
                if (j == 0) {
                    c8 = xmx::dpas<8, 8, int, int8_t, int8_t>(b, a8);
                    c1 = xmx::dpas<8, 1, int, int8_t, int8_t>(b, a1);
                } else {
                    c8 = xmx::dpas<8, 8, int, int, int8_t, int8_t>(c8, b, a8);
                    c1 = xmx::dpas<8, 1, int, int, int8_t, int8_t>(c1, b, a1);
                }
            }
            const simd<float, CC> ksg = ksh.template select<CC, 2>((g >> 1) * 2 * CC + (g & 1));
#pragma unroll
            for (int h = 0; h < HT; ++h) {   // rows limb * 3 + head: 0-7 in c8, row 8 (limb 2 of head 2) in c1
                const simd<float, CC> f0 = c8.template select<CC, 1>(h * CC);
                const simd<float, CC> f1 = c8.template select<CC, 1>((HT + h) * CC);
                simd<float, CC> f2;
                if (h < 2) f2 = c8.template select<CC, 1>((2 * HT + h) * CC);
                else f2 = c1;
                const int w = (g * HT + h) * 3;
                const float w0 = wq[w], w1 = wq[w + 1], w2 = wq[w + 2];
                const simd<float, CC> tt = f0 * w0 + f1 * w1 + f2 * w2;
                sc[h] += tt * ksg;
            }
        }
        // ---- the V scale bounds (every thread the same) and the online softmax of this thread's heads
        float beta = 1.0f;   // this thread's group
        simd<float, CC> vsg[4];
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            vsg[g] = vsh.template select<CC, 2>((g >> 1) * 2 * CC + (g & 1));
            vsg[g].merge(simd<float, CC>(0.0f), !live);
            const float vmax = es::hmax<float>(vsg[g]);
            const int eo = (int) ev[g];
            if (vmax > 0.0f) {
                const int e = (int) ((sycl::bit_cast<uint32_t>(vmax) >> 23) & 0xFFu) - 126;   // vmax < 2^e
                if (e > eo) {
                    if (eo != -1000 && g == t) beta = pow2i(eo - e);
                    ev.template select<1, 1>(g) = e;
                }
            }
            const int en = (int) ev[g];
            vsg[g] *= pow2i(14 - en < 127 ? 14 - en : 127);
        }
        simd<float, 4> alpha = 1.0f;
        simd<float, CC> p[HT];
#pragma unroll
        for (int h = 0; h < HT; ++h) {
            sc[h].merge(simd<float, CC>(-INFINITY), !live);
            const float mx = es::hmax<float>(sc[h]);
            const float mo = (float) m[h];
            const float mn = mo > mx ? mo : mx;
            if (mn > mo) {
                alpha.template select<1, 1>(h) = es::exp2<float>(mo - mn);
                m.template select<1, 1>(h) = mn;
            }
            p[h] = es::exp2(sc[h] - mn);
            p[h].merge(simd<float, CC>(0.0f), !live);
            const float lo = l[h], al = alpha[h];
            l.template select<1, 1>(h) = lo * al + es::reduce<float>(p[h], std::plus<>());
        }
        // ---- P' = P x V scale per group, hi + lo halves of this thread's 3 rows, and its alphas, to the step's buffer
        const uint32_t sb = (uint32_t) (buf * SLM_BUF);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            simd<half, HT * CC> hi3, lo3;
#pragma unroll
            for (int h = 0; h < HT; ++h) {
                const simd<float, CC> pv = p[h] * vsg[g];
                const simd<half, CC> hi = pv;
                const simd<float, CC> hif = hi;
                hi3.template select<CC, 1>(h * CC) = hi;
                lo3.template select<CC, 1>(h * CC) = simd<half, CC>(pv - hif);
            }
#pragma unroll
            for (int hl = 0; hl < 2; ++hl) {   // rows 3t, 3t + 1 and 3t + 2
                const uint32_t o = sb + (uint32_t) (((g * 2 + hl) * G + t * HT) * CC * 2);
                simd<half, HT * CC>& v = hl ? lo3 : hi3;
                slm_block_store<half, 2 * CC>(o, v.template select<2 * CC, 1>(0));
                slm_block_store<half, CC>(o + 2 * CC * 2, v.template select<CC, 1>(2 * CC));
            }
        }
        slm_block_store<float, 4>(sb + (uint32_t) (SLM_P + t * 16), alpha);
        es::barrier();
        // ---- O += P' V over this thread's 64 dims: the A operands of all 12 heads (rows 0-7, 8-11) of group t
        simd<half, 8 * CC> ah8 = slm_block_load<half, 8 * CC>(sb + (uint32_t) (((t * 2 + 0) * G) * CC * 2));
        simd<half, 4 * CC> ah4 = slm_block_load<half, 4 * CC>(sb + (uint32_t) (((t * 2 + 0) * G + 8) * CC * 2));
        simd<half, 8 * CC> al8 = slm_block_load<half, 8 * CC>(sb + (uint32_t) (((t * 2 + 1) * G) * CC * 2));
        simd<half, 4 * CC> al4 = slm_block_load<half, 4 * CC>(sb + (uint32_t) (((t * 2 + 1) * G + 8) * CC * 2));
        {   // one rescale when a running max (of any thread's heads) or this group's V bound moved
            simd<float, NT * 4> f = slm_block_load<float, NT * 4>(sb + (uint32_t) SLM_P);
            f *= beta;
            if ((f != 1.0f).any()) {
#pragma unroll
                for (int r = 0; r < G; ++r) {
                    const float fr = f[(r / HT) * 4 + r % HT];
#pragma unroll
                    for (int n4 = 0; n4 < 4; ++n4) O.template select<CC, 1>(n4 * G * CC + r * CC) *= fr;
                }
            }
        }
#pragma unroll
        for (int n4 = 0; n4 < 4; ++n4) {
            // the code bytes zero-extended into the pair's words, then FP16 1024 + (code + 128) by the bits and minus
            // 1152: the code, exact (one move a cell row, where a byte -> FP16 conversion took three)
            simd<uint16_t, 256> bw;
#pragma unroll
            for (int kr = 0; kr < 8; ++kr)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    bw.template select<CC, 2>(kr * 32 + j) = simd<uint8_t, CC>(vr.template select<CC, 1>((2 * kr + j) * 64 + n4 * 16));
            simd<uint32_t, 128> bd = bw.template bit_cast_view<uint32_t>();
            bd ^= 0x64806480u;
            simd<half, 256> b = bd.template bit_cast_view<half>();
            b -= half(1152.0f);
            simd<float, 8 * CC> o8 = O.template select<8 * CC, 1>(n4 * G * CC);
            simd<float, 4 * CC> o4 = O.template select<4 * CC, 1>(n4 * G * CC + 8 * CC);
            o8 = xmx::dpas<8, 8, float, float, half, half>(o8, b, ah8);
            o8 = xmx::dpas<8, 8, float, float, half, half>(o8, b, al8);
            o4 = xmx::dpas<8, 4, float, float, half, half>(o4, b, ah4);
            o4 = xmx::dpas<8, 4, float, float, half, half>(o4, b, al4);
            O.template select<8 * CC, 1>(n4 * G * CC) = o8;
            O.template select<4 * CC, 1>(n4 * G * CC + 8 * CC) = o4;
        }
        if (a.pf) prefetch_rows(a, row2, t);
        row = row1;
        live = live1;
        row1 = row2;
        live1 = live2;
        buf ^= 1;
    }
    // ---- out: every head's l from its thread, O / l, back from the V scale units
    slm_block_store<float, 4>((uint32_t) (SLM_L + t * 16), l);
    es::barrier();
    simd<float, NT * 4> la = slm_block_load<float, NT * 4>((uint32_t) SLM_L);
    const int e = (int) ev[t];
    const float vf = e == -1000 ? 0.0f : pow2i(e - 14);
    float* op = a.attn + head0 * HD + t * 64;
#pragma unroll
    for (int r = 0; r < G; ++r) {
        const float lr = la[(r / HT) * 4 + r % HT];
        const float f = lr > 0.0f ? vf / lr : 0.0f;
        simd<float, 64> o;
#pragma unroll
        for (int n4 = 0; n4 < 4; ++n4) o.template select<CC, 1>(n4 * CC) = O.template select<CC, 1>(n4 * G * CC + r * CC) * f;
        block_store<float, 64>(op + (size_t) r * HD, o);
    }
}

void launch(const Args& args, int64_t n_q, sycl::queue* st) {
    // (256 registers: Q's limbs, O and a step's V in registers; the 128-register mode spilled, 11.1 ms against 4.6)
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        Args a = args;
        a.q += q0 * a.nkv * G * HD;
        a.attn += q0 * a.nkv * G * HD;
        a.ids += q0 * a.cap;
        a.steps += q0 * kStepCount;
        st->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>((size_t) nb, (size_t) a.nkv, NT), sycl::range<3>(1, 1, NT)), props,
                             [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                                 slm_init<SLM_BYTES>();
                                 attn_thread(a, (int) it.get_global_id(0), (int) it.get_global_id(1), (int) it.get_local_id(2));
                             });
        });
    }
}
}  // namespace
#endif

bool qsa_prompt_attn_esimd(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
#ifdef STRATA_I8_NO_ESIMD
    (void) q; (void) pools; (void) ids; (void) steps; (void) cap; (void) s; (void) attn; (void) n_q; (void) stream;
    return false;
#else
    if (n_q <= 0) return false;
    // INT8 K and V with their scales only (not FP16, Q4_0 or the hybrid K8V4 pools)
    if (pools.k_q == nullptr || pools.v_q == nullptr || pools.k_scale == nullptr || pools.v_scale == nullptr ||
        pools.k_q4 != nullptr || pools.v_q4 != nullptr || !pools.page_table || !ids || !steps || cap <= 0)
        return false;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || s.page_size < 1 || (s.page_size & (s.page_size - 1)) != 0)
        return false;
    sycl::queue* st = strata::q_of(stream);
    // the default is Battlemage only (the card it was measured on); STRATA_PROMPT_ATTN_XMX=3 set explicitly: any device with
    // ESIMD and the s8 8x16x32 DPAS (Xe2)
    static std::mutex mu;
    static std::unordered_map<sycl::device, bool> per_dev;   // a layer split has one prompt path per card
    bool usable = false;
    {
        std::lock_guard<std::mutex> lk(mu);
        const sycl::device d = st->get_device();
        auto it = per_dev.find(d);
        if (it == per_dev.end()) {
            namespace ex = sycl::ext::oneapi::experimental;
            namespace mx = sycl::ext::oneapi::experimental::matrix;
            bool ok = d.is_gpu() && d.has(sycl::aspect::ext_intel_esimd) && d.has(sycl::aspect::fp16);
            if (ok) {
                const auto arch = d.get_info<ex::info::device::architecture>();
                ok = std::getenv("STRATA_PROMPT_ATTN_XMX") != nullptr || arch == ex::architecture::intel_gpu_bmg_g21 ||
                     arch == ex::architecture::intel_gpu_bmg_g31;
            }
            if (ok) {
                bool s8 = false;
                for (const auto& c : d.get_info<ex::info::device::matrix_combinations>())
                    if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 && c.nsize == 16 && c.ksize == 32) s8 = true;
                ok = s8;
            }
            it = per_dev.emplace(d, ok).first;
        }
        usable = it->second;
    }
    if (!usable) return false;
    Args a;
    a.q = q;
    a.kq = reinterpret_cast<const uint32_t*>(pools.k_q);
    a.vq = pools.v_q;
    a.ks = reinterpret_cast<const uint32_t*>(pools.k_scale);
    a.vs = reinterpret_cast<const uint32_t*>(pools.v_scale);
    a.table = pools.page_table;
    a.ids = ids;
    a.steps = steps;
    a.attn = attn;
    a.nkv = (int) s.n_head_kv;
    int sh = 0;
    while ((1ll << sh) < s.page_size) ++sh;
    a.ps_shift = sh;
    a.cap = (int) cap;
    a.scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    // STRATA_PA3_PF=0: no L1 prefetch (2048 queries: 4.6 ms against 4.6-4.9 at 118K, 1.40 against 1.13 at 1.5K; the engine's
    // selections at 114,688: 37.6 against 38.4 ms for 16,384 queries)
    static const int pf = [] { const char* v = std::getenv("STRATA_PA3_PF"); return v ? std::atoi(v) : 1; }();
    a.pf = pf;
    launch(a, n_q, st);
    return true;
#endif
}
}  // namespace strata::kernels
