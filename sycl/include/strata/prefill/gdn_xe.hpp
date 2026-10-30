// sycl/include/strata/prefill/gdn_xe.hpp - the prompt GDN recurrence for Intel GPUs (variants 6 and 7 of
// gdn_recurrence_variant, src/prefill/kernels.dp.cpp).  The column kernel's four row groups and their 32-long FMA
// chains, in the same order, but a 16-lane sub-group holds four columns x four row groups (lane % 4 = column,
// lane / 4 = row group), so the two cross-row-group sums are sub-group shuffles instead of SLM round trips, and a
// token needs one work-group barrier instead of five.  Bit-identical to variant 0 on an Arc A770.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <algorithm>
#include <cstdint>

namespace strata::prefill::detail {
constexpr int GDN_XE_S = 128, GDN_XE_HK = 16, GDN_XE_HV = 48, GDN_XE_C = 10240;
constexpr int GDN_XE_CB = 16, GDN_XE_NCB = GDN_XE_S / GDN_XE_CB;
constexpr int GDN_XE_NT = GDN_XE_CB * 4;

// Do not replace with reduce_over_group: keep ((p0+p1)+p2)+p3,
// exactly the old red[0][c]+red[1][c]+red[2][c]+red[3][c] expression.
inline float gdn_xe_sum4(sycl::sub_group sg, float x, unsigned c) {
    const float p0 = sycl::select_from_group(sg, x, c);
    const float p1 = sycl::select_from_group(sg, x, c + 4);
    const float p2 = sycl::select_from_group(sg, x, c + 8);
    const float p3 = sycl::select_from_group(sg, x, c + 12);
    return p0 + p1 + p2 + p3;
}

inline void gdn_xe_body(sycl::nd_item<1> it, float* __restrict__ state,
                        const float* __restrict__ h, const float* __restrict__ gate,
                        const float* __restrict__ beta, float* __restrict__ out,
                        int64_t T, sycl::local_accessor<float, 1> inputs) {
    const auto sg = it.get_sub_group();
    const unsigned lane = sg.get_local_linear_id();
    const unsigned rg = lane / 4, c = lane % 4;
    const unsigned tid = it.get_local_linear_id();
    const unsigned head = it.get_group_linear_id() / GDN_XE_NCB;
    const unsigned cb = it.get_group_linear_id() % GDN_XE_NCB;
    const unsigned col = cb * GDN_XE_CB + sg.get_group_linear_id() * 4 + c;
    const unsigned qh = head % GDN_XE_HK;
    constexpr unsigned RPG = GDN_XE_S / 4;
    constexpr size_t rs = GDN_XE_HV * GDN_XE_S;
    float* base = state + (rg * RPG * GDN_XE_HV + head) * GDN_XE_S + col;
    float s[RPG];
#pragma unroll
    for (unsigned r = 0; r < RPG; ++r) s[r] = base[r * rs];
    float nq[2], nk[2], nv = 0, ng = 0, nb = 0;
    auto fetch = [&](int64_t t) {
        const float* ht = h + t * GDN_XE_C;
#pragma unroll
        for (unsigned u = 0; u < 2; ++u) {
            nq[u] = ht[qh * GDN_XE_S + tid + u * GDN_XE_NT];
            nk[u] = ht[GDN_XE_HK * GDN_XE_S + qh * GDN_XE_S + tid + u * GDN_XE_NT];
        }
        nv = ht[2 * GDN_XE_HK * GDN_XE_S + head * GDN_XE_S + col];
        ng = gate[t * GDN_XE_HV + head];
        nb = beta[t * GDN_XE_HV + head];
    };
    if (T > 0) fetch(0);
    for (int64_t t = 0; t < T; ++t) {
        // Alternate q/k banks. The barrier at t-1 guarantees every read of
        // bank t-2 has finished before we overwrite it at t. No end barrier.
        const unsigned bank = (t & 1) * 2 * GDN_XE_S;
#pragma unroll
        for (unsigned u = 0; u < 2; ++u) {
            inputs[bank + tid + u * GDN_XE_NT] = nq[u];
            inputs[bank + GDN_XE_S + tid + u * GDN_XE_NT] = nk[u];
        }
        const float cv = nv, cg = ng, cbt = nb;
        it.barrier(sycl::access::fence_space::local_space);
        if (t + 1 < T) fetch(t + 1);
        const float g = sycl::native::exp(cg);
        float kv = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], inputs[bank + GDN_XE_S + rg * RPG + r], kv);
        const float kv_col = gdn_xe_sum4(sg, kv, c);
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < RPG; ++r) {
            s[r] = sycl::fma(g, s[r], inputs[bank + GDN_XE_S + rg * RPG + r] * delta);
            o = sycl::fma(s[r], inputs[bank + rg * RPG + r], o);
        }
        const float oc = gdn_xe_sum4(sg, o, c) * sycl::rsqrt(float(GDN_XE_S));
        if (rg == 0) out[t * GDN_XE_HV * GDN_XE_S + head * GDN_XE_S + col] = oc;
    }
#pragma unroll
    for (unsigned r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// Intel GPUs only (the caller checks gdn_xe_device).  256 GRFs a thread: an offline DG2 (acm-g10) compile of this kernel
// spilled 4000 bytes a thread at the default 128.  Keeping the state in SLM instead of registers removed the spill
// but ran ~4x slower on an A770 (13.7 vs 3.1-3.3 ms at T=1000, same bits), so the state stays in registers.
inline sycl::event gdn_xe_launch(sycl::queue& q, float* state, const float* h,
                                const float* gate, const float* beta, float* out, int64_t T) {
    const auto range = sycl::nd_range<1>(GDN_XE_HV * GDN_XE_NCB * GDN_XE_NT, GDN_XE_NT);
    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> inputs(4 * GDN_XE_S, cgh);   // two q/k banks, 2048 bytes a work-group
        cgh.parallel_for<class GdnXeLargeGrf>(range,
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                gdn_xe_body(it, state, h, gate, beta, out, T, inputs);
            });
    });
}

// Variant 7: the column kernel's arithmetic with the state and a token's q/k rows in registers.  A 16-lane sub-group
// holds 16 columns (a lane each) x one row group (32 rows, 32 GRFs), a work-group the four row groups of one 16-column
// block: 1536 threads at 128 GRFs, six a vector engine (the B70 has 256).  One pass over a thread's rows per token does
// the update, its q^T S partial and the next token's k^T S partial (each a 32-long FMA chain in the column kernel's row
// order); the q/k rows arrive as two coalesced 16-float loads each, two tokens ahead, and reach the FMAs as sub-group
// broadcasts with constant lanes (register regions, no SLM).  The partials meet in SLM (two banks by token parity)
// behind one barrier a token and are summed red[0] + red[1] + red[2] + red[3] as before: the same bits as
// gdn_rec_cols_pipe_kernel (gdn_parity).  B70, T = 16384, 48 heads: 11.1-12.0 ms against 21.3-21.6 ms.  (A sub-group
// holding all 128 rows of its columns, 384 threads at 256 GRFs, ran 21 ms: one thread's chains with nothing to hide
// them, the same time at 12 heads as at 48.)
constexpr int GDN_XE_RCB = 16, GDN_XE_RNCB = GDN_XE_S / GDN_XE_RCB;   // columns per block, blocks per head
inline void gdn_xe_rg_body(sycl::nd_item<1> it, float* __restrict__ state, const float* __restrict__ h,
                           const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ out,
                           int64_t T, float* __restrict__ red) {   // red: [2 banks][2: kv, o][4 rg][16]
    const auto sg = it.get_sub_group();
    const unsigned lane = sg.get_local_linear_id(), rg = sg.get_group_linear_id();
    const unsigned w = it.get_group_linear_id();
    const unsigned head = w / GDN_XE_RNCB, col = (w % GDN_XE_RNCB) * GDN_XE_RCB + lane;
    const unsigned qh = head % GDN_XE_HK;
    constexpr unsigned RPG = GDN_XE_S / 4, NR = RPG / 16;   // rows per row group, 16-row pieces of them
    constexpr size_t rs = GDN_XE_HV * GDN_XE_S;
    constexpr unsigned BANK = 2 * 4 * 16, KO = 4 * 16;
    float* base = state + (size_t) (rg * RPG) * rs + head * GDN_XE_S + col;
    float s[RPG];
#pragma unroll
    for (unsigned r = 0; r < RPG; ++r) s[r] = base[r * rs];
    // the inputs of tokens t (c*), t + 1 (n*: its k rows feed this token's pass) and t + 2 (f*, loading); zeros past T
    float cq[NR], ck[NR], nq[NR], nk[NR], cv = 0, cg = 0, cbt = 0, nv = 0, ng = 0, nb = 0;
    auto fetch = [&](int64_t t, float* q_, float* k_, float& v_, float& g_, float& b_) {
        if (t < T) {
            const float* ht = h + t * GDN_XE_C;
#pragma unroll
            for (unsigned u = 0; u < NR; ++u) {
                q_[u] = ht[qh * GDN_XE_S + rg * RPG + u * 16 + lane];
                k_[u] = ht[GDN_XE_HK * GDN_XE_S + qh * GDN_XE_S + rg * RPG + u * 16 + lane];
            }
            v_ = ht[2 * GDN_XE_HK * GDN_XE_S + head * GDN_XE_S + col];
            g_ = gate[t * GDN_XE_HV + head];
            b_ = beta[t * GDN_XE_HV + head];
        } else {
#pragma unroll
            for (unsigned u = 0; u < NR; ++u) q_[u] = k_[u] = 0.0f;
            v_ = g_ = b_ = 0.0f;
        }
    };
    fetch(0, cq, ck, cv, cg, cbt);
    fetch(1, nq, nk, nv, ng, nb);
    // row r of the row group in a token's k / q: lane r % 16 of piece r / 16
    auto bc = [&](const float* x, unsigned r) { return sycl::select_from_group(sg, x[r / 16], r % 16); };
    float* mine = red + rg * 16 + lane;   // this thread's slot ([bank][kv / o] added)
    const float* part = red + lane;       // the column's four row-group slots
    {   // k^T S of token 0
        float kv = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < RPG; ++r) kv = sycl::fma(s[r], bc(ck, r), kv);
        mine[0] = kv;
    }
    it.barrier(sycl::access::fence_space::local_space);
    float kv_col = part[0] + part[16] + part[32] + part[48];
    for (int64_t t = 0; t < T; ++t) {
        float fq[NR], fk[NR], fv, fg, fb;
        fetch(t + 2, fq, fk, fv, fg, fb);
        const unsigned b0 = (t & 1) * BANK, b1 = BANK - b0;   // this token's bank, the next one's
        const float g = sycl::native::exp(cg);
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f, kv = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < RPG; ++r) {
            s[r] = sycl::fma(g, s[r], bc(ck, r) * delta);
            o = sycl::fma(s[r], bc(cq, r), o);
            kv = sycl::fma(s[r], bc(nk, r), kv);
        }
        // bank b1's kv slots were last read before the previous barrier, bank b0's o slots (row group 0) before it too
        mine[b0 + KO] = o;
        mine[b1] = kv;
        it.barrier(sycl::access::fence_space::local_space);
        kv_col = part[b1] + part[b1 + 16] + part[b1 + 32] + part[b1 + 48];   // the next token's k^T S
        if (rg == 0)
            out[t * GDN_XE_HV * GDN_XE_S + head * GDN_XE_S + col] =
                (part[b0 + KO] + part[b0 + KO + 16] + part[b0 + KO + 32] + part[b0 + KO + 48]) * sycl::rsqrt(float(GDN_XE_S));
#pragma unroll
        for (unsigned u = 0; u < NR; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; nq[u] = fq[u]; nk[u] = fk[u]; }
        cv = nv; cg = ng; cbt = nb; nv = fv; ng = fg; nb = fb;
    }
#pragma unroll
    for (unsigned r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// Intel GPUs only (the caller checks gdn_xe_device), T > 0.  The default 128 GRFs (an IGC dump of the B70 build: no
// spill), 1 KB of SLM a work-group.
inline sycl::event gdn_xe_rg_launch(sycl::queue& q, float* state, const float* h, const float* gate,
                                    const float* beta, float* out, int64_t T) {
    const auto range = sycl::nd_range<1>(GDN_XE_HV * GDN_XE_RNCB * 4 * 16, 4 * 16);
    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> red(2 * 2 * 4 * 16, cgh);
        cgh.parallel_for<class GdnXeRg>(range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            gdn_xe_rg_body(it, state, h, gate, beta, out, T,
                           red.template get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}

// The output norm after variant 7: gdn_out_norm_kernel's arithmetic (its 32-column warp sums in their xor order,
// then wsum[0] + wsum[1] + wsum[2] + wsum[3]) with a 16-lane sub-group per (token, head) row, columns lane + 16 j, over a
// grid-stride walk of the rows (the old launch was a 128-thread work-group per row, T * 48 of them).  The same bits;
// B70, T = 16384: ~2.0 ms against ~2.6-3.2 ms, about 1 GB moved.
inline void gdn_xe_norm_body(sycl::nd_item<1> it, const float* __restrict__ z, const float* __restrict__ gamma,
                             float eps, const float* __restrict__ y, uint16_t* __restrict__ y16, int64_t rows,
                             int64_t ld16) {
    const auto sg = it.get_sub_group();
    const unsigned lane = sg.get_local_linear_id();
    constexpr unsigned NJ = GDN_XE_S / 16;
    float gm[NJ];
#pragma unroll
    for (unsigned j = 0; j < NJ; ++j) gm[j] = gamma[lane + 16 * j];
    const int64_t nsg = (int64_t) (it.get_global_range(0) / 16);
    for (int64_t row = (int64_t) (it.get_global_linear_id() / 16); row < rows; row += nsg) {
        const float* yr = y + row * GDN_XE_S;
        const float* zr = z + row * GDN_XE_S;
        float oc[NJ], zz[NJ];
#pragma unroll
        for (unsigned j = 0; j < NJ; ++j) { oc[j] = yr[lane + 16 * j]; zz[j] = zr[lane + 16 * j]; }
        float b[NJ / 2];   // warp w = columns 32 w .. 32 w + 31: its xor-16 step pairs j = 2 w and 2 w + 1 in a lane
#pragma unroll
        for (unsigned w = 0; w < NJ / 2; ++w) {
            const float a0 = oc[2 * w] * oc[2 * w], a1 = oc[2 * w + 1] * oc[2 * w + 1];
            b[w] = a0 + a1;
        }
#pragma unroll
        for (unsigned o = 8; o > 0; o >>= 1)
#pragma unroll
            for (unsigned w = 0; w < NJ / 2; ++w) b[w] += sycl::permute_group_by_xor(sg, b[w], o);
        const float ss = b[0] + b[1] + b[2] + b[3];
        uint16_t* o16 = y16 + (row / GDN_XE_HV) * ld16 + (row % GDN_XE_HV) * GDN_XE_S;
#pragma unroll
        for (unsigned j = 0; j < NJ; ++j) {
            const float v = oc[j] * sycl::rsqrt(ss / (float) GDN_XE_S + eps) * gm[j] *
                            (1.0f / (1.0f + sycl::native::exp(-zz[j])));
            o16[lane + 16 * j] = sycl::bit_cast<uint16_t>(sycl::vec<float, 1>(v).convert<sycl::half, sycl::rounding_mode::rte>()[0]);
        }
    }
}
inline sycl::event gdn_xe_norm_launch(sycl::queue& q, const float* z, const float* gamma, float eps, const float* y,
                                      uint16_t* y16, int64_t T, int64_t ld16) {
    const int64_t rows = T * GDN_XE_HV;
    const int64_t wgs = std::min<int64_t>((rows + 7) / 8, 1024);   // 8 sub-groups a work-group
    return q.parallel_for<class GdnXeNorm>(sycl::nd_range<1>(wgs * 128, 128),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            gdn_xe_norm_body(it, z, gamma, eps, y, y16, rows, ld16);
        });
}
} // namespace strata::prefill::detail
