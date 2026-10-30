// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/native_router.hpp"
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__dpct_inline__ float warp_sum(float value) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += sycl::permute_group_by_xor(sg, value, mask);
    return value;
}
__dpct_inline__ float warp_max(float value) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = sycl::fmax(value, sycl::permute_group_by_xor(sg, value, mask));
    return value;
}
/*
DPCT1110: The total declared local variable size in device function route
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
template <int NE>   // SYCL port: 512 (canonical) or 256 (the Coder) experts, 32 lanes x NE/32 values
__dpct_inline__ void route(const float *__restrict__ logits,
                           int32_t *__restrict__ ids,
                           float *__restrict__ weights) {
    // Preserve the pinned 32x8 block geometry; only row zero is active here.
    // blockIdx.x = the token (a multi-token launch; 0 for the single one)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    logits += (size_t)item_ct1.get_group(2) * NE;
        ids += (size_t)item_ct1.get_group(2) * 10;
        weights += (size_t)item_ct1.get_group(2) * 10;
    if (item_ct1.get_local_id(1) != 0) return;
    const int lane = item_ct1.get_local_id(2);
    float values[NE / 32];
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) values[i] = logits[lane + i * 32];
    // Only subgroup row zero reaches this point (the other rows returned above): a work-group barrier here is
    // divergent and hangs on Alchemist (A770); the subgroup barrier is the one the surviving row can meet.
    sycl::group_barrier(item_ct1.get_sub_group());
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) maximum = sycl::max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) {
        values[i] = sycl::native::exp(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) {
        values[i] *= reciprocal;
        if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < NE / 32; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            const float other = sycl::permute_group_by_xor(sg, best, mask);
            const int other_id = sycl::permute_group_by_xor(sg, expert, mask);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = sycl::max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
/*
DPCT1110: The total declared local variable size in device function
route_multi exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void route_multi(const float *__restrict__ logits,
                                 int32_t *__restrict__ ids,
                                 float *__restrict__ weights, int n_tok) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tk =
        (int)item_ct1.get_group(2) * 8 + (int)item_ct1.get_local_id(1);
    if (tk >= n_tok) return;
    logits += (size_t) tk * 512; ids += (size_t) tk * 10; weights += (size_t) tk * 10;
    const int lane = item_ct1.get_local_id(2);
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = sycl::max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = sycl::native::exp(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            const float other = sycl::permute_group_by_xor(sg, best, mask);
            const int other_id = sycl::permute_group_by_xor(sg, expert, mask);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = sycl::max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
// The same contract for other (experts, top-k) shapes - Qwen3.6-35B-A3B's 256 / 8 - one warp per token, n_tok tokens
// (8 per block).  The 512 / 10 path keeps its own kernels above (the Flash-Next bits), this template is only used
// for the other shapes.
template <int NE, int K>
__dpct_inline__ void route_topk(const float *__restrict__ logits,
                                int32_t *__restrict__ ids,
                                float *__restrict__ weights, int n_tok) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(NE % 32 == 0 && K <= 32,
                  "experts in whole warps, at most 32 picks");
    constexpr int V = NE / 32;
    const int tk =
        (int)item_ct1.get_group(2) * 8 + (int)item_ct1.get_local_id(1);
    if (tk >= n_tok) return;
    logits += (size_t) tk * NE; ids += (size_t) tk * K; weights += (size_t) tk * K;
    const int lane = item_ct1.get_local_id(2);
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < V; ++i) maximum = sycl::max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] = sycl::native::exp(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] *= reciprocal;
        if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float other = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, mask);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int other_id = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                expert, mask);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = sycl::max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < K) weights[lane] = selected * inverse_selected_sum;
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    ((sycl::queue *)(strata::q_of(stream)))
        ->parallel_for<dpct_kernel_name<class route_96e6fb>>(
            sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(32)]] {
                    route<512>(logits, ids, weights);
                });
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    // SYCL port: one 32-lane block per token running route<512>.
    ((sycl::queue *)(strata::q_of(stream)))
        ->parallel_for<dpct_kernel_name<class route_512_multi>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned) n_tok) * sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] { route<512>(logits, ids, weights); });
}

void native_router_top10_multi_ne(const float* logits, int32_t* ids, float* weights, int n_tok, int n_expert, void* stream) {
    if (n_expert == 512) { native_router_top10_multi(logits, ids, weights, n_tok, stream); return; }
    if (!stream || n_tok < 1 || n_expert != 256 || !valid(logits, (size_t) n_tok * 256 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi, 256): a stream, 256 experts and aligned [n,256]/[n,10] buffers");
    ((sycl::queue *)(strata::q_of(stream)))
        ->parallel_for<dpct_kernel_name<class route_256_multi>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned) n_tok) * sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] { route<256>(logits, ids, weights); });
}
bool native_router_topk_supported(int n_expert, int k) { return (n_expert == 512 && k == 10) || (n_expert == 256 && k == 8); }
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream) {
    if (n_expert == 512 && k == 10) {   // Flash-Next: the pinned kernels above, unchanged
        if (n_tok == 1) native_router_top10(logits, ids, weights, stream);
        else native_router_top10_multi(logits, ids, weights, n_tok, stream);
        return;
    }
    if (!stream || n_tok < 1 || !native_router_topk_supported(n_expert, k) ||
        !valid(logits, (size_t) n_tok * n_expert * 4) || !valid(ids, (size_t) n_tok * k * 4) ||
        !valid(weights, (size_t) n_tok * k * 4))
        throw std::invalid_argument("native_router_topk: a stream, a supported (experts, k) and aligned buffers");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class route_topk_ebc375,
                                            dpct_kernel_scalar<256>,
                                            dpct_kernel_scalar<8>>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((n_tok + 7) / 8)) *
                        sycl::range(1, 8, 32),
                    sycl::range(1, 8, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        route_topk<256, 8>(logits, ids, weights, n_tok);
                    });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}

// STRATA_ROUTE_RESIDENT (EXPERIMENTAL, opt-in, changes the output): residency-biased routing for 512 x 10.
// After the top-10: for the ranks lo..hi a non-resident expert is replaced by the best resident expert not already
// picked when its logit is within `margin` of the replaced one; the weights are then the softmax of the selected
// logits (the router's renormalisation).  Tokens without a swap keep the router's exact bits.
// stats (device, 4 x uint64): [0] tail entries seen, [1] swaps, [2] non-resident entries before, [3] after.
//
// One warp per token.  This used to be one THREAD per token scanning all 512 experts (4 ranks x 512 x 12 scalar
// ops on up to 6 threads): 45 us per layer, 2.2 ms of a 19.4 ms window, which is exactly what the option saved in
// PCIe experts (~2.4 ms) - the swap cost as much as the misses it removed.  The ranks are still visited in order
// (a swap changes the set already picked), but the scan is coalesced (lane j reads f = j, j + 32, ...) and reduced
// within the warp.  The bits are unchanged: a strictly greater logit wins, the smallest expert index breaks ties,
// and the softmax is the same expression.
/*
DPCT1110: The total declared local variable size in device function
route_resident_k exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void route_resident_k(const float *__restrict__ logits,
                                      int32_t *__restrict__ ids,
                                      float *__restrict__ weights,
                                      const int32_t *__restrict__ res,
                                      int n_tok, float margin, int lo, int hi,
                                      unsigned long long *__restrict__ stats) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    if (t >= n_tok) return;
    const int lane = (int)item_ct1.get_local_id(2);
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    int32_t my[10];
#pragma unroll
    for (int r = 0; r < 10; ++r) my[r] = id[r];
    int swaps = 0, tail = 0, before = 0;
    if (lane == 0)
#pragma unroll
        for (int r = 0; r < 10; ++r) before += res[my[r]] < 0;
    for (int r = lo; r <= hi && r < 10; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;                      // already resident: uniform across the warp
        if (lane == 0) ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = lane; f < 512; f += 32) {
            if (res[f] < 0) continue;
            bool used = false;
#pragma unroll
            for (int q = 0; q < 10; ++q) used |= my[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ov = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off);
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oi = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        if (lane == 0 && bi >= 0 && l[e] - bv <= margin) { my[r] = bi; ++swaps; }
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        my[r] = dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            my[r], 0); // all lanes keep the same picked set
    }
    int after = 0;
    if (lane == 0)
#pragma unroll
        for (int r = 0; r < 10; ++r) after += res[my[r]] < 0;
    /*
    DPCT1108: '__shfl_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    const int sw = dpct::experimental::select_from_sub_group(
        0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), swaps,
        0);
    if (sw) {
        float m = -INFINITY;
#pragma unroll
        for (int r = 0; r < 10; ++r) m = sycl::fmax(m, l[my[r]]);
        float ex[10], sum = 0.0f;
#pragma unroll
        for (int r = 0; r < 10; ++r) {
            ex[r] = sycl::native::exp(l[my[r]] - m); sum += ex[r];
        }
#pragma unroll
        for (int r = 0; r < 10; ++r) weights[(size_t)t * 10 + r] = ex[r] / sum;
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < 10; ++r) id[r] = my[r];
        if (stats) {
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                stats + 0, (unsigned long long)tail);
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                stats + 1, (unsigned long long)sw);
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                stats + 2, (unsigned long long)before);
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                stats + 3, (unsigned long long)after);
        }
    }
}
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream) {
    if (!stream || n_tok < 1 || !res_layer) throw std::invalid_argument("native_route_resident: bad arguments");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class route_resident_k_54fccc>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_tok) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        route_resident_k(logits, ids, weights, res_layer, n_tok,
                                         margin, rank_lo, rank_hi, stats);
                    });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}
}
