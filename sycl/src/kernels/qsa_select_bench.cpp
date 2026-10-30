#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstring>
// src/kernels/qsa_select_bench.cpp - the prompt path's QSA selection (qsa_select.hpp) timed per stage, block scores
// (the warp kernel and the tensor-core one) and top-k, for a batch of consecutive queries at a given context, and
// the two scorers compared: score difference and how many selections differ (GPU, synthetic, no model).
// Usage: qsa_select_bench [context=131072] [queries=256] [reps=10] [capacity_cells]
// capacity_cells (the engine's --max-context): the score buffers and the top-k dispatch follow the CAPACITY
// (max_blocks = capacity / 4 + 2), the work follows the context. Default: capacity = context (max_blocks = ctx / 4 + 1).
// BENCH_SEL_ENGINE=1: the prompt path's selection as prefill.cpp runs it on SYCL instead - `queries` consecutive
// positions ending at the context (a prompt chunk: 16384) in batches of 256 (BENCH_SEL_BATCH), the scores as oneMKL FP32
// GEMM tiles + qsa_block_scores_reduce (the warp kernel below 2,048 blocks, as there), then qsa_block_topk; each stage
// timed on its own (barriers between the launches) and the whole chunk without them, and every scorer's selections
// counted against an FP64 host selection of sampled queries (BENCH_SEL_REFQ of them, 64).
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select_xmx.hpp"
#include "strata/sycl_queue.hpp"
#include <oneapi/mkl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace strata::kernels {   // SYCL port, kept out of the shared headers (prefill.cpp declares it the same way)
void qsa_block_scores_reduce(const float* S, int64_t ld, int64_t b0, int64_t nb, const float* dead, const float* q_idx,
                             const int32_t* steps, int64_t nq, int64_t max_blocks, float* scores, void* stream);
}
namespace k = strata::kernels;

namespace {
void ck(dpct::err0 e, const char *w) {
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(DPCT_CHECK_ERROR(
           d = (T *)sycl::malloc_device(h.size() * sizeof(T) + 64,
                                        dpct::get_in_order_queue())),
       "malloc");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
           d, h.data(), h.size() * sizeof(T)).wait()),
       "upload");
    return d;
}

// the prompt path's selection rule on FP64 scores: every cell of a block above the width-th largest cell's key, ties to
// the lowest block (a block at the threshold may be cut), cells ascending; the tail block (weight n_kv - 4 n_bid) first
std::vector<int32_t> host_select(const std::vector<double>& sc, int64_t n_kv, int64_t n_bid, int64_t width) {
    std::vector<int32_t> out;
    if (n_kv <= width) {
        for (int64_t j = 0; j < n_kv; ++j) out.push_back((int32_t) j);
        return out;
    }
    std::vector<int64_t> ord((size_t) n_bid);
    for (int64_t b = 0; b < n_bid; ++b) ord[(size_t) b] = b;
    std::sort(ord.begin(), ord.end(), [&](int64_t a, int64_t b) { return sc[(size_t) a] != sc[(size_t) b] ? sc[(size_t) a] > sc[(size_t) b] : a < b; });
    int64_t left = width;
    const int64_t tail = n_kv - n_bid * 4;
    for (int64_t c = 0; c < tail && left > 0; ++c, --left) out.push_back((int32_t) (n_bid * 4 + c));
    for (size_t i = 0; i < ord.size() && left > 0; ++i)
        for (int c = 0; c < 4 && left > 0; ++c, --left) out.push_back((int32_t) (ord[i] * 4 + c));
    std::sort(out.begin(), out.end());
    return out;
}

double ms_between(const sycl::event& a, const sycl::event& b) {
    return (double) (b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                     a.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e6;
}

int engine_stages(int64_t ctx, int64_t nq, int reps, int64_t capacity) {
    const k::QsaShapes s = k::qsa_real_shapes();
    sycl::queue& Q = *strata::q_of(nullptr);
    const int64_t max_blocks = capacity > 0 ? capacity / 4 + 2 : ctx / 4 + 1, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t selb = std::getenv("BENCH_SEL_BATCH") ? std::atoll(std::getenv("BENCH_SEL_BATCH")) : 256;
    // prefill.cpp's attention batch (STRATA_ATTN_BATCH, default 32; BENCH_SEL_ATTN_BATCH here): the GEMM tiles live in its scratch
    const int64_t attn_batch = std::getenv("BENCH_SEL_ATTN_BATCH") ? std::atoll(std::getenv("BENCH_SEL_ATTN_BATCH")) : 32;
    const uint64_t scratch_f = (uint64_t) attn_batch * k::qsa_decode_attn_scratch_floats(cap, s);
    const int64_t sel_tile = std::min<int64_t>(8192, (int64_t) (scratch_f / (uint64_t) (selb * 4)));
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> dir(128), pooled((size_t) (max_blocks * 128)), dead(128), q((size_t) (nq * 512));
    for (auto& x : dir) x = nd(rng);
    for (int64_t b = 0; b < max_blocks; ++b) {
        const float a = nd(rng);
        for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * a * dir[d] + nd(rng);
    }
    for (auto& x : dead) x = nd(rng);
    for (int64_t i = 0; i < nq; ++i)
        for (int d = 0; d < 512; ++d) q[(size_t) (i * 512 + d)] = 0.2f * dir[d % 128] + 0.1f * nd(rng);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_pooled = up(pooled);
    const float* d_dead = up(dead);
    const float* d_q = up(q);
    const int32_t* d_steps = up(steps);
    float* sc = sycl::malloc_device<float>((size_t) (selb * max_blocks), Q);
    float* scratch = sycl::malloc_device<float>(scratch_f, Q);
    // the XMX scorer's own scratch here (prefill.cpp gives it the attention scratch): keys up to the chunk's last n_bid
    const int64_t key_blocks = (int64_t) steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1;
    k::QsaSelectXmx xs;
    xs.key_blocks = key_blocks;
    xs.max_q = selb;
    xs.scratch = sycl::malloc_device<uint8_t>(k::qsa_select_xmx_bytes(key_blocks, selb), Q);
    const bool topk_ref = std::getenv("BENCH_SEL_TOPK_REF") != nullptr;   // the 256-thread reference top-k instead
    std::printf("engine path ctx %lld, %lld queries (positions %lld..%lld) in batches of %lld, capacity %lld (max_blocks %lld), "
                "GEMM tile %lld blocks, XMX scratch %.1f MB (attention scratch %.1f MB)\n", (long long) ctx, (long long) nq,
                (long long) (ctx - nq), (long long) (ctx - 1), (long long) selb, (long long) capacity, (long long) max_blocks,
                (long long) sel_tile, k::qsa_select_xmx_bytes(key_blocks, selb) / 1e6, scratch_f * 4 / 1e6);
    // arms: 0 the FP32 GEMM tiles + reduce (prefill.cpp today), 1 the XMX scorer (BENCH_SEL_XMX), 2 the warp kernel for
    // every batch (BENCH_SEL_WARP: the FP32 reorder baseline, slow)
    std::vector<int> arms{0};
    if (std::getenv("BENCH_SEL_XMX")) arms.push_back(1);
    if (std::getenv("BENCH_SEL_WARP")) arms.push_back(2);
    const char* names[3] = {"FP32 GEMM + reduce", "XMX 3xFP16", "warp FP32"};
    std::vector<std::vector<int32_t>> h_ids(3);
    for (int arm : arms) {
        int32_t* ids = sycl::malloc_device<int32_t>((size_t) (nq * cap), Q);
        Q.memset(ids, 0xff, (size_t) (nq * cap) * 4).wait();
        // stage: 0 everything, 1 scores (GEMM tiles / XMX), 2 reduce (XMX: the key split), 3 top-k; `mark` (stage
        // attribution) puts a barrier after every launch
        double t_stage[4] = {0, 0, 0, 0};
        double gflop = 0;
        auto run = [&](bool mark) {
            sycl::event last = Q.ext_oneapi_submit_barrier();
            auto lap = [&](int st) { if (mark) { sycl::event e = Q.ext_oneapi_submit_barrier(); e.wait(); t_stage[st] += ms_between(last, e); last = e; } };
            gflop = 0;
            if (arm == 1) {
                if (!k::qsa_select_xmx_keys(xs, d_pooled, s, nullptr)) { std::fprintf(stderr, "XMX scorer refused\n"); std::exit(2); }
                lap(2);
            }
            for (int64_t t0 = 0; t0 < nq; t0 += selb) {
                const int64_t nb = std::min(selb, nq - t0);
                const int32_t* steps0 = d_steps + t0 * k::kStepCount;
                const int64_t active = (int64_t) steps[(size_t) ((t0 + nb - 1) * k::kStepCount + k::kStepNBid)] + 1;
                gflop += 2.0 * (double) active * (double) (nb * 4) * 128.0 / 1e9;
                if (arm == 1) {
                    if (!k::qsa_block_scores_xmx(xs, d_dead, d_q + t0 * 512, steps0, nb, max_blocks, active, sc, nullptr)) {
                        std::fprintf(stderr, "XMX scorer refused\n");
                        std::exit(2);
                    }
                    lap(1);
                } else if (arm == 0 && active >= 2048) {
                    for (int64_t b0 = 0; b0 < active; b0 += sel_tile) {
                        const int64_t nbk = std::min<int64_t>(sel_tile, active - b0);
                        oneapi::mkl::blas::column_major::gemm(Q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, nbk,
                                                              nb * 4, 128, 1.0f, d_pooled + b0 * 128, 128, d_q + t0 * 512, 128, 0.0f,
                                                              scratch, sel_tile);
                        lap(1);
                        k::qsa_block_scores_reduce(scratch, sel_tile, b0, nbk, d_dead, d_q + t0 * 512, steps0, nb, max_blocks, sc, nullptr);
                        lap(2);
                    }
                } else {
                    k::qsa_block_scores(d_pooled, d_dead, d_q + t0 * 512, steps0, nb, max_blocks, s, sc, nullptr, active);
                    lap(1);
                }
                if (topk_ref) k::qsa_block_topk_ref(sc, steps0, nb, max_blocks, cap, s, ids + t0 * cap, nullptr);
                else k::qsa_block_topk(sc, steps0, nb, max_blocks, cap, s, ids + t0 * cap, nullptr, active);
                lap(3);
            }
            sycl::event e = Q.ext_oneapi_submit_barrier();
            e.wait();
            return e;
        };
        const int r_arm = arm == 2 ? 1 : reps;
        run(false);   // warm-up (JIT, oneMKL's kernels)
        Q.wait();
        sycl::event e0 = Q.ext_oneapi_submit_barrier();
        sycl::event e1;
        for (int r = 0; r < r_arm; ++r) e1 = run(false);
        t_stage[0] = ms_between(e0, e1) / r_arm;
        for (int r = 0; r < r_arm; ++r) run(true);
        for (int i = 1; i < 4; ++i) t_stage[i] /= r_arm;
        std::printf("  %-18s chunk %8.3f ms | stages (barriers between): scores %.3f ms (%.1f TFLOP/s of FP32 work), %s %.3f, "
                    "top-k %.3f, sum %.3f\n", names[arm], t_stage[0], t_stage[1], gflop / t_stage[1],
                    arm == 1 ? "key split" : "reduce", t_stage[2], t_stage[3], t_stage[1] + t_stage[2] + t_stage[3]);
        {   // score-level error on the last batch (still in `sc`): max |score - FP64| over the score scale, 16 queries x 1024 blocks
            const int64_t t0 = (nq - 1) / selb * selb, nb = nq - t0;
            std::vector<float> h_sc((size_t) (nb * max_blocks));
            Q.memcpy(h_sc.data(), sc, h_sc.size() * 4).wait();
            std::mt19937 srng(11);
            double err = 0, scale = 0;
            for (int qs = 0; qs < 16; ++qs) {
                const int64_t qi = qs * nb / 16, i = t0 + qi;
                const int64_t n_bid = steps[(size_t) (i * k::kStepCount + k::kStepNBid)];
                for (int sidx = 0; sidx < 1024; ++sidx) {
                    const int64_t j = (int64_t) (srng() % (uint64_t) n_bid);
                    double ref = 0;
                    for (int h = 0; h < 4; ++h) {
                        double d = 0;
                        for (int c = 0; c < 128; ++c) d += (double) q[(size_t) (i * 512 + h * 128 + c)] * (double) pooled[(size_t) (j * 128 + c)];
                        ref += d > 0 ? d : 0;
                    }
                    scale = std::max(scale, std::fabs(ref));
                    err = std::max(err, std::fabs(ref - (double) h_sc[(size_t) (qi * max_blocks + j)]));
                }
            }
            std::printf("  %-18s scores vs FP64 (last batch, 16 x 1024 samples): max err %.3g of scale %.3g (%.2g)\n", names[arm],
                        err, scale, scale > 0 ? err / scale : 0.0);
        }
        h_ids[(size_t) arm].resize((size_t) (nq * cap));
        Q.memcpy(h_ids[(size_t) arm].data(), ids, (size_t) (nq * cap) * 4).wait();
        sycl::free(ids, Q);
    }
    auto diff = [&](const int32_t* a, const int32_t* b, int64_t i, int64_t& cd) {   // a, b: query i's selections
        const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
        std::vector<int32_t> x(a, a + w), y(b, b + w), d;
        std::set_symmetric_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(d));
        cd += (int64_t) d.size() / 2;
        return !d.empty();
    };
    auto fp64_select = [&](int64_t i) {   // query i's FP64 host selection
        const int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t n_bid = st[k::kStepNBid];
        std::vector<double> ref((size_t) n_bid);
        for (int64_t b = 0; b < n_bid; ++b) {
            double sum = 0;
            for (int h = 0; h < 4; ++h) {
                double d = 0;
                for (int c = 0; c < 128; ++c) d += (double) q[(size_t) (i * 512 + h * 128 + c)] * (double) pooled[(size_t) (b * 128 + c)];
                sum += d > 0 ? d : 0;
            }
            ref[(size_t) b] = sum;
        }
        return host_select(ref, st[k::kStepNKv], n_bid, st[k::kStepWidth]);
    };
    for (int arm : arms) {   // every query against arm 0; where they differ, which one the FP64 selection agrees with
        if (arm == 0) continue;
        int64_t qd = 0, cd = 0, ok0 = 0, ok1 = 0, cd0 = 0, cd1 = 0;
        for (int64_t i = 0; i < nq; ++i) {
            if (!diff(h_ids[0].data() + i * cap, h_ids[(size_t) arm].data() + i * cap, i, cd)) continue;
            ++qd;
            const std::vector<int32_t> x = fp64_select(i);
            ok0 += !diff(x.data(), h_ids[0].data() + i * cap, i, cd0);
            ok1 += !diff(x.data(), h_ids[(size_t) arm].data() + i * cap, i, cd1);
        }
        std::printf("  %s vs %s: %lld of %lld selections differ, %lld cells; of those the FP64 selection is %s's %lld times "
                    "(%lld cells off), %s's %lld times (%lld cells off)\n", names[arm], names[0], (long long) qd, (long long) nq,
                    (long long) cd, names[0], (long long) ok0, (long long) cd0, names[arm], (long long) ok1, (long long) cd1);
    }
    // sampled queries against the FP64 host selection
    const int64_t refq = std::min<int64_t>(nq, std::getenv("BENCH_SEL_REFQ") ? std::atoll(std::getenv("BENCH_SEL_REFQ")) : 64);
    int64_t q_diff[3] = {0, 0, 0}, c_diff[3] = {0, 0, 0}, c_all = 0;
    for (int64_t r = 0; r < refq; ++r) {
        const int64_t i = (r * nq) / refq + (nq / refq) - 1;   // the last query of each stretch (the deepest of its batch)
        const int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t n_kv = st[k::kStepNKv], n_bid = st[k::kStepNBid], w = st[k::kStepWidth];
        std::vector<double> ref((size_t) n_bid);
        for (int64_t b = 0; b < n_bid; ++b) {
            double sum = 0;
            for (int h = 0; h < 4; ++h) {
                double d = 0;
                for (int c = 0; c < 128; ++c) d += (double) q[(size_t) (i * 512 + h * 128 + c)] * (double) pooled[(size_t) (b * 128 + c)];
                sum += d > 0 ? d : 0;
            }
            ref[(size_t) b] = sum;
        }
        const std::vector<int32_t> x = host_select(ref, n_kv, n_bid, w);
        for (int arm : arms) q_diff[arm] += diff(x.data(), h_ids[(size_t) arm].data() + i * cap, i, c_diff[arm]);
        c_all += w;
    }
    for (int arm : arms)
        std::printf("  %s vs FP64 selection (%lld sampled queries): %lld differ, %lld of %lld cells\n", names[arm],
                    (long long) refq, (long long) q_diff[arm], (long long) c_diff[arm], (long long) c_all);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 131072;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : 0;
    if (std::getenv("BENCH_SEL_ENGINE")) return engine_stages(ctx, nq, reps, capacity);
    const int64_t max_blocks = capacity > 0 ? capacity / 4 + 2 : ctx / 4 + 1, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // keys with a shared direction plus noise, so the scores have a spread like a real indexer's
    std::vector<float> dir(128), pooled((size_t) (max_blocks * 128)), dead(128), q((size_t) (nq * 512));
    for (auto& x : dir) x = nd(rng);
    for (int64_t b = 0; b < max_blocks; ++b) {
        const float a = nd(rng);
        for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * a * dir[d] + nd(rng);
    }
    for (auto& x : dead) x = nd(rng);
    for (int64_t i = 0; i < nq; ++i)
        for (int d = 0; d < 512; ++d) q[(size_t) (i * 512 + d)] = 0.2f * dir[d % 128] + 0.1f * nd(rng);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {   // qsa_step_fill's arithmetic (kept here so the bench links only qsa_select)
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_pooled = up(pooled);
    const float* d_dead = up(dead);
    const float* d_q = up(q);
    const int32_t* d_steps = up(steps);
    float *sc_old = nullptr, *sc_new = nullptr;
    int32_t *ids_old = nullptr, *ids_new = nullptr;
    ck(DPCT_CHECK_ERROR(
           sc_old = (float *)sycl::malloc_device((size_t)(nq * max_blocks) * 4,
                                                 dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           sc_new = (float *)sycl::malloc_device((size_t)(nq * max_blocks) * 4,
                                                 dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           ids_old = (int32_t *)sycl::malloc_device(
               (size_t)(nq * cap) * 4, dpct::get_in_order_queue())),
       "malloc");
    ck(DPCT_CHECK_ERROR(
           ids_new = (int32_t *)sycl::malloc_device(
               (size_t)(nq * cap) * 4, dpct::get_in_order_queue())),
       "malloc");
    const int64_t active = steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1;
    if (std::getenv("QSA_BENCH_DECODE")) {   // a decode window's call: no block count (block_scores_multi_kernel)
        auto run_dec = [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_old, nullptr, -1); };
        run_dec();
        ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
           "sync");
        dpct::event_ptr a0, a1;
        a0 = new sycl::event(); a1 = new sycl::event();
        dpct::sync_barrier(a0);
        for (int r = 0; r < reps; ++r) run_dec();
        dpct::sync_barrier(a1); a1->wait_and_throw();
        float ms = 0; ms = (a1->get_profiling_info<
                                sycl::info::event_profiling::command_end>() -
                            a0->get_profiling_info<
                                sycl::info::event_profiling::command_start>()) /
                           1000000.0f;
        std::vector<float> h((size_t) (nq * max_blocks));
        ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                .memcpy(h.data(), sc_old, h.size() * 4)
                                .wait()),
           "down");
        unsigned long long x = 1469598103934665603ull;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t nbid = steps[(size_t) (i * k::kStepCount + k::kStepNBid)];
            for (int64_t j = 0; j <= nbid; ++j) { uint32_t u; std::memcpy(&u, &h[(size_t) (i * max_blocks + j)], 4); x = (x ^ u) * 1099511628211ull; }
        }
        const double gb = (double) active * 512.0 / 1e9;
        std::printf("decode scores ctx %lld, %lld queries: %.1f us (%.0f GB/s of pooled keys), hash %016llx\n", (long long) ctx,
                    (long long) nq, 1e3 * ms / reps, gb / (ms / reps / 1e3), x);
        return 0;
    }
    auto run_old = [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_old, nullptr, active); };
    // a device without the tensor-core scorer (HIP other than gfx12): time the warp scorer and the top-k only
    const bool have_tc = k::qsa_block_scores_tc(d_pooled, d_dead, d_q, d_steps, 1, max_blocks, s, sc_new, nullptr, active);
    auto run_new = [&] {
        if (!k::qsa_block_scores_tc(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_new, nullptr, active)) {
            std::fprintf(stderr, "tc scorer refused\n");
            std::exit(2);
        }
    };
    run_old();
    if (const char* tg = std::getenv("BENCH_SEL_TIES")) {   // the warp scores rounded to a grid (0: one value): ties for the top-k check
        const float g = (float) std::atof(tg);
        std::vector<float> h((size_t) (nq * max_blocks));
        dpct::get_in_order_queue().memcpy(h.data(), sc_old, h.size() * 4).wait();
        for (auto& x : h) x = g > 0 ? std::round(x / g) * g : 1.0f;
        dpct::get_in_order_queue().memcpy(sc_old, h.data(), h.size() * 4).wait();
    }
    if (have_tc) run_new();
    else ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                sc_new, sc_old, (size_t)(nq * max_blocks) * 4).wait()),
            "copy");
    k::qsa_block_topk_ref(sc_old, d_steps, nq, max_blocks, cap, s, ids_old, nullptr);
    k::qsa_block_topk(sc_new, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active);
    int32_t* ids_reg = nullptr;   // the register top-k on the OLD scores: must equal the reference exactly
    ck(DPCT_CHECK_ERROR(
           ids_reg = (int32_t *)sycl::malloc_device(
               (size_t)(nq * cap) * 4, dpct::get_in_order_queue())),
       "malloc");
    k::qsa_block_topk(sc_old, d_steps, nq, max_blocks, cap, s, ids_reg, nullptr, active);
    ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
       "warm");
    // compare
    std::vector<float> a((size_t) (nq * max_blocks)), b(a.size());
    std::vector<int32_t> ia((size_t) (nq * cap)), ib(ia.size());
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(a.data(), sc_old, a.size() * 4)
                            .wait()),
       "down");
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(b.data(), sc_new, b.size() * 4)
                            .wait()),
       "down");
    if (std::getenv("QSA_BENCH_HASH")) {   // the fast scorer's bytes (compare STRATA_SELECT_SIMT=1 and =2)
        unsigned long long h = 1469598103934665603ull;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t nbid = steps[(size_t) (i * k::kStepCount + k::kStepNBid)];
            for (int64_t j = 0; j < nbid; ++j) {
                uint32_t u;
                std::memcpy(&u, &b[(size_t) (i * max_blocks + j)], 4);
                h = (h ^ u) * 1099511628211ull;
            }
        }
        std::printf("fast scores hash %016llx\n", h);
    }
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(ia.data(), ids_old, ia.size() * 4)
                            .wait()),
       "down");
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(ib.data(), ids_new, ib.size() * 4)
                            .wait()),
       "down");
    std::vector<int32_t> ir(ia.size());
    ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                            .memcpy(ir.data(), ids_reg, ir.size() * 4)
                            .wait()),
       "down");
    int64_t reg_same = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
        reg_same += std::equal(ia.begin() + i * cap, ia.begin() + i * cap + w, ir.begin() + i * cap);
    }
    double max_rel = 0, sum_rel = 0, n_rel = 0;
    int64_t same_sel = 0, cells_diff = 0, cells_all = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* st = steps.data() + i * k::kStepCount;
        for (int64_t j = 0; j <= st[k::kStepNBid]; ++j) {
            const double x = a[(size_t) (i * max_blocks + j)], y = b[(size_t) (i * max_blocks + j)];
            const double r = std::fabs(x - y) / std::max(1e-6, std::fabs(x));
            max_rel = std::max(max_rel, r);
            sum_rel += r;
            n_rel += 1;
        }
        const int64_t w = st[k::kStepWidth];
        std::vector<int32_t> x(ia.begin() + i * cap, ia.begin() + i * cap + w), y(ib.begin() + i * cap, ib.begin() + i * cap + w);
        same_sel += x == y;
        std::vector<int32_t> d;
        std::set_symmetric_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(d));
        cells_diff += (int64_t) d.size() / 2;
        cells_all += w;
    }
    // accuracy against an FP64 host reference on a sample (blocks below n_bid; the tail block is the warp kernel's own
    // arithmetic in both scorers). Gate, as the prompt-attention harness's: the scorer under test is no worse than 4x
    // the warp kernel's error, floored at 1e-5 of the score scale.  The floor is the fast scorer's own: measured against the FP64
    // reference it sits at 1.0e-6 of the scale (RTX 5070 sm_120, ctx 32K: 0.99e-6, which a 1e-6 floor failed by rounding; 131K: 0.97e-6)
    // against the warp kernel's 7e-8, i.e. FP32-accumulated TF32-split products; 1e-5 keeps ten times that margin and is still
    // two orders below a plain single-TF32 product (~1e-3), which is the error this gate exists to catch.
    double err_old = 0, err_new = 0, scale = 0;
    {
        std::mt19937 srng(11);
        const int64_t nqs = std::min<int64_t>(nq, 32);
        for (int64_t qs = 0; qs < nqs; ++qs) {
            const int64_t i = qs * nq / nqs;
            const int64_t nbid = steps[(size_t) (i * k::kStepCount + k::kStepNBid)];
            if (nbid <= 0) continue;
            for (int sidx = 0; sidx < 1024; ++sidx) {
                const int64_t j = sidx < 64 ? std::min<int64_t>(nbid - 1, sidx) : (int64_t) (srng() % (uint64_t) nbid);
                double ref = 0;
                for (int h = 0; h < 4; ++h) {
                    double d = 0;
                    for (int c = 0; c < 128; ++c) d += (double) q[(size_t) (i * 512 + h * 128 + c)] * (double) pooled[(size_t) (j * 128 + c)];
                    ref += d > 0 ? d : 0;
                }
                scale = std::max(scale, std::fabs(ref));
                err_old = std::max(err_old, std::fabs(ref - (double) a[(size_t) (i * max_blocks + j)]));
                err_new = std::max(err_new, std::fabs(ref - (double) b[(size_t) (i * max_blocks + j)]));
            }
        }
    }
    const bool acc_ok = !have_tc || err_new <= std::max(4.0 * err_old, 1e-5 * scale);
    // time
    dpct::event_ptr e0, e1;
    e0 = new sycl::event(); e1 = new sycl::event();
    auto timed = [&](auto f) {
        try {
    dpct::sync_barrier(e0);
        for (int r = 0; r < reps; ++r) f();
        dpct::sync_barrier(e1);
        ck(DPCT_CHECK_ERROR(e1->wait_and_throw()), "time");
        float ms = 0;
        ms = (e1->get_profiling_info<
                  sycl::info::event_profiling::command_end>() -
              e0->get_profiling_info<
                  sycl::info::event_profiling::command_start>()) /
             1000000.0f;
        return ms / reps;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    const float t_old = timed(run_old), t_new = have_tc ? timed(run_new) : t_old;
    const float t_tk = timed([&] { k::qsa_block_topk_ref(sc_old, d_steps, nq, max_blocks, cap, s, ids_old, nullptr); });
    const float t_tk2 = timed([&] { k::qsa_block_topk(sc_old, d_steps, nq, max_blocks, cap, s, ids_reg, nullptr, active); });
    std::printf("top-k %.3f -> %.3f ms (%.1fx), register top-k identical to the reference %lld/%lld\n", t_tk, t_tk2,
                t_tk / t_tk2, (long long) reg_same, (long long) nq);
    std::printf("%s accuracy vs FP64 (score scale %.3g): warp kernel max err %.3g, tensor-core max err %.3g (%.2g of scale)\n",
                !have_tc ? "SKIP" : acc_ok ? "PASS" : "FAIL", scale, err_old, err_new, scale > 0 ? err_new / scale : 0.0);
    if (!have_tc) std::printf("tensor-core scorer not available on this device: warp scorer %.3f ms, top-k %.3f ms (%.0f%% of the two)\n", t_old, t_tk2, 100.0 * t_tk2 / (t_old + t_tk2));
    std::printf("ctx %lld, %lld queries x %lld blocks: scores %.3f -> %.3f ms (%.1fx), top-k %.3f ms; score rel diff "
                "mean %.2g max %.2g; selections identical %lld/%lld, cells differing %.4f%%\n", (long long) ctx,
                (long long) nq, (long long) active, t_old, t_new, t_old / t_new, t_tk, sum_rel / std::max(1.0, n_rel),
                max_rel, (long long) same_sel, (long long) nq, cells_all ? 100.0 * (double) cells_diff / (double) cells_all : 0.0);
    return acc_ok && reg_same == nq ? 0 : 1;
}
