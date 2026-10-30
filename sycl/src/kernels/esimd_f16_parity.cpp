// esimd_f16_parity: the FP16 expert products for the larger groups (prefill/esimd_moe.dp.cpp, gate_up_f16 / down_f16) against
//   (a) the engine's FP16 chain they replace (iq_dequant_expert_f16 + oneMKL FP16 GEMM, swiglu_interleaved, oneMKL GEMM),
//   (b) an FP64 host reference from the blobs with the FP16 activations (gate/up: H = silu(g) u in FP64; down: from that H),
//   (c) the down product on the FP16 chain's own H (isolates the kernel arithmetic: only the summation order differs),
// and times the pieces of a group: the FP16 chain (dequantize / gate-up GEMMs / SwiGLU / down GEMMs and the whole chain), the int8
// chain, and the FP16 products, with TFLOP/s.
//   esimd_f16_parity [mean_rows=320] [experts=16] [reps=10] [seed=1] [gate_up_type=22 down_type=42]
//   BENCH_NOREF=1: no FP64 reference (timing sweeps).
//   Row counts: gamma-distributed (a few large experts, many small), three edge counts (1, 33, 65), scaled to the mean.
// Exit status 1 when (c) is off by more than 1e-5 rms or (b) is more than 1.25x the FP16 chain's own error.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/prefill/esimd_moe.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "../prefill/iq_tables.inc"

namespace i8 = strata::prefill::i8;
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

static int block_bytes(int t) { return t == 22 ? 82 : t == 16 ? 66 : 18; }
static int block_vals(int t) { return (t == 22 || t == 16) ? 256 : 64; }

// host decode of one row (K values) -> double (the exact values; the FP16 chain rounds them to FP16)
static void dec_row(int t, const uint8_t* row, int K, double* out) {
    const int bb = block_bytes(t), bv = block_vals(t);
    for (int b = 0; b < K / bv; ++b) {
        const uint8_t* blk = row + (size_t) b * bb;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        const double d = h2f(dh);
        double* o = out + (size_t) b * bv;
        if (t == 22) {
            const uint8_t *qs = blk + 2, *qh = blk + 66, *scl = blk + 74, *sgn = blk + 34;
            for (int ib = 0; ib < 8; ++ib)
                for (int l = 0; l < 4; ++l) {
                    const int idx = qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
                    const uint8_t* g = (const uint8_t*) &iq2s_grid[idx];
                    const double dd = d * (0.5 + ((scl[ib] >> (4 * (l / 2))) & 0xf)) * 0.25;
                    const uint8_t sb = sgn[4 * ib + l];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + 8 * l + j] = dd * ((sb >> j & 1) ? -(int) g[j] : (int) g[j]);
                }
        } else if (t == 16) {
            const uint8_t* qs = blk + 2;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux; std::memcpy(&aux, qs + 8 * ib + 4, 4);
                const double dd = d * (0.5 + (aux >> 28)) * 0.25;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g = (const uint8_t*) &iq2xxs_grid[qs[8 * ib + l]];
                    const uint8_t sb = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + 8 * l + j] = dd * ((sb >> j & 1) ? -(int) g[j] : (int) g[j]);
                }
            }
        } else {   // 42 Q2_0
            for (int i = 0; i < 64; ++i) o[i] = d * (((blk[2 + i / 4] >> ((i % 4) * 2)) & 3) - 1);
        }
    }
}

static bool run_case(sycl::queue* q, dpct::blas::descriptor_ptr hd, int gt, int dt, int mean, int n, int reps, int seed) {
    const int K = 2560, NFF = 640, NE = 2560;
    std::mt19937 rng(seed);
    std::vector<int32_t> cnt(n);
    std::gamma_distribution<double> gd(1.3, 1.0);
    std::vector<double> w(n);
    for (auto& x : w) x = gd(rng);
    const int nedge = n >= 6 ? 3 : 0;
    const int edge[3] = {1, 33, 65};
    double wsum = 0;
    for (int i = nedge; i < n; ++i) wsum += w[i];
    const double rest = (double) mean * n - (nedge ? 99 : 0);
    int tot = 0;
    for (int i = 0; i < n; ++i) { cnt[i] = i < nedge ? edge[i] : std::max(1, (int) std::lround(w[i] / wsum * rest)); tot += cnt[i]; }
    std::printf("  rows per expert:");
    for (int c : cnt) std::printf(" %d", c);
    std::printf("  (total %d, mean %.0f)\n", tot, (double) tot / n);
    i8::Layout lo;
    i8::make_layout(lo, cnt.data(), n);

    const size_t rowb_gu = (size_t) (K / block_vals(gt)) * block_bytes(gt), rowb_dn = (size_t) (NFF / block_vals(dt)) * block_bytes(dt);
    const size_t gate_b = NFF * rowb_gu, down_b = NE * rowb_dn, blob_b = 2 * gate_b + down_b;
    std::vector<std::vector<uint8_t>> hb(n);
    std::vector<uint8_t*> db(n);
    for (int i = 0; i < n; ++i) {
        hb[i].resize(blob_b);
        for (auto& x : hb[i]) x = (uint8_t) rng();
        for (size_t o = 0; o < 2 * gate_b; o += block_bytes(gt)) { const uint16_t d = f2h(0.0004f + 0.00002f * (rng() % 50)); std::memcpy(&hb[i][o], &d, 2); }
        for (size_t o = 2 * gate_b; o < blob_b; o += block_bytes(dt)) { const uint16_t d = f2h(0.0004f + 0.00002f * (rng() % 50)); std::memcpy(&hb[i][o], &d, 2); }
        db[i] = sycl::malloc_device<uint8_t>(blob_b, *q);   // (no slack: the kernels must not read past the blob)
        q->memcpy(db[i], hb[i].data(), blob_b);
    }
    std::vector<uint16_t> hx((size_t) tot * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hx) { float v = nd(rng) * 0.5f; if (rng() % 97 == 0) v *= 12.f; x = f2h(v); }   // a few outliers
    uint16_t* dxh = sycl::malloc_device<uint16_t>(hx.size(), *q);
    q->memcpy(dxh, hx.data(), hx.size() * 2);
    float* GUr = sycl::malloc_device<float>((size_t) tot * 2 * NFF, *q);
    float *Dmr = sycl::malloc_device<float>((size_t) tot * NE, *q), *Dmn = sycl::malloc_device<float>((size_t) tot * NE, *q);
    float* Dmc = sycl::malloc_device<float>((size_t) tot * NE, *q);
    uint16_t *Hr = sycl::malloc_device<uint16_t>((size_t) tot * NFF, *q), *Hn = sycl::malloc_device<uint16_t>((size_t) tot * NFF, *q);
    uint16_t* dwgu = sycl::malloc_device<uint16_t>((size_t) 2 * NFF * K, *q);
    uint16_t* dwd = sycl::malloc_device<uint16_t>((size_t) NE * NFF, *q);
    i8::Workspace ws8;
    if (!i8::workspace(lo, K, NFF, ws8, q)) { std::printf("no int8 workspace\n"); std::exit(2); }
    q->wait();
    std::vector<const uint8_t*> pb(n);
    for (int i = 0; i < n; ++i) pb[i] = db[i];
    const float one = 1.f, zero = 0.f;
    auto gemm_gu = [&](int i, int64_t o) {
        dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, 2 * NFF, cnt[i], K, &one, dwgu,
                         dpct::library_data_t::real_half, K, dxh + o * K, dpct::library_data_t::real_half, K, &zero,
                         GUr + o * 2 * NFF, dpct::library_data_t::real_float, 2 * NFF, dpct::compute_type::f32);
    };
    auto gemm_dn = [&](int i, int64_t o, const uint16_t* H, float* D) {
        dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, NE, cnt[i], NFF, &one, dwd,
                         dpct::library_data_t::real_half, NFF, H + o * NFF, dpct::library_data_t::real_half, NFF, &zero,
                         D + o * NE, dpct::library_data_t::real_float, NE, dpct::compute_type::f32);
    };
    auto deq = [&](int i) { strata::kernels::iq_dequant_expert_f16(gt, dt, db[i], db[i] + gate_b, db[i] + 2 * gate_b, NFF, K, dwgu, dwd, q); };
    auto chain = [&]() {
        int64_t o = 0;
        for (int i = 0; i < n; ++i) {
            deq(i);
            gemm_gu(i, o);
            strata::prefill::swiglu_interleaved(GUr + o * 2 * NFF, Hr + o * NFF, cnt[i], q);
            gemm_dn(i, o, Hr, Dmr);
            o += cnt[i];
        }
    };
    auto newp = [&]() {
        if (!i8::gate_up_f16(gt, pb.data(), gate_b, NFF, lo, K, dxh, Hn, q)) { std::printf("gate_up_f16 refused (type %d)\n", gt); std::exit(2); }
        if (!i8::down_f16(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hn, Dmn, q)) { std::printf("down_f16 refused (type %d)\n", dt); std::exit(2); }
    };
    auto int8 = [&]() {
        i8::quantize_rows(dxh, lo, K, ws8.xq, ws8.dx, q);
        i8::gate_up(gt, pb.data(), gate_b, NFF, lo, K, ws8.xq, ws8.dx, GUr, q);
        i8::swiglu_quantize(GUr, lo, NFF, ws8.hq, ws8.dh, q);
        i8::down(dt, pb.data(), 2 * gate_b, NE, lo, NFF, ws8.hq, ws8.dh, Dmc, q);
    };
    auto tm = [&](auto f) {
        f(); q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.3) { f(); q->wait(); }   // GPU clocks up
        double best = 1e30;
        for (int k = 0; k < 3; ++k) {
            const auto a = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) f();
            q->wait();
            best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / reps);
        }
        return best;
    };

    chain(); q->wait();
    newp(); q->wait();
    // (c): the new down product on the FP16 chain's H
    if (!i8::down_f16(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hr, Dmc, q)) std::exit(2);
    q->wait();
    std::vector<uint16_t> hHr((size_t) tot * NFF), hHn((size_t) tot * NFF);
    std::vector<float> hDr((size_t) tot * NE), hDn((size_t) tot * NE), hDc((size_t) tot * NE);
    q->memcpy(hHr.data(), Hr, hHr.size() * 2);
    q->memcpy(hHn.data(), Hn, hHn.size() * 2);
    q->memcpy(hDr.data(), Dmr, hDr.size() * 4);
    q->memcpy(hDn.data(), Dmn, hDn.size() * 4);
    q->memcpy(hDc.data(), Dmc, hDc.size() * 4).wait();
    bool ok = true;
    {
        double e2 = 0, r2 = 0;
        size_t ulp_diff = 0;
        for (size_t i = 0; i < hHr.size(); ++i) {
            const double a = h2f(hHr[i]), b = h2f(hHn[i]);
            e2 += (a - b) * (a - b); r2 += a * a;
            ulp_diff += hHr[i] != hHn[i];
        }
        double f2 = 0, s2 = 0;
        for (size_t i = 0; i < hDr.size(); ++i) { f2 += (double) (hDr[i] - hDc[i]) * (hDr[i] - hDc[i]); s2 += (double) hDr[i] * hDr[i]; }
        double g2 = 0;
        for (size_t i = 0; i < hDr.size(); ++i) g2 += (double) (hDr[i] - hDn[i]) * (hDr[i] - hDn[i]);
        const bool okc = std::sqrt(f2 / s2) < 1e-5;
        ok = ok && okc;
        std::printf("  vs the FP16 chain: H rms rel diff %.2e (%zu of %zu FP16 values differ) | down on the chain's H rms rel diff %.2e %s | "
                    "chain end to end %.2e\n", std::sqrt(e2 / r2), ulp_diff, hHr.size(), std::sqrt(f2 / s2), okc ? "OK" : "FAIL",
                    std::sqrt(g2 / s2));
    }
    if (std::getenv("BENCH_NOREF") == nullptr) {   // (b) FP64 (BENCH_NOREF=1: skipped, for timing sweeps)
        std::vector<double> wgu((size_t) 2 * NFF * K), wd((size_t) NE * NFF), H(NFF), x(K);
        double eh_r = 0, eh_n = 0, rh = 0, ed_r = 0, ed_n = 0, rd = 0;
        int smp = 0;
        for (int e = 0; e < n; ++e) {
            for (int j = 0; j < 2 * NFF; ++j) dec_row(gt, &hb[e][(size_t) j * rowb_gu], K, &wgu[(size_t) j * K]);   // gate rows, then up
            for (int v = 0; v < NE; v += 41) dec_row(dt, &hb[e][2 * gate_b + (size_t) v * rowb_dn], NFF, &wd[(size_t) v * NFF]);
            for (int rl = 0; rl < cnt[e]; rl += (cnt[e] > 24 ? 11 : 1)) {
                const int row = lo.row0[e] + rl;
                for (int k = 0; k < K; ++k) x[k] = h2f(hx[(size_t) row * K + k]);
                for (int j = 0; j < NFF; ++j) {
                    const double* wg = &wgu[(size_t) j * K];
                    const double* wu = &wgu[(size_t) (NFF + j) * K];
                    double g = 0, u = 0;
                    for (int k = 0; k < K; ++k) { g += x[k] * wg[k]; u += x[k] * wu[k]; }
                    H[j] = g / (1.0 + std::exp(-g)) * u;
                    const double a = h2f(hHr[(size_t) row * NFF + j]), b = h2f(hHn[(size_t) row * NFF + j]);
                    eh_r += (a - H[j]) * (a - H[j]); eh_n += (b - H[j]) * (b - H[j]); rh += H[j] * H[j];
                }
                for (int v = 0; v < NE; v += 41) {
                    double ref = 0;
                    for (int k = 0; k < NFF; ++k) ref += H[k] * wd[(size_t) v * NFF + k];
                    const double a = hDr[(size_t) row * NE + v], b = hDn[(size_t) row * NE + v];
                    ed_r += (a - ref) * (a - ref); ed_n += (b - ref) * (b - ref); rd += ref * ref;
                }
                ++smp;
            }
        }
        const double hr = std::sqrt(eh_r / rh), hn = std::sqrt(eh_n / rh), dr = std::sqrt(ed_r / rd), dn = std::sqrt(ed_n / rd);
        const bool okb = hn <= 1.25 * hr + 1e-6 && dn <= 1.25 * dr + 1e-6;
        ok = ok && okb;
        std::printf("  vs FP64 (%d rows): H rms rel err chain %.3e new %.3e | down chain %.3e new %.3e  %s\n", smp, hr, hn, dr, dn,
                    okb ? "OK" : "FAIL");
    }
    // timings (us per group) and TFLOP/s
    const double fl_gu = 2.0 * tot * 2 * NFF * K, fl_dn = 2.0 * tot * NE * NFF;
    const double t_chain = tm(chain);
    const double t_deq = tm([&] { for (int i = 0; i < n; ++i) deq(i); });
    const double t_ggu = tm([&] { int64_t o = 0; for (int i = 0; i < n; ++i) { gemm_gu(i, o); o += cnt[i]; } });
    const double t_sw = tm([&] { int64_t o = 0; for (int i = 0; i < n; ++i) { strata::prefill::swiglu_interleaved(GUr + o * 2 * NFF, Hr + o * NFF, cnt[i], q); o += cnt[i]; } });
    const double t_gdn = tm([&] { int64_t o = 0; for (int i = 0; i < n; ++i) { gemm_dn(i, o, Hr, Dmr); o += cnt[i]; } });
    const double t_i8 = tm(int8);
    const double t_ngu = tm([&] { i8::gate_up_f16(gt, pb.data(), gate_b, NFF, lo, K, dxh, Hn, q); });
    const double t_ndn = tm([&] { i8::down_f16(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hn, Dmn, q); });
    const double t_new = tm(newp);
    const double fl = fl_gu + fl_dn;
    std::printf("  FP16 chain %.0f us (%.1f TFLOP/s): dequant %.0f, gate/up GEMMs %.0f (%.1f TFLOP/s), SwiGLU %.0f, down GEMMs %.0f "
                "(%.1f TFLOP/s), gaps %.0f\n", t_chain, fl / t_chain * 1e-6, t_deq, t_ggu, fl_gu / t_ggu * 1e-6, t_sw, t_gdn,
                fl_dn / t_gdn * 1e-6, t_chain - t_deq - t_ggu - t_sw - t_gdn);
    std::printf("  int8 chain %.0f us (%.1f TFLOP/s)\n", t_i8, fl / t_i8 * 1e-6);
    std::printf("  FP16 products %.0f us (%.1f TFLOP/s): gate/up %.0f (%.1f TFLOP/s), down %.0f (%.1f TFLOP/s)  -> %.2fx the FP16 chain, %.2fx int8\n",
                t_new, fl / t_new * 1e-6, t_ngu, fl_gu / t_ngu * 1e-6, t_ndn, fl_dn / t_ndn * 1e-6, t_chain / t_new, t_i8 / t_new);
    for (int i = 0; i < n; ++i) sycl::free(db[i], *q);
    sycl::free(dxh, *q); sycl::free(GUr, *q); sycl::free(Dmr, *q); sycl::free(Dmn, *q); sycl::free(Dmc, *q);
    sycl::free(Hr, *q); sycl::free(Hn, *q); sycl::free(dwgu, *q); sycl::free(dwd, *q);
    return ok;
}

int main(int argc, char** argv) {
    const int mean = argc > 1 ? std::atoi(argv[1]) : 320;
    const int n = argc > 2 ? std::atoi(argv[2]) : 16;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int seed = argc > 4 ? std::atoi(argv[4]) : 1;
    sycl::queue* q = &dpct::get_in_order_queue();
    std::vector<std::pair<int, int>> cases = {{22, 42}, {16, 42}};
    if (argc > 6) cases = {{std::atoi(argv[5]), std::atoi(argv[6])}};
    if (!i8::f16_handles(cases[0].first, cases[0].second, 640, 2560, q)) { std::printf("esimd_f16_parity: SKIP (no ESIMD DPAS device)\n"); return 0; }
    dpct::blas::descriptor_ptr hd = new dpct::blas::descriptor();
    hd->set_queue(q);
    int fails = 0;
    for (auto [gt, dt] : cases) {
        std::printf("gate/up %d, down %d, mean %d rows, %d experts\n", gt, dt, mean, n);
        fails += !run_case(q, hd, gt, dt, mean, n, reps, seed);
    }
    std::printf("%s\n", fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
