// sycl/src/kernels/gr_prompt_bench.cpp - the prompt path's hyper-connection (prefill.cpp's "hc read" and "hc write")
// at a chunk of T tokens (default 16384) on random weights at the model's shapes: every kernel of the default path
// timed alone, its bytes and GB/s, then the read against an FP64 reference on sampled tokens.
//   gr_prompt_bench [T] [reps] [samples]
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr float EPS = 1e-6f;
uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
float unbf(uint16_t h) { const uint32_t u = (uint32_t) h << 16; float f; std::memcpy(&f, &u, 4); return f; }
double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}  // namespace

int main(int argc, char** argv) {
    namespace pf = strata::prefill;
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 16384;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
    const int samples = argc > 3 ? std::atoi(argv[3]) : 48;
    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("gr_prompt_bench: T %lld, %s\n", (long long) T, q->get_device().get_info<sycl::info::device::name>().c_str());

    uint32_t r = 12345;
    auto rnd = [&] { r = r * 1664525u + 1013904223u; return ((r >> 8) & 0xffff) / 65536.0f - 0.5f; };
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& v : wd) v = bf16(rnd() * 0.05f);
    for (auto& v : wu) v = bf16(rnd() * 0.2f);
    for (auto& v : wi) v = bf16(rnd() * 0.05f);
    std::vector<float> wn(D), wnn(D);
    for (auto& v : wn) v = 1.0f + 0.2f * rnd();
    for (auto& v : wnn) v = 1.0f + 0.2f * rnd();
    // R: per-stream scales that differ (the streams of a real residual do), bo the half's output, inj the previous
    // half's injection logits
    std::vector<float> hR((size_t) T * D), hbo((size_t) T * N), hinj((size_t) T * HC);
    for (int64_t t = 0; t < T; ++t)
        for (int c = 0; c < HC; ++c) {
            const float sc = 0.5f + c + 0.25f * (float) (t % 7);
            for (int d = 0; d < N; ++d) hR[(size_t) t * D + c * N + d] = sc * rnd();
        }
    for (auto& v : hbo) v = rnd();
    for (auto& v : hinj) v = 4.0f * rnd();

    auto up = [&](const void* h, size_t b) { void* p = sycl::malloc_device(b, *q); q->memcpy(p, h, b).wait(); return p; };
    auto* d_wd = (uint16_t*) up(wd.data(), wd.size() * 2);
    auto* d_wu = (uint16_t*) up(wu.data(), wu.size() * 2);
    auto* d_wi = (uint16_t*) up(wi.data(), wi.size() * 2);
    auto* d_wn = (float*) up(wn.data(), wn.size() * 4);
    auto* d_wnn = (float*) up(wnn.data(), wnn.size() * 4);
    auto* d_R = (float*) up(hR.data(), hR.size() * 4);
    auto* d_R0 = (float*) up(hR.data(), hR.size() * 4);   // the pristine copy (the write changes R in place)
    auto* d_bo = (float*) up(hbo.data(), hbo.size() * 4);
    auto* d_inj0 = (float*) up(hinj.data(), hinj.size() * 4);
    float* d_inj = sycl::malloc_device<float>((size_t) T * HC, *q);
    float* d_rs = sycl::malloc_device<float>((size_t) T * HC, *q);
    uint16_t* d_xn16 = sycl::malloc_device<uint16_t>((size_t) T * D, *q);
    float* d_lo = sycl::malloc_device<float>((size_t) T * LR, *q);
    uint16_t* d_lo16 = sycl::malloc_device<uint16_t>((size_t) T * LR, *q);
    float* d_gated = sycl::malloc_device<float>((size_t) T * D, *q);
    float* d_mixed = sycl::malloc_device<float>((size_t) T * N, *q);
    uint16_t* d_mbf = sycl::malloc_device<uint16_t>((size_t) T * N, *q);
    uint16_t* d_mh = sycl::malloc_device<uint16_t>((size_t) T * N, *q);
    pf::Gemm gm;
    std::string err;
    if (!gm.init(q, 64ll << 20, err)) { std::printf("Gemm::init: %s\n", err.c_str()); return 2; }

    auto time = [&](auto&& fn) {   // the best of 3 batches of `reps`, after 3 warm-up calls
        for (int i = 0; i < 3; ++i) fn();
        q->wait();
        double best = 1e30;
        for (int b = 0; b < 3; ++b) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) fn();
            q->wait();
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
        }
        return best;
    };
    const double MB = 1e6;
    auto line = [&](const char* what, double ms, double bytes, double flop) {
        std::printf("  %-34s %8.3f ms  %8.1f MB  %6.0f GB/s", what, ms, bytes / MB, bytes / ms / 1e6);
        if (flop > 0) std::printf("  %6.1f TFLOP/s", flop / ms / 1e9);
        std::printf("\n");
    };
    const double fT = (double) T;
    // ---- the write (gr_write_norm_rs: R += bo * 2 sigmoid(inj / 4), then the next half's norm)
    std::printf("hc write (default path):\n");
    const double t_w = time([&] {
        pf::gr_write_norm_rs(d_R, d_bo, d_inj0, HC, d_wnn, EPS, d_rs, d_xn16, T, q);
    });
    line("gr_write_norm_rs", t_w, fT * (D * 4 * 2 + N * 4 + D * 2 + HC * 4), 0);
    const double t_wo = time([&] { pf::gr_write(d_R, d_bo, d_inj0, HC, T, q); });
    line("gr_write alone (no norm: reference)", t_wo, fT * (D * 4 * 2 + N * 4 + HC * 4), 0);
    // ---- the card's copy rate (a D2D copy of R's size: read + write)
    {
        const double t_c = time([&] { q->memcpy(d_gated, d_R0, (size_t) T * D * 4); });
        line("memcpy (R's size, D2D)", t_c, fT * D * 8, 0);
    }
    // ---- the read
    q->memcpy(d_R, d_R0, (size_t) T * D * 4).wait();
    std::printf("hc read (default path):\n");
    const double t_n = time([&] { pf::gr_norm_rs(d_R, d_wn, EPS, d_rs, d_xn16, T, q); });
    line("gr_norm_rs (not run: in the write)", t_n, fT * (D * 4 + D * 2), 0);
    const double t_d = time([&] { gm.bf16(d_xn16, d_wd, d_lo, T, LR, D); });
    line("down GEMM (oneMKL, N 320 K 10240)", t_d, fT * (D * 2 + LR * 4) + LR * D * 2.0, 2.0 * fT * LR * D);
    const double t_s = time([&] { pf::gr_silu(d_lo, d_lo16, T, q); });
    line("gr_silu", t_s, fT * LR * 6, 0);
    const double t_u = time([&] { gm.bf16(d_lo16, d_wu, d_gated, T, D, LR); });
    line("up GEMM (oneMKL, N 10240 K 320)", t_u, fT * (LR * 2 + D * 4) + LR * D * 2.0, 2.0 * fT * LR * D);
    const double t_i = time([&] { gm.bf16(d_xn16, d_wi, d_inj, T, HC, D); });
    line("inject GEMM (oneMKL, N 4 K 10240)", t_i, fT * (D * 2 + HC * 4), 2.0 * fT * HC * D);
    const double t_m = time([&] { pf::gr_mix_r(d_R, d_rs, d_wn, d_gated, d_mixed, d_mbf, T, q, d_mh); });
    line("gr_mix_r", t_m, fT * (D * 4 * 2 + N * 8 + HC * 4), 0);
    const double t_read = t_d + t_s + t_u + t_i + t_m;
    std::printf("  read total %.3f ms, write %.3f ms, both %.3f ms per half\n", t_read, t_w, t_read + t_w);

    // ---- the candidates
    std::printf("candidates:\n");
    constexpr int LRI = LR + HC;   // [w_down; w_inject] as one weight: lo and inj from one pass over xn16
    uint16_t* d_wdi = sycl::malloc_device<uint16_t>((size_t) (LR + 16) * D, *q);
    q->memset(d_wdi, 0, (size_t) (LR + 16) * D * 2).wait();
    float* d_loi = sycl::malloc_device<float>((size_t) T * (LR + 16), *q);
    {
        const double t_cp = time([&] {
            q->memcpy(d_wdi, d_wd, (size_t) LR * D * 2);
            q->memcpy(d_wdi + (size_t) LR * D, d_wi, (size_t) HC * D * 2);
        });
        line("copy [w_down; w_inject] (6.6 MB)", t_cp, 2.0 * LRI * D * 2, 0);
        const double t_di = time([&] { gm.bf16(d_xn16, d_wdi, d_loi, T, LRI, D); });
        line("down+inject GEMM (oneMKL, N 324)", t_di, fT * (D * 2 + LRI * 4) + LRI * D * 2.0, 2.0 * fT * LRI * D);
        const double t_d6 = time([&] { gm.bf16(d_xn16, d_wdi, d_loi, T, LR + 16, D); });
        line("down+inject GEMM (oneMKL, N 336)", t_d6, fT * (D * 2 + (LR + 16) * 4) + (LR + 16) * D * 2.0, 2.0 * fT * (LR + 16) * D);
        const double t_sl = time([&] { pf::gr_silu(d_loi, d_lo16, T, q, nullptr, LRI); });
        line("gr_silu (row stride 324)", t_sl, fT * LR * 6, 0);
    }
    {
        uint16_t* d_xp = sycl::malloc_device<uint16_t>((size_t) T * (D + 64), *q);
        q->memset(d_xp, 0, (size_t) T * (D + 64) * 2).wait();
        const double t_dp = time([&] { gm.bf16(d_xp, d_wd, d_lo, T, LR, D, 0, 0.0f, D + 64); });
        line("down GEMM, xn16 stride D + 64 (hc_pad)", t_dp, fT * (D * 2 + LR * 4) + LR * D * 2.0, 2.0 * fT * LR * D);
        sycl::free(d_xp, *q);
    }
    bool have_upmix = false;
    {
        q->memcpy(d_R, d_R0, (size_t) T * D * 4).wait();
        have_upmix = pf::gr_upmix(d_lo16, d_wu, d_R, d_rs, d_wn, d_mixed, d_mbf, d_mh, T, q);
        if (have_upmix) {
            const double t_um = time([&] { pf::gr_upmix(d_lo16, d_wu, d_R, d_rs, d_wn, d_mixed, d_mbf, d_mh, T, q); });
            line("gr_upmix (XMX up + mix)", t_um, fT * (LR * 2 + D * 4 + N * 8 + HC * 4) + LR * D * 2.0, 2.0 * fT * LR * D);
        } else {
            std::printf("  gr_upmix: refused on this device\n");
        }
    }

    // ---- the read's outputs against FP64 on sampled tokens (from the pristine R)
    std::vector<int64_t> ts((size_t) samples);
    for (int si = 0; si < samples; ++si) ts[(size_t) si] = (int64_t) ((si * 2654435761ull + 977) % (uint64_t) T);
    if (T >= 2) ts[0] = T - 1;   // the last token (a partial tile when T is not a tile multiple)
    // two references: exact math (FP64 throughout), and FP64 sums over the path's own BF16 roundings (xn16, lo16)
    std::vector<double> m_ex((size_t) samples * N), m_bi((size_t) samples * N), i_ex((size_t) samples * HC), i_bi((size_t) samples * HC);
    {
        std::vector<double> xn(D), xb(D), lo(LR), lob(LR);
        for (int si = 0; si < samples; ++si) {
            const int64_t t = ts[(size_t) si];
            for (int c = 0; c < HC; ++c) {
                double ss = 0;
                for (int d = 0; d < N; ++d) { const double v = hR[(size_t) t * D + c * N + d]; ss += v * v; }
                const double rsd = 1.0 / std::sqrt(ss / N + EPS);
                const float rsf = 1.0f / std::sqrt((float) (ss / N) + EPS);
                for (int d = 0; d < N; ++d) {
                    const float rv = hR[(size_t) t * D + c * N + d];
                    xn[c * N + d] = rv * rsd * wn[c * N + d];
                    xb[c * N + d] = unbf(bf16(rv * rsf * wn[c * N + d]));
                }
            }
            for (int j = 0; j < LR; ++j) {
                double a = 0, b = 0;
                for (int k = 0; k < D; ++k) { const double w = unbf(wd[(size_t) j * D + k]); a += w * xn[k]; b += w * xb[k]; }
                a /= HC; b /= HC;
                lo[j] = a * sig(a);
                lob[j] = unbf(bf16((float) (b * sig(b))));
            }
            for (int c = 0; c < HC; ++c) {
                double a = 0, b = 0;
                for (int k = 0; k < D; ++k) { const double w = unbf(wi[(size_t) c * D + k]); a += w * xn[k]; b += w * xb[k]; }
                i_ex[(size_t) si * HC + c] = a; i_bi[(size_t) si * HC + c] = b;
            }
            for (int d = 0; d < N; ++d) {
                double m0 = 0, m1 = 0;
                for (int c = 0; c < HC; ++c) {
                    double a = 0, b = 0;
                    for (int k = 0; k < LR; ++k) { const double w = unbf(wu[(size_t) (c * N + d) * LR + k]); a += w * lo[k]; b += w * lob[k]; }
                    m0 += xn[c * N + d] * sig(a);
                    m1 += xn[c * N + d] * sig(b);
                }
                m_ex[(size_t) si * N + d] = m0 / HC; m_bi[(size_t) si * N + d] = m1 / HC;
            }
        }
    }
    std::vector<float> ref_mixed((size_t) T * N), ref_inj((size_t) T * HC);
    double ref_err = 0, ref_ierr = 0;   // the default path's error against FP64 over the BF16 inputs (mixed, inject)
    bool pass = true;
    std::vector<uint16_t> ref_mbf((size_t) T * N), ref_mh((size_t) T * N);
    auto report = [&](const char* name, const float* d_m, const float* d_i, int64_t ld_i, const uint16_t* d_b, const uint16_t* d_h, bool is_ref) {
        std::vector<float> gm_((size_t) T * N), gi((size_t) T * ld_i);
        std::vector<uint16_t> gb((size_t) T * N), gh((size_t) T * N);
        q->memcpy(gm_.data(), d_m, gm_.size() * 4);
        q->memcpy(gi.data(), d_i, gi.size() * 4);
        q->memcpy(gb.data(), d_b, gb.size() * 2);
        q->memcpy(gh.data(), d_h, gh.size() * 2).wait();
        double e2[2] = {0, 0}, r2 = 0, emax[2] = {0, 0}, ie2[2] = {0, 0}, ir2 = 0;
        for (int si = 0; si < samples; ++si) {
            const int64_t t = ts[(size_t) si];
            for (int d = 0; d < N; ++d) {
                const double g = gm_[(size_t) t * N + d], a = m_ex[(size_t) si * N + d], b = m_bi[(size_t) si * N + d];
                e2[0] += (g - a) * (g - a); e2[1] += (g - b) * (g - b); r2 += a * a;
                emax[0] = std::max(emax[0], std::fabs(g - a)); emax[1] = std::max(emax[1], std::fabs(g - b));
            }
            for (int c = 0; c < HC; ++c) {
                const double g = gi[(size_t) t * ld_i + c], a = i_ex[(size_t) si * HC + c], b = i_bi[(size_t) si * HC + c];
                ie2[0] += (g - a) * (g - a); ie2[1] += (g - b) * (g - b); ir2 += a * a;
            }
        }
        std::printf("  %-26s mixed vs FP64 rel RMS %.3e (max %.2e), vs FP64 over the BF16 inputs %.3e (max %.2e); inject %.3e / %.3e\n",
                    name, std::sqrt(e2[0] / r2), emax[0], std::sqrt(e2[1] / r2), emax[1], std::sqrt(ie2[0] / ir2), std::sqrt(ie2[1] / ir2));
        bool finite = true;
        for (float v : gm_) finite = finite && std::isfinite(v);
        const double em = std::sqrt(e2[1] / r2), ei = std::sqrt(ie2[1] / ir2);
        if (!finite || (!is_ref && (em > 2.0 * ref_err + 1e-7 || ei > 2.0 * ref_ierr + 1e-7))) {
            std::printf("  %-26s FAIL: %s\n", name, finite ? "error above twice the default path's" : "non-finite output");
            pass = false;
        }
        if (is_ref) {
            ref_err = em; ref_ierr = ei;
            ref_mixed = gm_; ref_mbf = gb; ref_mh = gh;
            for (int64_t t = 0; t < T; ++t) for (int c = 0; c < HC; ++c) ref_inj[(size_t) t * HC + c] = gi[(size_t) t * ld_i + c];
            return;
        }
        double d2 = 0, rr = 0, dmax = 0; int64_t same = 0, sb = 0, sh = 0, isame = 0;
        for (size_t i = 0; i < gm_.size(); ++i) {
            const double dd = (double) gm_[i] - ref_mixed[i];
            d2 += dd * dd; rr += (double) ref_mixed[i] * ref_mixed[i]; dmax = std::max(dmax, std::fabs(dd));
            same += gm_[i] == ref_mixed[i]; sb += gb[i] == ref_mbf[i]; sh += gh[i] == ref_mh[i];
        }
        for (int64_t t = 0; t < T; ++t) for (int c = 0; c < HC; ++c) isame += gi[(size_t) t * ld_i + c] == ref_inj[(size_t) t * HC + c];
        std::printf("  %-26s vs the default path: mixed rel RMS %.3e, max |diff| %.2e; identical: mixed %.4f, bf16 %.4f, f16 %.4f, inject %.4f\n",
                    "", std::sqrt(d2 / std::max(rr, 1e-300)), dmax, (double) same / gm_.size(), (double) sb / gb.size(),
                    (double) sh / gh.size(), (double) isame / ((double) T * HC));
    };
    std::printf("accuracy (%d sampled tokens; the full chunk for the comparisons):\n", samples);
    q->memcpy(d_R, d_R0, (size_t) T * D * 4).wait();
    pf::gr_norm_rs(d_R, d_wn, EPS, d_rs, d_xn16, T, q);
    gm.bf16(d_xn16, d_wd, d_lo, T, LR, D);
    pf::gr_silu(d_lo, d_lo16, T, q);
    gm.bf16(d_lo16, d_wu, d_gated, T, D, LR);
    gm.bf16(d_xn16, d_wi, d_inj, T, HC, D);
    pf::gr_mix_r(d_R, d_rs, d_wn, d_gated, d_mixed, d_mbf, T, q, d_mh);
    report("default", d_mixed, d_inj, HC, d_mbf, d_mh, true);
    if (have_upmix) {
        q->memset(d_mixed, 0, (size_t) T * N * 4).wait();
        pf::gr_upmix(d_lo16, d_wu, d_R, d_rs, d_wn, d_mixed, d_mbf, d_mh, T, q);
        report("up+mix fused", d_mixed, d_inj, HC, d_mbf, d_mh, false);
        // the whole candidate read: [w_down; w_inject] in one GEMM, silu over its stride, the fused up + mix
        q->memset(d_mixed, 0, (size_t) T * N * 4).wait();
        gm.bf16(d_xn16, d_wdi, d_loi, T, LRI, D);
        pf::gr_silu(d_loi, d_lo16, T, q, nullptr, LRI);
        pf::gr_upmix(d_lo16, d_wu, d_R, d_rs, d_wn, d_mixed, d_mbf, d_mh, T, q);
        report("down+inject, up+mix", d_mixed, d_loi + LR, LRI, d_mbf, d_mh, false);
        const double t_new = time([&] {
            q->memcpy(d_wdi, d_wd, (size_t) LR * D * 2);
            q->memcpy(d_wdi + (size_t) LR * D, d_wi, (size_t) HC * D * 2);
            gm.bf16(d_xn16, d_wdi, d_loi, T, LRI, D);
            pf::gr_silu(d_loi, d_lo16, T, q, nullptr, LRI);
            pf::gr_upmix(d_lo16, d_wu, d_R, d_rs, d_wn, d_mixed, d_mbf, d_mh, T, q);
        });
        const double t_old = time([&] {
            gm.bf16(d_xn16, d_wd, d_lo, T, LR, D);
            pf::gr_silu(d_lo, d_lo16, T, q);
            gm.bf16(d_lo16, d_wu, d_gated, T, D, LR);
            gm.bf16(d_xn16, d_wi, d_inj, T, HC, D);
            pf::gr_mix_r(d_R, d_rs, d_wn, d_gated, d_mixed, d_mbf, T, q, d_mh);
        });
        std::printf("read, back to back: default %.3f ms, candidate %.3f ms (%.2fx)\n", t_old, t_new, t_old / t_new);
    }
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
