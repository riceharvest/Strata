// ds4_xmx_test: the prompt path's expert sets (ds4_iq's set_gate_up / set_down, the GGUF low-bit experts) on Xe2's XMX
// (set_xmx_kernel, joint_matrix) against the F32 kernel and a host reference, on synthetic experts of DeepSeek-V4.1-Flash's
// shape (n_embd 5120, n_ff 2304) in the official sets' types (IQ2_XXS, IQ2_XS, IQ1_M) and IQ3_XXS.
//   ds4_xmx_test                       the check (ctest): rows per expert 1 .. 130 (partial tiles), the light kernels off
//   ds4_xmx_test --bench TYPE E R [N]  us per set_gate_up / set_down call, F32 kernel vs XMX, E experts of R rows each
//   ds4_xmx_test --ixbench T N         ms per index_score (every key, 32 heads x 128, fp4p keys) of T queries over N keys
// The check also runs index_score's XMX kernel against its FMA kernel (bf16 and fp4p keys, causal limits).
// About 0.2 GB of device memory for the check; the bench E x 9 MB (IQ2_XXS).
#include "ds4_sycl.hpp"
#include "ds4_iq.cuh"
#include "ds4_kernels.cuh"

#include <chrono>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace iq = strata::ds4::gguf::iq;

// a normal deviate (Box-Muller on the generator's 32-bit words).  std::normal_distribution<float> looped forever in this build
// (icpx 2026.1, -O3 -fsycl: both tests that used it hung inside it, 2026-10-24); this needs nothing from it
struct Normal {
    float operator()(std::mt19937& g) const {
        const double u1 = ((double) g() + 1.0) / 4294967297.0, u2 = (double) g() / 4294967296.0;
        return (float) (std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
};

static void check_index(cudaStream_t s);   // (below: index_score's XMX kernel against its FMA kernel)
static void check_attn(cudaStream_t s);    // (below: a chunk's sparse attention, XMX against FMA)
static int fails = 0;
#define CHECK(c, ...)                                         \
    do {                                                      \
        if (!(c)) {                                           \
            ++fails;                                          \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);  \
            std::printf(__VA_ARGS__);                         \
            std::printf("\n");                                \
        }                                                     \
    } while (0)

constexpr int kD = 5120, kF = 2304;

static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }

// random blocks of the type with a sane block scale (the codes may be anything)
static void fill_blocks(int type, uint8_t* p, size_t bytes, std::mt19937& rng) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    const size_t bb = (size_t) iq::block_bytes(type);
    for (size_t b = 0; b + bb <= bytes; b += bb) {
        const uint16_t d = (uint16_t) (0x1C00u + (rng() & 0x3FFu));   // [2^-8, 2^-7)
        if (type == iq::kIQ1_M) {   // the scale's four nibbles on top of the four scale words (bytes 48 ..)
            uint16_t* sc = (uint16_t*) (p + b + 48);
            for (int i = 0; i < 4; ++i) sc[i] = (uint16_t) ((sc[i] & 0x0FFFu) | (((d >> (4 * i)) & 0xFu) << 12));
        } else {
            std::memcpy(p + b, &d, 2);
        }
    }
}

struct Set {
    int type = 0, E = 0;
    size_t rbD = 0, rbF = 0, tr[3] = {0, 0, 0}, stride = 0;
    uint8_t* w = nullptr;   // device: E blobs
    std::vector<int> cnt, bounds;
    int rows = 0, max_rows = 0, ntok = 0;
    int* d_bounds = nullptr;
    int* d_idx = nullptr;
    float* d_x = nullptr;     // [ntok][kD]
    uint16_t* d_x16 = nullptr;  // [ntok][kD] the same in F16 (the model's c.x16; DS4_TEST_X16=0: the tiles get the F32 rows only)
    uint16_t* d_h = nullptr;  // [rows][kF] gate/up out
    uint16_t* d_hin = nullptr;  // [rows][kF] down in
    float* d_y = nullptr;     // [rows][kD]
    std::vector<int> idx;
    std::vector<float> x;
    std::vector<uint16_t> hin;
};

static bool make_set(Set& S, int type, const std::vector<int>& cnt, std::mt19937& rng) {
    S.type = type;
    S.E = (int) cnt.size();
    S.cnt = cnt;
    S.rbD = iq::row_bytes(type, kD);
    S.rbF = iq::row_bytes(type, kF);
    S.tr[0] = 0;
    S.tr[1] = (size_t) kF * S.rbD;
    S.tr[2] = 2 * (size_t) kF * S.rbD;
    S.stride = (S.tr[2] + (size_t) kD * S.rbF + 255) / 256 * 256;
    std::vector<uint8_t> blob(S.stride);
    if (cudaMalloc(&S.w, S.stride * S.E) != cudaSuccess) return false;
    for (int e = 0; e < S.E; ++e) {
        fill_blocks(type, blob.data(), S.tr[2], rng);   // gate, up: rows of kD
        fill_blocks(type, blob.data() + S.tr[2], (size_t) kD * S.rbF, rng);
        cudaMemcpy(S.w + (size_t) e * S.stride, blob.data(), S.stride, cudaMemcpyHostToDevice);
    }
    S.bounds.assign(1, 0);
    for (int c : cnt) {
        S.bounds.push_back(S.bounds.back() + c);
        S.max_rows = std::max(S.max_rows, c);
    }
    S.rows = S.bounds.back();
    S.ntok = std::max(1, S.rows / 6 + 1);
    const Normal nd;
    S.x.resize((size_t) S.ntok * kD);
    for (float& v : S.x) v = nd(rng);
    S.idx.resize((size_t) S.rows);
    for (int& v : S.idx) v = (int) (rng() % (unsigned) S.ntok);
    S.hin.resize((size_t) S.rows * kF);
    for (uint16_t& v : S.hin) v = f2h(0.5f * nd(rng));
    cudaMalloc(&S.d_bounds, S.bounds.size() * 4);
    cudaMalloc(&S.d_idx, S.idx.size() * 4);
    cudaMalloc(&S.d_x, S.x.size() * 4);
    cudaMalloc(&S.d_h, (size_t) S.rows * kF * 2);
    cudaMalloc(&S.d_hin, S.hin.size() * 2);
    cudaMalloc(&S.d_y, (size_t) S.rows * kD * 4);
    cudaMemcpy(S.d_bounds, S.bounds.data(), S.bounds.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(S.d_idx, S.idx.data(), S.idx.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(S.d_x, S.x.data(), S.x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(S.d_hin, S.hin.data(), S.hin.size() * 2, cudaMemcpyHostToDevice);
    {
        std::vector<uint16_t> x16(S.x.size());
        for (size_t i = 0; i < x16.size(); ++i) x16[i] = f2h(S.x[i]);
        cudaMalloc(&S.d_x16, x16.size() * 2);
        cudaMemcpy(S.d_x16, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice);
    }
    return true;
}
static void free_set(Set& S) {
    for (void* p : {(void*) S.w, (void*) S.d_bounds, (void*) S.d_idx, (void*) S.d_x, (void*) S.d_x16, (void*) S.d_h, (void*) S.d_hin, (void*) S.d_y}) cudaFree(p);
}
static iq::SetArgs args(const Set& S, bool gu, float limit) {
    iq::SetArgs a;
    a.wbase = S.w;
    a.stride = S.stride;
    for (int r = 0; r < 3; ++r) a.tr[r] = S.tr[r];
    a.bounds = S.d_bounds;
    a.row0 = 0;
    a.max_rows = S.max_rows;
    a.n_exp = S.E;
    a.n_embd = kD;
    a.n_ff = kF;
    a.limit = limit;
    a.tg = a.td = S.type;
    if (gu) {
        a.xf = S.d_x;
        static const bool x16 = !(std::getenv("DS4_TEST_X16") && std::atoi(std::getenv("DS4_TEST_X16")) == 0);
        if (x16) a.xh = S.d_x16;
        a.idx = S.d_idx;
        a.ldx = kD;
        a.h = S.d_h;
        a.ldo = kF;
    } else {
        a.xh = S.d_hin;
        a.ldx = kF;
        a.y = S.d_y;
        a.ldo = kD;
    }
    return a;
}

static double swiglu_ref(double g, double u, double lim) {
    if (lim > 0) {
        g = std::min(g, lim);
        u = std::min(std::max(u, -lim), lim);
    }
    return g / (1.0 + std::exp(-g)) * u;
}

// one expert's sampled outputs against the host: weights as F16 (dequant_rows), rows as F16 (xmx: the F16 operands) or F32
struct Err {
    double h = 0, y = 0;   // max |got - ref| / rms(ref)
};
static Err check_expert(const Set& S, int e, const std::vector<uint16_t>& h, const std::vector<float>& y, bool f16_rows, float limit,
                        cudaStream_t s) {
    const int lo = S.bounds[(size_t) e], hi = S.bounds[(size_t) e + 1];
    Err out;
    if (hi <= lo) return out;
    const int nout = 96;   // sampled output columns: the first, the last, some between
    std::vector<int> cols;
    for (int i = 0; i < nout; ++i) cols.push_back((int) ((int64_t) i * 6007 % kF));
    std::vector<int> dcols;
    for (int i = 0; i < nout; ++i) dcols.push_back((int) ((int64_t) i * 9973 % kD));
    // the weight rows needed, as F16 from the device's own dequantisation
    uint16_t* dw = nullptr;
    cudaMalloc(&dw, (size_t) std::max(kD, kF) * 2);
    std::vector<uint16_t> wrow((size_t) std::max(kD, kF));
    const uint8_t* blob = S.w + (size_t) e * S.stride;
    auto row = [&](size_t off, size_t rb, int k) {
        iq::dequant_rows(S.type, blob + off, rb, k, 1, dw, false, s);
        cudaMemcpyAsync(wrow.data(), dw, (size_t) k * 2, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        std::vector<double> r((size_t) k);
        for (int i = 0; i < k; ++i) r[(size_t) i] = h2f(wrow[(size_t) i]);
        return r;
    };
    const int nr = std::min(hi - lo, 9);
    std::vector<int> rs;
    for (int i = 0; i < nr; ++i) rs.push_back(lo + (int) ((int64_t) i * (hi - lo - 1) / std::max(1, nr - 1)));
    double ss = 0, mx = 0;
    int n = 0;
    std::vector<std::vector<double>> wg, wu;
    for (int c : cols) {
        wg.push_back(row(S.tr[0] + (size_t) c * S.rbD, S.rbD, kD));
        wu.push_back(row(S.tr[1] + (size_t) c * S.rbD, S.rbD, kD));
    }
    for (int r : rs) {
        const float* xr = S.x.data() + (size_t) S.idx[(size_t) r] * kD;
        for (size_t ci = 0; ci < cols.size(); ++ci) {
            double g = 0, u = 0;
            for (int k = 0; k < kD; ++k) {
                const double xv = f16_rows ? (double) h2f(f2h(xr[k])) : (double) xr[k];
                g += xv * wg[ci][(size_t) k];
                u += xv * wu[ci][(size_t) k];
            }
            const double ref = swiglu_ref(g, u, limit);
            ss += ref * ref;
            ++n;
            mx = std::max(mx, std::fabs((double) h2f(h[(size_t) r * kF + cols[ci]]) - ref));
        }
    }
    out.h = mx / std::sqrt(ss / std::max(1, n) + 1e-30);
    ss = mx = 0;
    n = 0;
    for (int c : dcols) {
        const std::vector<double> wd = row(S.tr[2] + (size_t) c * S.rbF, S.rbF, kF);
        for (int r : rs) {
            double acc = 0;
            for (int k = 0; k < kF; ++k) acc += (double) h2f(S.hin[(size_t) r * kF + k]) * wd[(size_t) k];
            ss += acc * acc;
            ++n;
            mx = std::max(mx, std::fabs((double) y[(size_t) r * kD + c] - acc));
        }
    }
    out.y = mx / std::sqrt(ss / std::max(1, n) + 1e-30);
    cudaFree(dw);
    return out;
}

static void run(const Set& S, float limit, cudaStream_t s, std::vector<uint16_t>& h, std::vector<float>& y) {
    cudaMemsetAsync(S.d_h, 0xff, (size_t) S.rows * kF * 2, s);   // (NaN halves: an output left unwritten shows)
    cudaMemsetAsync(S.d_y, 0xff, (size_t) S.rows * kD * 4, s);
    const bool ok = iq::set_gate_up(args(S, true, limit), s) && iq::set_down(args(S, false, limit), s);
    CHECK(ok, "type %d: set_gate_up / set_down not covered", S.type);
    h.resize((size_t) S.rows * kF);
    y.resize((size_t) S.rows * kD);
    cudaMemcpyAsync(h.data(), S.d_h, h.size() * 2, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(y.data(), S.d_y, y.size() * 4, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
}

static int check_main() {
    setenv("STRATA_DS4_NO_LIGHT", "1", 1);   // (every expert through the tiles, the few-row ones too)
    std::printf("device: %s\n", dpct::get_current_device().get_info<sycl::info::device::name>().c_str());
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    std::mt19937 rng(7);
    const float limit = 7.0f;
    const std::vector<int> cnt = {1, 7, 8, 9, 0, 63, 64, 65, 130, 16};
    iq::force_tiles(1);
    if (!iq::tiles_xmx()) {
        std::printf("this build has no XMX sets (not an AOT bmg / lnl build): nothing to check\n");
        return 0;
    }
    for (int type : {iq::kIQ2_XXS, iq::kIQ2_XS, iq::kIQ1_M, iq::kIQ3_XXS}) {
        Set S;
        CHECK(make_set(S, type, cnt, rng), "set of type %d", type);
        std::vector<uint16_t> hx, hf;
        std::vector<float> yx, yf;
        iq::force_tiles(1);
        run(S, limit, s, hx, yx);
        iq::force_tiles(0);
        run(S, limit, s, hf, yf);
        int nan_x = 0, nan_f = 0;
        double dd = 0, dn = 0;
        for (size_t i = 0; i < hx.size(); ++i) {
            nan_x += !std::isfinite(h2f(hx[i]));
            nan_f += !std::isfinite(h2f(hf[i]));
        }
        for (size_t i = 0; i < yx.size(); ++i) {
            nan_x += !std::isfinite(yx[i]);
            nan_f += !std::isfinite(yf[i]);
            dd += (double) (yx[i] - yf[i]) * (yx[i] - yf[i]);
            dn += (double) yf[i] * yf[i];
        }
        double eh = 0, ey = 0, fh = 0, fy = 0;
        for (int e = 0; e < S.E; ++e) {
            const Err a = check_expert(S, e, hx, yx, true, limit, s);    // XMX against F16 rows
            const Err b = check_expert(S, e, hf, yf, false, limit, s);   // F32 kernel against F32 rows
            eh = std::max(eh, a.h), ey = std::max(ey, a.y), fh = std::max(fh, b.h), fy = std::max(fy, b.y);
        }
        std::printf("%-8s XMX vs F16-row reference: h %.2e y %.2e | F32 kernel vs F32 reference: h %.2e y %.2e | XMX vs F32 kernel, down: "
                    "rms rel %.2e | non-finite %d / %d\n",
                    iq::name(type), eh, ey, fh, fy, std::sqrt(dd / (dn + 1e-30)), nan_x, nan_f);
        CHECK(nan_x == 0 && nan_f == 0, "%s: %d / %d outputs not finite (unwritten)", iq::name(type), nan_x, nan_f);
        // h is F16 (2^-11 of the value, the largest a few rms); y: the XMX sums are the reference's operands exactly (F16 weights and
        // rows), F32 sums; the F32 kernel's weights are not rounded to F16 as the reference's are (~2^-11 a weight)
        CHECK(eh < 1e-2 && ey < 1e-4, "%s XMX: h %.2e y %.2e", iq::name(type), eh, ey);
        CHECK(fh < 1e-2 && fy < 3e-3, "%s F32: h %.2e y %.2e", iq::name(type), fh, fy);
        free_set(S);
    }
    iq::force_tiles(-1);
    check_index(s);
    check_attn(s);
    cudaStreamDestroy(s);
    std::printf(fails == 0 ? "ALL OK\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}


namespace ds4k = strata::ds4;
// index_score's inputs: T queries (32 heads x 128, F32), their head weights (64 a row), n keys in format kf
struct IxSet {
    float *q = nullptr, *w = nullptr, *out = nullptr;
    uint8_t* k = nullptr;
};
static IxSet make_ix(int T, int n, int kf, std::mt19937& rng) {
    IxSet x;
    Normal nd;
    std::vector<float> hq((size_t) T * 32 * 128), hw((size_t) T * 64);
    for (float& v : hq) v = nd(rng);
    for (float& v : hw) v = std::fabs(nd(rng)) * 0.1f;
    const size_t rb = ds4k::kv_row_bytes(kf, 128);
    std::vector<uint8_t> hk((size_t) n * rb, 0);
    for (int p = 0; p < n; ++p) {
        uint8_t* r = hk.data() + (size_t) p * rb;
        if (kf == ds4k::kKvFp4p) {
            for (int i = 0; i < 64; ++i) r[i] = (uint8_t) rng();
            for (int i = 0; i < 4; ++i) r[64 + i] = (uint8_t) (120 + rng() % 8);   // (UE8M0: 2^-7 .. 2^0)
        } else
            for (int i = 0; i < 128; ++i) {
                const float v = nd(rng);
                uint32_t b;
                std::memcpy(&b, &v, 4);
                const uint16_t h = (uint16_t) (b >> 16);
                std::memcpy(r + 2 * i, &h, 2);
            }
    }
    cudaMalloc(&x.q, hq.size() * 4);
    cudaMalloc(&x.w, hw.size() * 4);
    cudaMalloc(&x.k, hk.size());
    cudaMalloc(&x.out, (size_t) T * n * 4);
    cudaMemcpy(x.q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(x.w, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(x.k, hk.data(), hk.size(), cudaMemcpyHostToDevice);
    return x;
}
static void free_ix(IxSet& x) {
    cudaFree(x.q);
    cudaFree(x.w);
    cudaFree(x.k);
    cudaFree(x.out);
}
static ds4k::Rows ix_rows(int p0) {   // gen 1: query t sees (p0 + t + 1) / 2 keys
    ds4k::Rows r;
    r.gen = 1;
    r.p0 = p0;
    r.R = 2;
    r.kcap = 512;
    return r;
}
// XMX vs FMA: the same -inf positions, every finite score within 2% of its row's largest magnitude (BF16 operands)
static void check_index(cudaStream_t s) {
    std::mt19937 rng(5);
    const int T = 37, p0 = 3001, n = (p0 + T) / 2;
    for (int kf : {ds4k::kKvBf16, ds4k::kKvFp4p}) {
        IxSet x = make_ix(T, n, kf, rng);
        const ds4k::Rows lim = ix_rows(p0);
        std::vector<float> a((size_t) T * n), b((size_t) T * n);
        for (int mode : {0, 1}) {
            ds4k::index_force_xmx(mode);
            cudaMemsetAsync(x.out, 0, (size_t) T * n * 4, s);
            ds4k::index_score(x.q, x.w, x.k, 32, 128, T, n, nullptr, 0, 1, lim, x.out, n, s, kf);
            cudaMemcpyAsync(mode ? b.data() : a.data(), x.out, (size_t) T * n * 4, cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
        }
        ds4k::index_force_xmx(-1);
        int inf_bad = 0;
        double worst = 0;
        for (int t = 0; t < T; ++t) {
            double mx = 1e-30;
            for (int p = 0; p < n; ++p)
                if (std::isfinite(a[(size_t) t * n + p])) mx = std::max(mx, (double) std::fabs(a[(size_t) t * n + p]));
            for (int p = 0; p < n; ++p) {
                const float u = a[(size_t) t * n + p], v = b[(size_t) t * n + p];
                if (std::isfinite(u) != std::isfinite(v)) ++inf_bad;
                else if (std::isfinite(u)) worst = std::max(worst, std::fabs((double) u - v) / mx);
            }
        }
        std::printf("index_score %s keys, %d queries x %d keys: XMX vs FMA worst %.2e of the row's largest, %d -inf mismatches\n",
                    kf == ds4k::kKvFp4p ? "fp4p" : "bf16", T, n, worst, inf_bad);
        CHECK(inf_bad == 0 && worst < 2e-2, "index_score XMX (%s keys): worst %.2e, %d -inf mismatches",
              kf == ds4k::kKvFp4p ? "fp4p" : "bf16", worst, inf_bad);
        free_ix(x);
    }
}

// a prompt chunk's sparse attention over packed rows: XMX vs FMA on T queries of 64 heads, the window's BF16 rows and n_sel
// selected compressed rows each (random codes, E4M3 scales near 1), sinks; every output within 2% of its head's largest
static void check_attn(cudaStream_t s) {
    std::mt19937 rng(17);
    Normal nd;
    const int T = 9, H = 64, D = 512, WIN = 128, NC = 700, KC = 512, R = 4;
    const long long p0 = 2600;   // (query t sees min(512, (p0 + t + 1) / 4) selected rows and the window)
    std::vector<float> hq((size_t) T * H * D), hs(H);
    for (float& v : hq) v = nd(rng) * 0.05f;
    for (float& v : hs) v = nd(rng);
    std::vector<uint16_t> hw((size_t) (WIN + T) * D);
    for (uint16_t& v : hw) {
        const float f = nd(rng);
        uint32_t b;
        std::memcpy(&b, &f, 4);
        v = (uint16_t) (b >> 16);
    }
    const size_t crow = ds4k::kv_row_bytes(ds4k::kKvFp4, D);
    std::vector<uint8_t> hc((size_t) NC * crow, 0);
    for (int r = 0; r < NC; ++r) {
        uint8_t* row = hc.data() + (size_t) r * crow;
        for (int i = 0; i < 256; ++i) row[i] = (uint8_t) rng();
        for (int i = 0; i < 32; ++i) row[256 + i] = (uint8_t) (0x30 + rng() % 16);   // (E4M3 0.5 .. ~2)
    }
    std::vector<int> hsel((size_t) T * KC);
    for (int& v : hsel) v = (int) (rng() % NC);
    float *q, *sk, *o;
    uint16_t* w;
    uint8_t* c;
    int* sel;
    cudaMalloc(&q, hq.size() * 4);
    cudaMalloc(&sk, hs.size() * 4);
    cudaMalloc(&o, hq.size() * 4);
    cudaMalloc(&w, hw.size() * 2);
    cudaMalloc(&c, hc.size());
    cudaMalloc(&sel, hsel.size() * 4);
    cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(sk, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(c, hc.data(), hc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(sel, hsel.data(), hsel.size() * 4, cudaMemcpyHostToDevice);
    ds4k::AttnRows ar;
    ar.gen = 1;
    ar.p0 = p0;
    ar.t0 = 0;
    ar.win = WIN;
    ar.R = R;
    ar.kcap = KC;
    std::vector<float> a(hq.size()), b(hq.size());
    for (int mode : {0, 1}) {
        ds4k::attn_force_xmx(mode);
        cudaMemsetAsync(o, 0, hq.size() * 4, s);
        // (the window buffer: the ring's 128 rows before p0, then the chunk's rows - here random rows all of them)
        ds4k::sparse_attn(q, w, c, sel, KC, ar, sk, H, D, T, 1.0f / std::sqrt((float) D), o, s, ds4k::kKvFp4, 0, nullptr);
        cudaMemcpyAsync(mode ? b.data() : a.data(), o, hq.size() * 4, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
    }
    ds4k::attn_force_xmx(-1);
    double worst = 0;
    int bad = 0;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const float* u = a.data() + ((size_t) t * H + h) * D;
            const float* v = b.data() + ((size_t) t * H + h) * D;
            double mx = 1e-30;
            for (int i = 0; i < D; ++i) mx = std::max(mx, (double) std::fabs(u[i]));
            for (int i = 0; i < D; ++i) {
                if (!std::isfinite(v[i])) ++bad;
                else worst = std::max(worst, std::fabs((double) u[i] - v[i]) / mx);
            }
        }
    std::printf("sparse_attn (chunk rows, packed): XMX vs FMA worst %.2e of the head's largest, %d non-finite\n", worst, bad);
    CHECK(bad == 0 && worst < 2e-2, "sparse_attn XMX: worst %.2e, %d non-finite", worst, bad);
    cudaFree(q), cudaFree(sk), cudaFree(o), cudaFree(w), cudaFree(c), cudaFree(sel);
}

// ms per sparse_attn call of T chunk queries (64 heads, the window + 512 selected rows each): FMA, XMX 16 heads a block, XMX 64
static int attn_bench(int T) {
    std::mt19937 rng(3);
    Normal nd;
    const int H = 64, D = 512, WIN = 128, NC = 4096, KC = 512, R = 4;
    std::vector<float> hq((size_t) T * H * D), hs(H, 0.0f);
    for (float& v : hq) v = nd(rng) * 0.05f;
    std::vector<uint16_t> hw((size_t) (WIN + T) * D, 0x3f80);
    const size_t crow = ds4k::kv_row_bytes(ds4k::kKvFp4, D);
    std::vector<uint8_t> hc((size_t) NC * crow);
    for (size_t i = 0; i < hc.size(); ++i) hc[i] = (uint8_t) ((i % crow) >= 256 ? 0x38 : rng());
    std::vector<int> hsel((size_t) T * KC);
    for (int& v : hsel) v = (int) (rng() % NC);
    float *q, *sk, *o;
    uint16_t* w;
    uint8_t* c;
    int* sel;
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaMalloc(&q, hq.size() * 4), cudaMalloc(&sk, hs.size() * 4), cudaMalloc(&o, hq.size() * 4);
    cudaMalloc(&w, hw.size() * 2), cudaMalloc(&c, hc.size()), cudaMalloc(&sel, hsel.size() * 4);
    cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(sk, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(c, hc.data(), hc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(sel, hsel.data(), hsel.size() * 4, cudaMemcpyHostToDevice);
    ds4k::AttnRows ar;
    ar.gen = 1, ar.p0 = 100000, ar.t0 = 0, ar.win = WIN, ar.R = R, ar.kcap = KC;
    const bool x16 = !(std::getenv("DS4_ATTN_XMX64") != nullptr && std::atoi(std::getenv("DS4_ATTN_XMX64")) != 0);
    for (int m = 0; m < 2; ++m) {
        ds4k::attn_force_xmx(m);
        auto call = [&] { ds4k::sparse_attn(q, w, c, sel, KC, ar, sk, H, D, T, 0.0442f, o, s, ds4k::kKvFp4, 0, nullptr); };
        call();
        cudaStreamSynchronize(s);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) call();
        cudaStreamSynchronize(s);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5;
        std::printf("sparse_attn %-13s %d queries x 640 rows: %.2f ms (%.1f TFLOP/s)\n", m == 0 ? "FMA" : x16 ? "XMX 16 heads" : "XMX 64 heads",
                    T, ms, 4.0 * T * 640.0 * H * D / (ms * 1e-3) / 1e12);
    }
    ds4k::attn_force_xmx(-1);
    cudaFree(q), cudaFree(sk), cudaFree(o), cudaFree(w), cudaFree(c), cudaFree(sel);
    cudaStreamDestroy(s);
    return 0;
}
static int ix_bench(int T, int n) {
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    std::mt19937 rng(9);
    IxSet x = make_ix(T, n, ds4k::kKvFp4p, rng);
    const ds4k::Rows lim = ix_rows(2 * n - T);   // (the slab's last query sees every key)
    for (int mode : {0, 1}) {
        ds4k::index_force_xmx(mode);
        ds4k::index_score(x.q, x.w, x.k, 32, 128, T, n, nullptr, 0, 1, lim, x.out, n, s, ds4k::kKvFp4p);
        cudaStreamSynchronize(s);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 3; ++i) ds4k::index_score(x.q, x.w, x.k, 32, 128, T, n, nullptr, 0, 1, lim, x.out, n, s, ds4k::kKvFp4p);
        cudaStreamSynchronize(s);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 3;
        std::printf("index_score %-4s %d queries x %d keys: %.1f ms (%.1f TFLOP/s)\n", mode ? "XMX" : "FMA", T, n, ms,
                    2.0 * T * (double) n * 32 * 128 / (ms * 1e-3) / 1e12);
    }
    ds4k::index_force_xmx(-1);
    if (T <= 4) {   // the decode's top-k over the scores: the 512 positions, and the candidate source's 2048 blocks of 8
        int *idx = nullptr, *cand = nullptr;
        float* bm = nullptr;
        cudaMalloc(&idx, (size_t) T * 512 * 4), cudaMalloc(&cand, (size_t) T * 2048 * 4), cudaMalloc(&bm, (size_t) T * (n / 8 + 64) * 4);
        ds4k::Rows r, rc;
        for (int t = 0; t < T; ++t) r.n[t] = rc.n[t] = n, r.k[t] = 512, rc.k[t] = 2048;
        auto tm = [&](const char* nm, const std::function<void()>& f) {
            f();
            cudaStreamSynchronize(s);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 20; ++i) f();
            cudaStreamSynchronize(s);
            std::printf("%-28s %d x %d: %.1f us\n", nm, T, n,
                        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20);
        };
        tm("topk 512", [&] { ds4k::topk(x.out, n, T, r, nullptr, 0, 1, idx, 512, s); });
        {   // against the host's: the 512 largest (ties: the lowest positions) in position order
            std::vector<float> sc((size_t) n);
            std::vector<int> got(512);
            cudaMemcpy(sc.data(), x.out, (size_t) n * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(got.data(), idx, 512 * 4, cudaMemcpyDeviceToHost);
            std::vector<int> ord((size_t) n);
            for (int i = 0; i < n; ++i) ord[(size_t) i] = i;
            auto key = [&](int i) {   // (the kernel's order: the float's bits, -0 below +0)
                uint32_t u;
                std::memcpy(&u, &sc[(size_t) i], 4);
                return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            };
            std::stable_sort(ord.begin(), ord.end(), [&](int a2, int b2) { return key(a2) > key(b2); });
            std::vector<int> want(ord.begin(), ord.begin() + 512);
            std::sort(want.begin(), want.end());
            std::sort(got.begin(), got.end());   // (the kernel's: the ones above the threshold in position order, then its ties)
            std::printf("   topk vs host: %s\n", want == got ? "same" : "FAIL (differs)");
            if (want != got) {
                int shown = 0;
                for (int i = 0; i < 512 && shown < 4; ++i)
                    if (want[(size_t) i] != got[(size_t) i]) {
                        std::printf("     [%d] want %d (%g) got %d (%g)\n", i, want[(size_t) i], sc[(size_t) want[(size_t) i]], got[(size_t) i],
                                    got[(size_t) i] >= 0 && got[(size_t) i] < n ? sc[(size_t) got[(size_t) i]] : 0.0);
                        ++shown;
                    }
                std::printf("     512th largest %g, 513th %g\n", sc[(size_t) ord[511]], sc[(size_t) ord[512]]);
            }
            if (want != got) ++fails;
        }
        tm("cand_blocks 2048 of 8", [&] { ds4k::cand_blocks(x.out, n, T, rc, 8, bm, n / 8 + 64, cand, 2048, s); });
        tm("index_score FMA (default T=1)", [&] { ds4k::index_score(x.q, x.w, x.k, 32, 128, T, n, nullptr, 0, 1, lim, x.out, n, s, ds4k::kKvFp4p); });
        cudaFree(idx), cudaFree(cand), cudaFree(bm);
    }
    free_ix(x);
    cudaStreamDestroy(s);
    return 0;
}

static int bench_main(int type, int E, int R, int N) {
    std::printf("device: %s; %s, %d experts x %d rows, n_embd %d n_ff %d, light kernels %s\n",
                dpct::get_current_device().get_info<sycl::info::device::name>().c_str(), iq::name(type), E, R, kD, kF,
                std::getenv("STRATA_DS4_NO_LIGHT") ? "off" : "on (experts <= 16 rows)");
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    std::mt19937 rng(11);
    Set S;
    if (!make_set(S, type, std::vector<int>((size_t) E, R), rng)) {
        std::printf("allocation failed\n");
        return 1;
    }
    const float limit = 7.0f;
    for (int mode : {0, 1}) {
        iq::force_tiles(mode);
        if (mode == 1 && !iq::tiles_xmx()) break;
        double us[2];
        for (int gu = 1; gu >= 0; --gu) {
            const iq::SetArgs a = args(S, gu != 0, limit);
            for (int i = 0; i < 3; ++i) gu ? iq::set_gate_up(a, s) : iq::set_down(a, s);
            cudaStreamSynchronize(s);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) gu ? iq::set_gate_up(a, s) : iq::set_down(a, s);
            cudaStreamSynchronize(s);
            us[gu] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
        }
        const double flop = 2.0 * (double) E * R * kD * kF * 3;
        std::printf("%-10s gate_up %9.1f us  down %9.1f us  (%.1f TFLOP/s over both)\n", mode ? "XMX" : "F32 kernel", us[1], us[0],
                    flop / ((us[0] + us[1]) * 1e-6) / 1e12);
    }
    iq::force_tiles(-1);
    free_set(S);
    cudaStreamDestroy(s);
    return 0;
}


// --gemv: the decode's GEMVs at their shapes (one token): the dense Q5_K matrices (iq::mv) and 6 experts' gate / up and down
// (iq::gate_up / down, plan of 6) - GB/s of weights read
static int gemv_bench(int iters) {
    std::FILE* dump = std::getenv("DS4_GEMV_DUMP") ? std::fopen(std::getenv("DS4_GEMV_DUMP"), "wb") : nullptr;   // (outputs, to cmp two builds)
    auto put = [&](const void* dp, size_t bytes) {
        if (dump == nullptr) return;
        std::vector<uint8_t> v(bytes);
        cudaMemcpy(v.data(), dp, bytes, cudaMemcpyDeviceToHost);
        std::fwrite(v.data(), 1, bytes, dump);
    };
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    std::mt19937 rng(7);
    auto timeit = [&](const std::function<void()>& f) {
        f();
        cudaStreamSynchronize(s);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) f();
        cudaStreamSynchronize(s);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    };
    struct Sh {
        int n, k, cnt;
        const char* nm;
    } shapes[] = {{32768, 1280, 1, "wq_b"}, {5120, 8192, 1, "wo_b"}, {1024, 4096, 8, "wo_a x8"}, {2304, 5120, 2, "s1+s3"},
                  {5120, 2304, 1, "s2"}, {1280, 5120, 1, "wq_a"}};
    for (const Sh& sh : shapes) {
        const size_t rb = iq::row_bytes(iq::kQ5_K, sh.k), mb = rb * sh.n;
        constexpr int kRot = 8;   // (8 copies taken in turn: the smaller matrices otherwise come from the L2 cache)
        std::vector<uint8_t> hw(mb * sh.cnt * kRot);
        for (size_t i = 0; i < hw.size(); ++i) hw[i] = (uint8_t) rng();
        for (size_t b = 0; b < hw.size(); b += 176) hw[b] = 0x00, hw[b + 1] = 0x20, hw[b + 2] = 0x00, hw[b + 3] = 0x20;   // (d, dmin small)
        std::vector<int8_t> hx((size_t) sh.cnt * sh.k);
        for (int8_t& v : hx) v = (int8_t) (rng() % 255 - 127);
        std::vector<float> hd((size_t) sh.cnt * sh.k / 32, 0.01f);
        uint8_t* w;
        int8_t* xq;
        float *xd, *y;
        cudaMalloc(&w, hw.size()), cudaMalloc(&xq, hx.size()), cudaMalloc(&xd, hd.size() * 4), cudaMalloc(&y, (size_t) sh.cnt * sh.n * 4);
        cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(xq, hx.data(), hx.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(xd, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice);
        iq::MvJob J[8];
        int rot = 0;
        for (int i = 0; i < sh.cnt; ++i) {
            J[i].w = w + (size_t) i * mb;
            J[i].ld = rb;
            J[i].type = iq::kQ5_K;
            J[i].k = sh.k;
            J[i].n = sh.n;
            J[i].ncol = 1;
            J[i].xq = xq + (size_t) i * sh.k;
            J[i].xd = xd + (size_t) i * sh.k / 32;
            J[i].y = y + (size_t) i * sh.n;
            J[i].ycs = (size_t) sh.n;
        }
        auto turn = [&] {   // (the next copy)
            rot = (rot + 1) % kRot;
            for (int i = 0; i < sh.cnt; ++i) J[i].w = w + ((size_t) rot * sh.cnt + i) * mb;
        };
        const double us = timeit([&] { turn(); iq::mv(J, sh.cnt, s); });
        std::printf("mv Q5_K %-8s %5d x %5d x %d: %8.1f us  %6.1f GB/s\n", sh.nm, sh.n, sh.k, sh.cnt, us, mb * sh.cnt / us / 1e3);
        put(y, (size_t) sh.cnt * sh.n * 4);
        {   // the rows-a-warp forms against one row a warp: the same sums in another order
            std::vector<float> y0((size_t) sh.cnt * sh.n), y1(y0.size());
            iq::mv(J, sh.cnt, s);   // (this copy's, both forms)
            cudaStreamSynchronize(s);
            cudaMemcpy(y0.data(), y, y0.size() * 4, cudaMemcpyDeviceToHost);
            iq::mv_rows_per_warp(1);
            iq::mv(J, sh.cnt, s);
            cudaStreamSynchronize(s);
            iq::mv_rows_per_warp(0);
            cudaMemcpy(y1.data(), y, y1.size() * 4, cudaMemcpyDeviceToHost);
            double mx = 0, ref = 0;
            for (size_t i = 0; i < y0.size(); ++i) mx = std::max(mx, (double) std::fabs(y0[i] - y1[i])), ref = std::max(ref, (double) std::fabs(y1[i]));
            std::printf("   vs one row a warp: worst %.2e of the largest %.3e%s\n", mx / (ref + 1e-30), ref, mx <= 1e-5 * ref ? "" : "  FAIL");
        }
        cudaFree(w), cudaFree(xq), cudaFree(xd), cudaFree(y);
    }
    for (int type : {iq::kIQ2_XXS, iq::kIQ2_XS, iq::kIQ1_M, iq::kIQ2_XXS}) {
        Set S;
        if (!make_set(S, type, std::vector<int>(6, 1), rng)) continue;
        std::vector<unsigned long long> plan(8, 0ull);
        for (int e = 0; e < 6; ++e) plan[(size_t) e] = (unsigned long long) (S.w + (size_t) e * S.stride);
        unsigned long long* dplan;
        int8_t *xq, *hq;
        float *xd, *hd, *h, *part;
        cudaMalloc(&dplan, 64), cudaMalloc(&xq, kD), cudaMalloc(&xd, kD / 32 * 4), cudaMalloc(&hq, 8 * kF), cudaMalloc(&hd, 8 * kF / 32 * 4);
        cudaMalloc(&h, 8 * kF * 4), cudaMalloc(&part, 8 * kD * 4);
        cudaMemcpy(dplan, plan.data(), 64, cudaMemcpyHostToDevice);
        std::vector<int8_t> hx(8 * kF);
        for (int8_t& v : hx) v = (int8_t) (rng() % 255 - 127);
        std::vector<float> hs(8 * kF / 32, 0.01f);
        cudaMemcpy(xq, hx.data(), kD, cudaMemcpyHostToDevice), cudaMemcpy(hq, hx.data(), 8 * kF, cudaMemcpyHostToDevice);
        cudaMemcpy(xd, hs.data(), kD / 32 * 4, cudaMemcpyHostToDevice), cudaMemcpy(hd, hs.data(), 8 * kF / 32 * 4, cudaMemcpyHostToDevice);
        iq::Args a;
        a.plan_ptr = dplan;
        for (int r = 0; r < 3; ++r) a.tr[r] = S.tr[r];
        a.n_embd = kD, a.n_ff = kF, a.limit = 10.0f, a.xq = xq, a.xd = xd, a.hq = hq, a.hd = hd, a.h = h, a.part = part;
        a.tg = a.td = type;
        const double gu = timeit([&] { iq::gate_up(a, 6, 1, s); }), dn = timeit([&] { iq::down(a, 6, 1, s); });
        std::printf("experts %-8s x6: gate_up %7.1f us %6.1f GB/s | down %7.1f us %6.1f GB/s\n", iq::name(type), gu,
                    6.0 * 2 * kF * S.rbD / gu / 1e3, dn, 6.0 * kD * S.rbF / dn / 1e3);
        put(h, 6 * kF * 4);
        put(part, 6 * kD * 4);
        cudaFree(dplan), cudaFree(xq), cudaFree(xd), cudaFree(hq), cudaFree(hd), cudaFree(h), cudaFree(part);
        free_set(S);
    }
    cudaStreamDestroy(s);
    if (dump != nullptr) std::fclose(dump);
    return 0;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // (a line at a time: a timeout keeps what was printed)
    if (argc >= 5 && std::strcmp(argv[1], "--bench") == 0) {
        const std::string t = argv[2];
        const int type = t == "IQ2_XS" ? iq::kIQ2_XS : t == "IQ1_M" ? iq::kIQ1_M : t == "IQ3_XXS" ? iq::kIQ3_XXS : iq::kIQ2_XXS;
        return bench_main(type, std::atoi(argv[3]), std::atoi(argv[4]), argc > 5 ? std::atoi(argv[5]) : 20);
    }
    if (argc >= 2 && std::strcmp(argv[1], "--gemv") == 0) return gemv_bench(argc > 2 ? std::atoi(argv[2]) : 200);
    if (argc >= 3 && std::strcmp(argv[1], "--attnbench") == 0) return attn_bench(std::atoi(argv[2]));
    if (argc >= 4 && std::strcmp(argv[1], "--ixbench") == 0) return ix_bench(std::atoi(argv[2]), std::atoi(argv[3]));
    return check_main();
}
