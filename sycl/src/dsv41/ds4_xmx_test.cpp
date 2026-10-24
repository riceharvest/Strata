// ds4_xmx_test: the prompt path's expert sets (ds4_iq's set_gate_up / set_down, the GGUF low-bit experts) on Xe2's XMX
// (set_xmx_kernel, joint_matrix) against the F32 kernel and a host reference, on synthetic experts of DeepSeek-V4.1-Flash's
// shape (n_embd 5120, n_ff 2304) in the official sets' types (IQ2_XXS, IQ2_XS, IQ1_M) and IQ3_XXS.
//   ds4_xmx_test                       the check (ctest): rows per expert 1 .. 130 (partial tiles), the light kernels off
//   ds4_xmx_test --bench TYPE E R [N]  us per set_gate_up / set_down call, F32 kernel vs XMX, E experts of R rows each
// About 0.2 GB of device memory for the check; the bench E x 9 MB (IQ2_XXS).
#include "ds4_sycl.hpp"
#include "ds4_iq.cuh"

#include <chrono>
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
    return true;
}
static void free_set(Set& S) {
    for (void* p : {(void*) S.w, (void*) S.d_bounds, (void*) S.d_idx, (void*) S.d_x, (void*) S.d_h, (void*) S.d_hin, (void*) S.d_y}) cudaFree(p);
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
    cudaStreamDestroy(s);
    std::printf(fails == 0 ? "ALL OK\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
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

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // (a line at a time: a timeout keeps what was printed)
    if (argc >= 5 && std::strcmp(argv[1], "--bench") == 0) {
        const std::string t = argv[2];
        const int type = t == "IQ2_XS" ? iq::kIQ2_XS : t == "IQ1_M" ? iq::kIQ1_M : t == "IQ3_XXS" ? iq::kIQ3_XXS : iq::kIQ2_XXS;
        return bench_main(type, std::atoi(argv[3]), std::atoi(argv[4]), argc > 5 ? std::atoi(argv[5]) : 20);
    }
    return check_main();
}
