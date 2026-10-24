// ds4_prune_test: --prune-experts end to end on synthetic data - the plan (ds4_prune.hpp: the profile file, the refusal,
// the water-filling, the mask) on the host, then the engine's own routers (route, route_publish, route_ahead from
// ds4_kernels) on the device with the masked biases, against a host reference.  DeepSeek-V4.1-Flash's router shape: 40
// layers of 384 experts, top 6, plus one drafter (mtp) layer.  About 3 MB of device memory.
#include "ds4_sycl.hpp"
#include "ds4_kernels.cuh"
#include "ds4_prune.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace pr = strata::ds4::prune;
using strata::ds4::kPicks;
using strata::ds4::kWin;
using strata::ds4::VlBits;

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

constexpr int kNl = 40, kMtp = 1, kNexp = 384, kK = 6, kBudget = 160, kNoBias = 2;   // (layer 2: no router bias)
constexpr float kScale = 2.5f;

// the router's arithmetic, the kernels' order: score = sqrt(softplus(logit)), the k best of score + bias (ties: the lower
// index), weights = score / sum(scores of the picks) * scale
struct Ref {
    std::vector<int> idx;
    std::vector<double> sc, bs;
};
static Ref ref_route(const float* logit, const float* bias, int n, int k) {
    Ref r;
    r.sc.resize((size_t) n);
    r.bs.resize((size_t) n);
    for (int i = 0; i < n; ++i) {
        const double l = logit[i];
        r.sc[(size_t) i] = std::sqrt(l > 20.0 ? l : std::log1p(std::exp(l)));
        r.bs[(size_t) i] = r.sc[(size_t) i] + (double) bias[i];
    }
    std::vector<double> b = r.bs;
    for (int j = 0; j < k; ++j) {
        int best = 0;
        for (int i = 1; i < n; ++i)
            if (b[(size_t) i] > b[(size_t) best]) best = i;
        r.idx.push_back(best);
        b[(size_t) best] = -INFINITY;
    }
    return r;
}

// the device's picks against the reference: the same experts in the same order, or a pick whose biased score is within a
// float rounding of the reference's (a near-tie the F32 softplus may order the other way)
static bool same_picks(const int* got, const Ref& r, int k) {
    for (int j = 0; j < k; ++j)
        if (got[j] != r.idx[(size_t) j] && std::fabs(r.bs[(size_t) got[j]] - r.bs[(size_t) r.idx[(size_t) j]]) > 1e-5) return false;
    return true;
}
// weights renormalised over the picks: w_j = sc_j / sum sc * scale, so they sum to scale
static double weight_err(const int* got, const float* w, const Ref& r, int k) {
    double s = 0, e = 0;
    for (int j = 0; j < k; ++j) s += r.sc[(size_t) got[j]];
    for (int j = 0; j < k; ++j) e = std::max(e, std::fabs((double) w[j] - r.sc[(size_t) got[j]] / s * kScale));
    return e;
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // (a line at a time: a timeout keeps what was printed)
    std::mt19937 rng(20261024);
    const Normal nd;
    const int n_rows = kNl + kMtp;

    // ------------------------------------------------------------------ the plan on the host
    std::vector<pr::LayerInfo> L;
    for (int il = 0; il < n_rows; ++il) L.push_back({il, kNexp, kK, il < kNl && il != kNoBias});   // (il kNl: the drafter)
    // a profile: per layer a Zipf over a shuffled expert order (the hot experts are not the low indices), the skew by layer
    std::vector<uint32_t> prof((size_t) n_rows * kNexp, 0u);
    for (int il = 0; il < n_rows; ++il) {
        std::vector<int> perm(kNexp);
        for (int i = 0; i < kNexp; ++i) perm[(size_t) i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        const double s = 0.5 + 1.0 * (double) (il % 7) / 6.0;
        for (int r = 0; r < kNexp; ++r) prof[(size_t) il * kNexp + perm[(size_t) r]] = 1u + (uint32_t) (200000.0 / std::pow(r + 1.0, s));
    }
    // the profile file: whole, the trunk's rows only, another size, missing
    {
        const char* td = std::getenv("TMPDIR");
        const std::string base = std::string(td != nullptr && td[0] != 0 ? td : "/tmp") + "/ds4_prune_test." + std::to_string(getpid());
        auto write = [&](const std::string& p, size_t vals) {
            std::FILE* f = std::fopen(p.c_str(), "wb");
            if (f == nullptr) return false;
            const bool ok = std::fwrite(prof.data(), 4, vals, f) == vals;
            return std::fclose(f) == 0 && ok;
        };
        const size_t trunk = (size_t) kNl * kNexp;
        struct Case { const char* name; size_t vals; bool whole, trunk_only; };
        const Case cases[] = {{"whole", prof.size(), true, false}, {"trunk rows only", trunk, false, true},
                              {"another size", trunk - 7, false, false}, {"missing", 0, false, false}};
        for (const Case& c : cases) {
            const std::string p = base + "." + std::to_string(c.vals);
            if (c.vals > 0) CHECK(write(p, c.vals), "writing %s", p.c_str());
            std::vector<uint32_t> got(prof.size(), 0u);
            pr::load_profile(p, got, trunk);
            if (c.vals > 0) std::remove(p.c_str());
            size_t same_trunk = 0, nz_mtp = 0, nz = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                if (i < trunk) same_trunk += got[i] == prof[i];
                else nz_mtp += got[i] != 0;
                nz += got[i] != 0;
            }
            const int bad = pr::refused_layer(L, got.data(), kNexp);
            if (c.whole) CHECK(got == prof && bad == -1, "profile %s: read back %d, refused at %d", c.name, (int) (got == prof), bad);
            else if (c.trunk_only) CHECK(same_trunk == trunk && nz_mtp == 0 && bad == -1, "profile %s: %zu of %zu, drafter %zu, refused %d", c.name, same_trunk, trunk, nz_mtp, bad);
            else CHECK(nz == 0 && bad == 0, "profile %s: %zu values kept, refused at %d (want all zeros, refused at layer 0)", c.name, nz, bad);
            std::printf("profile file %-16s -> %s\n", c.name, bad < 0 ? "pruning allowed" : ("refused at layer " + std::to_string(bad)).c_str());
        }
    }
    // the refusal: a trunk row without hits refuses; the drafter's row and the bias-less layer's do not count
    {
        std::vector<uint32_t> p = prof;
        std::fill(p.begin() + (size_t) kNl * kNexp, p.end(), 0u);
        std::fill(p.begin() + (size_t) kNoBias * kNexp, p.begin() + (size_t) (kNoBias + 1) * kNexp, 0u);
        CHECK(pr::refused_layer(L, p.data(), kNexp) == -1, "zero drafter / bias-less rows refused (%d)", pr::refused_layer(L, p.data(), kNexp));
        std::fill(p.begin() + (size_t) 17 * kNexp, p.begin() + (size_t) 18 * kNexp, 0u);
        CHECK(pr::refused_layer(L, p.data(), kNexp) == 17, "zero row 17 -> refused at %d", pr::refused_layer(L, p.data(), kNexp));
    }
    // the water-filling
    const std::vector<int> alloc = pr::allocate(L, prof.data(), kNexp, kBudget);
    auto coverage = [&](int il, int keep) {
        const uint32_t* p = prof.data() + (size_t) il * kNexp;
        const std::vector<int> o = pr::hot_order(p, kNexp);
        double kept = 0, tot = 0;
        for (int i = 0; i < kNexp; ++i) {
            tot += p[o[(size_t) i]];
            if (i < keep) kept += p[o[(size_t) i]];
        }
        return kept / tot;
    };
    int lo_l = -1, hi_l = -1, total = 0, n_moe = 0;
    double min_cov = 1, min_cov_flat = 1;
    for (const pr::LayerInfo& l : L) {
        const int a = alloc[(size_t) l.il];
        if (!l.prunable) {
            CHECK(a == 0, "layer %d (%s) got %d experts in the budget", l.il, l.il == kNl ? "drafter" : "no bias", a);
            CHECK(!pr::masks(l, a), "layer %d masked", l.il);
            continue;
        }
        ++n_moe;
        total += a;
        CHECK(a >= kK && a <= kNexp, "layer %d keeps %d", l.il, a);
        if (lo_l < 0 || a < alloc[(size_t) lo_l]) lo_l = l.il;
        if (hi_l < 0 || a > alloc[(size_t) hi_l]) hi_l = l.il;
        min_cov = std::min(min_cov, coverage(l.il, a));
        min_cov_flat = std::min(min_cov_flat, coverage(l.il, kBudget));
    }
    CHECK(total == kBudget * n_moe, "budget %d, want %d", total, kBudget * n_moe);
    CHECK(min_cov >= min_cov_flat - 1e-12, "water-filling's least covered layer %.4f below a flat %d's %.4f", min_cov, kBudget, min_cov_flat);
    std::printf("water-filling: %d layers, %d..%d experts a layer (layers %d / %d), total %d; least covered layer %.1f%% (flat %d: %.1f%%)\n",
                n_moe, alloc[(size_t) lo_l], alloc[(size_t) hi_l], lo_l, hi_l, total, 100 * min_cov, kBudget, 100 * min_cov_flat);
    {
        const std::vector<int> none = pr::allocate(L, prof.data(), kNexp, 0);
        const std::vector<int> all = pr::allocate(L, prof.data(), kNexp, 1000);
        int bad = 0;
        for (const pr::LayerInfo& l : L) bad += none[(size_t) l.il] != 0 || pr::masks(l, all[(size_t) l.il]);
        CHECK(bad == 0, "--prune-experts 0 / >= 384: %d layers would be masked", bad);
    }

    // the masked biases (each layer's router bias and image-token bias, as every member's copy is masked), on the host
    std::vector<float> bias((size_t) n_rows * kNexp), bias_vl((size_t) n_rows * kNexp);
    for (float& v : bias) v = 0.1f * nd(rng);
    for (float& v : bias_vl) v = 0.1f * nd(rng);
    const std::vector<float> bias0 = bias, bias_vl0 = bias_vl;
    std::vector<std::vector<int>> masked((size_t) n_rows);   // per layer: the masked experts
    for (const pr::LayerInfo& l : L) {
        const uint32_t* p = prof.data() + (size_t) l.il * kNexp;
        const std::vector<int> order = pr::hot_order(p, kNexp);
        for (int i = 0; i + 1 < kNexp; ++i) {
            const int a = order[(size_t) i], b = order[(size_t) i + 1];
            if (p[a] < p[b] || (p[a] == p[b] && a > b)) {
                CHECK(false, "hot_order layer %d at %d", l.il, i);
                break;
            }
        }
        const int keep = alloc[(size_t) l.il];
        if (!pr::masks(l, keep)) continue;
        for (float* b : {bias.data() + (size_t) l.il * kNexp, bias_vl.data() + (size_t) l.il * kNexp}) pr::mask(b, order, keep, kNexp);
        uint32_t min_kept = UINT32_MAX, max_masked = 0;
        for (int i = 0; i < kNexp; ++i) {
            const int e = order[(size_t) i];
            if (i < keep) min_kept = std::min(min_kept, p[e]);
            else {
                max_masked = std::max(max_masked, p[e]);
                masked[(size_t) l.il].push_back(e);
            }
        }
        CHECK(max_masked <= min_kept, "layer %d masks an expert with %u hits, keeps one with %u", l.il, max_masked, min_kept);
    }
    {   // exactly the masked entries changed; the drafter's and the bias-less layer's rows bit for bit as they were
        int bad = 0;
        for (const pr::LayerInfo& l : L) {
            const size_t o = (size_t) l.il * kNexp;
            int n_masked = 0;
            for (int e = 0; e < kNexp; ++e) {
                const bool m = bias[o + e] == pr::kMaskBias;
                n_masked += m;
                bad += m != (bias_vl[o + e] == pr::kMaskBias);
                if (!m) bad += std::memcmp(&bias[o + e], &bias0[o + e], 4) != 0 || std::memcmp(&bias_vl[o + e], &bias_vl0[o + e], 4) != 0;
            }
            bad += n_masked != (int) masked[(size_t) l.il].size();
            if (!l.prunable) bad += n_masked != 0;
        }
        CHECK(bad == 0, "mask: %d entries wrong", bad);
    }

    // ------------------------------------------------------------------ the routers on the device
    std::printf("device: %s\n", dpct::get_current_device().get_info<sycl::info::device::name>().c_str());
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    const int T = 512;
    float *d_log = nullptr, *d_b = nullptr, *d_bvl = nullptr, *d_w = nullptr;
    int* d_idx = nullptr;
    cudaMalloc((void**) &d_log, (size_t) T * 1024 * 4);
    cudaMalloc((void**) &d_b, kNexp * 4);
    cudaMalloc((void**) &d_bvl, kNexp * 4);
    cudaMalloc((void**) &d_w, (size_t) T * kPicks * 4);
    cudaMalloc((void**) &d_idx, (size_t) T * kPicks * 4);
    VlBits vb{};
    for (int t = 0; t < T; t += 3) vb.w[t >> 5] |= 1u << (t & 31);   // every third row an image token
    std::vector<float> logit((size_t) T * 1024, 0.0f);
    std::vector<int> idx((size_t) T * kPicks);
    std::vector<float> w((size_t) T * kPicks);

    for (const int il : {lo_l, hi_l, 20, kNl}) {   // the most and the least pruned layers, one more, the drafter
        const float* b = bias.data() + (size_t) il * kNexp;
        const float* bvl = bias_vl.data() + (size_t) il * kNexp;
        const std::vector<int>& mk = masked[(size_t) il];
        std::vector<char> is_m(kNexp, 0);
        for (int e : mk) is_m[(size_t) e] = 1;
        // logits: noise, and per token three of the layer's cold experts (masked where the layer is pruned) pushed up, so
        // that the unpruned router would take them - the mask has to move those picks
        const std::vector<int> cold = [&] {
            const std::vector<int> o = pr::hot_order(prof.data() + (size_t) il * kNexp, kNexp);
            const int nc = mk.empty() ? 64 : std::min(64, (int) mk.size());   // (the masked ones where there are)
            return std::vector<int>(o.end() - nc, o.end());
        }();
        for (int t = 0; t < T; ++t) {
            float* lg = logit.data() + (size_t) t * 1024;
            for (int e = 0; e < kNexp; ++e) lg[e] = 1.5f * nd(rng);
            for (int j = 0; j < 3; ++j) lg[cold[(size_t) (rng() % cold.size())]] += 5.0f;
        }
        cudaMemcpy(d_log, logit.data(), logit.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, kNexp * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bvl, bvl, kNexp * 4, cudaMemcpyHostToDevice);
        strata::ds4::route(d_log, d_b, kNexp, kK, T, kScale, d_idx, d_w, s, d_bvl, &vb);
        cudaMemcpyAsync(idx.data(), d_idx, idx.size() * 4, cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(w.data(), d_w, w.size() * 4, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        int picked_masked = 0, mismatch = 0, would = 0, moved = 0;
        double werr = 0, wsum_err = 0;
        for (int t = 0; t < T; ++t) {
            const bool img = (vb.w[t >> 5] >> (t & 31)) & 1u;
            const float* lg = logit.data() + (size_t) t * 1024;
            const int* gi = idx.data() + (size_t) t * kPicks;
            const float* gw = w.data() + (size_t) t * kPicks;
            const Ref r = ref_route(lg, img ? bvl : b, kNexp, kK);
            const Ref u = ref_route(lg, (img ? bias_vl0 : bias0).data() + (size_t) il * kNexp, kNexp, kK);   // unpruned
            double ws = 0;
            bool u_m = false;
            for (int j = 0; j < kK; ++j) {
                picked_masked += is_m[(size_t) gi[j]];
                ws += gw[j];
                u_m = u_m || is_m[(size_t) u.idx[(size_t) j]];
            }
            would += u_m;
            moved += u.idx != r.idx;
            mismatch += !same_picks(gi, r, kK);
            werr = std::max(werr, weight_err(gi, gw, r, kK));
            wsum_err = std::max(wsum_err, std::fabs(ws - kScale));
        }
        std::printf("route layer %2d keeps %3d of %d: %4d of %d tokens would pick a masked expert unpruned, %4d changed; picked masked %d, "
                    "mismatches %d, max |w - ref| %.2e, max |sum w - %.1f| %.2e\n",
                    il, alloc[(size_t) il] > 0 ? alloc[(size_t) il] : kNexp, kNexp, would, T, moved, picked_masked, mismatch, werr, kScale, wsum_err);
        CHECK(picked_masked == 0, "layer %d: %d picks of masked experts", il, picked_masked);
        CHECK(mismatch == 0, "layer %d: %d tokens' picks differ from the reference", il, mismatch);
        CHECK(werr < 2e-5 && wsum_err < 2e-5, "layer %d: weights off by %.2e (sum %.2e)", il, werr, wsum_err);
        if (il == kNl) CHECK(mk.empty() && moved == 0, "drafter layer: %zu masked, %d tokens changed", mk.size(), moved);
        else CHECK(would > T / 2 && moved == would, "layer %d: only %d of %d tokens test the mask (%d changed)", il, would, T, moved);

        // route_publish (the async decode's router): a window's T rows, the image bits as a word, the plan's pointers
        {
            const int Tw = kWin;
            const unsigned vl = 0x5555u;
            std::vector<unsigned long long> vram(kNexp);
            for (int e = 0; e < kNexp; ++e) vram[(size_t) e] = e % 2 ? 0ull : 0x100000ull + 4096ull * (unsigned long long) e;
            unsigned long long *d_vram = nullptr, *d_plan = nullptr;
            float* d_x = nullptr;
            cudaMalloc((void**) &d_vram, kNexp * 8);
            cudaMalloc((void**) &d_plan, (size_t) Tw * kPicks * 8);
            cudaMalloc((void**) &d_x, (size_t) Tw * 64 * 4);
            cudaMemcpy(d_vram, vram.data(), kNexp * 8, cudaMemcpyHostToDevice);
            strata::ds4::route_publish(d_log, d_b, kNexp, kK, Tw, kScale, d_vram, d_x, 64, d_idx, d_w, d_plan, nullptr, 0u, s,
                                       d_bvl, vl);
            std::vector<unsigned long long> plan((size_t) Tw * kPicks);
            cudaMemcpyAsync(idx.data(), d_idx, (size_t) Tw * kPicks * 4, cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(w.data(), d_w, (size_t) Tw * kPicks * 4, cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(plan.data(), d_plan, plan.size() * 8, cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            int bad = 0, pm = 0;
            for (int t = 0; t < Tw; ++t) {
                const int* gi = idx.data() + (size_t) t * kPicks;
                const Ref r = ref_route(logit.data() + (size_t) t * 1024, (vl >> t) & 1u ? bvl : b, kNexp, kK);
                bad += !same_picks(gi, r, kK) || weight_err(gi, w.data() + (size_t) t * kPicks, r, kK) > 2e-5;
                for (int j = 0; j < kK; ++j) {
                    pm += is_m[(size_t) gi[j]];
                    bad += plan[(size_t) t * kPicks + j] != vram[(size_t) gi[j]];
                }
            }
            CHECK(bad == 0 && pm == 0, "route_publish layer %d: %d wrong, %d masked picks", il, bad, pm);
            cudaFree(d_vram);
            cudaFree(d_plan);
            cudaFree(d_x);
        }
    }

    // route_ahead (the lookahead's guesses, m up to 16 with DS4_AHEAD_EXTRA): a layer that keeps 8 guesses its 8 kept experts
    // in score order and no masked one (-1 for the rest)
    {
        const int keep = 8, m = strata::ds4::kAhead1, Tw = kWin;
        const int il = 20;
        const std::vector<int> order = pr::hot_order(prof.data() + (size_t) il * kNexp, kNexp);
        std::vector<float> b(bias0.begin() + (size_t) il * kNexp, bias0.begin() + (size_t) (il + 1) * kNexp);
        pr::mask(b.data(), order, keep, kNexp);
        cudaMemcpy(d_b, b.data(), kNexp * 4, cudaMemcpyHostToDevice);
        strata::ds4::Req* d_req = nullptr;
        cudaMalloc((void**) &d_req, sizeof(strata::ds4::Req));
        cudaMemset(d_req, 0x7f, sizeof(strata::ds4::Req));
        strata::ds4::route_ahead(d_log, d_b, kNexp, m, Tw, d_req, 0, s);
        std::vector<strata::ds4::Req> h(1);
        cudaMemcpyAsync(h.data(), d_req, sizeof(strata::ds4::Req), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        int bad = 0, masked_guess = 0;
        for (int t = 0; t < Tw; ++t) {
            const Ref r = ref_route(logit.data() + (size_t) t * 1024, b.data(), kNexp, keep);
            for (int i = 0; i < m; ++i) {
                const int e = h[0].ahead[t][i];
                if (i < keep) bad += e != r.idx[(size_t) i] && (e < 0 || std::fabs(r.bs[(size_t) e] - r.bs[(size_t) r.idx[(size_t) i]]) > 1e-5);
                else bad += e != -1;
                masked_guess += e >= 0 && b[(size_t) e] == pr::kMaskBias;
            }
        }
        std::printf("route_ahead: m %d on a layer keeping %d: %d wrong guesses, %d masked guesses\n", m, keep, bad, masked_guess);
        CHECK(bad == 0 && masked_guess == 0, "route_ahead: %d wrong, %d masked guesses", bad, masked_guess);
        cudaFree(d_req);
    }
    cudaFree(d_log);
    cudaFree(d_b);
    cudaFree(d_bvl);
    cudaFree(d_w);
    cudaFree(d_idx);
    cudaStreamDestroy(s);
    std::printf(fails == 0 ? "ALL OK\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
