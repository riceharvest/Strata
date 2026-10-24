// --prune-experts' plan, apart from the model so it can be tested on its own (ds4_prune_test): the expert profile read
// from its file, the refusal without a profile, the water-filling of the experts kept per layer and the router mask.
// Pure host code.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace strata::ds4::prune {

/// A masked expert's router bias: score + bias never wins against a kept expert's (scores are sqrt(softplus) of the
/// logits, real biases O(1)), and stays above the -FLT_MAX the routers use for "already picked".
constexpr float kMaskBias = -1e9f;

/// One layer as the plan sees it: its index into the profile (rows n_exp_stride apart), its experts, its picks, and
/// whether it may be pruned (a routed trunk layer: not the drafter's mtp layers, not a layer without a router bias).
struct LayerInfo {
    int il = 0;
    int n_exp = 0;
    int k_exp = 0;
    bool prunable = false;
};

/// The profile file into prof (sized by the caller, zeros): read whole when it holds every row (trunk + drafter) or
/// exactly the trunk's (trunk_vals values); a missing file or one of another size leaves prof all zeros.
inline void load_profile(const std::string& path, std::vector<uint32_t>& prof, size_t trunk_vals) {
    if (path.empty()) return;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return;
    std::vector<uint32_t> p(prof.size());
    const size_t n = std::fread(p.data(), 4, p.size(), f);
    if (n == p.size() || n == trunk_vals) std::copy(p.begin(), p.begin() + (long) n, prof.begin());
    std::fclose(f);
}

/// The first prunable layer whose profile row has no hits (its "coldest" experts would be its last indices), -1: none.
inline int refused_layer(const std::vector<LayerInfo>& L, const uint32_t* prof, int stride) {
    for (const LayerInfo& l : L) {
        if (!l.prunable) continue;
        const uint32_t* p = prof + (size_t) l.il * stride;
        uint64_t sum = 0;
        for (int e = 0; e < l.n_exp; ++e) sum += p[e];
        if (sum == 0) return l.il;
    }
    return -1;
}

/// A layer's experts, the profile's hottest first (ties by index).
inline std::vector<int> hot_order(const uint32_t* p, int n) {
    std::vector<int> order((size_t) n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
    return order;
}

/// The experts each layer keeps (index il; 0: not pruned) for a target of prune_experts a layer on average over the
/// prunable layers: the largest coverage floor (the share of a layer's profile hits its kept experts take) that fits the
/// budget, every layer at least k_exp, then one expert at a time to the least covered layer (or from the most covered)
/// until the total is the budget.
inline std::vector<int> allocate(const std::vector<LayerInfo>& L, const uint32_t* prof, int stride, int prune_experts) {
    std::vector<int> alloc(L.size(), 0);
    if (prune_experts <= 0 || L.empty()) return alloc;
    int n_moe = 0;
    for (const LayerInfo& l : L)
        if (l.prunable) ++n_moe;
    if (n_moe == 0) return alloc;
    const int target_budget = std::min(prune_experts, L[0].n_exp) * n_moe;
    std::vector<std::vector<double>> cum_p(L.size());
    for (const LayerInfo& l : L) {
        if (!l.prunable) continue;
        const uint32_t* p = prof + (size_t) l.il * stride;
        std::vector<uint32_t> sorted_p(p, p + l.n_exp);
        std::sort(sorted_p.begin(), sorted_p.end(), std::greater<uint32_t>());
        double sum = 0;
        for (uint32_t x : sorted_p) sum += x;
        if (sum <= 0) sum = 1.0;
        cum_p[(size_t) l.il].resize((size_t) l.n_exp);
        double cur = 0;
        for (int i = 0; i < l.n_exp; ++i) {
            cur += sorted_p[(size_t) i];
            cum_p[(size_t) l.il][(size_t) i] = cur / sum;
        }
    }
    // Binary search for coverage floor F in [0.0, 1.0]
    double low = 0.0, high = 1.0;
    for (int iter = 0; iter < 50; ++iter) {
        const double mid = 0.5 * (low + high);
        int needed = 0;
        for (const LayerInfo& l : L) {
            if (!l.prunable) continue;
            const auto& c = cum_p[(size_t) l.il];
            auto it = std::lower_bound(c.begin(), c.end(), mid);
            int idx = (int) std::distance(c.begin(), it) + 1;
            needed += std::max(l.k_exp, std::min(l.n_exp, idx));
        }
        if (needed > target_budget) high = mid;
        else low = mid;
    }
    int cur_budget = 0;
    for (const LayerInfo& l : L) {
        if (!l.prunable) continue;
        const auto& c = cum_p[(size_t) l.il];
        auto it = std::lower_bound(c.begin(), c.end(), low);
        int idx = (int) std::distance(c.begin(), it) + 1;
        alloc[(size_t) l.il] = std::max(l.k_exp, std::min(l.n_exp, idx));
        cur_budget += alloc[(size_t) l.il];
    }
    while (cur_budget < target_budget) {
        int best_l = -1;
        double min_cov = 2.0;
        for (const LayerInfo& l : L) {
            if (!l.prunable || alloc[(size_t) l.il] >= l.n_exp) continue;
            const double cov = cum_p[(size_t) l.il][(size_t) (alloc[(size_t) l.il] - 1)];
            if (cov < min_cov) { min_cov = cov; best_l = l.il; }
        }
        if (best_l < 0) break;
        alloc[(size_t) best_l]++;
        cur_budget++;
    }
    while (cur_budget > target_budget) {
        int best_l = -1;
        double max_cov = -1.0;
        for (const LayerInfo& l : L) {
            if (!l.prunable || alloc[(size_t) l.il] <= l.k_exp) continue;
            const double cov = cum_p[(size_t) l.il][(size_t) (alloc[(size_t) l.il] - 1)];
            if (cov > max_cov) { max_cov = cov; best_l = l.il; }
        }
        if (best_l < 0) break;
        alloc[(size_t) best_l]--;
        cur_budget--;
    }
    return alloc;
}

/// Whether a layer is masked at all: pruned, with fewer experts kept than it has.
inline bool masks(const LayerInfo& l, int keep) { return l.prunable && keep > 0 && keep < l.n_exp; }

/// The mask over n values (a router bias, the expert cache's heat): order[keep .. n) (the coldest) set to kMaskBias.
inline void mask(float* v, const std::vector<int>& order, int keep, int n) {
    for (int i = keep; i < n; ++i) v[(size_t) order[(size_t) i]] = kMaskBias;
}

}  // namespace strata::ds4::prune
