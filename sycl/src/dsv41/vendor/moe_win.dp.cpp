// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT).
// Ported into Strata for Intel Arc Pro B70 (Xe2 / bmg-g31).
// sycl/src/dsv41/vendor/moe_win.dp.cpp - the window's expert grouping kernel.
#include "ds4_sycl.hpp"
#include "strata/dsv41/moe_win.hpp"
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41k::glmf {
namespace {
constexpr int kWinEnt = kMaxWinTokens * 8;

void moe_win_group_kernel(const MoeWin w) {
    SYCL_SHARED(unsigned long long[kWinEnt], gp);
    SYCL_SHARED(int[kWinEnt], cnt);
    SYCL_SHARED(int[kWinEnt], gof);
    SYCL_SHARED(int[kWinEnt + 1], fill);

    if (threadIdx.x != 0) return;
    int ng = 0;
    for (int t = 0; t < w.T; ++t)
        for (int i = 0; i < 8; ++i) {
            const int e = t * 8 + i;
            gof[e] = -1;
            const unsigned long long p = i < w.k ? w.plan_ptr[e] : 0ull;
            if (p == 0ull) continue;
            int g = 0;
            while (g < ng && gp[g] != p) ++g;
            if (g == ng) {
                gp[ng] = p;
                cnt[ng++] = 0;
            }
            ++cnt[g];
            gof[e] = g;
        }
    fill[0] = 0;
    for (int g = 0; g < ng; ++g) {
        w.grp_ptr[g] = gp[g];
        w.grp_start[g] = fill[g];
        fill[g + 1] = fill[g] + cnt[g];
    }
    w.grp_start[ng] = fill[ng];
    for (int e = 0; e < w.T * 8; ++e)
        if (gof[e] >= 0) w.grp_ent[fill[gof[e]]++] = e;
    *w.n_groups = ng;
}
}  // namespace

void moe_win_group(const MoeWin& w, cudaStream_t s) {
    STRATA_LAUNCH(s, 1, 32, 0, moe_win_group_kernel, w);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dsv41: moe_win_group: %s\n", cudaGetErrorString(e));
        std::abort();
    }
}
}  // namespace strata::dsv41k::glmf
