// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT).
// Ported into Strata for Intel Arc Pro B70 (Xe2 / bmg-g31).
#pragma once

namespace hipk {
template <int T, bool GU>
inline bool launch(const SetArgs& a, cudaStream_t s) {
    (void) a;
    (void) s;
    return false;
}
}  // namespace hipk
