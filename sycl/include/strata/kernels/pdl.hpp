// include/strata/kernels/pdl.hpp - programmatic dependent launch (PDL) for the verify window's kernel chain: the SYCL port.
//
// Upstream's PDL lets an sm_90+ CUDA kernel start while the one before it still runs (griddepcontrol.wait /
// launch_dependents, cudaLaunchAttributeProgrammaticStreamSerialization).  Level Zero has no counterpart: on SYCL every
// launch is ordinary, pdl_wait() / pdl_trigger() are no-ops and the kernels keep their plain loops (kPdlPrefetch), as
// upstream's HIP build does.  pdl_scope() / pdl_supported() stay so the verify window's recorder reads the same.
#pragma once

#include <sycl/sycl.hpp>

namespace strata::kernels {

/// Whether launches on this thread may carry the PDL attribute (set by the verify window's recorder).
bool& pdl_scope();

/// Always false on SYCL.
bool pdl_supported();

#define STRATA_PDL_RESTRICT __restrict__

inline constexpr bool kPdlPrefetch = false;   // no PDL: the prefetch before the wait would only cost registers

inline void pdl_wait() {}
inline void pdl_trigger() {}

}  // namespace strata::kernels
