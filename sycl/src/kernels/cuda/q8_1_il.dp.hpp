// src/kernels/cuda/q8_1_il.cuh - the interleaved copy of 2..8 q8_1 activation columns (native_mmvq.hpp) as the
// multi-column dot products read it, for native_mmvq.cu and iq_kernels.cu.
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_mmvq.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels {
namespace {

// NC columns of the block-major (PM false) or position-major copy: u(b, p, o) = int p of q8_1 block b of every column,
// scales(b, o) = their scales
template <int NC, bool PM> struct Q81Il {
    static constexpr int CP = native_q8_1_il_cp(NC);
    const int* __restrict__ q;
    const float* __restrict__ d;
    int nb;   // blocks a column
    __dpct_inline__ void u(int b, int p, int (&o)[NC]) const {
        const int* s = q + std::size_t(PM ? p * nb + b : b * 8 + p) * CP;
        if constexpr (CP == 2) {
            const sycl::int2 a = *reinterpret_cast<const sycl::int2 *>(s);
            o[0] = a.x();
            o[1] = a.y();
        } else {
            const sycl::int4 a = *reinterpret_cast<const sycl::int4 *>(s);
            o[0] = a.x();
            if constexpr (NC > 1) o[1] = a.y();
            if constexpr (NC > 2) o[2] = a.z();
            if constexpr (NC > 3) o[3] = a.w();
            if constexpr (NC > 4) {
                const sycl::int4 e =
                    *reinterpret_cast<const sycl::int4 *>(s + 4);
                o[4] = e.x();
                if constexpr (NC > 5) o[5] = e.y();
                if constexpr (NC > 6) o[6] = e.z();
                if constexpr (NC > 7) o[7] = e.w();
            }
        }
    }
    // P1 (native_mmvq_il_pf_kernel): the same loads from a pointer the caller strength-reduces - qat(qbase(b, p) + k * QB)
    // is u(b + k, p), dat(dbase(b) + k * DB) is scales(b + k)
    static constexpr int QB = PM ? CP : 8 * CP, DB = CP;
    __dpct_inline__ int qp(int dp) const {
        return PM ? dp * nb * CP : dp * CP;
    } // ints from p to p + dp
    __dpct_inline__ const int *qbase(int b, int p) const {
        return q + std::size_t(PM ? p * nb + b : b * 8 + p) * CP;
    }
    __dpct_inline__ const float *dbase(int b) const {
        return d + std::size_t(b) * CP;
    }
    __dpct_inline__ void
    qat(const int *s,
        int (&o)[NC]) const { // __ldg: s comes from an asm (madw), so the
        if constexpr (CP == 2) {                                                // front end no longer knows it is global
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const sycl::int2 a = *s;
            o[0] = a.x();
            o[1] = a.y();
        } else {
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const sycl::int4 a = *s;
            o[0] = a.x();
            if constexpr (NC > 1) o[1] = a.y();
            if constexpr (NC > 2) o[2] = a.z();
            if constexpr (NC > 3) o[3] = a.w();
            if constexpr (NC > 4) {
                /*
                DPCT1098: The '*' expression is used instead of the __ldg
                call. These two expressions do not provide the exact same
                functionality. Check the generated code for potential precision
                and/or performance issues.
                */
                const sycl::int4 e = *(s + 4);
                o[4] = e.x();
                if constexpr (NC > 5) o[5] = e.y();
                if constexpr (NC > 6) o[6] = e.z();
                if constexpr (NC > 7) o[7] = e.w();
            }
        }
    }
    __dpct_inline__ void dat(const float *s, float (&o)[NC]) const {
        if constexpr (CP == 2) {
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const sycl::float2 a = *s;
            o[0] = a.x();
            o[1] = a.y();
        } else {
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const sycl::float4 a = *s;
            o[0] = a.x();
            if constexpr (NC > 1) o[1] = a.y();
            if constexpr (NC > 2) o[2] = a.z();
            if constexpr (NC > 3) o[3] = a.w();
            if constexpr (NC > 4) {
                /*
                DPCT1098: The '*' expression is used instead of the __ldg
                call. These two expressions do not provide the exact same
                functionality. Check the generated code for potential precision
                and/or performance issues.
                */
                const sycl::float4 e = *(s + 4);
                o[4] = e.x();
                if constexpr (NC > 5) o[5] = e.y();
                if constexpr (NC > 6) o[6] = e.z();
                if constexpr (NC > 7) o[7] = e.w();
            }
        }
    }
    __dpct_inline__ void scales(int b, float (&o)[NC]) const {
        const float* s = d + std::size_t(b) * CP;
        if constexpr (CP == 2) {
            const sycl::float2 a = *reinterpret_cast<const sycl::float2 *>(s);
            o[0] = a.x();
            o[1] = a.y();
        } else {
            const sycl::float4 a = *reinterpret_cast<const sycl::float4 *>(s);
            o[0] = a.x();
            if constexpr (NC > 1) o[1] = a.y();
            if constexpr (NC > 2) o[2] = a.z();
            if constexpr (NC > 3) o[3] = a.w();
            if constexpr (NC > 4) {
                const sycl::float4 e =
                    *reinterpret_cast<const sycl::float4 *>(s + 4);
                o[4] = e.x();
                if constexpr (NC > 5) o[5] = e.y();
                if constexpr (NC > 6) o[6] = e.z();
                if constexpr (NC > 7) o[7] = e.w();
            }
        }
    }
};

// One column's plain q8_1 blocks (36 bytes: the half2 scale and sum, then 32 codes) behind Q81Il's interface
struct Q81One {
    const uint8_t* __restrict__ x;
    __dpct_inline__ void u(int b, int p, int (&o)[1]) const {
        o[0] = reinterpret_cast<const int*>(x + std::size_t(b) * 36 + 4)[p];
    }
    __dpct_inline__ void scales(int b, float (&o)[1]) const {
        o[0] = (*reinterpret_cast<const sycl::half2 *>(x +
                                                       std::size_t(b) * 36))[0];
    }
    static constexpr int QB = 9, DB = 9;   // a 36-byte block in ints / floats
    __dpct_inline__ int qp(int dp) const { return dp; }
    __dpct_inline__ const int *qbase(int b, int p) const {
        return reinterpret_cast<const int *>(x + std::size_t(b) * 36 + 4) + p;
    }
    __dpct_inline__ const float *dbase(int b) const {
        return reinterpret_cast<const float *>(x + std::size_t(b) * 36);
    }
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    __dpct_inline__ void qat(const int *s, int (&o)[1]) const { o[0] = *s; }
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    __dpct_inline__ void dat(const float *s, float (&o)[1]) const {
        o[0] = *s[0];
    }
};

// The activations a kernel for NC columns reads: one column's q8_1 blocks, 2+ columns' interleaved copy (nb blocks a
// column)
template <int NC, bool PM>
__dpct_inline__ auto q8_1_cols(const void *xq, const float *xd, int nb) {
    if constexpr (NC == 1) return Q81One{static_cast<const uint8_t*>(xq)};
    else return Q81Il<NC, PM>{static_cast<const int*>(xq), xd, nb};
}

// The copy's parts inside the buffer native_q8_1_il_bytes sizes: bm, pm (int32) and d (float).
struct Q81IlParts {
    const int* bm;
    const int* pm;
    const float* d;
};
inline Q81IlParts q8_1_il_parts(const void* x_il, int n_in, int ncols) {
    const std::size_t words = std::size_t(n_in / 32) * 8 * std::size_t(native_q8_1_il_cp(ncols));
    const int* bm = static_cast<const int*>(x_il);
    return {bm, bm + words, reinterpret_cast<const float*>(bm + 2 * words)};
}

}  // namespace
}  // namespace strata::kernels
