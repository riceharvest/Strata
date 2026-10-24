// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT).
// Ported into Strata for Intel Arc Pro B70 (Xe2 / bmg-g31).
// sycl/src/dsv41/ds4_sycl.hpp - SYCL runtime and math shim for strata-dsv41.
#pragma once

#ifndef STRATA_USE_HIP
#define STRATA_USE_HIP 1
#endif

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/sycl_math.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------- Hardware DP4A & B70 SWAR
#ifndef __dp4a
#define __dp4a(a, b, c) strata::dp4a((a), (b), (c))
#endif

template <typename T>
__dpct_inline__ T atomicOr(T* addr, T val) {
    return dpct::atomic_fetch_or<T, sycl::access::address_space::generic_space>(addr, val);
}
template <typename T>
__dpct_inline__ T atomicAdd(T* addr, T val) {
    return dpct::atomic_fetch_add<T, sycl::access::address_space::generic_space>(addr, val);
}
template <typename T>
__dpct_inline__ T atomicSub(T* addr, T val) {
    return dpct::atomic_fetch_sub<T, sycl::access::address_space::generic_space>(addr, val);
}
template <typename T>
__dpct_inline__ T atomicMax(T* addr, T val) {
    return dpct::atomic_fetch_max<T, sycl::access::address_space::generic_space>(addr, val);
}
template <typename T>
__dpct_inline__ T atomicMin(T* addr, T val) {
    return dpct::atomic_fetch_min<T, sycl::access::address_space::generic_space>(addr, val);
}

__dpct_inline__ int swar_ne4(unsigned x) {
    x = (x | (x >> 4)) & 0x0F0F0F0Fu;
    x = (x | (x >> 2)) & 0x03030303u;
    x = (x | (x >> 1)) & 0x01010101u;
    return (int) (x * 0xFFu);
}

__dpct_inline__ int swar_sub4(unsigned a, unsigned s) {
    return (int) (((a & 0x7F7F7F7Fu) + (s & 0x01010101u)) ^ (a & 0x80808080u));
}

__dpct_inline__ uint32_t iq4nl_lut4(uint32_t q4) {
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t n = (q4 >> (8 * i)) & 0xF;
        const uint32_t t = n < 4 ? t0 : n < 8 ? t1 : n < 12 ? t2 : t3;
        r |= ((t >> (8 * (n & 3))) & 0xFF) << (8 * i);
    }
    return r;
}

__dpct_inline__ sycl::int2 iq4nl_pair(int aux) {
    return sycl::int2((int) iq4nl_lut4((uint32_t) aux & 0x0f0f0f0fu),
                      (int) iq4nl_lut4(((uint32_t) aux >> 4) & 0x0f0f0f0fu));
}

__dpct_inline__ int load4_a2(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0];
    if (!(a & 2)) return (int) w0;
    return (int) ((w0 >> 16) | (q[1] << 16));
}

__dpct_inline__ sycl::int2 load8_a2(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0], w1 = q[1];
    if (!(a & 2)) return sycl::int2((int) w0, (int) w1);
    const uint32_t w2 = q[2];
    return sycl::int2((int) ((w0 >> 16) | (w1 << 16)), (int) ((w1 >> 16) | (w2 << 16)));
}

// ---------------------------------------------------------------- Subgroup Shuffles & Sync
template <typename T>
__dpct_inline__ T __shfl_xor_sync(unsigned mask, T var, int lane_mask, int width = 32) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::permute_group_by_xor(sg, var, lane_mask);
}

template <typename T>
__dpct_inline__ T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = 32) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::shift_group_left(sg, var, delta);
}

template <typename T>
__dpct_inline__ T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width = 32) {
    (void)mask; (void)width;
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::shift_group_right(sg, var, delta);
}

template <typename T>
__dpct_inline__ T __shfl_sync(unsigned mask, T var, int src_lane, int width = 32) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return sycl::select_from_group(sg, var, src_lane);
}

template <typename T>
__dpct_inline__ T __shfl_xor(T var, int lane_mask) {
    return __shfl_xor_sync(0xffffffffu, var, lane_mask);
}

template <typename T>
__dpct_inline__ T __shfl_down(T var, unsigned int delta) {
    return __shfl_down_sync(0xffffffffu, var, delta);
}

__dpct_inline__ unsigned int __ballot_sync(unsigned int mask, int predicate) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto m = sycl::ext::oneapi::group_ballot(sg, predicate != 0);
    unsigned int bits = 0;
    m.extract_bits(bits, 0);
    return bits & mask;
}

__dpct_inline__ int __popc(unsigned int x) {
    return std::popcount(x);
}

__dpct_inline__ void __syncthreads() {
    sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier(sycl::access::fence_space::local_space);
}

__dpct_inline__ void __syncwarp(unsigned mask = 0xffffffffu) {
    (void)mask;
}

__dpct_inline__ void __threadfence_system() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

__dpct_inline__ void __threadfence() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
}

__dpct_inline__ void __threadfence_block() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
}

// ---------------------------------------------------------------- Thread / Work-Item Queries
struct dim3 {
    unsigned int x = 1, y = 1, z = 1;
    constexpr dim3() = default;
    constexpr dim3(unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

struct SyclId3 {
    int x = 0, y = 0, z = 0;
    constexpr SyclId3(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}
};

__dpct_inline__ SyclId3 get_sycl_thread_idx() {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return SyclId3((int) item.get_local_id(2), (int) item.get_local_id(1), (int) item.get_local_id(0));
}

__dpct_inline__ SyclId3 get_sycl_block_idx() {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return SyclId3((int) item.get_group(2), (int) item.get_group(1), (int) item.get_group(0));
}

__dpct_inline__ SyclId3 get_sycl_block_dim() {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return SyclId3((int) item.get_local_range(2), (int) item.get_local_range(1), (int) item.get_local_range(0));
}

__dpct_inline__ SyclId3 get_sycl_grid_dim() {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return SyclId3((int) item.get_group_range(2), (int) item.get_group_range(1), (int) item.get_group_range(0));
}

#ifndef threadIdx
#define threadIdx (get_sycl_thread_idx())
#endif
#ifndef blockIdx
#define blockIdx  (get_sycl_block_idx())
#endif
#ifndef blockDim
#define blockDim  (get_sycl_block_dim())
#endif
#ifndef gridDim
#define gridDim   (get_sycl_grid_dim())
#endif

#define SYCL_SHARED(Type, var) auto& var = *sycl::ext::oneapi::group_local_memory_for_overwrite<Type>(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group())

// ---------------------------------------------------------------- Math / Conversion Helpers
__dpct_inline__ float __uint_as_float(uint32_t u) { return sycl::bit_cast<float>(u); }
__dpct_inline__ uint32_t __float_as_uint(float f) { return sycl::bit_cast<uint32_t>(f); }
__dpct_inline__ float __int_as_float(int32_t i) { return sycl::bit_cast<float>(i); }
__dpct_inline__ int32_t __float_as_int(float f) { return sycl::bit_cast<int32_t>(f); }

using __half = sycl::half;

__dpct_inline__ float __half2float(sycl::half h) { return (float) h; }
__dpct_inline__ sycl::half __float2half_rn(float f) { return sycl::half(f); }
__dpct_inline__ sycl::half __ushort_as_half(unsigned short u) { return sycl::bit_cast<sycl::half>(u); }
__dpct_inline__ unsigned short __half_as_ushort(sycl::half h) { return sycl::bit_cast<unsigned short>(h); }

__dpct_inline__ uint16_t __float2bfloat16_rn(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__dpct_inline__ uint16_t __bfloat16_as_ushort(uint16_t u) { return u; }

__dpct_inline__ float fmaxf(float a, float b) { return std::max(a, b); }
__dpct_inline__ float fminf(float a, float b) { return std::min(a, b); }
template <typename A, typename B>
__dpct_inline__ auto min(A a, B b) { return (a < b) ? a : b; }
template <typename A, typename B>
__dpct_inline__ auto max(A a, B b) { return (a > b) ? a : b; }
__dpct_inline__ float fabsf(float a) { return std::abs(a); }
__dpct_inline__ float rsqrtf(float a) { return 1.0f / std::sqrt(a); }
__dpct_inline__ float __fmul_rn(float a, float b) { return a * b; }
__dpct_inline__ float __fdiv_rn(float a, float b) { return a / b; }
__dpct_inline__ float __fsub_rn(float a, float b) { return a - b; }
__dpct_inline__ float __fadd_rn(float a, float b) { return a + b; }
__dpct_inline__ float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
__dpct_inline__ float expf(float x) { return std::exp(x); }
__dpct_inline__ float __expf(float x) { return std::exp(x); }
__dpct_inline__ float logf(float x) { return std::log(x); }
__dpct_inline__ float __logf(float x) { return std::log(x); }
template <typename T>
__dpct_inline__ T __ldg(const T* ptr) { return *ptr; }
__dpct_inline__ float erff(float x) { return std::erf(x); }
__dpct_inline__ void sincosf(float x, float* s, float* c) {
    *s = std::sin(x);
    *c = std::cos(x);
}
__dpct_inline__ int __float2int_rn(float f) {
#if defined(__SYCL_DEVICE_ONLY__)
    return (int) sycl::rint(f);
#else
    return (int) std::rint(f);
#endif
}

__dpct_inline__ unsigned int __vsub4(unsigned int a, unsigned int b) {
    uint8_t a0 = a & 0xff, a1 = (a >> 8) & 0xff, a2 = (a >> 16) & 0xff, a3 = (a >> 24) & 0xff;
    uint8_t b0 = b & 0xff, b1 = (b >> 8) & 0xff, b2 = (b >> 16) & 0xff, b3 = (b >> 24) & 0xff;
    return (uint32_t)(uint8_t)(a0 - b0) |
           ((uint32_t)(uint8_t)(a1 - b1) << 8) |
           ((uint32_t)(uint8_t)(a2 - b2) << 16) |
           ((uint32_t)(uint8_t)(a3 - b3) << 24);
}
__dpct_inline__ int __vsub4(int a, int b) {
    return (int) __vsub4((unsigned int) a, (unsigned int) b);
}

__dpct_inline__ uint32_t __byte_perm(uint32_t x, uint32_t y, unsigned int s) {
    uint8_t bytes[8];
    bytes[0] = x & 0xff;
    bytes[1] = (x >> 8) & 0xff;
    bytes[2] = (x >> 16) & 0xff;
    bytes[3] = (x >> 24) & 0xff;
    bytes[4] = y & 0xff;
    bytes[5] = (y >> 8) & 0xff;
    bytes[6] = (y >> 16) & 0xff;
    bytes[7] = (y >> 24) & 0xff;
    uint32_t res = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int sel = (s >> (4 * i)) & 0xf;
        uint32_t b = (sel < 8) ? bytes[sel] : 0;
        res |= (b << (8 * i));
    }
    return res;
}

__dpct_inline__ uint32_t __umulhi(uint32_t a, uint32_t b) {
    return (uint32_t) (((uint64_t) a * (uint64_t) b) >> 32);
}

__dpct_inline__ uint32_t __funnelshift_r(uint32_t low, uint32_t high, uint32_t shift) {
    return (uint32_t)(((static_cast<uint64_t>(high) << 32) | low) >> (shift & 31U));
}

__dpct_inline__ uint32_t __funnelshift_l(uint32_t low, uint32_t high, uint32_t shift) {
    return (uint32_t)((((static_cast<uint64_t>(high) << 32) | low) << (shift & 31U)) >> 32);
}

struct alignas(8) float2 {
    float x;
    float y;
    float2() = default;
    constexpr float2(float _x, float _y) : x(_x), y(_y) {}
};
struct alignas(16) float4 {
    float x, y, z, w;
    float4() = default;
    constexpr float4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
};
struct alignas(8) uint2 {
    uint32_t x, y;
    uint2() = default;
    constexpr uint2(uint32_t _x, uint32_t _y) : x(_x), y(_y) {}
};
struct alignas(16) uint4 {
    uint32_t x, y, z, w;
    uint4() = default;
    constexpr uint4(uint32_t _x, uint32_t _y, uint32_t _z, uint32_t _w) : x(_x), y(_y), z(_z), w(_w) {}
};
struct alignas(8) int2 {
    int32_t x, y;
    int2() = default;
    constexpr int2(int32_t _x, int32_t _y) : x(_x), y(_y) {}
};
struct alignas(16) int4 {
    int32_t x, y, z, w;
    int4() = default;
    constexpr int4(int32_t _x, int32_t _y, int32_t _z, int32_t _w) : x(_x), y(_y), z(_z), w(_w) {}
};

struct alignas(4) __half2 {
    sycl::half x;
    sycl::half y;
    __half2() = default;
    constexpr __half2(sycl::half _x, sycl::half _y) : x(_x), y(_y) {}
};

__dpct_inline__ __half2 __halves2half2(sycl::half a, sycl::half b) { return __half2(a, b); }
__dpct_inline__ __half2 __half2half2(sycl::half a) { return __half2(a, a); }
__dpct_inline__ __half2 __floats2half2_rn(float a, float b) { return __half2(sycl::half(a), sycl::half(b)); }
__dpct_inline__ __half2 __hfma2(__half2 a, __half2 b, __half2 c) { return __half2(a.x * b.x + c.x, a.y * b.y + c.y); }
__dpct_inline__ __half2 __hadd2(__half2 a, __half2 b) { return __half2(a.x + b.x, a.y + b.y); }
__dpct_inline__ __half2 __hsub2(__half2 a, __half2 b) { return __half2(a.x - b.x, a.y - b.y); }
__dpct_inline__ __half2 __hmul2(__half2 a, __half2 b) { return __half2(a.x * b.x, a.y * b.y); }
__dpct_inline__ float __low2float(__half2 h) { return (float) h.x; }
__dpct_inline__ float __high2float(__half2 h) { return (float) h.y; }
__dpct_inline__ sycl::half __low2half(__half2 h) { return h.x; }
__dpct_inline__ sycl::half __high2half(__half2 h) { return h.y; }
__dpct_inline__ float2 __half22float2(__half2 h) { return float2((float) h.x, (float) h.y); }

__dpct_inline__ float2 make_float2(float x, float y) { return float2(x, y); }
__dpct_inline__ float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }
__dpct_inline__ uint2 make_uint2(uint32_t x, uint32_t y) { return uint2(x, y); }
__dpct_inline__ uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return uint4(x, y, z, w); }
__dpct_inline__ int2 make_int2(int x, int y) { return int2(x, y); }
__dpct_inline__ int4 make_int4(int x, int y, int z, int w) { return int4(x, y, z, w); }

// ---------------------------------------------------------------- CUDA Runtime Emulation
using cudaStream_t = sycl::queue*;
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorNotReady = 600;
constexpr cudaError_t cudaErrorUnknown = 999;
constexpr cudaError_t cudaErrorInvalidConfiguration = 9;

inline const char* cudaGetErrorString(cudaError_t) { return "sycl success"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }

template <typename F>
inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, F func, int blockSize, size_t dynamicSMemSize) {
    (void)func; (void)blockSize; (void)dynamicSMemSize;
    if (numBlocks) *numBlocks = 2;
    return cudaSuccess;
}

struct CudaEventWrapper {
    sycl::event ev;
    std::chrono::steady_clock::time_point tp;
    bool recorded = false;
};
using cudaEvent_t = CudaEventWrapper*;

inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
    *e = new CudaEventWrapper();
    return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned int) {
    *e = new CudaEventWrapper();
    return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) {
    delete e;
    return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = nullptr) {
    if (e) {
        auto q = strata::q_of(s);
        e->ev = q->ext_oneapi_submit_barrier();
        e->tp = std::chrono::steady_clock::now();
        e->recorded = true;
    }
    return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    if (e && e->recorded) {
        e->ev.wait();
    }
    return cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t e) {
    if (!e || !e->recorded) return cudaSuccess;
    auto st = e->ev.get_info<sycl::info::event::command_execution_status>();
    return st == sycl::info::event_command_status::complete ? cudaSuccess : cudaErrorNotReady;
}
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (start && end) {
        double d = std::chrono::duration<double, std::milli>(end->tp - start->tp).count();
        *ms = (float) d;
    } else {
        *ms = 0.0f;
    }
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int = 0) {
    if (e && e->recorded) {
        strata::q_of(s)->ext_oneapi_submit_barrier({e->ev});
    }
    return cudaSuccess;
}

// USM allocations
template <typename T>
inline cudaError_t cudaMalloc(T** p, size_t n) {
    *p = reinterpret_cast<T*>(sycl::malloc_device(n, dpct::get_in_order_queue()));
    return *p ? cudaSuccess : 1;
}
inline cudaError_t cudaMalloc(void** p, size_t n) {
    *p = sycl::malloc_device(n, dpct::get_in_order_queue());
    return *p ? cudaSuccess : 1;
}
inline cudaError_t cudaFree(void* p) {
    if (p) sycl::free(p, dpct::get_in_order_queue());
    return cudaSuccess;
}
inline cudaError_t cudaHostAlloc(void** p, size_t n, unsigned int = 0) {
    *p = sycl::malloc_host(n, dpct::get_in_order_queue());
    return *p ? cudaSuccess : 1;
}
inline cudaError_t cudaFreeHost(void* p) {
    if (p) sycl::free(p, dpct::get_in_order_queue());
    return cudaSuccess;
}
inline cudaError_t cudaHostGetDevicePointer(void** pDev, void* pHost, unsigned int = 0) {
    *pDev = pHost;
    return cudaSuccess;
}
inline cudaError_t cudaHostRegister(void*, size_t, unsigned int = 0) {
    return cudaSuccess;
}
inline cudaError_t cudaHostUnregister(void*) {
    return cudaSuccess;
}

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    auto& q = dpct::get_in_order_queue();
    q.memcpy(dst, src, n).wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t s = nullptr) {
    auto q = strata::q_of(s);
    q->memcpy(dst, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* p, int v, size_t n) {
    auto& q = dpct::get_in_order_queue();
    q.memset(p, v, n).wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t s = nullptr) {
    auto q = strata::q_of(s);
    q->memset(p, v, n);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind) {
    auto& q = dpct::get_in_order_queue();
    for (size_t r = 0; r < height; ++r) {
        q.memcpy((char*) dst + r * dpitch, (const char*) src + r * spitch, width);
    }
    q.wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind, cudaStream_t s = nullptr) {
    auto q = strata::q_of(s);
    for (size_t r = 0; r < height; ++r) {
        q->memcpy((char*) dst + r * dpitch, (const char*) src + r * spitch, width);
    }
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int) {
    auto ctx = dpct::get_current_device().get_context();
    sycl::device dev = dpct::get_current_device();
    *s = new sycl::queue(ctx, dev, sycl::property::queue::in_order());
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    if (s && s != &dpct::get_in_order_queue()) delete s;
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    strata::q_of(s)->wait();
    return cudaSuccess;
}
inline cudaError_t cudaStreamQuery(cudaStream_t) {
    return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() {
    dpct::get_current_device().queues_wait_and_throw();
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int dev_id) {
    dpct::select_device(dev_id);
    return cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* dev_id) {
    *dev_id = dpct::dev_mgr::instance().current_device_id();
    return cudaSuccess;
}
inline cudaError_t cudaGetDeviceCount(int* count) {
    *count = dpct::dev_mgr::instance().device_count();
    return cudaSuccess;
}

struct cudaDeviceProp {
    char name[256] = "Intel Arc Pro B70";
    size_t totalGlobalMem = 32ULL * 1024 * 1024 * 1024;
    int major = 8;
    int minor = 9;
    int multiProcessorCount = 32;
    int integrated = 0;
};
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int dev_id) {
    if (!prop) return 1;
    std::strncpy(prop->name, "Intel Arc Pro B70", sizeof(prop->name));
    prop->totalGlobalMem = 32ULL * 1024 * 1024 * 1024;
    prop->major = 8;
    prop->minor = 9;
    prop->multiProcessorCount = 32;
    prop->integrated = 0;
    return cudaSuccess;
}

enum cudaDeviceAttr {
    cudaDevAttrComputeCapabilityMajor = 1,
    cudaDevAttrMultiProcessorCount = 2,
    cudaDevAttrIntegrated = 3,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 4
};
inline cudaError_t cudaDeviceGetAttribute(int* val, cudaDeviceAttr attr, int dev_id) {
    if (attr == cudaDevAttrComputeCapabilityMajor) *val = 8;
    else if (attr == cudaDevAttrMultiProcessorCount) *val = 32;
    else if (attr == cudaDevAttrIntegrated) *val = 0;
    else if (attr == cudaDevAttrMaxSharedMemoryPerBlockOptin) *val = 64 * 1024;
    else *val = 0;
    return cudaSuccess;
}
inline cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    *total_bytes = 32ULL * 1024 * 1024 * 1024;
    *free_bytes = 28ULL * 1024 * 1024 * 1024;
    return cudaSuccess;
}

enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 1
};
template <typename F>
inline cudaError_t cudaFuncSetAttribute(F, cudaFuncAttribute, int) {
    return cudaSuccess;
}

enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0
};
inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* status) {
    *status = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

constexpr unsigned int cudaHostAllocDefault = 0;
constexpr unsigned int cudaHostAllocPortable = 1;
constexpr unsigned int cudaHostAllocMapped = 2;
constexpr unsigned int cudaHostRegisterPortable = 1;
constexpr unsigned int cudaHostRegisterMapped = 2;
constexpr unsigned int cudaStreamNonBlocking = 1;
constexpr unsigned int cudaEventDisableTiming = 2;

#ifndef __global__
#define __global__
#endif
#ifndef __constant__
#define __constant__ const
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ __dpct_inline__
#endif
#ifndef __grid_constant__
#define __grid_constant__
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __align__
#define __align__(x) alignas(x)
#endif

// ---------------------------------------------------------------- cuBLAS / oneMKL Shim
using cublasHandle_t = dpct::blas::descriptor_ptr;
using cublasStatus_t = int;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
constexpr int CUBLAS_OP_N = 0;
constexpr int CUBLAS_OP_T = 1;
constexpr int CUBLAS_COMPUTE_32F = 0;
constexpr int CUBLAS_GEMM_DEFAULT = 0;
constexpr int CUBLAS_GEMM_DEFAULT_TENSOR_OP = 0;
enum cudaDataType {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
    CUDA_R_16BF = 14
};

inline cublasStatus_t cublasCreate(cublasHandle_t* h) {
    *h = new dpct::blas::descriptor();
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t h) {
    if (h) delete h;
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetStream(cublasHandle_t h, cudaStream_t s) {
    if (h) h->set_queue(strata::q_of(s));
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) {
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasSgemm(cublasHandle_t h, int transa, int transb, int m, int n, int k,
                                  const float* alpha, const float* A, int lda,
                                  const float* B, int ldb,
                                  const float* beta, float* C, int ldc) {
    auto ta = transa == CUBLAS_OP_T ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
    auto tb = transb == CUBLAS_OP_T ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
    dpct::blas::gemm(h, ta, tb, m, n, k, alpha, A, dpct::library_data_t::real_float, lda,
                     B, dpct::library_data_t::real_float, ldb, beta,
                     C, dpct::library_data_t::real_float, ldc, dpct::compute_type::f32);
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasGemmEx(cublasHandle_t h, int transa, int transb, int m, int n, int k,
                                   const void* alpha, const void* A, cudaDataType Atype, int lda,
                                   const void* B, cudaDataType Btype, int ldb,
                                   const void* beta, void* C, cudaDataType Ctype, int ldc,
                                   int computeType = 0, int algo = 0) {
    auto ta = transa == CUBLAS_OP_T ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
    auto tb = transb == CUBLAS_OP_T ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
    auto map_dt = [](cudaDataType dt) {
        if (dt == CUDA_R_16F) return dpct::library_data_t::real_half;
        if (dt == CUDA_R_16BF) return dpct::library_data_t::real_bfloat16;
        return dpct::library_data_t::real_float;
    };
    dpct::blas::gemm(h, ta, tb, m, n, k, alpha, A, map_dt(Atype), lda,
                     B, map_dt(Btype), ldb, beta,
                     C, map_dt(Ctype), ldc, dpct::compute_type::f32);
    return CUBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------- SYCL Kernel Launch Helper
template <typename KernelLaunchLambda>
inline void strata_launch_impl(cudaStream_t s, dim3 grid, dim3 block, size_t shmem, KernelLaunchLambda&& fn) {
    auto q = strata::q_of(s);
    dim3 g(grid), b(block);
    sycl::range<3> global(g.z * b.z, g.y * b.y, g.x * b.x);
    sycl::range<3> local(b.z, b.y, b.x);
    if (shmem > 0) {
        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(shmem), cgh);
            cgh.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                fn(&slm[0]);
            });
        });
    } else {
        q->parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            fn(nullptr);
        });
    }
}

// The kernel's arguments are evaluated HERE, on the host, as a <<<>>> launch does, and the values go to the device: an argument
// such as scr->cap or *vl (host memory) read inside the device lambda would page-fault on the GPU.
#define STRATA_LAUNCH(stream, grid, block, shmem, kernel, ...) \
    strata_launch_impl(stream, grid, block, shmem, [_ds4_args = std::make_tuple(__VA_ARGS__)](uint8_t* _slm) { \
        (void)_slm; \
        std::apply([](const auto&... _ds4_a) { kernel(_ds4_a...); }, _ds4_args); \
    })

#define STRATA_LAUNCH_SLM(stream, grid, block, shmem, kernel, ...) \
    strata_launch_impl(stream, grid, block, shmem, [_ds4_args = std::make_tuple(__VA_ARGS__)](uint8_t* _slm) { \
        std::apply([_slm](const auto&... _ds4_a) { kernel(_ds4_a..., _slm); }, _ds4_args); \
    })

