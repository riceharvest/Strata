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
#include "strata/sycl_doorbell.hpp"
#include "strata/sycl_math.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <execinfo.h>
#if defined(__linux__)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#include <cerrno>
#endif
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

// CUDA's __syncthreads orders the block's global stores as well as its shared ones (the split-K partials, the all-reduce rows)
__dpct_inline__ void __syncthreads() {
    sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier(sycl::access::fence_space::global_and_local);
}

// a sub-group (warp) barrier: the lanes hand data to each other through shared memory around it.  Every call site runs
// with the whole warp (a lane mask other than all lanes is not supported)
__dpct_inline__ void __syncwarp(unsigned mask = 0xffffffffu) {
    (void)mask;
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
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

// __shared__ arrays: group_local_memory aligns to alignof(Type), and the kernels read several of them as uint2 / uint4 / float4
// (CUDA's __align__(16) on them was dropped in the port), so each is boxed at 16 bytes at least
template <class T>
struct alignas(alignof(T) > 16 ? alignof(T) : 16) ds4_slm_box {
    T v;
};
#define SYCL_SHARED(Type, var) auto& var = sycl::ext::oneapi::group_local_memory_for_overwrite<ds4_slm_box<Type>>(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group())->v

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

// CUDA's fmaxf / fminf return the other operand when one is NaN (std::max / std::min do not)
__dpct_inline__ float fmaxf(float a, float b) { return sycl::fmax(a, b); }
__dpct_inline__ float fminf(float a, float b) { return sycl::fmin(a, b); }
template <typename A, typename B>
__dpct_inline__ auto min(A a, B b) { return (a < b) ? a : b; }
template <typename A, typename B>
__dpct_inline__ auto max(A a, B b) { return (a > b) ? a : b; }
// These names are libm's C functions (<cmath> declares them extern "C"), so each definition here IS that function.  A body
// through std:: lowers back to a call of the same symbol under -fp-model=precise: on the B70 expf called itself until the GPU
// was killed (hc_split, swiglu: the first layer of every prompt never finished).  The sycl:: built-ins never come back here.
__dpct_inline__ float fabsf(float a) { return sycl::fabs(a); }
__dpct_inline__ float rsqrtf(float a) { return 1.0f / std::sqrt(a); }
__dpct_inline__ float __fmul_rn(float a, float b) { return a * b; }
__dpct_inline__ float __fdiv_rn(float a, float b) { return a / b; }
__dpct_inline__ float __fsub_rn(float a, float b) { return a - b; }
__dpct_inline__ float __fadd_rn(float a, float b) { return a + b; }
__dpct_inline__ float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
__dpct_inline__ float expf(float x) { return sycl::exp(x); }
__dpct_inline__ float __expf(float x) { return sycl::exp(x); }
__dpct_inline__ float logf(float x) { return sycl::log(x); }
__dpct_inline__ float __logf(float x) { return sycl::log(x); }
template <typename T>
__dpct_inline__ T __ldg(const T* ptr) { return *ptr; }
__dpct_inline__ float erff(float x) { return sycl::erf(x); }
__dpct_inline__ void sincosf(float x, float* s, float* c) {
    *s = sycl::sin(x);
    *c = sycl::cos(x);
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
        const int sel = (s >> (4 * i)) & 7;   // (CUDA: the low three bits of each selector nibble; the fourth is ignored)
        res |= ((uint32_t) bytes[sel] << (8 * i));
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
// DS4_PROF: the streams cudaStreamCreate* makes are created with profiling, so an event's barrier carries the GPU's timestamp
inline bool ds4_prof_queues() {   // (DS4_PROF=sync: none - the host's clock at a sync after each stage instead)
    static const bool on = std::getenv("DS4_PROF") != nullptr && std::string(std::getenv("DS4_PROF")) != "sync";
    return on;
}
// The GPU's time between the two barriers where their queues profile (DS4_PROF), else the host's between the records: the
// host's is when the work was queued, not when it ran - every DS4_PROF stage and kernel time read so before 2026-10-25
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    *ms = 0.0f;
    if (!start || !end) return cudaSuccess;
    if (ds4_prof_queues() && start->recorded && end->recorded) {
        try {
            const uint64_t a = start->ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            const uint64_t b = end->ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            *ms = (float) ((double) ((int64_t) (b - a)) / 1e6);
            return cudaSuccess;
        } catch (const sycl::exception&) {   // (a queue without profiling: the host's times below)
        }
    }
    *ms = (float) std::chrono::duration<double, std::milli>(end->tp - start->tp).count();
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int = 0) {
    if (e && e->recorded) {
        strata::q_of(s)->ext_oneapi_submit_barrier({e->ev});
    }
    return cudaSuccess;
}

constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorNotSupported = 801;

constexpr unsigned int cudaHostAllocDefault = 0;
constexpr unsigned int cudaHostAllocPortable = 1;
constexpr unsigned int cudaHostAllocMapped = 2;
constexpr unsigned int cudaHostRegisterPortable = 1;
constexpr unsigned int cudaHostRegisterMapped = 2;
constexpr unsigned int cudaStreamNonBlocking = 1;
constexpr unsigned int cudaEventDisableTiming = 2;

// The shim's own bookkeeping: the queues cudaStreamCreate* made (CUDA's device-wide waits must cover them), the bytes
// cudaMalloc holds per device (cudaMemGetInfo's fallback), and the host ranges cudaHostRegister pinned.
namespace ds4_shim {
struct State {
    std::mutex mu;
    std::vector<sycl::queue*> queues;
    std::unordered_map<const void*, std::pair<unsigned, size_t>> dev_allocs;   // pointer -> (dpct device id, bytes)
    std::unordered_map<unsigned, size_t> dev_used;
    std::map<const char*, size_t> registered;                                  // cudaHostRegister'd ranges
};
inline State& st() {
    static State s;
    return s;
}
inline unsigned cur_dev() { return dpct::dev_mgr::instance().current_device_id(); }
// wait for the queues the shim made (all of them, or those of one device) and dpct's own
inline void wait_queues(bool all_devices) {
    std::vector<sycl::queue*> qs;
    {
        std::lock_guard<std::mutex> lk(st().mu);
        qs = st().queues;
    }
    const sycl::device cur = dpct::get_current_device();
    for (sycl::queue* q : qs)
        if (all_devices || q->get_device() == cur) q->wait();
    if (all_devices) {
        for (unsigned i = 0; i < dpct::dev_mgr::instance().device_count(); ++i) dpct::get_device(i).queues_wait_and_throw();
    } else {
        dpct::get_current_device().queues_wait_and_throw();
    }
}
inline bool registered(const void* p) {
    std::lock_guard<std::mutex> lk(st().mu);
    auto& r = st().registered;
    auto it = r.upper_bound((const char*) p);
    if (it == r.begin()) return false;
    --it;
    return (const char*) p < it->first + it->second;
}
// host memory that is neither USM nor cudaHostRegister'd: CUDA copies it synchronously (the caller may reuse it at once)
inline bool pageable(const void* p, const sycl::context& ctx) {
    return sycl::get_pointer_type(p, ctx) == sycl::usm::alloc::unknown && !registered(p);
}
constexpr size_t kBigFill = (size_t) 256 << 20;   // memsets of zero past this go through strata::big_fill_zero (xe: no multi-GiB blit)
// DS4_COPY_CHECK=1: every copy's and memset's device side must lie inside one cudaMalloc'd buffer, a USM host range inside
// its allocation - else the call is printed with its stack before it is queued (a bad one kills the copy engine: DEVICE_LOST)
inline bool copy_check_on() {
    static const bool on = std::getenv("DS4_COPY_CHECK") != nullptr && std::atoi(std::getenv("DS4_COPY_CHECK")) != 0;
    return on;
}
inline bool in_dev_alloc(const void* p, size_t n) {
    std::lock_guard<std::mutex> lk(st().mu);
    for (const auto& a : st().dev_allocs)
        if ((const char*) p >= (const char*) a.first && (const char*) p + n <= (const char*) a.first + a.second.second) return true;
    return false;
}
inline void check_range(const char* what, const void* p, size_t n, bool device_side, const sycl::context& ctx) {
    if (!copy_check_on() || n == 0) return;
    const sycl::usm::alloc t = sycl::get_pointer_type(p, ctx);
    bool ok = true;
    if (t == sycl::usm::alloc::device) ok = in_dev_alloc(p, n);
    else if (t == sycl::usm::alloc::unknown) ok = !device_side && (!registered(p) || registered((const char*) p + n - 1));
    else ok = sycl::get_pointer_type((const char*) p + n - 1, ctx) == t;
    if (ok) return;
    static std::atomic<int> reported{0};
    if (reported.fetch_add(1) >= 20) return;
    std::fprintf(stderr, "DS4_COPY_CHECK: bad %s %p + %zu (usm type %d, device side %d)\n", what, p, n, (int) t, (int) device_side);
    void* bt[32];
    const int k = backtrace(bt, 32);
    backtrace_symbols_fd(bt, k, 2);
}
// (DS4_COPY_CHECK: the last 256 copies, with their callers, printed when the process dies - with the buffers nearest
//  DS4_FAULT_VA, the address the kernel's xe fault line names)
struct CopyRec {
    const void *dst, *src;
    size_t n;
    int kind;
    void* bt[4];
};
inline CopyRec* copy_ring() {
    static CopyRec r[256];
    return r;
}
inline std::atomic<unsigned>& copy_ring_i() {
    static std::atomic<unsigned> i{0};
    return i;
}
inline void copy_dump() {
    std::fprintf(stderr, "DS4_COPY_CHECK: the last copies (oldest first):\n");
    const unsigned e = copy_ring_i().load();
    for (unsigned k = e > 256 ? e - 256 : 0; k < e; ++k) {
        const CopyRec& c = copy_ring()[k % 256];
        std::fprintf(stderr, "  #%u kind %d dst %p src %p n %zu  callers %p %p %p\n", k, c.kind, c.dst, c.src, c.n, c.bt[1], c.bt[2],
                     c.bt[3]);
    }
    if (const char* v = std::getenv("DS4_FAULT_VA")) {
        uintptr_t a = (uintptr_t) std::strtoull(v, nullptr, 16);
        if (a < ((uintptr_t) 1 << 48) && (a >> 47) != 0) a |= ~(((uintptr_t) 1 << 48) - 1);   // (xe prints 48 bits: canonical form)
        for (const auto& d : st().dev_allocs)
            std::fprintf(stderr, "DS4_ALLOC %p + %zu\n", d.first, d.second.second);
        uintptr_t best_lo = 0, best_hi = ~(uintptr_t) 0;
        std::string lo_s = "-", hi_s = "-";
        for (const auto& d : st().dev_allocs) {
            const uintptr_t b = (uintptr_t) d.first, end = b + d.second.second;
            char t[128];
            std::snprintf(t, sizeof t, "device %p + %zu", d.first, d.second.second);
            if (b <= a && a < end) std::fprintf(stderr, "DS4_FAULT_VA %p: INSIDE %s\n", (void*) a, t);
            if (end <= a && end > best_lo) best_lo = end, lo_s = t;
            if (b > a && b < best_hi) best_hi = b, hi_s = t;
        }
        for (const auto& r : st().registered) {
            const uintptr_t b = (uintptr_t) r.first, end = b + r.second;
            char t[128];
            std::snprintf(t, sizeof t, "host range %p + %zu", (const void*) r.first, r.second);
            if (b <= a && a < end) std::fprintf(stderr, "DS4_FAULT_VA %p: INSIDE %s\n", (void*) a, t);
            if (end <= a && end > best_lo) best_lo = end, lo_s = t;
            if (b > a && b < best_hi) best_hi = b, hi_s = t;
        }
        std::fprintf(stderr, "DS4_FAULT_VA %p: below it %s (ends %p), above it %s\n", (void*) a, lo_s.c_str(), (void*) best_lo,
                     hi_s.c_str());
    }
}
inline void copy_record(const void* dst, const void* src, size_t n, int kind) {
    static const bool hooked = [] {
        static std::terminate_handler prev = std::set_terminate([] {
            copy_dump();
            if (prev) prev();
            std::abort();
        });
        return true;
    }();
    (void) hooked;
    const unsigned k = copy_ring_i().fetch_add(1);
    CopyRec& c = copy_ring()[k % 256];
    c.dst = dst, c.src = src, c.n = n, c.kind = kind;
    backtrace(c.bt, 4);
    static const bool all = std::getenv("DS4_COPY_LOG") != nullptr;   // (every copy as it is queued)
    if (all) std::fprintf(stderr, "DS4_COPY #%u kind %d dst %p src %p n %zu callers %p %p %p\n", k, kind, dst, src, n, c.bt[1], c.bt[2], c.bt[3]);
}
inline void check_copy(void* dst, const void* src, size_t n, int kind, const sycl::context& ctx) {
    if (!copy_check_on()) return;
    copy_record(dst, src, n, kind);
    check_range("copy dst", dst, n, kind == 1 || kind == 3, ctx);
    check_range("copy src", src, n, kind == 2 || kind == 3, ctx);
}
}  // namespace ds4_shim

// USM allocations.  Device memory through strata::malloc_device_guarded (a big buffer is checked for aliased pages, as the main
// engine's are); CUDA's cudaFree / cudaFreeHost wait for the device first, so these wait for every queue the shim made.
template <typename T>
inline cudaError_t cudaMalloc(T** p, size_t n) {
    void* v = strata::malloc_device_guarded(n, dpct::get_in_order_queue(), "dsv41 device buffer");
    *p = reinterpret_cast<T*>(v);
    if (v == nullptr) return cudaErrorMemoryAllocation;
    std::lock_guard<std::mutex> lk(ds4_shim::st().mu);
    const unsigned d = ds4_shim::cur_dev();
    ds4_shim::st().dev_allocs[v] = {d, n};
    ds4_shim::st().dev_used[d] += n;
    if (ds4_shim::copy_check_on() && std::getenv("DS4_COPY_LOG")) {
        void* bt[5];
        backtrace(bt, 5);
        std::fprintf(stderr, "DS4_MALLOC %p n %zu callers %p %p %p %p\n", v, n, bt[1], bt[2], bt[3], bt[4]);
    }
    return cudaSuccess;
}
inline cudaError_t cudaFree(void* p) {
    if (p == nullptr) return cudaSuccess;
    ds4_shim::wait_queues(true);
    {
        std::lock_guard<std::mutex> lk(ds4_shim::st().mu);
        auto it = ds4_shim::st().dev_allocs.find(p);
        if (it != ds4_shim::st().dev_allocs.end()) {
            if (ds4_shim::copy_check_on() && std::getenv("DS4_COPY_LOG")) {
                void* bt[5];
                backtrace(bt, 5);
                std::fprintf(stderr, "DS4_FREE %p n %zu callers %p %p %p %p\n", p, it->second.second, bt[1], bt[2], bt[3], bt[4]);
            }
            ds4_shim::st().dev_used[it->second.first] -= it->second.second;
            ds4_shim::st().dev_allocs.erase(it);
        }
    }
    strata::device_alloc_unregister(p);
    sycl::free(p, dpct::get_in_order_queue());
    return cudaSuccess;
}
// cudaHostAllocMapped: memory a running kernel polls for the host's stores (the decode rings, the CPU lane's rows), from
// strata::host_malloc_polled as in the main engine (docs/INTEL.md, "Host-mapped flags")
inline cudaError_t cudaHostAlloc(void** p, size_t n, unsigned int flags = 0) {
    auto& q = dpct::get_in_order_queue();
    *p = (flags & cudaHostAllocMapped) ? strata::host_malloc_polled(n, q) : sycl::malloc_host(n, q);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFreeHost(void* p) {
    if (p == nullptr) return cudaSuccess;
    ds4_shim::wait_queues(true);
    strata::host_free_polled(p, dpct::get_in_order_queue());   // (either kind: it frees ordinary host USM too)
    return cudaSuccess;
}
// Only USM is visible to the device: an ordinary host pointer handed to a kernel page-faults on xe.
inline cudaError_t cudaHostGetDevicePointer(void** pDev, void* pHost, unsigned int = 0) {
    const auto t = sycl::get_pointer_type(pHost, dpct::get_in_order_queue().get_context());
    if (t == sycl::usm::alloc::unknown) {
        *pDev = nullptr;
        return cudaErrorInvalidValue;
    }
    *pDev = pHost;
    return cudaSuccess;
}
// Mapping ordinary (mmap'd) memory into the device is not available here: cudaHostRegisterMapped fails, so the callers fall back
// (DS4_STORE_ZC keeps the CPU lane).  A plain registration is remembered: CUDA copies from it asynchronously (the memory outlives
// the copy), where unregistered host memory is copied synchronously (ds4_shim::pageable).
inline cudaError_t cudaHostRegister(void* p, size_t n, unsigned int flags = 0) {
    if (flags & cudaHostRegisterMapped) return cudaErrorNotSupported;
    std::lock_guard<std::mutex> lk(ds4_shim::st().mu);
    ds4_shim::st().registered[(const char*) p] = n;
    return cudaSuccess;
}
inline cudaError_t cudaHostUnregister(void* p) {
    std::lock_guard<std::mutex> lk(ds4_shim::st().mu);
    ds4_shim::st().registered.erase((const char*) p);
    return cudaSuccess;
}
// (no CUDA name) A host range imported into the driver for device copies - sycl_ext_oneapi_copy_optimize, Level Zero's external
// pointer import: a copy from it is a DMA as from USM host memory, where cudaHostRegister above is bookkeeping only (it is what
// the model files' mappings get - 100+ GB the import would pin).  For the RAM store's arena chunks (2 GB each).
// The import does not pin: xe maps the range as a userptr that follows the CPU's page tables, and when the kernel migrates a
// page under a running copy (compaction, khugepaged) the copy engine faults (CAT error, -EBUSY / -ENOENT: DEVICE_LOST).  So the
// range is pinned long-term first, as cudaHostRegister pins on CUDA: registered as io_uring fixed buffers (FOLL_LONGTERM), its
// pages cannot move while the ring's file is open.  A stress test (copies from an imported 8 GB range while another process
// made the kernel compact, 1-2 M pages migrated): import alone DEVICE_LOST, pinned none.  Needs RLIMIT_MEMLOCK for the range
// (docker --ulimit memlock=-1) and io_uring allowed (seccomp); else a warning and the range unpinned, as before.  DS4_PIN=0: off
inline int ds4_host_pin(void* p, size_t n) {
#if defined(__linux__)
    if (std::getenv("DS4_PIN") != nullptr && std::atoi(std::getenv("DS4_PIN")) == 0) return -1;
    io_uring_params prm{};
    const int fd = (int) syscall(__NR_io_uring_setup, 1, &prm);
    int err = fd < 0 ? errno : 0;
    if (fd >= 0) {
        std::vector<iovec> iov;
        const size_t piece = (size_t) 1 << 30;   // (a fixed buffer is at most 1 GB)
        for (size_t o = 0; o < n; o += piece) iov.push_back({(char*) p + o, std::min(piece, n - o)});
        if (syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS, iov.data(), (unsigned) iov.size()) == 0) return fd;
        err = errno;
        ::close(fd);
    }
    static std::atomic<bool> told{false};
    if (!told.exchange(true))
        std::fprintf(stderr, "ds4: WARNING: the device-copied RAM could not be pinned (io_uring: %s) - a page the kernel moves under a "
                             "copy can stop the GPU (DEVICE_LOST); give the container --ulimit memlock=-1 and io_uring\n",
                     std::strerror(err));
    return -1;
#else
    (void) p, (void) n;
    return -1;
#endif
}
inline std::map<const char*, int>& ds4_host_pins() {
    static std::map<const char*, int> m;
    return m;
}
inline bool ds4_host_import(void* p, size_t n) {
#if defined(__linux__)
    if (!(std::getenv("DS4_THP") != nullptr && std::atoi(std::getenv("DS4_THP")) != 0)) madvise(p, n, MADV_NOHUGEPAGE);   // (ds4_os.hpp thp_on)
#endif
    const int pin = ds4_host_pin(p, n);
    try {
        sycl::ext::oneapi::experimental::prepare_for_device_copy(p, n, dpct::get_in_order_queue().get_context());
    } catch (const sycl::exception&) {
        if (pin >= 0) ::close(pin);
        return false;
    }
    std::lock_guard<std::mutex> lk(ds4_shim::st().mu);
    ds4_shim::st().registered[(const char*) p] = n;
    if (pin >= 0) ds4_host_pins()[(const char*) p] = pin;
    return true;
}
inline void ds4_host_unimport(void* p) {
    int pin = -1;
    {
        std::lock_guard<std::mutex> lk(ds4_shim::st().mu);
        if (ds4_shim::st().registered.erase((const char*) p) == 0) return;
        auto it = ds4_host_pins().find((const char*) p);
        if (it != ds4_host_pins().end()) pin = it->second, ds4_host_pins().erase(it);
    }
    ds4_shim::wait_queues(true);   // (no copy still reads it)
    try {
        sycl::ext::oneapi::experimental::release_from_device_copy(p, dpct::get_in_order_queue().get_context());
    } catch (const sycl::exception&) {
    }
    if (pin >= 0) ::close(pin);   // (the pages movable again: after the release)
}

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
    auto& q = dpct::get_in_order_queue();
    ds4_shim::check_copy(dst, src, n, (int) kind, q.get_context());
    q.memcpy(dst, src, n).wait();
    return cudaSuccess;
}
// CUDA: an async copy to or from pageable host memory returns once the host side is no longer needed - a temporary may be
// reused or freed right after (the RAM store's table, ds4_model 4752).  SYCL's copy reads it later, so those wait.
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind kind, cudaStream_t s = nullptr) {
    auto q = strata::q_of(s);
    ds4_shim::check_copy(dst, src, n, (int) kind, q->get_context());
    sycl::event e = q->memcpy(dst, src, n);
    const sycl::context ctx = q->get_context();
    if (ds4_shim::pageable(src, ctx) || ds4_shim::pageable(dst, ctx)) e.wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* p, int v, size_t n) {
    auto& q = dpct::get_in_order_queue();
    ds4_shim::check_range("memset", p, n, true, q.get_context());
    if (ds4_shim::copy_check_on()) ds4_shim::copy_record(p, nullptr, n, 9);
    if (v == 0 && n > ds4_shim::kBigFill) strata::big_fill_zero(q, p, n);
    else q.memset(p, v, n).wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t s = nullptr) {
    auto q = strata::q_of(s);
    ds4_shim::check_range("memset", p, n, true, q->get_context());
    if (ds4_shim::copy_check_on()) ds4_shim::copy_record(p, nullptr, n, 9);
    if (v == 0 && n > ds4_shim::kBigFill) strata::big_fill_zero(*q, p, n);   // (waits: compute kernels, not one huge blit)
    else q->memset(p, v, n);
    return cudaSuccess;
}
inline sycl::event ds4_copy2d(sycl::queue& q, void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
    if (dpitch == width && spitch == width) return q.memcpy(dst, src, width * height);   // (one contiguous copy)
    sycl::event e;
    for (size_t r = 0; r < height; ++r) e = q.memcpy((char*) dst + r * dpitch, (const char*) src + r * spitch, width);
    return e;
}
inline cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
    auto& q = dpct::get_in_order_queue();
    if (ds4_shim::copy_check_on() && height > 0) {
        ds4_shim::check_range("copy2d dst", dst, dpitch * (height - 1) + width, kind == 1 || kind == 3, q.get_context());
        ds4_shim::check_range("copy2d src", src, spitch * (height - 1) + width, kind == 2 || kind == 3, q.get_context());
    }
    ds4_copy2d(q, dst, dpitch, src, spitch, width, height);
    q.wait();
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t s = nullptr) {
    auto q = strata::q_of(s);
    if (ds4_shim::copy_check_on() && height > 0) {
        ds4_shim::check_range("copy2d dst", dst, dpitch * (height - 1) + width, kind == 1 || kind == 3, q->get_context());
        ds4_shim::check_range("copy2d src", src, spitch * (height - 1) + width, kind == 2 || kind == 3, q->get_context());
    }
    sycl::event e = ds4_copy2d(*q, dst, dpitch, src, spitch, width, height);
    const sycl::context ctx = q->get_context();
    if (height > 0 && (ds4_shim::pageable(src, ctx) || ds4_shim::pageable(dst, ctx))) e.wait();   // (in order: the last row's)
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int) {
    auto ctx = dpct::get_current_device().get_context();
    sycl::device dev = dpct::get_current_device();
    *s = ds4_prof_queues() ? new sycl::queue(ctx, dev, sycl::property_list{sycl::property::queue::in_order(),
                                                                          sycl::property::queue::enable_profiling()})
                           : new sycl::queue(ctx, dev, sycl::property::queue::in_order());
    std::lock_guard<std::mutex> lk(ds4_shim::st().mu);
    ds4_shim::st().queues.push_back(*s);
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    if (s && s != &dpct::get_in_order_queue()) {
        {
            std::lock_guard<std::mutex> lk(ds4_shim::st().mu);
            auto& v = ds4_shim::st().queues;
            v.erase(std::remove(v.begin(), v.end(), s), v.end());
        }
        s->wait();
        delete s;
    }
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    strata::q_of(s)->wait();
    return cudaSuccess;
}
inline cudaError_t cudaStreamQuery(cudaStream_t s) {
    return strata::q_of(s)->ext_oneapi_empty() ? cudaSuccess : cudaErrorNotReady;
}
inline cudaError_t cudaDeviceSynchronize() {
    ds4_shim::wait_queues(false);
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
// The device's own free-memory figure where the runtime has it (ext_intel_free_memory; Level Zero needs ZES_ENABLE_SYSMAN=1,
// which ds4_main sets), never more than the card's size less what this process's cudaMalloc holds, less a margin for what is not
// counted (the runtime, the kernels, oneMKL's scratch).  The engine sizes its expert slots from this, then takes --reserve-mib off.
inline cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    constexpr size_t kMargin = (size_t) 512 << 20;
    auto& dev = dpct::get_current_device();
    const size_t total = dev.get_global_mem_size();
    size_t used = 0;
    {
        std::lock_guard<std::mutex> lk(ds4_shim::st().mu);
        auto it = ds4_shim::st().dev_used.find(ds4_shim::cur_dev());
        if (it != ds4_shim::st().dev_used.end()) used = it->second;
    }
    size_t fr = total > used ? total - used : 0;
    try {
        if (dev.has(sycl::aspect::ext_intel_free_memory)) {
            const size_t q = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            if (q > 0) fr = std::min(fr, q);   // (0: the driver does not know - keep the count)
        }
    } catch (...) {
    }
    *total_bytes = total;
    *free_bytes = fr > kMargin ? fr - kMargin : 0;
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
        // (in 16-byte elements: CUDA's dynamic shared memory is 16-byte aligned and the kernels read it as uint4 / float4)
        const size_t n16 = (shmem + 15) / 16;
        q->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::uint4, 1> slm(sycl::range<1>(n16), cgh);
            cgh.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                fn(reinterpret_cast<uint8_t*>(&slm[0]));
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

// ---------------------------------------------------------------- host-mapped hand-offs (docs/INTEL.md: "Host-mapped flags",
// "Bounded spins")
// A volatile load of memory the host or another device writes is served from the GPU's caches on Intel, and a spin on it never
// ends - the xe driver then resets the GT and the card stays wedged.  Every flag goes through the main engine's uncached
// sys_load / sys_store, payloads through the same hints a 32-bit word at a time, and every spin is bounded: on the bound it
// goes on, as the main engine's doorbell waits do.
__dpct_inline__ unsigned ds4_ld_flag(const void* p) { return strata::sys_load((const volatile uint32_t*) p); }
__dpct_inline__ void ds4_st_flag(void* p, unsigned v) { strata::sys_store((volatile uint32_t*) p, v); }
template <class T>
__dpct_inline__ T ds4_ld_mapped(const T* p) {
    static_assert(sizeof(T) % 4 == 0, "32-bit words");
    T v;
    uint32_t* d = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
    for (int i = 0; i < (int) (sizeof(T) / 4); ++i) d[i] = strata::sys_load_mapped(reinterpret_cast<const uint32_t*>(p) + i);
    return v;
}
template <class T>
__dpct_inline__ void ds4_st_mapped(T* p, const T& v) {
    static_assert(sizeof(T) % 4 == 0, "32-bit words");
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
    for (int i = 0; i < (int) (sizeof(T) / 4); ++i) strata::sys_store_mapped(reinterpret_cast<uint32_t*>(p) + i, w[i]);
}
// The bound, in reads, passed to each spinning kernel by its launcher.  strata::spin_max picks 20,000 on an xe card for the main
// engine's device-plan windows, which spin only when something failed; here the GPU waits for the CPU lane's experts at every
// layer, which can take tens of milliseconds when they come from the SSD, and a spin that gives up goes on without them (the
// A-series failure spin_max describes).  So these take the main engine's other bound, 2,000,000 reads (a few seconds);
// STRATA_SPIN_MAX or -DSTRATA_SYCL_SPIN_MAX set it through strata::spin_max.
inline uint32_t ds4_spin_max(cudaStream_t s) {
#ifdef STRATA_SYCL_SPIN_MAX
    return strata::spin_max(*strata::q_of(s));
#else
    static const bool env = [] {
        const char* e = std::getenv("STRATA_SPIN_MAX");
        return e != nullptr && std::atol(e) > 0;
    }();
    return env ? strata::spin_max(*strata::q_of(s)) : 2000000u;
#endif
}
