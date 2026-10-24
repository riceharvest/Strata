// ds4_shim_test: checks of sycl/src/dsv41/ds4_sycl.hpp: host-evaluated launch args, __byte_perm vs CUDA, a polled-flag handshake with a
// bounded spin, fmaxf NaN, SLM alignment, cudaMemGetInfo / cudaStreamQuery, the device list, host registration, a
// 300 MiB zero fill, the libm-named math wrappers in a kernel.  About 300 MB of device memory.
#include "ds4_sycl.hpp"

#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <thread>

static int fails = 0;
#define CHECK(c, ...)                                   \
    do {                                                \
        if (!(c)) {                                     \
            ++fails;                                    \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                   \
            std::printf("\n");                          \
        }                                               \
    } while (0)

struct HostS {
    int cap;
    int* rows;
};

__global__ void args_kernel(int cap, int vl, int* out) {
    if (threadIdx.x == 0) {
        out[0] = cap;
        out[1] = vl;
    }
}
__global__ void slm_kernel(int* out, uint8_t* dyn) {
    SYCL_SHARED(char, c);
    SYCL_SHARED(uint16_t[7], a);
    SYCL_SHARED(int, b);
    SYCL_SHARED(float4[3], f);
    if (threadIdx.x == 0) {
        c = 1;
        b = 2;
        out[0] = (int) ((uintptr_t) &a[0] & 15);
        out[1] = (int) ((uintptr_t) &f[0] & 15);
        out[2] = (int) ((uintptr_t) dyn & 15);
        out[3] = c + b;
    }
}
__global__ void perm_kernel(const uint32_t* x, const uint32_t* y, const uint32_t* s, int n, uint32_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __byte_perm(x[i], y[i], s[i]);
}
__global__ void fmax_kernel(float* out) {
    if (threadIdx.x == 0) {
        const float nan = __uint_as_float(0x7fc00000u);
        out[0] = fmaxf(nan, 1.0f);
        out[1] = fmaxf(1.0f, nan);
        out[2] = fminf(nan, 2.0f);
        out[3] = fminf(2.0f, nan);
    }
}
// the wait_seq_kernel shape: bounded spin on a host-written flag, then report through host-mapped memory
__global__ void wait_kernel(const unsigned* flag, unsigned want, uint32_t spin_max, unsigned* out) {
    if (threadIdx.x != 0) return;
    uint32_t spin = 0;
    for (; spin < spin_max && ds4_ld_flag(flag) != want; ++spin) {
    }
    __threadfence_system();
    ds4_st_mapped(out, spin);
    ds4_st_flag(out + 1, ds4_ld_flag(flag) == want ? 1u : 2u);
}

static uint32_t perm_ref(uint32_t x, uint32_t y, uint32_t s) {   // CUDA __byte_perm, default mode
    const uint64_t v = ((uint64_t) y << 32) | x;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= (uint32_t) ((v >> (8 * ((s >> (4 * i)) & 7))) & 0xff) << (8 * i);
    return r;
}

// the shim's libm-named wrappers in a kernel (expf called itself on the B70 until the GPU was killed: a hang here is that bug back)
__global__ void math_kernel(const float* x, float* y) {
    const int i = threadIdx.x;
    float sn, cs;
    sincosf(x[i], &sn, &cs);
    y[i * 7 + 0] = expf(x[i]);
    y[i * 7 + 1] = __expf(x[i]);
    y[i * 7 + 2] = logf(fabsf(x[i]) + 1.0f);
    y[i * 7 + 3] = __logf(fabsf(x[i]) + 1.0f);
    y[i * 7 + 4] = erff(x[i]);
    y[i * 7 + 5] = sn;
    y[i * 7 + 6] = cs;
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // (a line at a time: a timeout keeps what was printed)
    // as ds4_main does before its first SYCL call (dpct lists every SYCL device otherwise: the card through Level Zero, the
    // same card through OpenCL, the CPU - and the engine gave each of them layers)
    setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu", 0);
    {
        int n = 0, bad = 0;
        cudaGetDeviceCount(&n);
        for (int i = 0; i < n; ++i) {
            const sycl::device dv = dpct::dev_mgr::instance().get_device(i);
            bad += !dv.is_gpu() || dv.get_backend() != sycl::backend::ext_oneapi_level_zero;
        }
        std::printf("dpct devices under ONEAPI_DEVICE_SELECTOR=%s: %d (%d not a Level Zero GPU)\n", std::getenv("ONEAPI_DEVICE_SELECTOR"), n, bad);
        CHECK(n >= 1 && bad == 0, "dpct lists %d devices, %d of them not Level Zero GPUs", n, bad);
    }
    std::printf("device: %s\n", dpct::get_current_device().get_info<sycl::info::device::name>().c_str());
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

    // 1. launch arguments evaluated on the host (plain new'd memory: a device read of it page-faults on xe)
    {
        HostS* h = new HostS{1234, nullptr};
        int* vl = new int(77);
        int* d = nullptr;
        cudaMalloc(&d, 2 * sizeof(int));
        STRATA_LAUNCH(s, 1, 32, 0, args_kernel, h->cap, *vl, d);
        delete h;   // (gone before the kernel runs: the launch holds the values, not the pointers)
        delete vl;
        int out[2] = {0, 0};
        cudaMemcpyAsync(out, d, sizeof out, cudaMemcpyDeviceToHost, s);   // (pageable: waits)
        CHECK(out[0] == 1234 && out[1] == 77, "launch args %d %d", out[0], out[1]);
        cudaFree(d);
    }
    // 2. SLM alignment
    {
        int* d = nullptr;
        cudaMalloc(&d, 4 * sizeof(int));
        STRATA_LAUNCH_SLM(s, 1, 32, 37, slm_kernel, d);
        int out[4] = {-1, -1, -1, -1};
        cudaMemcpyAsync(out, d, sizeof out, cudaMemcpyDeviceToHost, s);
        CHECK(out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 3, "slm align %d %d %d (%d)", out[0], out[1], out[2], out[3]);
        cudaFree(d);
    }
    // 3. __byte_perm against CUDA's rule, every selector nibble 0-15 in every byte position
    {
        const int n = 4096;
        std::vector<uint32_t> x(n), y(n), sel(n), got(n);
        uint32_t r = 12345;
        for (int i = 0; i < n; ++i) {
            r = r * 1664525u + 1013904223u; x[i] = r;
            r = r * 1664525u + 1013904223u; y[i] = r;
            r = r * 1664525u + 1013904223u; sel[i] = i < 16 ? 0x1111u * (uint32_t) i : (r & 0xffffu);
        }
        uint32_t *dx, *dy, *ds, *dout;
        cudaMalloc(&dx, n * 4); cudaMalloc(&dy, n * 4); cudaMalloc(&ds, n * 4); cudaMalloc(&dout, n * 4);
        cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dy, y.data(), n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, sel.data(), n * 4, cudaMemcpyHostToDevice);
        STRATA_LAUNCH(s, n / 256, 256, 0, perm_kernel, dx, dy, ds, n, dout);
        cudaMemcpyAsync(got.data(), dout, n * 4, cudaMemcpyDeviceToHost, s);
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += got[i] != perm_ref(x[i], y[i], sel[i]);
        CHECK(bad == 0, "__byte_perm: %d of %d differ (e.g. sel %04x: %08x vs %08x)", bad, n, sel[8], got[8], perm_ref(x[8], y[8], sel[8]));
        // host side too (the shim is host-callable)
        int hbad = 0;
        for (int i = 0; i < n; ++i) hbad += __byte_perm(x[i], y[i], sel[i]) != perm_ref(x[i], y[i], sel[i]);
        CHECK(hbad == 0, "__byte_perm on the host: %d differ", hbad);
        cudaFree(dx); cudaFree(dy); cudaFree(ds); cudaFree(dout);
    }
    // 4. fmaxf / fminf with a NaN: the other operand
    {
        float* d = nullptr;
        cudaMalloc(&d, 16);
        STRATA_LAUNCH(s, 1, 32, 0, fmax_kernel, d);
        float out[4];
        cudaMemcpyAsync(out, d, sizeof out, cudaMemcpyDeviceToHost, s);
        CHECK(out[0] == 1.0f && out[1] == 1.0f && out[2] == 2.0f && out[3] == 2.0f, "fmaxf/fminf NaN: %f %f %f %f", out[0], out[1], out[2], out[3]);
        cudaFree(d);
    }
    // 5. the polled-flag handshake: the host writes the flag 50 ms into the kernel's spin; the kernel sees it
    {
        unsigned *flag = nullptr, *out = nullptr;
        CHECK(cudaHostAlloc((void**) &flag, 64, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess, "hostalloc flag");
        CHECK(cudaHostAlloc((void**) &out, 64, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess, "hostalloc out");
        for (int round = 0; round < 3; ++round) {
            flag[0] = 0;
            out[0] = out[1] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const unsigned want = 100 + round;
            const uint32_t bound = ds4_spin_max(s);
            STRATA_LAUNCH(s, 1, 32, 0, wait_kernel, flag, want, bound, out);
            CHECK(cudaStreamQuery(s) == cudaErrorNotReady, "cudaStreamQuery while the kernel spins");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            ((volatile unsigned*) flag)[0] = want;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const auto t0 = std::chrono::steady_clock::now();
            while (((volatile unsigned*) out)[1] == 0 &&
                   std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) std::this_thread::yield();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            cudaStreamSynchronize(s);
            CHECK(cudaStreamQuery(s) == cudaSuccess, "cudaStreamQuery after sync");
            std::printf("handshake %d: kernel saw the flag after %u reads (bound %u), host saw the reply %.2f ms after the flag\n",
                        round, ((volatile unsigned*) out)[0], bound, ms);
            CHECK(((volatile unsigned*) out)[1] == 1 && ((volatile unsigned*) out)[0] < bound, "handshake %d: status %u spins %u", round,
                  out[1], out[0]);
        }
        // the bound: a flag that never comes ends the spin (a small bound: tens of ms)
        flag[0] = 0;
        out[0] = out[1] = 0;
        const auto t0 = std::chrono::steady_clock::now();
        STRATA_LAUNCH(s, 1, 32, 0, wait_kernel, flag, 999u, 20000u, out);
        cudaStreamSynchronize(s);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("bounded spin: %u reads, status %u, %.1f ms\n", out[0], out[1], ms);
        CHECK(out[1] == 2 && out[0] == 20000u, "bounded spin: status %u spins %u", out[1], out[0]);
        cudaFreeHost(flag);
        cudaFreeHost(out);
    }
    // 6. cudaMemGetInfo: real numbers, and a cudaMalloc shows in them
    {
        size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
        cudaMemGetInfo(&f0, &t0);
        void* p = nullptr;
        cudaMalloc(&p, (size_t) 64 << 20);
        cudaMemGetInfo(&f1, &t1);
        std::printf("cudaMemGetInfo: total %.2f GiB, free %.2f GiB -> %.2f GiB after a 64 MiB cudaMalloc\n", t0 / 1073741824.0,
                    f0 / 1073741824.0, f1 / 1073741824.0);
        CHECK(t0 > ((size_t) 1 << 30) && f0 < t0 && f1 + ((size_t) 60 << 20) <= f0, "meminfo %zu %zu %zu", t0, f0, f1);
        cudaFree(p);
        int n = 0;
        cudaGetDeviceCount(&n);
        std::printf("devices: %d\n", n);
    }
    // 7. a pageable async copy of a temporary: the data is the temporary's at the call
    {
        int* d = nullptr;
        cudaMalloc(&d, 1024 * sizeof(int));
        {
            std::vector<int> tmp(1024, 5);
            cudaMemcpyAsync(d, tmp.data(), 1024 * sizeof(int), cudaMemcpyHostToDevice, s);
            std::fill(tmp.begin(), tmp.end(), 9);
        }
        std::vector<int> back(1024, 0);
        cudaMemcpy(back.data(), d, 1024 * sizeof(int), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int v : back) bad += v != 5;
        CHECK(bad == 0, "pageable async copy: %d of 1024 changed", bad);
        cudaFree(d);
    }
    // 8. host registration: Mapped is refused (the device cannot read plain mmap'd memory on xe), a plain registration is
    // kept; cudaHostGetDevicePointer only for USM
    {
        std::vector<char> plain(1 << 16);
        CHECK(cudaHostRegister(plain.data(), plain.size(), cudaHostRegisterMapped) != cudaSuccess, "cudaHostRegister(Mapped) succeeded");
        void* dp = (void*) 1;
        CHECK(cudaHostGetDevicePointer(&dp, plain.data(), 0) != cudaSuccess && dp == nullptr, "cudaHostGetDevicePointer(malloc'd) gave %p", dp);
        CHECK(cudaHostRegister(plain.data(), plain.size(), 0u) == cudaSuccess, "cudaHostRegister(Default)");
        CHECK(!ds4_shim::pageable(plain.data() + 100, dpct::get_in_order_queue().get_context()), "a registered range still pageable");
        cudaHostUnregister(plain.data());
        CHECK(ds4_shim::pageable(plain.data() + 100, dpct::get_in_order_queue().get_context()), "an unregistered range not pageable");
        void* h = nullptr;
        cudaHostAlloc(&h, 4096, cudaHostAllocMapped);
        dp = nullptr;
        CHECK(cudaHostGetDevicePointer(&dp, h, 0) == cudaSuccess && dp == h, "cudaHostGetDevicePointer(Mapped USM) %p for %p", dp, h);
        cudaFreeHost(h);
    }
    // 9. a zero fill past 256 MiB (strata::big_fill_zero's kernels, not one blit) of an odd size: every byte, the tail too
    {
        const size_t n = ((size_t) 300 << 20) + 5;
        uint8_t* d = nullptr;
        CHECK(cudaMalloc(&d, n) == cudaSuccess, "cudaMalloc 300 MiB");
        for (int pass = 0; pass < 2; ++pass) {
            cudaMemset(d, 0xab, n);
            const auto t0 = std::chrono::steady_clock::now();
            if (pass == 0) cudaMemset(d, 0, n);
            else {
                cudaMemsetAsync(d, 0, n, s);
                cudaStreamSynchronize(s);
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            unsigned* nz = nullptr;
            cudaHostAlloc((void**) &nz, 64, cudaHostAllocMapped);
            nz[0] = 0;
            const uint8_t* cd = d;
            dpct::get_in_order_queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                if (cd[i] != 0) sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::system>(nz[0]).fetch_add(1u);
            }).wait();
            std::printf("%s of %zu bytes (300 MiB + 5) to 0: %.1f ms, %u bytes not 0\n", pass ? "cudaMemsetAsync" : "cudaMemset", n, ms, nz[0]);
            CHECK(nz[0] == 0, "big zero fill (%s): %u bytes not 0", pass ? "async" : "sync", nz[0]);
            cudaFreeHost(nz);
        }
        cudaFree(d);
    }
    // 10. a strided 2-D copy (a TP member's part of each down row: 5,120 rows of 297 of 594 bytes, IQ2_XXS) host -> device -> host,
    // from pinned and from pageable memory, against the bytes.  (Its speed: ~12 ms on the B70 either as one ext_oneapi_memcpy2d or
    // as a memcpy a row, measured 2026-10-24 - so the shim keeps the memcpy a row)
    {
        const size_t W = 297, SP = 594, H = 5120, DP = 304;
        std::vector<uint8_t> src(SP * H), back(W * H);
        for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t) (i * 2654435761u >> 13);
        uint8_t *pin = nullptr, *d = nullptr;
        cudaHostAlloc((void**) &pin, SP * H, 0);
        std::memcpy(pin, src.data(), SP * H);
        cudaMalloc(&d, DP * H);
        for (int from_pin = 0; from_pin < 2; ++from_pin) {
            cudaMemset(d, 0, DP * H);
            cudaMemcpy2DAsync(d, DP, from_pin ? (const uint8_t*) pin : src.data(), SP, W, H, cudaMemcpyHostToDevice, s);
            cudaMemcpy2DAsync(back.data(), W, d, DP, W, H, cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            size_t bad = 0;
            for (size_t r = 0; r < H; ++r) bad += std::memcmp(back.data() + r * W, src.data() + r * SP, W) != 0;
            CHECK(bad == 0, "cudaMemcpy2DAsync (%s): %zu of %zu rows wrong", from_pin ? "pinned" : "pageable", bad, H);
        }
        cudaFree(d);
        cudaFreeHost(pin);
    }
    // 11. expf / __expf / logf / __logf / erff / sincosf / fabsf on the device against the host's libm (a few ulp)
    {
        constexpr int n = 32;
        float hx[n], hy[n * 7];
        for (int i = 0; i < n; ++i) hx[i] = -4.0f + 0.25f * (float) i;
        float *dx = nullptr, *dy = nullptr;
        cudaMalloc((void**) &dx, sizeof hx);
        cudaMalloc((void**) &dy, sizeof hy);
        cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice);
        STRATA_LAUNCH(s, 1, n, 0, math_kernel, (const float*) dx, dy);
        cudaMemcpyAsync(hy, dy, sizeof hy, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            const double x = hx[i], want[7] = {std::exp(x), std::exp(x), std::log(std::fabs(x) + 1), std::log(std::fabs(x) + 1),
                                               std::erf(x), std::sin(x), std::cos(x)};
            for (int j = 0; j < 7; ++j) bad += std::fabs(hy[i * 7 + j] - want[j]) > 1e-5 * (1.0 + std::fabs(want[j]));
        }
        CHECK(bad == 0, "math wrappers: %d of %d values off", bad, n * 7);
        cudaFree(dx);
        cudaFree(dy);
    }
    cudaStreamDestroy(s);
    std::printf(fails == 0 ? "ALL OK\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
