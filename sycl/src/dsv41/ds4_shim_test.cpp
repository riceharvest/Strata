// ds4_shim_test: checks of sycl/src/dsv41/ds4_sycl.hpp: host-evaluated launch args, __byte_perm vs CUDA, a polled-flag handshake with a
// bounded spin, fmaxf NaN, SLM alignment, cudaMemGetInfo / cudaStreamQuery.  Small: < 1 MB of device memory.
#include "ds4_sycl.hpp"

#include <atomic>
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

int main() {
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
    cudaStreamDestroy(s);
    std::printf(fails == 0 ? "ALL OK\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
