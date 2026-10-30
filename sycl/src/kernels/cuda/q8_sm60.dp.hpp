#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
// src/kernels/cuda/q8_sm60.cuh - a Q8_0 decode GEMV for Pascal GP100 (compute
// capability 6.0), opt-in with STRATA_Q8_SM60=1.  Included by native_mmvq.cu,
// and by a standalone parity/speed check, so both run this code.
//
// Reads the packed planes of STRATA_Q8_PACKED (the SAME int8 values row-major,
// n_in bytes per row, then one fp16 scale per 32 of them) for every Q8_0 dense
// matrix and the head, whatever their shape.  Why the packed kernel above is
// slow on a GP100 die (measured on PH402, 1050 MHz: ~137 GB/s for the dense
// projections, ~160 for the head, against 220-270 for this layout): one
// workgroup per row, eight bytes per thread per pass and a __syncthreads() per
// row - a 2560-wide row is 80 blocks, so the reduction costs more than the row
// - and the window's activations read as single ints, because a q8_1 block is
// 36 bytes and its values start at byte 4.
//
// Here:
//   * a workgroup first copies the window's activations into shared memory as
//   an int8 plane per column and a float
//     scale per 32 (once; the workgroup then loops over rows), so every
//     activation read is a 16-byte shared load;
//   * TPR threads own a row (a whole warp for n_in >= 2048, a half or a quarter
//   for narrower rows) and read its
//     weights 16 bytes at a time through the read-only cache; the dot of 4 ints
//     is STRATA_DP4A (vmad on sm_60);
//   * the sum is a shuffle within those TPR lanes - no shared memory, no
//   barrier per row.
// The per-32 products are added in another order than the exact kernels', so
// outputs are not bitwise theirs (float rounding only; the int8 dots are
// exact).
#pragma once

namespace q8sm60 {

constexpr int THREADS = 256;
constexpr std::size_t MAX_SMEM = 48 * 1024;   // a Pascal workgroup's static + dynamic limit
constexpr int MAX_COLS = 8;                        // columns of one launch (a window of 9..16 rows runs in passes)

inline std::size_t smem_bytes(int ncols, int n_in) {
    return std::size_t(ncols) * std::size_t(n_in) + std::size_t(ncols) * std::size_t(n_in / 32) * sizeof(float);
}

template <int NCOLS, int TPR>
/*
DPCT1110: The total declared local variable size in device function kernel
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
kernel(const int8_t *__restrict__ qs, const sycl::half *__restrict__ dpl,
       const Q81Block *__restrict__ x, float *__restrict__ y, int n_in,
       int n_out, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto q8sm60_smem = (sycl::int4 *)dpct_local;
    int8_t* sx = reinterpret_cast<int8_t*>(q8sm60_smem);                         // [NCOLS][n_in]
    float* sd = reinterpret_cast<float*>(sx + std::size_t(NCOLS) * n_in);       // [NCOLS][n_in / 32]
    const int nb = n_in / 32;
    for (int b = int(item_ct1.get_local_id(2)); b < NCOLS * nb;
         b += THREADS) { // column j = b / nb, block b % nb
        const Q81Block* xb = x + b;
        const int* src = reinterpret_cast<const int*>(xb->qs);
        int* dst = reinterpret_cast<int*>(sx + std::size_t(b) * 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = src[i];
        sd[b] = xb->ds[0];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    constexpr int RPB = THREADS / TPR;
    const int lane = int(item_ct1.get_local_id(2)) % TPR;
    const int rloc = int(item_ct1.get_local_id(2)) / TPR;
    const int nch = n_in / 16;                                                   // 16-byte chunks, two per block
    const int groups = (n_out + RPB - 1) / RPB;
    for (int g = int(item_ct1.get_group(2)); g < groups;
         g += int(item_ct1.get_group_range(2))) {
        const int row = g * RPB + rloc;
        float acc[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] = 0.0f;
        if (row < n_out) {
            const sycl::int4 *w = reinterpret_cast<const sycl::int4 *>(
                qs + std::size_t(row) * n_in);
            const sycl::half *d = dpl + std::size_t(row) * nb;
            for (int c = lane; c < nch; c += TPR) {
                /*
                DPCT1098: The '*' expression is used instead of the __ldg
                call. These two expressions do not provide the exact same
                functionality. Check the generated code for potential precision
                and/or performance issues.
                */
                const sycl::int4 wv = *(w + c);
                const float dw =
                    sycl::vec<sycl::half, 1>(d[c >> 1])
                        .convert<float, sycl::rounding_mode::automatic>()[0];
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const sycl::int4 xv = reinterpret_cast<const sycl::int4 *>(
                        sx + std::size_t(j) * n_in)[c];
                    int s = STRATA_DP4A(wv.x(), xv.x(), 0);
                    s = STRATA_DP4A(wv.y(), xv.y(), s);
                    s = STRATA_DP4A(wv.z(), xv.z(), s);
                    s = STRATA_DP4A(wv.w(), xv.w(), s);
                    acc[j] += dw * sd[j * nb + (c >> 1)] * float(s);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {                                       // every lane takes part, row or not
            float v = acc[j];
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = TPR / 2; o > 0; o >>= 1) v +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), v, o,
                    TPR);
            if (lane == 0 && row < n_out) y[std::size_t(j) * n_out + row] = v;
        }
    }
}

// One workgroup per SM would leave too few rows in flight on a 48-SM die; four per SM is what the shapes of
// Qwen3.8-Flash-Next measured best (STRATA_Q8_SM60_BPS overrides it).
inline int blocks_per_sm() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_Q8_SM60_BPS");
        const int n = e ? std::atoi(e) : 0;
        return n > 0 ? n : 4;
    }();
    return v;
}

inline int sm_count() try {
    int dev = 0, n = 0;
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 ||
        DPCT_CHECK_ERROR(n = dpct::get_device(dev).get_max_compute_units()) !=
            0)
        return 48;
    return n > 0 ? n : 48;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <int NCOLS, int TPR>
void launch_one(const int8_t *qs, const sycl::half *d, const Q81Block *x,
                float *y, int n_in, int n_out, dpct::queue_ptr s) {
    constexpr int RPB = THREADS / TPR;
    const int groups = (n_out + RPB - 1) / RPB;
    const int grid = (std::min)(groups, sm_count() * blocks_per_sm());
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(smem_bytes(NCOLS, n_in)), cgh);

            cgh.parallel_for<
                dpct_kernel_name<class kernel_205b2e, dpct_kernel_scalar<NCOLS>,
                                 dpct_kernel_scalar<TPR>>>(
                sycl::nd_range<3>(sycl::range(1, 1, unsigned(grid)) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        kernel<NCOLS, TPR>(
                            qs, d, x, y, n_in, n_out,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
}

template <int NCOLS>
void launch_cols(const int8_t *qs, const sycl::half *d, const Q81Block *x,
                 float *y, int n_in, int n_out, dpct::queue_ptr s) {
    if (n_in >= 2048) launch_one<NCOLS, 32>(qs, d, x, y, n_in, n_out, s);
    else if (n_in >= 1024) launch_one<NCOLS, 16>(qs, d, x, y, n_in, n_out, s);
    else launch_one<NCOLS, 8>(qs, d, x, y, n_in, n_out, s);
}

// false: a shape this kernel does not take (the caller runs its usual kernel)
inline bool launch(const int8_t *qs, const sycl::half *d, const Q81Block *x,
                   float *y, int n_in, int n_out, int ncols,
                   dpct::queue_ptr s) {
    if (n_in % 32 != 0 || n_in < 128 || ncols < 1 || ncols > MAX_COLS || smem_bytes(ncols, n_in) > MAX_SMEM) return false;
    switch (ncols) {
        case 1: launch_cols<1>(qs, d, x, y, n_in, n_out, s); break;
        case 2: launch_cols<2>(qs, d, x, y, n_in, n_out, s); break;
        case 3: launch_cols<3>(qs, d, x, y, n_in, n_out, s); break;
        case 4: launch_cols<4>(qs, d, x, y, n_in, n_out, s); break;
        case 5: launch_cols<5>(qs, d, x, y, n_in, n_out, s); break;
        case 6: launch_cols<6>(qs, d, x, y, n_in, n_out, s); break;
        case 7: launch_cols<7>(qs, d, x, y, n_in, n_out, s); break;
        default: launch_cols<8>(qs, d, x, y, n_in, n_out, s); break;
    }
    return true;
}

// GGUF Q8_0 blocks (34 bytes, 2-byte aligned) -> the packed planes, on the device (the head's 0.6 GiB stays off the host)
__dpct_inline__ void pack_kernel(const Q80Block *__restrict__ blocks,
                                 int8_t *__restrict__ qs,
                                 sycl::half *__restrict__ d,
                                 std::size_t count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const std::size_t b =
        std::size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (b >= count) return;
    const Q80Block& src = blocks[b];
    const uint16_t* in = reinterpret_cast<const uint16_t*>(src.qs);
    uint16_t* out = reinterpret_cast<uint16_t*>(qs + b * 32);
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = in[i];
    d[b] = src.d;
}

} // namespace q8sm60
