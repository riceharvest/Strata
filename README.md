<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<h2 align="center">Strata — Intel Arc Pro B70 (<code>bmg-g31</code> / Xe2) Edition</h2>

<p align="center">
  <b>Hyper-optimized SYCL 2020 + Xe2 ESIMD DPAS + AVX-512 VNNI inference engine for Intel Arc Pro B70 (32 GB GDDR6) &amp; Battlemage</b><br>
  <sub>Stream 125B–1T+ Mixture-of-Experts models (<b>Qwen3.8-Flash-Next</b> &amp; <b>DeepSeek-V4.1-Flash-1017</b>) at full PCIe 5.0 + XMX speed on Linux</sub>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/GPU-Intel%20Arc%20Pro%20B70%20(bmg--g31%20%2F%2032GB)-0071C5" alt="Intel Arc Pro B70">
  <img src="https://img.shields.io/badge/Backend-oneAPI%202026%20SYCL%20%2B%20ESIMD%20DPAS-0071C5" alt="SYCL 2020 + ESIMD">
  <img src="https://img.shields.io/badge/CPU%20Lane-AVX--512%20VNNI%20%2B%20VPOPCNTDQ-orange" alt="AVX-512">
</p>

---

## Overview

**`riceharvest/Strata`** is an architecture-specific fork of [Strata](https://github.com/Niko1221/Strata) engineered specifically for the **Intel Arc Pro B70 (`bmg-g31`, 32 GB GDDR6, 32 Xe2 cores / 256 XMX engines, PCIe 5.0 x16)** and Battlemage (`bmg-g21` / Arc B580 / B60) on Linux (`xe` kernel driver + Intel oneAPI 2026).

Upstream Strata's SYCL backend was produced via automated CUDA-to-SYCL (`dpct`) migration, leaving substantial Level Zero submission overhead, host-stalling synchronization primitives, unaligned byte-pair loads on Xe2's 32-bit LSC data port, incomplete ESIMD DPAS prefill format coverage, and no SYCL port for `DeepSeek-V4.1-Flash` (`strata-dsv41`).

This fork strips out the DPCT translation overhead, rewires the decode, MTP verify, and ESIMD DPAS prefill pipelines specifically for **Battlemage Xe2 (`sub_group_size(16)`, `8x16x32` int8 XMX DPAS)**, fixes Linux `xe` + IOMMU `DMA-FQ` host-mirror page fragmentation (`Strata#3681`), and ports the entire **16-translation-unit `DeepSeek-V4.1-Flash` (`strata-dsv41`)** engine to native SYCL 2020.

---

## Supported Models & Arc Pro B70 (32 GB VRAM) Placement

Strata runs sparse Mixture-of-Experts (MoE) models that far exceed GPU VRAM by keeping dense attention/MLA layers, KV cache, and the hottest routed experts resident in **32 GB GDDR6 VRAM**, holding warm experts in **2 MiB hugepage-backed pinned USM host RAM** (streamed over PCIe 5.0 DMA or co-processed on the CPU AVX-512 lane), and reading cold experts / Engram lookup tables from NVMe SSD.

### 1. `Qwen3.8-Flash-Next` (125B-A3B, 512 experts/layer, top-10 routed, MTP speculative verify)
Engine binary: **`strata-generate`**

| Quantization | Total Footprint | B70 (32 GB VRAM) + 64 GB RAM Placement | ESIMD DPAS Prefill | Decode / MTP Verify Kernels |
| :--- | ---: | :--- | :---: | :---: |
| **`Q2_0`** (`GSQ-RCO`) | ~35 GB | Trunk + ~45% experts in VRAM; rest in pinned USM host RAM | `T_Q2_0` (Down) | Native SWAR + `Multi` |
| **`IQ2_XS`** (`GSQ-RCO` / `Swift 1.5`) | ~39.2 GB | Trunk + ~40% experts in VRAM; rest in pinned USM host RAM | `T_IQ2XS` (Gate/Up) + `T_IQ4NL`/`T_Q2_0` (Down) | `Multi<17>` + `load4_a2` |
| **`IQ3_XXS`** (`GSQ-RCO`) | ~47.0 GB | Trunk + ~33% experts in VRAM; rest in pinned USM host RAM | `T_IQ3XXS` (Gate/Up) + `T_IQ4NL` (Down) | `Multi<18>` + `load4_a2` |
| **`IQ3_S`** (`GSQ-RCO`) | ~54.8 GB | Trunk + ~28% experts in VRAM; rest in pinned USM host RAM | `T_IQ3S` (Gate/Up) + `T_IQ4NL` (Down) | `Multi<21>` + `load4_a2` |
| **`UD-IQ4_XS`** / **`UD-Q4_K_XL`** | ~68–72 GB | Trunk + hot experts in VRAM; warm experts in RAM / SSD | `T_IQ4XS` (Gate/Up) + `T_Q8_0` (Down) | `Multi<23>` + in-reg LUT + SWAR `q6_K` |
| **`Qwen3-Coder-Next`** | ~32 GB | Trunk + ~50% experts in VRAM; rest in pinned USM host RAM | Full Xe2 DPAS | Full Xe2 Native |

### 2. `DeepSeek-V4.1-Flash-1017` (`ISTA-DASLab` GSQ-RCO + 111 GB `IQ4_NL` Engram)
Engine binary: **`strata-dsv41`** *(ported to SYCL 2020 in this fork)*

| Quantization (`ISTA-DASLab`) | Download Size | B70 (32 GB VRAM, 1 GiB reserved) Residency | RAM + NVMe Tiering (64 GB Host RAM) | Quality vs FP8 Base |
| :--- | ---: | :--- | :--- | :--- |
| **`ist-das-1.75bpw`** (`IQ1_M`) | 239 GB | Trunk (`4.65 GiB`) + KV + **~16,300 hot experts** (`~24.5 GiB`) in VRAM | ~20,900 experts in 2 MiB hugepage RAM; cold experts + 111 GB Engram on NVMe | 98.7% task average (AIME25, GPQA-D, LCB v6) |
| **`ist-das-2.00bpw`** (`IQ2_XXS`) | 257 GB | Trunk (`4.65 GiB`) + KV + **~14,350 hot experts** (`~24.5 GiB`) in VRAM | ~18,400 experts in 2 MiB hugepage RAM; cold experts + 111 GB Engram on NVMe | KL 0.170; code +8.6%, agent +5.3% |
| **`ist-das-2.25bpw`** (`IQ2_XS`) | 274 GB | Trunk (`4.65 GiB`) + KV + **~12,800 hot experts** (`~24.5 GiB`) in VRAM | ~16,400 experts in 2 MiB hugepage RAM; cold experts + 111 GB Engram on NVMe | KL 0.107; code +3.2%, agent +2.2% |

---

## Key Optimizations in This Fork

### 1. Zero-Overhead SYCL / Level Zero Dispatch & Lean MTP Verify
| Subsystem | Upstream Bottleneck | B70 Fork Optimization | Impact |
| :--- | :--- | :--- | :--- |
| **Kernel Launch Properties** (`sycl/src/**/*.dp.cpp`) | DPCT attached `sycl::ext::oneapi::experimental::use_root_sync` to **394 kernel launches** across 49 `.dp.cpp` files even though zero kernels call `get_root_group()`. | Stripped all 394 `use_root_sync` properties so Level Zero uses standard `zeCommandListAppendLaunchKernel` instead of `zeCommandListAppendLaunchCooperativeKernel`. | Eliminates Level Zero cooperative residency/occupancy queries and implicit slice serialization across ~600+ launches/token. |
| **MTP Verify Window MoE** (`sycl/src/core/verify.cpp`) | On every MoE layer (48 layers), `verify.cpp` launched `grouped` -> `hit_out`, `copy_or_zero_from_mapped` -> zero `parts_out`, and `moe_hit_add` -> `parts_out += hit_out` even when CPU host misses were disabled. | In `lean` mode (`STRATA_VERIFY_DEVICE_PLAN=1` + `STRATA_VERIFY_NO_HOST=1`), `grouped` writes directly into `parts_out`, bypassing `copy_or_zero_from_mapped` and `moe_hit_add`. | Eliminates **96 to 192 kernel launches and ~3.8 MB of redundant VRAM traffic** per verify step. |
| **Device Barrier Sync** (`sycl/include/dpct/device.hpp`) | `dpct::sync_barrier()` called `dev_mgr::instance().queues_wait_and_throw()` on the CPU host before submitting a dummy `single_task` kernel. | Replaced with pure non-blocking `queue->ext_oneapi_submit_barrier()`. | Prevents full-device CPU host stalls during multi-queue prefill and expert staging. |
| **Queue Resolution** (`sycl/include/strata/sycl_queue.hpp`) | `strata::q_of(nullptr)` locked `dev_mgr::m_mutex` twice per kernel submission (`current_device_id()` + `get_device()`). | Added lock-free `thread_local` fast path caching the default in-order queue pointer per host thread. | Eliminates ~1,200 global mutex locks per decode step. |
| **B70 Auto-Defaults** (`sycl_queue.hpp`, `generate.cpp`) | Required manual wrapper env vars to enable device-side verify planning and tune PCIe stager thread count. | `strata::apply_sycl_arch_defaults()` detects Battlemage (`xe` driver) at startup and enables `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1`, and `STRATA_STAGER_THREADS=12`. | Out-of-the-box optimal B70 execution even when running `strata-generate` directly. |

### 2. USM Host Expert Mirror 2 MiB Hugepage Advisory (`Strata#3681`)
On Linux with the `xe` driver and IOMMU `DMA-FQ`, allocating a 25–42 GiB USM host expert mirror (`sycl::malloc_host` in `gguf_expert_source.cpp` or `mmap` in `ds4_os.hpp`) after page-cache pressure causes the kernel to back pinned DMA mappings with fragmented 4 KiB (`order-0`) pages. When the B70 DMAs cold experts over PCIe 5.0 during decode, 4 KiB scatter-gather lists thrash the GPU/IOMMU IOTLB:
* **Fragmented 4 KiB USM Host Pages:** `9.6–15.0 GB/s` PCIe read bandwidth → **26.9 tok/s** decode (`37 ms/step`).
* **2 MiB Contiguous Hugepage USM Host Mirror:** `24.0 GB/s` PCIe read bandwidth → **42.7 tok/s** decode (`23 ms/step`, **+58% throughput**).

**What this fork does:**
* Calls `madvise(p, bytes, MADV_HUGEPAGE)` on every 2 GiB `sycl::malloc_host` chunk in `sycl/src/core/gguf_expert_source.cpp` immediately upon allocation before reader threads fault in the pages.
* Aligns `range_reserve` to 2 MiB (`kHugePageAlign`) and calls `madvise(..., MADV_HUGEPAGE)` inside `range_commit` in `src/dsv41/ds4_os.hpp` and `sycl/src/dsv41/ds4_os.hpp`.
* Samples `/proc/meminfo` (`AnonHugePages`) and `/proc/buddyinfo` (`Normal` zone `order-9` 2 MiB + `order-10` 4 MiB blocks) before and after mirror population, emitting an explicit actionable warning if `xe` DMA will be IOTLB-throttled.
* Adds preflight THP & buddy allocator checks to `sycl/serve/strata-sycl.sh`.

### 3. Full-Format Xe2 Int8 ESIMD DPAS Prefill (`esimd_moe.dp.cpp`, `prefill.cpp`)
Battlemage's Xe2 XMX systolic array executes `8x16x32` int8 DPAS instructions (`esimd::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>`). Upstream only whitelisted a subset of quant formats and fell back to slow scalar per-token GEMV on `IQ4_XS`, `IQ2_XS`, `IQ2_XXS`, and `Q8_0`.
* **Complete Gate/Up Format Coverage:** `T_IQ3S (21)`, `T_IQ3XXS (18)`, `T_IQ2S (22)`, `T_IQ2XS (17)`, `T_IQ2XXS (16)`, and `T_IQ4XS (23)`.
* **Complete Down Format Coverage:** `T_IQ4NL (20)`, `T_Q2_0 (42)`, and `T_Q8_0 (8)`.
* **SLM Activation Tile Sharing (`launch_gu_wg`):** Enabled workgroup-cooperative SLM activation caching for 2-bit formats (`T_IQ2S`, `T_IQ2XS`, `T_IQ2XXS`) with `tab_bytes = 0` (saving 2 KiB of SLM per WG), and coalesced per-expert barriers in `prefill.cpp`.

### 4. Battlemage Decode & MTP Verify Kernel Optimizations (`iq_kernels.dp.cpp`, `router_top10.dp.cpp`, `native_router.dp.cpp`)
* **Aligned 32-Bit LSC Dword Loads (`load4_a2` / `load8_a2`):** Replaces 16-bit byte-pair loads (`get_int_b2`) with aligned 32-bit loads across `vec_dot_iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s_q8_1`, `vec_dot_q4_0/q5_0/q6_K/q8_0_q8_1`, `KqSplit<14, 8>::load`, and `dq_q4_k`, cutting LSC load instructions in half.
* **In-Register `iq4nl_pair` LUT & 3-Op SWAR `q6_k_centred`:** Replaces global-memory `kvalues_iq4nl[]` table lookups in `vec_dot_iq4_xs_q8_1` with an in-register arithmetic lookup, and replaces DPCT's 4-byte `dpct::vectorized_binary<sycl::char4>(..., dpct::sub_sat())` loop in `q6_K` with 3-instruction SWAR centering (`v - 32`).
* **SWAR Sign Application (`swar_ne4` / `swar_sub4`) & Multi-Token Weight Reuse (`Multi<16, 17, 23>`):** Eliminates per-byte `uchar4` loops in `Split<16, 17, 22, 18, 21, 20>` and enables `kEntryMulti` for `IQ2_XXS`, `IQ2_XS`, and `IQ4_XS` in `native_gu_kernel` and `native_down_kernel` so MTP verify tokens reuse dequantized weights in registers.
* **Native SYCL 2020 Group Collectives & Single-Subgroup Routers:** Replaces `dpct::experimental::permute_sub_group_by_xor` / `shift_sub_group_left` (which emitted bounds-checking branches on every shuffle) with native SYCL 2020 `sycl::permute_group_by_xor` / `sycl::inclusive_scan_over_group` across 9 hot kernel files, enables the S6 single-subgroup top-10 router (`router_top10.dp.cpp`), and shrinks `native_router.dp.cpp` from 8 warps (7 idle) to 1 subgroup.

### 5. AVX-512 (`VPOPCNTDQ` + `VNNI`) CPU Expert Co-Processing Lane (`iq_avx512.cpp`, `pool.hpp`)
* **Hardware `VPOPCNTDQ` + `VNNI` (`_mm512_dpwssd_epi32`) in `iq512_gu_rows` & `iq512_rows`:** Enables `FmtX<TY>::decode` (`_mm256_popcnt_epi32` parity sign extraction + `_mm512_dpwssd_epi32` int16 dot-product accumulation) in `row_dot` instead of scalar `kSigns` table gathers and 3-instruction `maddubs + madd + add` sequences.
* **Fast Scale Broadcast & Cache-Line Isolation:** Replaces cross-lane `_mm256_permutevar8x32_epi32` scale expansion in `scales4` with `vpshufb + vpmovzxbd`, and aligns `ExpertPool::wstate_` to 64-byte cache lines (`alignas(64)`) to prevent false sharing across worker threads.

### 6. Full 16-TU `DeepSeek-V4.1-Flash` (`strata-dsv41`) SYCL 2020 Port (`sycl/src/dsv41/`)
Upstream only built `strata-dsv41` for CUDA (`src/dsv41/`) and HIP (`hip/src/dsv41/`). This fork ports all 16 translation units to SYCL 2020 in `sycl/src/dsv41/`:
* **6 Device Kernel TUs:** `ds4_gemv.dp.cpp`, `ds4_mla_ops.dp.cpp` (Multi-head Latent Attention + RoPE + KV cache), `ds4_moe_ops.dp.cpp` (256-expert group router + GSQ-RCO expert kernels), `ds4_SanityCheck.dp.cpp`, `ds4_prefill_sm120.dp.cpp`, `ds4_trellis_tricache.dp.cpp`.
* **10 Host Engine TUs:** `ds4_model.cpp`, `ds4_generate.cpp`, `ds4_prefill.cpp`, `ds4_verify.cpp`, `ds4_mla.cpp`, `ds4_moe.cpp`, `ds4_CPU_moe.cpp`, `ds4_embed.cpp`, `ds4_sampler.cpp`, `ds4_tok.cpp`.
* **USM Device + Pinned USM Host Expert Pool (`ds4_moe.cpp`):** Replaces CUDA Driver API `cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` virtual memory stitching with a contiguous USM device pool (`sycl::malloc_device`) and a 2 MiB hugepage-advised pinned USM host pool (`sycl::malloc_host`) for PCIe 5.0 DMA expert streaming.

---

## Quickstart: Building & Running on Intel Arc Pro B70

### 1. Host Prerequisites (Linux + `xe` Driver)
Ensure transparent hugepages are enabled (`always` or `madvise`) and proactive memory compaction is active so the 25–42 GiB USM host expert mirror receives contiguous 2 MiB pages:

```bash
# Enable madvise THP and compact fragmented 4 KiB pages into 2 MiB blocks
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
sudo sysctl -w vm.compaction_proactiveness=20
echo 1 | sudo tee /proc/sys/vm/compact_memory
```

### 2. Build `strata-generate` & `strata-dsv41`
You can build either natively with **Intel oneAPI 2026.0+** (`icpx` + MKL) installed on the host, or inside Intel's official `intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04` container:

#### Option A: Native Build (with oneAPI 2026 installed)
```bash
source /opt/intel/oneapi/setvars.sh
cmake -B sycl/build -S sycl -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -C sycl/build strata-generate strata-dsv41
```

#### Option B: Containerized Build (no host oneAPI installation required)
```bash
docker run --rm -v "$PWD":/work -w /work \
  intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04 \
  bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && \
           cmake -B sycl/build -S sycl -GNinja -DCMAKE_BUILD_TYPE=Release && \
           ninja -C sycl/build strata-generate strata-dsv41'
```

### 3. Run `Qwen3.8-Flash-Next` on Arc Pro B70
Use the guided Intel setup script (`sycl/setup_intel.py`) or launch the OpenAI/Anthropic-compatible server directly via `sycl/serve/strata-sycl.sh`:

```bash
# Guided setup (downloads model, detects B70 32 GB VRAM, configures expert cache & launches server)
python3 sycl/setup_intel.py

# Or launch directly with strata-sycl.sh (runs THP/buddyinfo preflight checks automatically)
./sycl/serve/strata-sycl.sh
```

### 4. Run `DeepSeek-V4.1-Flash-1017` on Arc Pro B70 (`strata-dsv41`)
Compute the exact B70 VRAM / Host RAM / NVMe expert placement plan with `strata_control.dsv41_fit` (automatically detects Intel Arc Pro B70 via `sycl-ls` / sysfs and reserves `1024 MiB` for Level Zero context & scratch buffers):

```bash
# Preview the B70 hardware placement plan and generated strata-dsv41 command line
python3 -m strata_control.dsv41_fit \
  --model-dir /path/to/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --variant 1.75bpw \
  --ctx 32768
```

#### Zero-SSD Execution via Dynamic Router Pruning (`--prune-experts 160`)
DeepSeek-V4.1 has 384 experts/layer, but activation profiles show a heavy power law: the top 160 experts account for **71.0% of all activations** (and top 192 account for **77.9%**). By setting `--prune-experts 160` (or `STRATA_DS4_PRUNE_EXPERTS=160`):
- **3,789 active experts (~28.3 GB)** fit completely in the B70's 32 GB GDDR6 VRAM.
- The remaining **2,611 active experts (~19.5 GB)** fit 100% in Host RAM.
- **SSD spill drops to 0 GB!** Cold experts outside the top $N$ are masked out of routing decisions with a $-10^9$ router bias, eliminating NVMe read stalls completely from the critical token decode path.

```bash
# Run with zero SSD spill on an Intel Arc Pro B70 + 64 GB RAM
./sycl/build/strata-dsv41 \
  --pack /path/to/DeepSeek-V4.1-Flash-1.75bpw/pack \
  --prune-experts 160 \
  --reserve-mib 1024 \
  --engram-cache-mb 4096 \
  --serve
```

Or select **DeepSeek V4.1-Flash (`1.75bpw` / `2.00bpw` / `2.25bpw`)** interactively in `python3 sycl/setup_intel.py`.

---

## Arc Pro B70 Runtime Environment Variables

`strata::apply_sycl_arch_defaults()` automatically configures optimal defaults when it detects an Intel `xe` GPU at startup. Any setting can still be overridden via environment variables:

| Environment Variable | B70 Default | Description |
| :--- | :---: | :--- |
| `STRATA_DS4_PRUNE_EXPERTS` | `0` (off, or `160`/`192`) | Masks bottom $384 - N$ experts per layer in router logits ($-10^9$ bias). `--prune-experts 160` fits all active experts in 32 GB VRAM + 64 GB RAM with **0 GB SSD spill**. |
| `STRATA_VERIFY_DEVICE_PLAN` | `1` (auto) | Builds the MTP speculative verify expert routing plan on the GPU (`verify_plan_build`) and uses the `lean` direct-to-`parts_out` verify graph. |
| `STRATA_VERIFY_NO_HOST` | `1` (auto) | Keeps MTP verify MoE dispatch purely on the GPU + asynchronous PCIe DMA stager without stalling on host CPU misses. |
| `STRATA_STAGER_THREADS` | `12` (auto) | Number of host threads copying missed experts into the pinned staging ring (`12` saturates PCIe 5.0 x16 without scheduling contention). |
| `STRATA_ESIMD_PREFILL` | `1` (default) | Uses fused Int8 Xe2 ESIMD DPAS (`8x16x32` XMX) for MoE prefill across all supported quant types (`IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `IQ4_XS`, `IQ4_NL`, `Q2_0`, `Q8_0`). |
| `STRATA_DPAS_DENSE` | `1` (default) | Uses Xe2 BF16 XMX DPAS (`esimd_gemm.dp.cpp`) for dense trunk / attention / LM-head GEMMs. |
| `STRATA_EXPERT_BUDGET_MB` | Auto (~22–26 GB) | Override the GDDR6 VRAM budget (in MiB) dedicated to the GPU hot-expert cache. |
| `STRATA_DS4_DISK_SKIP_W` | `0.025` (auto on B70) | Skips blocking NVMe SSD reads for cold routed experts carrying <2.5% token weight (cuts SSD stall time by ~40–50%). |
| `DS4_ENGRAM_CACHE_MB` | `4096` (auto on 64 GB) | In-memory cache for the 111 GB `IQ4_NL` Engram table (uncapped from upstream's 256 MB default). |
| `DS4_ENGRAM_THREADS` | `32–64` (auto on Linux) | Number of asynchronous reader threads to saturate NVMe controller queue depth for parallel IOPS. |
| `DS4_AHEAD_EXTRA` | `2` (auto with cache) | Lookahead expert predictor padding (predicts top-8 instead of top-6), raising next-layer lookahead hit rate to >92%. |
| `STRATA_DS4_SLOT_KEEP` | `1` | Keeps multi-turn agent conversation KV slots cached to avoid re-reading long prompts across turns. |
| `NEOReadDebugKeys=1` `UseKmdMigration=0` | Set in `strata-sycl.sh` | Prevents the Level Zero / NEO runtime from migrating USM allocations through KMD page-fault handlers. |

---

## Repository Structure

```text
Strata/
├── sycl/
│   ├── CMakeLists.txt                   # Builds strata-generate & strata-dsv41 with icpx (-fsycl -mavx512vnni -mavx512vpopcntdq)
│   ├── setup_intel.py                   # Interactive Intel Arc B70 builder, model downloader & launcher (Qwen3.8 + DeepSeek-V4.1)
│   ├── include/
│   │   ├── dpct/device.hpp              # Non-blocking ext_oneapi_submit_barrier() sync_barrier implementation
│   │   └── strata/
│   │       ├── sycl_queue.hpp           # Lock-free thread-local queue lookup & apply_sycl_arch_defaults() for Battlemage
│   │       └── core/gguf_expert_source.hpp
│   ├── src/
│   │   ├── core/
│   │   │   ├── gguf_expert_source.cpp   # 2 MiB MADV_HUGEPAGE USM host expert mirror + /proc/buddyinfo diagnostics
│   │   │   └── verify.cpp               # Lean direct-to-parts_out MTP speculative verify pipeline
│   │   ├── prefill/
│   │   │   ├── esimd_moe.dp.cpp         # Xe2 Int8 8x16x32 ESIMD DPAS MoE prefill (IQ2/IQ3/IQ4/Q2_0/Q8_0)
│   │   │   ├── esimd_gemm.dp.cpp        # Xe2 BF16 XMX ESIMD DPAS dense GEMM
│   │   │   └── prefill.cpp              # Prefill dispatcher with coalesced barriers & full ESIMD format routing
│   │   ├── kernels/cuda/
│   │   │   ├── iq_kernels.dp.cpp        # B70-optimized decode/verify kernels (load4_a2, SWAR, Multi<16/17/23>, native SYCL shuffles)
│   │   │   ├── router_top10.dp.cpp      # S6 single-subgroup top-10 router
│   │   │   └── native_router.dp.cpp     # Single-subgroup native router
│   │   └── dsv41/                       # Full 16-TU SYCL 2020 port of DeepSeek-V4.1-Flash (strata-dsv41)
│   └── serve/
│       ├── strata-sycl.sh               # B70 server launcher with THP & /proc/buddyinfo preflight checks
│       └── server_intel.py              # OpenAI & Anthropic API server with cached token-byte decoding
├── src/kernels/cpu/
│   └── iq_avx512.cpp                    # AVX-512 VPOPCNTDQ + VNNI (_mm512_dpwssd_epi32) CPU expert co-processing lane
└── strata_control/
    └── dsv41_fit.py                     # Hardware discovery & VRAM/RAM/SSD placement planner for DeepSeek-V4.1-Flash
```

---

## Running CPU Unit & Bit-Exactness Verification Tests

All B70 SWAR bit-manipulation math (`q6_k_centred`, `swar_ne4`, `swar_sub4`, `iq4nl_pair`, `load4_a2`) and `/proc/buddyinfo` + `/proc/meminfo` hugepage diagnostics have standalone CPU unit tests that run without touching `/dev/dri`:

```bash
g++ -O3 -std=c++20 sycl/src/kernels/iq_b70_math_test.cpp -o /tmp/iq_b70_math_test && /tmp/iq_b70_math_test
g++ -O2 -std=c++20 sycl/src/core/gguf_expert_source_test.cpp -o /tmp/gguf_expert_source_test && /tmp/gguf_expert_source_test
```

---

## Credits & License

* **Upstream Strata Engine:** [Niko1221/Strata](https://github.com/Niko1221/Strata)
* **Models & Quantizations:** [Qwen Team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [DeepSeek-AI](https://huggingface.co/deepseek-ai), [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) (`GSQ-RCO` quantizations for Qwen3.8-Flash-Next & DeepSeek-V4.1-Flash-1017), UkisAI, agentionai, Unsloth.
* **License:** [MIT License](LICENSE).
