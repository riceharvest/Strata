# Strata for the Intel Arc Pro B70

A fork of [Strata](https://github.com/Niko1221/Strata) that runs on one GPU only: the **Intel Arc Pro B70** (32 GB,
Battlemage `bmg-g31`) on Linux with the `xe` driver.

The NVIDIA (CUDA), AMD (HIP/ROCm) and Apple (Metal) runtimes are removed. What's left is Strata's engine ported to SYCL
(`sycl/`), tuned for this one card, plus Strata's server, setup and APIs. If you have a different GPU, use
[upstream Strata](https://github.com/Niko1221/Strata).

Not affiliated with Intel or with upstream Strata.

## What runs

| Model | Status on one B70 |
| --- | --- |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [ISTA-DASLab GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | Works. Numbers below |
| DeepSeek V4.1-Flash, [ISTA-DASLab sets](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), **pruned** (`--prune-experts`) | In progress. Pruning changes the answers; quality and speed not measured yet |
| DeepSeek V4.1-Flash, unpruned | Not recommended. On one card with 64 GB of RAM most experts come off the disk on every token: about 1 tok/s from a SATA SSD, about 5 tok/s from NVMe |

## What counts as usable here

A model and setting is called usable when it does all of this on one B70:

- 128K to 256K tokens of context
- at least 1,000 tokens/s reading the prompt
- at least 10 tokens/s writing the answer

Anything that misses one of these is labelled "works, not usable".

## Speed

Qwen3.8-Flash-Next IQ2_XS on one Arc Pro B70, Ryzen 9 5950X, 64 GB RAM. Settings:

- MTP draft, `--spec 4`, int8 KV
- 7 GB of VRAM kept free, because the card also drives a desktop
- page cache dropped before every start
- Strata's `standard-v1` benchmark, two runs each

| Context | Prompt (tok/s) | Answer (tok/s) |
| --- | ---: | ---: |
| short | | 57.4 / 57.7 |
| 4K | 1,542 / 1,612 | 51.2 / 51.5 |
| 16K | 1,689 / 1,701 | 51.9 / 51.9 |
| 128K, 256K | not measured yet | not measured yet |

That's +12% (4K) and +17% (16K) prompt speed over upstream Strata on the same card, with identical greedy answers.
[docs/B70.md](docs/B70.md) has the comparison, every change against upstream and who wrote it.

## What you need

- An Intel Arc Pro B70, Linux with the `xe` kernel driver and the Intel compute runtime (`ls /dev/dri` shows
  `renderD128`)
- Docker (or Podman). The engine builds and runs in a container, so the host needs no oneAPI
- 64 GB of RAM (what it's measured on)
- Disk: about 80 GB for Qwen3.8-Flash-Next IQ2_XS, about 260 GB for a DeepSeek V4.1-Flash set

## Install

1. Docker, once:

       sudo apt install docker.io git        # or your distro's equivalent
       sudo usermod -aG docker,render $USER  # then log out and in again

2. Get the code and build the engine. The image takes about 6 minutes once; the engine takes a few minutes:

       git clone -b b70 https://github.com/riceharvest/strata-b70-only && cd strata-b70-only
       docker build --pull -t strata-sycl-dev sycl/tools
       H=$(basename "$PWD")
       docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
           -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

   The build ends with `BUILD EXIT 0` and `errors: 0`. Build `strata-dsv41` the same way for DeepSeek.

3. Set up a model and serve it:

       ./setup.sh --backend sycl --model IQ2_XS

   Setup downloads the model, builds its pack, writes a config and a `run-<model>.sh` that starts the server.

Details, every engine switch and what was measured: [docs/INTEL.md](docs/INTEL.md). DeepSeek: [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

## Using it

The server speaks the OpenAI, Anthropic and OpenAI Responses APIs on the port setup prints:

| Your app speaks | Base URL |
| --- | --- |
| OpenAI API (most apps and agents) | `http://127.0.0.1:PORT/v1` |
| Anthropic API (Claude Code) | `http://127.0.0.1:PORT` (no `/v1`) |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:PORT/v1/responses` |

## Things that matter on this card

- **Keep VRAM headroom.** Near 32 GB the `xe` driver can wedge the card, and only a reboot or driver rebind brings it
  back. Leave a few GB free (`--vram-reserve-mib`).
- **Keep RAM headroom on a desktop.** On a 64 GB machine leave `--ple-io` at its default (`direct`); `ram` locks a
  28.8 GB table into memory. Size `STRATA_MIRROR_MIB` so a few GB stay free.
- **B70 defaults are on.** `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1` and `STRATA_STAGER_THREADS` are set
  unless you set them yourself. `0` turns one off; `STRATA_B70_DEFAULTS=0` turns all three off.

## Credits and license

- The engine and server are [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors. Several B70
  changes here are community pull requests to upstream, credited in [docs/B70.md](docs/B70.md).
- The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. The quantized sets
  are by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab), who also made the DeepSeek V4.1-Flash sets.
- Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp).

MIT License ([LICENSE](LICENSE)). The models have their own licenses.
