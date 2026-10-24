"""Which picture reader an engine folder gets: the GPU one (`strata-vision-sycl`, llama.cpp's SYCL backend) or the CPU one
(`strata-vision`).

An Intel Arc engine folder carries the CPU reader, `strata-vision`: it needs nothing from the GPU runtime, so it starts on every
PC.  `strata-vision-sycl` (tools/vision with STRATA_VISION_SYCL) is the same reader on the card.  Setup writes the run config with
the GPU reader first and the CPU reader as its fallback (`vision.cpu_exe`); the server (serve/server.py, Vision) starts the CPU
reader by itself, with one log line, when the GPU reader does not start.  Nothing is asked of the user.

`choose` is the whole decision; setup calls it, and tools/test_vision_reader.py checks every branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CPU_READER = "strata-vision"
SYCL_READER = "strata-vision-sycl"
GPU_READERS = (SYCL_READER,)                  # the GPU reader of an engine folder

#: Owner decision 2026-10-18: pictures are ON by default, and the reader runs on the CPU for a card under 24 GB (the GPU reader keeps about
#: 1-1.4 GB of VRAM that the experts would use: a few % of decode, more on some PCs, #2406) and on the GPU for 24 GB and above - on every make
#: of card.  A person can switch it per model (--vision gpu / --vision cpu, the Pictures step); what was asked for is kept.
GPU_MIN_VRAM_GB = 23.5

#: The one exception to that rule.  An Intel Arc of 24 GB or more keeps the CPU reader by default: the SYCL GPU reader works (tested on an A750,
#: forced with --vision gpu: 0.16 s a picture against 1.7 s on the CPU, embeddings cosine >= 0.995 to the CPU reader's) but it needs the oneAPI
#: runtime on the PC and has not run on a 24 GB+ card (an Arc Pro B70 is allowed since 2026-10-22; the CPU reader is what it gets).  --vision gpu asks for it.  Set True when a 24 GB+ Arc has run it.
INTEL_GPU_READER_BY_DEFAULT = False


def default_mode(vram_gb: float | None, make: str = "") -> str:
    """"cpu" or "gpu": the reader an unspecific "pictures on" gets on a card with this much memory (unknown memory: "gpu").  `make` is "intel"
    for an Arc, which stays on the CPU reader until INTEL_GPU_READER_BY_DEFAULT."""
    vram = float(vram_gb or 0.0)
    if make == "intel" and not INTEL_GPU_READER_BY_DEFAULT:
        return "cpu"
    return "cpu" if 0 < vram < GPU_MIN_VRAM_GB else "gpu"


def wanted(asked: str | None, vram_gb: float | None, make: str = "") -> str:
    """What is wanted for `asked` (None, "yes", "on", "gpu", "cpu", "no"/"none"/"off"): "gpu", "cpu" or "none".  Nothing asked and "yes" mean
    the default for this card; "gpu" and "cpu" are kept as asked."""
    a = (asked or "").lower()
    if a in ("no", "none", "off"):
        return "none"
    if a in ("gpu", "cpu"):
        return a
    return default_mode(vram_gb, make)


def reader_name(base: str, windows: bool) -> str:
    return base + (".exe" if windows else "")


def find(folder: Path | str, windows: bool) -> dict:
    """The readers in an engine folder: {"cpu": Path | None, "gpu": Path | None}."""
    folder = Path(folder)
    found = {}
    for key, bases in (("cpu", (CPU_READER,)), ("gpu", GPU_READERS)):
        paths = [folder / reader_name(base, windows) for base in bases]
        found[key] = next((p for p in paths if p.is_file()), None)
    return found


@dataclass
class Reader:
    mode: str                     # "gpu" | "cpu" | "none"
    exe: Path | None = None       # the program the run config starts
    cpu_exe: Path | None = None   # the CPU reader to fall back on when a GPU reader does not start (mode "gpu" only)
    note: str = ""                # one plain sentence for the setup output when it is not what was asked for


def choose(folder: Path | str, asked: str, *, windows: bool, gpu_arch: str = "", archs: list | None = None) -> Reader:
    """The reader for `asked` ("yes"/"on"/"gpu" = the best the folder has, "cpu", "no"/"none"/"off").  The GPU reader is tried
    and falls back to the CPU reader by itself (see the module doc).  gpu_arch / archs are accepted for older callers and not used."""
    asked = (asked or "").lower()
    if asked in ("no", "none", "off", ""):
        return Reader("none")
    have = find(folder, windows)
    if asked == "cpu":
        if have["cpu"]:
            return Reader("cpu", have["cpu"])
        return Reader("none", note="this engine has no picture reader: pictures off")
    # yes / on / gpu
    if have["gpu"] is None:
        if have["cpu"]:
            return Reader("cpu", have["cpu"], note="this engine has no GPU picture reader: pictures are read on the CPU "
                                                    "(a few seconds each)" if asked == "gpu" else "")
        return Reader("none", note="this engine has no picture reader: pictures off")
    return Reader("gpu", have["gpu"], have["cpu"])


def vision_section(reader: Reader, base: dict) -> dict:
    """The run config's "vision" section for `reader` on top of `base` (mmproj, model, max_tokens ...): exe, gpu, and for the
    GPU reader the CPU reader to fall back on."""
    out = dict(base)
    out["exe"] = str(reader.exe)
    out["gpu"] = reader.mode == "gpu"
    if reader.mode == "gpu" and reader.cpu_exe:
        out["cpu_exe"] = str(reader.cpu_exe)
    else:
        out.pop("cpu_exe", None)
    return out


def cpu_fallback(vision_cfg: dict) -> dict | None:
    """The same section for the CPU reader, when the GPU reader has one to fall back on; None otherwise."""
    if not vision_cfg.get("gpu") or not vision_cfg.get("cpu_exe"):
        return None
    out = {k: v for k, v in vision_cfg.items() if k not in ("cpu_exe", "elastic", "elastic_mib", "elastic_idle_s")}
    out["exe"], out["gpu"] = vision_cfg["cpu_exe"], False
    return out
