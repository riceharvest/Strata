"""The SYCL engine built from this checkout: the one engine of this fork that is not a package from the update feed.

``sycl/tools/build.sh`` (in the strata-sycl-dev image, docs/INTEL.md) leaves the binary in ``build-sycl-aot/strata`` (or ``build-sycl/strata``);
``sycl/serve/strata-sycl.sh`` runs it in that image and is the run config's ``exe``, and ``sycl/serve/server_intel.py`` (serve/server.py plus the
Intel pieces) is the server. sycl/setup_intel.py writes a config and a ``run-<model>.sh`` from these; `strata launch` finds the engine here and the
runtime host starts the same server for a config whose backend is "sycl". Nothing here runs anything: it only looks for files.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARIES = ("build-sycl-aot/strata", "build-sycl/strata")     # the AOT build first (sycl/tools/build.sh's default), then the JIT one
WRAPPER = "sycl/serve/strata-sycl.sh"
SERVER = "sycl/serve/server_intel.py"
BACKEND = "sycl"
MOUNT = "/work"                                                  # where strata-sycl.sh mounts the run config's "sycl_root" in the container


def _root(root: str | os.PathLike | None) -> Path:
    return Path(root) if root is not None else ROOT


def binary(root: str | os.PathLike | None = None) -> Path | None:
    """The built engine binary under `root` (default: this checkout), or None when it is not built."""
    base = _root(root)
    return next((base / rel for rel in BINARIES if (base / rel).is_file()), None)


def wrapper(root: str | os.PathLike | None = None) -> Path:
    return _root(root) / WRAPPER


def server(root: str | os.PathLike | None = None) -> Path:
    return _root(root) / SERVER


def version(root: str | os.PathLike | None = None) -> str:
    """The engine version the source tree builds (CMakeLists.txt's project version, as setup.source_version reads it), "0" when unreadable."""
    try:
        m = re.search(r"project\(strata VERSION ([\d.]+)", (_root(root) / "CMakeLists.txt").read_text(encoding="utf-8"))
    except OSError:
        return "0"
    return m.group(1) if m else "0"


def find(root: str | os.PathLike | None = None) -> dict | None:
    """The source-built SYCL engine of the checkout at `root`, or None: Linux only, and the binary, its launcher and the Intel server must all be
    there. The dict has the keys `strata components engine` reports for a package (version, backend, exe) plus where it came from."""
    if os.name == "nt":
        return None
    exe = binary(root)
    if exe is None or not wrapper(root).is_file() or not server(root).is_file():
        return None
    return {"version": version(root), "backend": BACKEND, "exe": str(exe), "launcher": str(wrapper(root)),
            "server": str(server(root)), "source": "build", "root": str(_root(root))}


def is_sycl(cfg: object) -> bool:
    """A run config for the SYCL engine (sycl/setup_intel.py writes "backend": "sycl")."""
    return isinstance(cfg, dict) and cfg.get("backend") == BACKEND


def server_command(cfg: object, root: str | os.PathLike | None = None) -> list[str]:
    """What follows the Python interpreter to start the server for `cfg`: sycl/serve/server_intel.py for a SYCL config (what run-<model>.sh
    runs), else the shared server module."""
    if is_sycl(cfg) and server(root).is_file():
        return [str(server(root))]
    return ["-m", "serve.server"]


def host_path(value: object, cfg: object) -> str:
    """A path from a SYCL run config's arguments as this computer sees it. sycl/setup_intel.py writes the model files as the engine's container
    sees them (/work/...), and /work is the config's "sycl_root" (the folder above the Strata checkout by default). Anything else is unchanged."""
    text = str(value)
    if not is_sycl(cfg) or not text.startswith(MOUNT + "/"):
        return text
    root = cfg.get("sycl_root") or str(ROOT.parent)              # type: ignore[union-attr]
    return str(Path(str(root)) / text[len(MOUNT) + 1:])


def host_args(cfg: object) -> list[str]:
    """The run config's arguments with every container path as this computer sees it (host_path)."""
    args = cfg.get("args") if isinstance(cfg, dict) else None
    return [host_path(a, cfg) for a in args] if isinstance(args, list) else []
