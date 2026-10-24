#!/usr/bin/env python3
"""Make a local, unpublished engine package for the component folder (strata_control/engine_component.py).

    python packaging/package_engine.py --src <folder with strata and its libraries> --backend sycl \
        --version 0.1.45 --out <components>/engine --arch intel-arc \
        [--build-id <id>] [--platform linux-x86_64] [--tree] [--include GLOB ...]

packaging/build_sycl_linux.sh runs it for the Intel Arc (SYCL) engine.

    python packaging/package_engine.py --verify <components>/engine/<package folder>     # full hash check, exit 0 / 1

The result is one folder `<out>/<version>-<platform>-<backend>/` holding the engine binary, the libraries it loads,
BUILD.json when the source has one, and `engine-component.json` (schema 2: every file's SHA-256 and size, the arch
list, the oldest driver, the engine build id, and an empty `signature` for S6). It is the thing Control resolves,
checks and starts; it is not signed, not published, and nothing here uploads it. Signing is stream S6
(`engine_component.register_signature_verifier`); `--sign-with` is deliberately absent until then.

Files taken from --src, by default: the entrypoint, every `*.so` / `*.so.*` beside it, `BUILD.json`, `LICENSE*`, the picture
reader `strata-vision` when the build has one (without it the engine answers text only: setup.py Control mode keeps the person's
picture choice only when it is there) and the DeepSeek-V4.1 engine `strata-dsv41` (setup.py dsv41_exe() looks for it beside the
engine; `--require-dsv41` refuses a package without it). `--include GLOB` (repeatable, relative to --src, `**` allowed) adds more,
for example `SYSTEM-LIBS.txt`. `--tree` takes every regular file under --src. Symlinks are never followed or copied.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import binary_deps  # noqa: E402
from strata_control import engine_component as ec  # noqa: E402
from strata_control.components import BACKEND_MATRIX, host_platform  # noqa: E402


DSV41_EXE = "strata-dsv41"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wanted(path: Path) -> bool:
    name = path.name.lower()
    return (name.endswith(".so") or ".so." in name or name == "build.json"
            or name.startswith("license") or name in ("strata-vision", DSV41_EXE))   # the picture reader and the DeepSeek-V4.1 engine beside the engine


def git_build_id(repo: Path = ROOT) -> str:
    """The source commit of the engine build, `-dirty` when the tree had changes. `unknown` outside a checkout."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (head + ("-dirty" if dirty else "")) if head else "unknown"


def select(src: Path, entrypoint: str, *, tree: bool, include: list[str]) -> list[str]:
    """Relative POSIX paths to package, entrypoint first, the rest sorted."""
    chosen: set[str] = set()
    for path in sorted(src.rglob("*") if (tree or include) else src.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        if rel == ec.MANIFEST:
            continue
        top_level = "/" not in rel
        if tree or (top_level and wanted(path)) or any(fnmatch.fnmatch(rel, pattern) for pattern in include):
            chosen.add(rel)
    chosen.discard(entrypoint)
    return [entrypoint] + sorted(chosen)


def build(src: Path, backend: str, version: str, target: str, out: Path, *, entry: str | None = None,
          archs: tuple[str, ...] = (), min_driver: dict | None = None, build_id: str | None = None,
          min_runtime: str | None = None, notes: str = "", tree: bool = False, include: list[str] | None = None,
          variant: str = "", allow_libs: list[str] | None = None, require_dsv41: bool = False) -> Path:
    if backend not in BACKEND_MATRIX:
        raise SystemExit(f"unknown backend {backend!r}")
    if backend != "cpu" and not archs:
        raise SystemExit(f"a {backend} package needs at least one --arch (intel-arc, bmg-g31, ...)")
    entrypoint = entry or "strata"
    if not (src / entrypoint).is_file():
        raise SystemExit(f"{src / entrypoint} does not exist")
    if require_dsv41 and not (src / DSV41_EXE).is_file():
        raise SystemExit(f"refusing to package: {src} has no {DSV41_EXE} (--require-dsv41); build it with -DSTRATA_BUILD_DSV41=ON "
                         "(--target strata-dsv41) and copy it beside the engine")
    folder = out / (f"{version}-{target}-{backend}" + (f"-{variant}" if variant else ""))
    if folder.exists():
        raise SystemExit(f"{folder} already exists; remove it first (packages are immutable)")
    staging = out / f".{folder.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        files = []
        for rel in select(src, entrypoint, tree=tree, include=include or []):
            destination = staging / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, destination)
            files.append({"path": rel, "sha256": sha256(destination), "bytes": destination.stat().st_size})
        _refuse_missing_libraries(staging, allow_libs or [])
        manifest = {"schema_version": ec.SCHEMA_VERSION, "id": ec.COMPONENT_ID, "name": ec.COMPONENT_ID,
                    "version": version, "platform": target, "backend": backend, "archs": list(archs),
                    "min_driver": dict(min_driver or {}), "build_id": build_id or git_build_id(),
                    "entrypoint": entrypoint, "files": files, "signature": None}
        if variant:
            manifest["variant"] = variant
        if min_runtime:
            manifest["min_runtime"] = min_runtime
        if notes:
            manifest["notes"] = notes
        (staging / ec.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        if os.name != "nt":
            for exe_name in (entrypoint, DSV41_EXE, "strata-vision"):      # the programs setup starts: executable in the tar
                if (staging / exe_name).is_file():
                    (staging / exe_name).chmod(0o755)
        ec.verify(ec.read_manifest(staging), full=True)
        staging.rename(folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return folder


def _refuse_missing_libraries(staging: Path, allow: list[str]) -> None:
    """Every library the package's binaries import must be in the package, or belong to the OS or the GPU driver
    (binary_deps.py). Anything missing refuses the package."""
    left = [(f, n) for f, n in binary_deps.uncovered(staging) if n.lower() not in {a.lower() for a in allow}]
    if left:
        lines = "\n".join(f"  {f} needs {n}" for f, n in sorted(set(left)))
        raise SystemExit("refusing to package: these libraries are neither bundled nor part of the OS or the GPU driver, so a clean PC could not "
                         f"start the engine ('library not found'):\n{lines}\nBundle them (packaging/collect_libs.py) or fix the build.")


def package_sha256(folder: Path) -> str:
    """One digest for the whole package: SHA-256 of the manifest file (it holds every file's hash)."""
    return sha256(folder / ec.MANIFEST)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", type=Path, metavar="FOLDER", help="hash-check an existing package and exit")
    ap.add_argument("--src", type=Path)
    ap.add_argument("--backend", choices=sorted(BACKEND_MATRIX))
    ap.add_argument("--version")
    ap.add_argument("--platform", default=None, help="default: this machine")
    ap.add_argument("--out", type=Path, help="the component folder's engine/ directory")
    ap.add_argument("--entrypoint", default=None)
    ap.add_argument("--arch", dest="arch", action="append", default=[],
                    help="intel-arc (JIT from SPIR-V), bmg-g31 (AOT), ...; repeat for several")
    ap.add_argument("--min-driver", action="append", default=[], metavar="VENDOR=VERSION",
                    help="the oldest driver, VENDOR=VERSION; repeat for several vendors")
    ap.add_argument("--variant", default="", help="tag when one backend has several builds; goes in the folder name")
    ap.add_argument("--build-id", default=None, help="default: the source commit of this checkout")
    ap.add_argument("--min-runtime", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--tree", action="store_true", help="every regular file under --src, recursively")
    ap.add_argument("--allow-lib", action="append", default=[], metavar="NAME",
                    help="an imported library that really is provided by the target system (rare; named so the check passes)")
    ap.add_argument("--require-dsv41", action="store_true",
                    help="refuse the package when --src has no strata-dsv41 (the DeepSeek-V4.1 engine that setup.py looks for beside the engine)")
    ap.add_argument("--include", action="append", default=[], metavar="GLOB", help="more files, relative to --src")
    a = ap.parse_args(argv)
    if a.verify:
        try:
            component = ec.read_manifest(a.verify)
            ec.verify(component, full=True)
        except ec.ComponentProblem as exc:
            print(f"{a.verify}: REFUSED: {exc}")
            return 1
        if a.require_dsv41 and not any(f[0] == DSV41_EXE for f in component.files):
            print(f"{a.verify}: REFUSED: no {DSV41_EXE} in the manifest (--require-dsv41)")
            return 1
        bad = binary_deps.uncovered(a.verify)
        if bad:
            for rel, name in bad:
                print(f"{a.verify}: REFUSED: {rel} needs {name}, which is neither bundled nor an OS / GPU driver library")
            return 1
        print(f"{a.verify}: ok ({len(component.files)} files, manifest sha256 {package_sha256(a.verify)})")
        return 0
    for needed in ("src", "backend", "version", "out"):
        if getattr(a, needed) is None:
            ap.error(f"--{needed} is required")
    drivers = {}
    for item in a.min_driver:
        vendor, _, version = item.partition("=")
        if not vendor or not version:
            ap.error(f"--min-driver {item!r} is not VENDOR=VERSION")
        drivers[vendor] = version
    a.out.mkdir(parents=True, exist_ok=True)
    folder = build(a.src, a.backend, a.version, a.platform or host_platform(), a.out, entry=a.entrypoint,
                   archs=tuple(a.arch), min_driver=drivers, build_id=a.build_id, min_runtime=a.min_runtime,
                   notes=a.notes, tree=a.tree, include=a.include, variant=a.variant,
                   allow_libs=a.allow_lib, require_dsv41=a.require_dsv41)
    entries = [p for p in folder.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in entries)
    print(f"{folder}  ({total / (1 << 20):.1f} MiB, {len(entries)} files, manifest sha256 {package_sha256(folder)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
