#!/usr/bin/env python3
"""Collect the shared libraries an engine binary needs into one folder, ready for packaging/package_engine.py (Linux).

    python packaging/collect_libs.py --binary build-sycl/strata --search /opt/intel/oneapi/compiler/latest/lib --out stage/ \
        [--also 'libur_adapter_level_zero*.so*']

What it does: runs `ldd` on the binary with LD_LIBRARY_PATH set to --search, copies every dependency that is NOT
supplied by the distribution (anything resolved outside /lib, /lib64, /usr/lib, /usr/lib64 and their x86_64-linux-gnu
folders) into --out under its soname (symlinks are resolved, the real file is copied once per name), repeats for each
copied library (their own dependencies), and adds libraries that are loaded with dlopen and so never show up in ldd
(--also, a glob searched in --search). packaging/build_sycl_linux.sh runs it for the oneAPI runtime of the SYCL engine.

Distribution libraries (libc, libm, libstdc++, libdrm, libnuma, ...) are not copied: they are listed in
`SYSTEM-LIBS.txt` beside the output so a missing one on a target PC can be named. The GPU driver's own libraries
(libze_intel_gpu, libigdrcl, libOpenCL ICDs) are never bundled: they belong to the driver.
Nothing is downloaded; this only reads files on the build machine.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys


def _copy(src, dst) -> None:
    """shutil.copy2, except onto itself: a library that an earlier binary's pass already put in --out is found there again by ldd (RUNPATH $ORIGIN)
    and is the same file (SameFileError broke the second binary's pass for the AMD packages)."""
    try:
        if os.path.exists(dst) and os.path.samefile(src, dst):
            return
    except OSError:
        pass
    shutil.copy2(src, dst)
from pathlib import Path

SYSTEM_DIRS = ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")
DRIVER_PATTERNS = ("libze_intel_gpu*", "libigdrcl*", "libOpenCL.so*", "libGLX*", "libEGL*", "libvulkan*", "libdrm.so*")
LDD_LINE = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(0x")


def is_system(path: str) -> bool:
    return path.startswith(SYSTEM_DIRS)


def is_driver(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in DRIVER_PATTERNS)


ELF_MAGIC = bytes([0x7F]) + b"ELF"


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == ELF_MAGIC
    except OSError:
        return False


def soname(path: Path) -> str | None:
    """The library's own name (DT_SONAME), which is the name its users ask the loader for."""
    done = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True)
    match = re.search(r"\(SONAME\)\s+Library soname: \[([^\]]+)\]", done.stdout)
    return match.group(1) if match else None


def dependencies(binary: Path, search: list[str]) -> tuple[dict[str, str], list[str]]:
    """{soname: resolved path} and the sonames ldd could not find."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(search + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    done = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, env=env, timeout=120)
    found, missing = {}, []
    for line in done.stdout.splitlines():
        if "not found" in line:
            missing.append(line.split("=>")[0].strip())
            continue
        match = LDD_LINE.match(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--search", action="append", default=[], help="a library folder to resolve against (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--also", action="append", default=[], help="glob of dlopen'd libraries to add, searched in --search")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    system: set[str] = set()
    queue = [a.binary]
    seen = set()
    while queue:
        item = queue.pop()
        if item in seen:
            continue
        seen.add(item)
        found, missing = dependencies(item, a.search)
        for name in missing:
            if not is_driver(name):
                print(f"WARNING: {item.name} needs {name}, which ldd cannot find", file=sys.stderr)
        for name, path in found.items():
            if is_driver(name):
                continue
            if is_system(path):
                system.add(name)
                continue
            if name in copied:
                continue
            real = Path(path).resolve()
            _copy(real, a.out / name)
            copied[name] = str(real)
            queue.append(a.out / name)
    for pattern in a.also:
        hits = [Path(d) / n for d in a.search if os.path.isdir(d) for n in sorted(os.listdir(d)) if fnmatch.fnmatch(n, pattern)]
        if not hits:
            print(f"WARNING: --also {pattern!r} matched nothing in {a.search}", file=sys.stderr)
        for hit in hits:
            if not hit.is_file() or not is_elf(hit):
                continue
            real = hit.resolve()
            if str(real) in copied.values():          # libX.so -> libX.so.1 -> libX.so.1.2.3: one file, one name (the soname)
                continue
            name = soname(real) or hit.name
            if name in copied:
                continue
            _copy(real, a.out / name)
            copied[name] = str(real)
            found, missing = dependencies(a.out / name, a.search)
            for dep in missing:
                if not is_driver(dep):
                    print(f"WARNING: {name} needs {dep}, which ldd cannot find", file=sys.stderr)
            for dep, path in found.items():
                if is_driver(dep) or dep in copied:
                    continue
                if is_system(path):
                    system.add(dep)
                else:
                    _copy(Path(path).resolve(), a.out / dep)
                    copied[dep] = str(Path(path).resolve())
    (a.out / "SYSTEM-LIBS.txt").write_text("\n".join(sorted(system)) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in a.out.rglob("*") if p.is_file())
    print(f"copied {len(copied)} libraries ({total / (1 << 20):.1f} MiB) to {a.out}; {len(system)} system libraries left to the OS")
    for name in sorted(copied):
        print(f"  {name} <- {copied[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
