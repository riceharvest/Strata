"""An engine package must carry every library its binaries import, except the OS's and the GPU driver's (packaging/binary_deps.py).

Tester report 2026-10-17: a clean Windows 11 PC failed with 0xC0000135 because strata.exe imported a toolkit DLL the package did
not hold. These tests build tiny real PE files (the format whose rule is the subtlest: case, lib_dirs), so they run anywhere and need
no GPU or toolkit; the ELF reader is checked on a system binary."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import binary_deps  # noqa: E402


def make_pe(imports, delay=()):
    """A minimal PE32+ file with a .idata section that imports the named DLLs (and delay-loads `delay`)."""
    va, raw = 0x1000, 0x400
    body = bytearray()
    descriptors = (len(imports) + 1) * 20
    delays = (len(delay) + 1) * 32
    names_at = descriptors + delays
    name_rvas = []
    blob = bytearray()
    for n in list(imports) + list(delay):
        name_rvas.append(va + names_at + len(blob))
        blob += n.encode("ascii") + b"\0"
    for i, _ in enumerate(imports):
        body += struct.pack("<IIIII", 0, 0, 0, name_rvas[i], 0)
    body += b"\0" * 20
    for j, _ in enumerate(delay):
        body += struct.pack("<II", 1, name_rvas[len(imports) + j]) + b"\0" * 24
    body += b"\0" * 32
    body += blob
    size = len(body)
    dos = bytearray(0x40)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)
    coff = b"PE\0\0" + struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 240, 0x22)
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<II", opt, 112 + 8, va if imports else 0, descriptors if imports else 0)
    struct.pack_into("<II", opt, 112 + 13 * 8, va + descriptors if delay else 0, delays if delay else 0)
    section = b".idata\0\0" + struct.pack("<IIIIIIHHI", size, va, size, raw, 0, 0, 0, 0, 0x40000040)
    head = bytes(dos) + coff + bytes(opt) + section
    return head + b"\0" * (raw - len(head)) + bytes(body)


class Imports(unittest.TestCase):
    def test_pe_reader_lists_load_time_and_delay_load_imports(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.exe"
            p.write_bytes(make_pe(["sycl8.dll", "KERNEL32.dll"], delay=["ur_loader.dll"]))
            self.assertEqual(binary_deps.pe_imports(p), ["sycl8.dll", "KERNEL32.dll", "ur_loader.dll"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "needs a real ELF")
    def test_elf_reader_on_a_system_binary(self):
        names = binary_deps.elf_needed(Path("/bin/ls") if Path("/bin/ls").exists() else Path(sys.executable))
        self.assertTrue(any(n.startswith("libc.so") for n in names), names)


class Rule(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.pkg = Path(self.dir.name) / "pkg"
        self.pkg.mkdir()

    def test_a_missing_runtime_dll_is_reported_and_the_driver_and_os_are_not(self):
        (self.pkg / "strata.exe").write_bytes(make_pe(["sycl8.dll", "KERNEL32.dll", "ADVAPI32.dll", "ze_loader.dll", "api-ms-win-crt-heap-l1-1-0.dll"]))
        self.assertEqual(binary_deps.uncovered(self.pkg), [("strata.exe", "sycl8.dll")])

    def test_the_visual_cpp_runtime_is_not_an_os_library(self):
        (self.pkg / "strata.exe").write_bytes(make_pe(["VCRUNTIME140.dll", "MSVCP140.dll"]))
        self.assertEqual({n for _, n in binary_deps.uncovered(self.pkg)}, {"VCRUNTIME140.dll", "MSVCP140.dll"})

    def test_bundled_libraries_count_and_their_own_imports_are_checked(self):
        (self.pkg / "strata.exe").write_bytes(make_pe(["sycl8.dll"]))
        (self.pkg / "sycl8.dll").write_bytes(make_pe(["ur_loader.dll", "KERNEL32.dll"]))
        self.assertEqual(binary_deps.uncovered(self.pkg), [("sycl8.dll", "ur_loader.dll")])
        (self.pkg / "UR_LOADER.DLL").write_bytes(make_pe(["KERNEL32.dll"]))
        self.assertEqual(binary_deps.uncovered(self.pkg), [])


class SubfolderDlls(unittest.TestCase):
    """A DLL only in a sub-folder is found only when BUILD.json lists the folder (Control puts those on PATH)."""

    def make(self, d, lib_dirs):
        import json
        d = Path(d)
        (d / "strata.exe").write_bytes(make_pe(["mkl_sycl.dll", "KERNEL32.dll"]))
        (d / "oneapi" / "bin").mkdir(parents=True)
        (d / "oneapi" / "bin" / "mkl_sycl.dll").write_bytes(make_pe(["KERNEL32.dll"]))
        (d / "BUILD.json").write_text(json.dumps({"lib_dirs": lib_dirs}))

    def test_subfolder_dll_not_on_the_path_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d, [])
            self.assertEqual(binary_deps.uncovered(Path(d)), [("strata.exe", "mkl_sycl.dll")])

    def test_subfolder_dll_listed_in_lib_dirs_passes(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d, ["oneapi/bin"])
            self.assertEqual(binary_deps.uncovered(Path(d)), [])


if __name__ == "__main__":
    unittest.main()
