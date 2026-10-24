"""packaging/feed_tool.py packs a package's programs executable wherever it runs. (The engine choice for the Intel Arc package is
tests/terminal/test_sycl_package.py's.)"""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import feed_tool  # noqa: E402

class PackedOnWindows(unittest.TestCase):
    def test_programs_are_recognised_by_their_header_so_a_linux_package_packed_on_windows_keeps_them_executable(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "strata-vision").write_bytes(b"\x7fELF\x02\x01" + bytes(20))
            (root / "libx.dylib").write_bytes(b"\xcf\xfa\xed\xfe" + bytes(20))
            (root / "run.sh").write_bytes(b"#!/bin/sh\n")
            (root / "BUILD.json").write_text("{}")
            (root / "kernel.hsaco").write_bytes(b"\x00\x01\x02\x03")
            names = ("strata-vision", "libx.dylib", "run.sh", "BUILD.json", "kernel.hsaco")
            self.assertEqual([feed_tool._is_native_executable(root / n) for n in names], [True, True, True, False, False])
            if os.name == "nt":                     # the owner's release PC: the archive is what matters
                archive = root / "a.tar.gz"
                feed_tool._write_archive([(n, root / n) for n in ("strata-vision", "BUILD.json")], archive, "tar.gz", None)
                with tarfile.open(archive) as tar:
                    modes = {m.name: m.mode for m in tar.getmembers()}
                self.assertEqual(modes, {"strata-vision": 0o755, "BUILD.json": 0o644})


if __name__ == "__main__":
    unittest.main()
