"""The picture reader choice for an Intel Arc engine folder (strata_control/vision_reader.py) and the server's fallback from the GPU
reader to the CPU reader (serve/server.py, Vision._start): sycl present, sycl missing, sycl start failure, then cpu.

    python -m unittest tools.test_vision_reader
"""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import vision_reader as VR
import serve.server as server


def folder(*names: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="vr-"))
    for n in names:
        (d / n).write_bytes(b"x")
    return d


class Choose(unittest.TestCase):
    def test_the_default_reader_by_card_memory_is_one_rule_for_every_make(self):
        self.assertEqual(VR.default_mode(12), "cpu")
        self.assertEqual(VR.default_mode(16), "cpu")
        self.assertEqual(VR.default_mode(23.5), "gpu")
        self.assertEqual(VR.default_mode(24), "gpu")
        self.assertEqual(VR.default_mode(32), "gpu")
        self.assertEqual(VR.default_mode(None), "gpu")                  # memory unknown: the GPU reader falls back by itself
        for asked, vram, want in ((None, 16, "cpu"), ("yes", 12, "cpu"), ("on", 32, "gpu"), ("gpu", 12, "gpu"), ("cpu", 32, "cpu"),
                                  ("no", 32, "none"), ("off", 12, "none")):
            self.assertEqual(VR.wanted(asked, vram), want, (asked, vram))

    def test_an_intel_arc_keeps_the_cpu_reader_by_default_and_gpu_is_asked_for(self):
        """The one exception to the 24 GB rule: the SYCL reader has run on a 8 GB A750 only (INTEL_GPU_READER_BY_DEFAULT)."""
        self.assertFalse(VR.INTEL_GPU_READER_BY_DEFAULT)
        for vram in (8, 16, 24, 32):
            self.assertEqual(VR.default_mode(vram, "intel"), "cpu")
            self.assertEqual(VR.wanted("yes", vram, "intel"), "cpu")
            self.assertEqual(VR.wanted("gpu", vram, "intel"), "gpu")
        self.assertEqual(VR.wanted("yes", 32), "gpu")                        # make unknown: the 24 GB rule
        with mock.patch.object(VR, "INTEL_GPU_READER_BY_DEFAULT", True):
            self.assertEqual(VR.default_mode(32, "intel"), "gpu")
            self.assertEqual(VR.default_mode(8, "intel"), "cpu")              # with the exception off, the 24 GB rule is the same as for every make

    def test_sycl_present_gives_the_gpu_reader_with_the_cpu_one_to_fall_back_on(self):
        d = folder("strata-vision", "strata-vision-sycl")
        for ask in ("yes", "on", "gpu"):
            r = VR.choose(d, ask, windows=False)
            self.assertEqual((r.mode, r.exe.name, r.cpu_exe.name), ("gpu", "strata-vision-sycl", "strata-vision"), ask)
            self.assertEqual(r.note, "")

    def test_sycl_missing_gives_the_cpu_reader(self):
        d = folder("strata-vision")
        r = VR.choose(d, "yes", windows=False)
        self.assertEqual((r.mode, r.exe.name, r.cpu_exe), ("cpu", "strata-vision", None))
        self.assertEqual(r.note, "")                               # "yes" never asked for the GPU in particular
        self.assertIn("no GPU picture reader", VR.choose(d, "gpu", windows=False).note)   # --vision gpu: said once

    def test_cpu_asked_and_off(self):
        d = folder("strata-vision", "strata-vision-sycl")
        self.assertEqual(VR.choose(d, "cpu", windows=False).exe.name, "strata-vision")
        for ask in ("no", "none", "off", None):
            self.assertEqual(VR.choose(d, ask, windows=False).mode, "none")

    def test_no_reader_at_all(self):
        d = folder("strata")
        for ask in ("yes", "cpu", "gpu"):
            r = VR.choose(d, ask, windows=False)
            self.assertEqual(r.mode, "none", ask)
            self.assertIn("no picture reader", r.note)

    def test_windows_names(self):
        d = folder("strata-vision.exe", "strata-vision-sycl.exe")
        r = VR.choose(d, "yes", windows=True)
        self.assertEqual((r.mode, r.exe.name, r.cpu_exe.name), ("gpu", "strata-vision-sycl.exe", "strata-vision.exe"))
        self.assertEqual(VR.choose(folder("strata-vision-sycl"), "yes", windows=True).mode, "none")   # a Linux name on Windows

    def test_run_config_section(self):
        d = folder("strata-vision", "strata-vision-sycl")
        base = {"mmproj": "m.gguf", "model": "t.gguf", "max_tokens": 1024}
        sec = VR.vision_section(VR.choose(d, "yes", windows=False), base)
        self.assertTrue(sec["gpu"])
        self.assertEqual(Path(sec["exe"]).name, "strata-vision-sycl")
        self.assertEqual(Path(sec["cpu_exe"]).name, "strata-vision")
        cpu = VR.cpu_fallback(sec)
        self.assertFalse(cpu["gpu"])
        self.assertEqual(Path(cpu["exe"]).name, "strata-vision")
        self.assertNotIn("cpu_exe", cpu)
        self.assertEqual(cpu["mmproj"], "m.gguf")
        cpu_only = VR.vision_section(VR.choose(folder("strata-vision"), "yes", windows=False), base)
        self.assertFalse(cpu_only["gpu"])
        self.assertNotIn("cpu_exe", cpu_only)
        self.assertIsNone(VR.cpu_fallback(cpu_only))
        self.assertIsNone(VR.cpu_fallback({**sec, "cpu_exe": ""}))
        self.assertIsNone(VR.cpu_fallback({"exe": "strata-vision", "gpu": True}))   # a GPU reader without a CPU one: no fallback


class FakeProc:
    def __init__(self, first_line: str):
        self.stdin, self.stdout = io.StringIO(), io.StringIO(first_line)
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None


def make_vision(cfg: dict) -> "server.Vision":
    with mock.patch.object(server.Vision, "work_dir", staticmethod(lambda: Path(tempfile.gettempdir()))):
        return server.Vision(cfg, lazy=True)


class Fallback(unittest.TestCase):
    CFG = {"exe": "/e/strata-vision-sycl", "cpu_exe": "/e/strata-vision", "gpu": True, "mmproj": "/m/mm.gguf",
           "model": "/m/t.gguf", "max_tokens": 1024, "threads": 6}

    def run_start(self, answers: dict, cfg: dict | None = None):
        """answers: exe name -> its first stdout line, or an exception to raise from popen."""
        started = []

        def popen(what, args, **kw):
            name = Path(args[0]).name
            started.append(args)
            a = answers[name]
            if isinstance(a, Exception):
                raise a
            return FakeProc(a)

        v = make_vision(cfg or self.CFG)
        out = io.StringIO()
        with mock.patch.object(server, "popen", popen), mock.patch("builtins.print", lambda *a, **k: out.write(" ".join(map(str, a)) + "\n")):
            err = None
            try:
                v._start()
            except RuntimeError as e:
                err = e
        return v, started, out.getvalue(), err

    def test_sycl_present_and_starts(self):
        v, started, log, err = self.run_start({"strata-vision-sycl": "READY 2048\n"})
        self.assertIsNone(err)
        self.assertEqual([Path(a[0]).name for a in started], ["strata-vision-sycl"])
        self.assertIn("--gpu", started[0])
        self.assertEqual(log, "")
        self.assertFalse(v.stopped)

    def test_sycl_start_failure_falls_back_to_cpu_with_one_line(self):
        v, started, log, err = self.run_start({"strata-vision-sycl": "ERR no SYCL device the picture reader can use\n",
                                               "strata-vision": "READY 2048\n"})
        self.assertIsNone(err)
        self.assertEqual([Path(a[0]).name for a in started], ["strata-vision-sycl", "strata-vision"])
        self.assertNotIn("--gpu", started[1])
        self.assertIn("--threads", started[1])
        self.assertEqual(log.count("\n"), 1)
        self.assertIn("no SYCL device", log)
        self.assertIn("CPU", log)
        self.assertFalse(v.stopped)
        self.assertIsNone(v.fallback)

    def test_sycl_process_that_dies_silently_falls_back(self):                  # a GPU runtime that aborts: stdout closes empty
        v, started, log, err = self.run_start({"strata-vision-sycl": "", "strata-vision": "READY 2048\n"})
        self.assertIsNone(err)
        self.assertEqual(len(started), 2)
        self.assertEqual(log.count("\n"), 1)

    def test_sycl_missing_file_falls_back(self):                               # popen raises (exe gone, not executable)
        v, started, log, err = self.run_start({"strata-vision-sycl": FileNotFoundError("strata-vision-sycl"),
                                               "strata-vision": "READY 2048\n"})
        self.assertIsNone(err)
        self.assertEqual(len(started), 2)

    def test_restart_after_a_fallback_stays_on_the_cpu_reader(self):
        v, started, log, err = self.run_start({"strata-vision-sycl": "ERR nope\n", "strata-vision": "READY 2048\n"})
        started.clear()
        with mock.patch.object(server, "popen", lambda what, args, **kw: started.append(args) or FakeProc("READY 2048\n")):
            v.restart()
        self.assertEqual([Path(a[0]).name for a in started], ["strata-vision"])

    def test_cpu_failure_after_the_fallback_is_an_error(self):
        v, started, log, err = self.run_start({"strata-vision-sycl": "ERR nope\n", "strata-vision": "ERR cannot load the vision encoder\n"})
        self.assertIsNotNone(err)
        self.assertIn("cannot load", str(err))

    def test_a_config_without_cpu_exe_has_no_fallback(self):                  # no CPU reader beside it: the error is shown as before
        cfg = {k: v for k, v in self.CFG.items() if k != "cpu_exe"}
        v, started, log, err = self.run_start({"strata-vision-sycl": "ERR nope\n"}, cfg)
        self.assertIsNotNone(err)
        self.assertEqual(len(started), 1)
        self.assertEqual(log, "")

    def test_cpu_config_is_unchanged(self):
        cfg = {"exe": "/e/strata-vision", "gpu": False, "mmproj": "/m/mm.gguf", "model": "/m/t.gguf"}
        v, started, log, err = self.run_start({"strata-vision": "READY 2048\n"}, cfg)
        self.assertIsNone(err)
        self.assertNotIn("--gpu", started[0])


if __name__ == "__main__":
    unittest.main()
