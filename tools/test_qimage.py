"""Tests for image generation's setup side (tools/image/qimage.py, setup.py --images): the pins, the quant and backend
choices, the CMake definitions, the source unpacking and the run config's "images" section.  No GPU, no network.

    python -m unittest tools.test_qimage
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "image"))
import qimage as QI  # noqa: E402
import setup  # noqa: E402


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class Pins(unittest.TestCase):
    def test_shapes(self):
        for c in (QI.SD_CPP_COMMIT, QI.SD_GGML_COMMIT, *QI.REVISIONS.values()):
            self.assertRegex(c, r"^[0-9a-f]{40}$")
        for name, (repo, path, size, sha) in QI.FILES.items():
            self.assertIn(repo, QI.REVISIONS, name)
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertGreater(size, 100 << 20)
        self.assertEqual(QI.file_url("Q8_0"), "https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF/resolve/"
                         "5d1f9f759ea877ae556cda88dbec958b62ad3809/qwen-image-2.1-turbo-Q8_0.gguf")
        self.assertEqual(QI.local_path(Path("m"), "vae"),
                         Path("m") / "qwen-image-2.1-turbo" / "qwen_image_2.1_vae_bf16.safetensors")

    def test_sigmas_are_the_models(self):
        """serve/images.py's schedule is model_index.json's sample_sigmas plus the terminal 0."""
        from serve import images
        sample = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        self.assertEqual([float(x) for x in images.TURBO_SIGMAS.split(",")], sample + [0.0])


class Choices(unittest.TestCase):
    def test_quant_by_memory(self):
        self.assertEqual([QI.pick_quant(g) for g in (4, 8, 8.5, 12, 12.1, 16, 32)],
                         ["Q4_K_M", "Q4_K_M", "Q6_K", "Q6_K", "Q8_0", "Q8_0", "Q8_0"])

    def test_backend(self):
        self.assertEqual(QI.pick_backend(None, False), "cpu")
        self.assertEqual(QI.pick_backend({"vendor": "intel", "name": "Intel Arc Pro B70", "vram_gb": 32.0}, False), "vulkan")
        self.assertEqual(QI.BACKENDS, ("vulkan", "cpu"))
        self.assertTrue(QI.is_apu({"uma": True}))
        self.assertFalse(QI.is_apu({"vendor": "intel", "vram_gb": 32.0}))
        self.assertFalse(QI.is_apu(None))

    def test_default_args(self):
        self.assertEqual(QI.default_args("cpu"), ["--diffusion-fa"])
        self.assertEqual(QI.default_args("vulkan"), ["--diffusion-fa", "--vae-tiling", "--vae-tile-overlap", "0.25"])

    def test_setup_recommends_never_forces(self):
        ns = argparse.Namespace(image_backend=None, image_quant=None)
        b70 = {"vendor": "intel", "name": "Intel Arc Pro B70", "vram_gb": 32.0}
        self.assertEqual(quiet(setup.image_backend_choice, ns, b70, False)[:2], ("vulkan", "Q8_0"))
        self.assertEqual(quiet(setup.image_backend_choice, ns, {**b70, "vram_gb": 12.0}, False)[:2], ("vulkan", "Q6_K"))
        self.assertEqual(quiet(setup.image_backend_choice, ns, None, False)[:2], ("cpu", "Q4_K_M"))
        ns.image_quant = "Q8_0"                      # more than recommended: kept, with a warning
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = setup.image_backend_choice(ns, {**b70, "vram_gb": 8.0}, False)
        self.assertEqual(r[:2], ("vulkan", "Q8_0"))
        self.assertIn("recommends Q4_K_M", out.getvalue())
        ns.image_quant, ns.image_backend = None, "cpu"
        self.assertEqual(quiet(setup.image_backend_choice, ns, {"vram_gb": 24.0}, False)[:2], ("cpu", "Q4_K_M"))


class Build(unittest.TestCase):
    def test_cmake_defs(self):
        d = QI.cmake_defs("vulkan")
        self.assertIn("-DSD_VULKAN=ON", d)
        self.assertIn("-DSD_WEBP=OFF", d)
        self.assertNotIn("-DSD_VULKAN=ON", QI.cmake_defs("cpu"))
        for other in ("cuda", "hip", "metal"):
            with self.assertRaises(ValueError):
                QI.cmake_defs(other)

    def test_unpack_puts_ggml_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            z1, z2 = d / "sd.zip", d / "ggml.zip"
            with zipfile.ZipFile(z1, "w") as z:
                z.writestr(f"stable-diffusion.cpp-{QI.SD_CPP_COMMIT}/CMakeLists.txt", "project(sd)")
                z.writestr(f"stable-diffusion.cpp-{QI.SD_CPP_COMMIT}/ggml/", "")
            with zipfile.ZipFile(z2, "w") as z:
                z.writestr(f"ggml-{QI.SD_GGML_COMMIT}/CMakeLists.txt", "project(ggml)")
            src = QI.unpack_source(d, z1, z2)
            self.assertEqual((src / "ggml" / "CMakeLists.txt").read_text(), "project(ggml)")
            self.assertTrue(QI.source_ok(src))
            (src / ".strata-pin").write_text("0" * 40 + "\n" + QI.SD_GGML_COMMIT + "\n")
            self.assertFalse(QI.source_ok(src))          # another pin: unpacked again

    def test_build_is_skipped_when_the_stamp_matches(self):
        b70 = {"vendor": "intel", "name": "Intel Arc Pro B70", "vram_gb": 32.0}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "IMAGE_ENGINE", Path(d)), \
                mock.patch.object(setup, "ROOT", Path(d)):
            (Path(d) / QI.SD_EXE).write_bytes(b"x")
            (Path(d) / "BUILD.json").write_text(json.dumps({**QI.build_stamp("vulkan", None), "lib_dirs": [d]}))
            with mock.patch.object(setup, "cmake_build") as cb:
                exe, libs = quiet(setup.build_image_engine, "vulkan", b70)
            cb.assert_not_called()
            self.assertEqual((exe, libs), (Path(d) / QI.SD_EXE, [d]))
            # another pin: compiled again (nothing downloaded or built here: the source, the SDK and the build are stand-ins, and
            # the build leaves no sd-cli, so setup stops)
            (Path(d) / "BUILD.json").write_text(json.dumps({**QI.build_stamp("vulkan", None), "sd_cpp": "0" * 40}))
            with mock.patch.object(setup, "cmake_build") as cb, mock.patch.object(setup, "get_sd_cpp", return_value=Path(d) / "src"), \
                    mock.patch.object(setup, "vulkan_sdk", return_value=None) as sdk:
                with self.assertRaises(SystemExit):
                    quiet(setup.build_image_engine, "vulkan", b70)
            sdk.assert_called_once()
            self.assertIn("-DSD_VULKAN=ON", cb.call_args[0][3])


class Config(unittest.TestCase):
    def test_section(self):
        c = QI.image_config("vulkan", "Q8_0", Path("/e/sd-cli"), Path("/m"), uma=True, lib_dirs=["/l"])
        self.assertEqual(c["diffusion_model"], str(Path("/m/qwen-image-2.1-turbo/qwen-image-2.1-turbo-Q8_0.gguf")))
        self.assertEqual(c["args"], ["--diffusion-fa", "--vae-tiling", "--vae-tile-overlap", "0.25"])
        self.assertTrue(c["uma"])
        self.assertEqual(c["gpu_policy"], "auto")
        from serve.images import ImageGen
        with tempfile.TemporaryDirectory() as d:
            ImageGen({**c, "out_dir": d})              # serve accepts what setup writes

    def test_images_no_drops_the_section(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"exe": "x", "images": {"exe": "sd-cli"}, "mine": 1}))
            quiet(setup.write_setup_config, p, {"exe": "y", "images": None})
            cfg = json.loads(p.read_text())
            self.assertNotIn("images", cfg)
            self.assertEqual(cfg["mine"], 1)
            p.write_text(json.dumps({"exe": "x", "images": {"exe": "sd-cli"}}))
            quiet(setup.write_setup_config, p, {"exe": "y"})      # not asked: carried over
            self.assertEqual(json.loads(p.read_text())["images"], {"exe": "sd-cli"})

    def test_flags(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        for flag in ("--images", "--image-quant", "--image-backend", "--image-gpu-policy"):
            self.assertRegex(src, re.escape(f'ap.add_argument("{flag}"'))


if __name__ == "__main__":
    unittest.main()
