"""Graphics card roles in setup (strata_control/gpu_roles.py): one card asks nothing, --gpu-roles names the roles, and
--gpus / --gpu name the cards.

    python tools/test_setup_gpu_fit.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from test_setup_golden import card, install  # noqa: E402


def run(ram, found, model, answers=None, extra=()):
    argv = ["--family", "qwen", "--no-start", "--model", model, "--context", "32768", *extra]
    return install(ram, found, argv, answers=answers or {})


class SetupFitRule(unittest.TestCase):
    def test_one_card_has_no_question(self):
        code, out, cfg, asked = run(62.0, [card(0, "Intel Arc card", 8.0)], "IQ3_XXS")
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0) if "gpu" in cfg else None
        self.assertFalse(any("Which GPUs" in q for q in asked), asked)

    def test_gpu_roles_flag(self):
        found = [card(0, "Intel Arc card", 12.0), card(1, "Intel Arc card", 16.0)]
        code, out, cfg, _ = run(62.0, found, "IQ3_XXS", extra=("--gpu-roles", "0=main,1=helper"))
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [1, 0])                              # the slower card is Main on purpose: it goes last
        self.assertEqual(cfg["gpu_roles"], {"0": "main", "1": "helper"})
        code, out, cfg, _ = run(62.0, found, "IQ3_XXS", extra=("--gpu-roles", "1=main,0=off"))
        self.assertEqual((cfg["gpu"], cfg["gpu_roles"]), (1, {"0": "off", "1": "main"}))
        self.assertNotIn("layer_split", cfg)
        code, out, cfg, _ = run(62.0, found, "IQ3_XXS", extra=("--gpu-roles", "0=off,1=off"))
        self.assertNotEqual(code, 0)

    def test_naming_the_cards_still_wins(self):
        found = [card(0, "Intel Arc card", 16.0), card(1, "Intel Arc card", 8.0)]
        code, out, cfg, _ = run(62.0, found, "IQ3_XXS", extra=("--gpus", "0,1"))
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(cfg["gpu"]), [0, 1])
        code, out, cfg, _ = run(62.0, found, "IQ3_XXS", extra=("--gpu", "1"))
        self.assertEqual(cfg["gpu"], 1)



if __name__ == "__main__":
    unittest.main()
