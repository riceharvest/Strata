"""serve/server.py: the run settings' "gpu_roles" (Main / Helper / Off) become the engine's card list, Main LAST, Off left out."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import card_roles_lines, gpu_list  # noqa: E402


class Roles(unittest.TestCase):
    def test_roles_alone_give_the_order_main_last_and_off_left_out(self):
        cfg = {"gpu_roles": {"0": "helper", "1": "main", "2": "off"}, "args": []}
        self.assertEqual(gpu_list(cfg), [0, 1])
        self.assertEqual(gpu_list({"gpu_roles": {"0": "main", "1": "helper"}, "args": []}), [1, 0])

    def test_one_card_on(self):
        cfg = {"gpu_roles": {"0": "off", "1": "main"}, "gpu": 1, "gpu_order": "as_given", "args": []}
        self.assertEqual(gpu_list(cfg), [1])

    def test_a_hand_edited_gpu_wins(self):
        cfg = {"gpu_roles": {"0": "helper", "1": "main"}, "gpu": [1, 0], "gpu_order": "as_given", "args": []}
        self.assertEqual(gpu_list(cfg), [1, 0])
        self.assertTrue(any("edited by hand" in l for l in card_roles_lines(cfg)))

    def test_bad_roles_are_ignored(self):
        for bad in ({"0": "boss"}, {"0": "main", "1": "main"}, "x", {}):
            self.assertEqual(gpu_list({"gpu_roles": bad, "gpu": 1, "args": []}), [1])

    def test_legacy_settings_are_untouched(self):
        self.assertEqual(gpu_list({"gpu": "0,2", "args": []}), [0, 2])
        self.assertEqual(gpu_list({"args": []}), [])
        self.assertEqual(card_roles_lines({"gpu": [1, 0], "args": []}), [])

    def test_the_log_line_names_every_role(self):
        lines = card_roles_lines({"gpu_roles": {"0": "helper", "1": "main", "2": "off"}, "gpu": [0, 1], "args": []})
        self.assertEqual(lines, ["[strata] card roles: Main GPU 1, Helper GPU 0, Off GPU 2 (numbered as setup numbers them)"])

    def test_every_card_off(self):
        cfg = {"gpu_roles": {"0": "off", "1": "off"}, "gpu": [], "gpu_none": True, "args": []}
        self.assertEqual(gpu_list(cfg), [])
        with self.assertRaises(SystemExit) as e:
            card_roles_lines(cfg)                                                   # the SYCL engine cannot start without a card: said plainly
        self.assertIn("Turn a card on", str(e.exception))
        cpu = card_roles_lines({**cfg, "backend": "cpu"})
        self.assertIn("Performance is degraded on CPU only.", cpu[0])


if __name__ == "__main__":
    unittest.main()
