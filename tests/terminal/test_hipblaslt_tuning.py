"""The hipBLASLt tuning tables in the AMD engine packages: what a package holds per card family, and the one rule that decides at a start
whether a table is used (strata_control/hipblaslt_tuning.py). Folders are throwaways; no GPU tool is run.

The packages' hipBLASLt versions below are the ones the 0.2.x release packages bundle (docs/AMD_HIP.md, Ready-made packages): Windows ROCm
10.2.0a20260930 carries 1.5.0 (100500), the Linux 7.10.0a20251120 wheels 1.2.0 (100200), Linux Strix Halo's 7.14.1 1.4.1 (100401)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import hipblaslt_tuning as ht

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tools" / "hip"

#: (platform, variant) -> (archs of the package, hipBLASLt version it bundles, architectures that must carry a table)
PACKAGES = {
    ("windows-x86_64", "rx9070"): (("gfx1200", "gfx1201"), 100500, {"gfx1201"}),
    ("windows-x86_64", "rdna3"): (("gfx1100", "gfx1101", "gfx1102", "gfx1151"), 100500, {"gfx1100", "gfx1151"}),
    ("linux-x86_64", "rx9070"): (("gfx1200", "gfx1201"), 100200, set()),                    # no table for 1.2.0 on RDNA4 (docs/AMD_HIP.md)
    ("linux-x86_64", "rx7900"): (("gfx1100", "gfx1101", "gfx1102"), 100200, {"gfx1100"}),
    ("linux-x86_64", "halo"): (("gfx1151",), 100401, {"gfx1151"}),
    # 0.2.2 RX 6000 packages: hipBLASLt ships no gfx1030 / gfx1031 kernels at any version, so no table, plain hipBLAS
    ("windows-x86_64", "rdna2"): (("gfx1030", "gfx1031"), 100500, set()),
    ("linux-x86_64", "rx6000"): (("gfx1030", "gfx1031"), 100400, set()),
}


ON = {ht.SWITCH: "on"}            # the switch that uses a matching table whatever "auto" does
AUTO = {}


def fake_table(folder: Path, arch: str, version: int, *, header: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ht.table_name(arch, version)
    path.write_text(f"# solution ids are scoped to this hipBLASLt version and device architecture\n"
                    f"{header or f'{ht.HEADER} {arch} {version}'}\nbf16 128 2560 128 4096 73754\n", encoding="utf-8")
    return path


def fake_package(root: Path, *, version: int | None = 100500, tables=(("gfx1201", 100500),), build=True) -> Path:
    folder = root / "0.2.1-pkg"
    folder.mkdir(parents=True)
    meta = {"version": "0.2.1", "backend": "hip"}
    if version is not None:
        meta["hipblaslt_version"] = version
    if build:
        (folder / "BUILD.json").write_text(json.dumps(meta), encoding="utf-8")
    for arch, ver in tables:
        fake_table(folder / ht.TABLE_DIR, arch, ver)
    return folder


class TempDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)


class Decision(TempDir):
    """Match, mismatch, switch, the person's own choice."""

    def test_a_table_for_this_card_and_this_hipblaslt_is_used(self):
        pkg = fake_package(self.root)
        d = ht.resolve(pkg, ["gfx1201"], environ=ON)
        self.assertEqual(d.path, pkg / "tuning" / "gfx1201-hipblaslt-100500.txt")
        self.assertIn("gfx1201-hipblaslt-100500.txt", d.message)
        self.assertIn(ht.SWITCH, d.message)               # the line tells how to turn it off
        self.assertEqual(d.level, "info")

    def test_a_table_for_another_hipblaslt_version_is_not_used_and_the_log_says_why(self):
        pkg = fake_package(self.root, version=100200, tables=(("gfx1201", 100500),))
        d = ht.resolve(pkg, ["gfx1201"], environ=ON)
        self.assertIsNone(d.path)
        self.assertEqual(d.level, "warn")
        self.assertIn("100200", d.message)
        self.assertIn("100500", d.message)
        self.assertIn("plain hipBLAS", d.message)

    def test_a_card_without_a_table_in_the_package_gets_plain_hipblas_and_one_line(self):
        pkg = fake_package(self.root)
        d = ht.resolve(pkg, ["gfx1200"], environ=ON)
        self.assertIsNone(d.path)
        self.assertIn("none for gfx1200", d.message)

    def test_the_card_that_has_a_table_is_found_among_several(self):
        pkg = fake_package(self.root, tables=(("gfx1100", 100500), ("gfx1151", 100500)))
        self.assertEqual(ht.resolve(pkg, ["gfx1101", "gfx1151"], environ=ON).path.name, "gfx1151-hipblaslt-100500.txt")

    def test_a_package_that_does_not_say_its_hipblaslt_is_not_guessed_at(self):
        pkg = fake_package(self.root, version=None)
        d = ht.resolve(pkg, ["gfx1201"], environ=ON)
        self.assertIsNone(d.path)
        self.assertEqual(d.level, "warn")
        self.assertIn("hipblaslt_version", d.message)

    def test_an_unreadable_card_is_a_note_not_a_guess(self):
        d = ht.resolve(fake_package(self.root), [], environ=ON)
        self.assertIsNone(d.path)
        self.assertIn("could not be read", d.message)

    def test_a_table_whose_header_disagrees_with_its_name_is_not_used(self):
        pkg = fake_package(self.root, tables=())
        fake_table(pkg / "tuning", "gfx1201", 100500, header=f"{ht.HEADER} gfx1201 100401")
        d = ht.resolve(pkg, ["gfx1201"], environ=ON)
        self.assertIsNone(d.path)
        self.assertIn("header", d.message)

    def test_the_switch_turns_it_off_in_the_environment_and_in_the_run_config(self):
        pkg = fake_package(self.root)
        for word in ("off", "OFF", "0", "no", "false", "none"):
            d = ht.resolve(pkg, ["gfx1201"], environ={ht.SWITCH: word})
            self.assertIsNone(d.path, word)
            self.assertIn("off", d.message)
        self.assertIsNone(ht.resolve(pkg, ["gfx1201"], environ=AUTO, cfg={"hipblaslt_table": False}).path)
        self.assertIsNotNone(ht.resolve(pkg, ["gfx1201"], environ=AUTO, cfg={"hipblaslt_table": True}).path)
        self.assertIsNone(ht.resolve(pkg, ["gfx1201"], environ={ht.SWITCH: "off"}, cfg={"hipblaslt_table": True}).path)   # the environment wins
        for word in ("on", "ON", "1", "yes", "true"):
            self.assertIsNotNone(ht.resolve(pkg, ["gfx1201"], environ={ht.SWITCH: word}).path, word)

    def test_auto_follows_the_one_constant_and_says_what_is_available(self):
        pkg = fake_package(self.root)
        with mock.patch.object(ht, "AUTO_USES_TABLE", False):
            d = ht.resolve(pkg, ["gfx1201"], environ=AUTO)
            self.assertIsNone(d.path)
            self.assertIn("gfx1201-hipblaslt-100500.txt", d.message)
            self.assertIn(f"{ht.SWITCH}=on", d.message)
            self.assertEqual(ht.mode(AUTO), "auto")
        with mock.patch.object(ht, "AUTO_USES_TABLE", True):
            d = ht.resolve(pkg, ["gfx1201"], environ=AUTO)
            self.assertEqual(d.path, pkg / "tuning" / "gfx1201-hipblaslt-100500.txt")
            self.assertIsNone(ht.resolve(pkg, ["gfx1201"], environ={ht.SWITCH: "off"}).path)

    def test_the_default_is_the_owners_decision(self):
        """A table changes the bits of the prompt's dense GEMMs (notes-021-hip.md); the owner chose: on by default for Strix Halo only."""
        self.assertFalse(ht.AUTO_USES_TABLE)
        self.assertEqual(ht.AUTO_ARCHS, ("gfx1151",))

    def test_auto_uses_the_table_on_strix_halo_only(self):
        for arch, used in (("gfx1151", True), ("gfx1201", False), ("gfx1100", False)):
            root = self.root / arch
            root.mkdir()
            pkg = fake_package(root, tables=((arch, 100500),))
            d = ht.resolve(pkg, [arch], environ=AUTO)
            self.assertEqual(d.path is not None, used, (arch, d.message))
            self.assertIsNone(ht.resolve(pkg, [arch], environ={ht.SWITCH: "off"}).path)

    def test_a_table_the_person_set_is_never_replaced(self):
        pkg = fake_package(self.root)
        mine = str(self.root / "mine.txt")
        for environ, cfg in (({ht.ENGINE_VAR: mine}, {}), ({}, {"env": {ht.ENGINE_VAR: mine}})):
            d = ht.resolve(pkg, ["gfx1201"], environ=environ, cfg=cfg)
            self.assertIsNone(d.path)
            self.assertIn(mine, d.message)
            out = ht.apply({"backend": "hip", **cfg}, pkg, ["gfx1201"], environ=environ, say=lambda line: None)
            self.assertEqual((out.get("env") or {}).get(ht.ENGINE_VAR, mine), mine)


class Apply(TempDir):
    def test_apply_puts_the_table_in_a_copy_of_the_config_and_says_so_once(self):
        pkg = fake_package(self.root)
        cfg = {"backend": "hip", "env": {"STRATA_RESIDENT_PIN": "0"}}
        lines = []
        out = ht.apply(cfg, pkg, ["gfx1201"], environ=ON, say=lines.append)
        self.assertEqual(out["env"][ht.ENGINE_VAR], str(pkg / "tuning" / "gfx1201-hipblaslt-100500.txt"))
        self.assertEqual(out["env"]["STRATA_RESIDENT_PIN"], "0")
        self.assertNotIn(ht.ENGINE_VAR, cfg["env"])        # the caller's config is untouched
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("[strata] hipBLASLt tuning table"))

    def test_apply_with_a_mismatch_leaves_the_config_alone_and_warns(self):
        pkg = fake_package(self.root, version=100200)
        cfg = {"backend": "hip"}
        lines = []
        self.assertIs(ht.apply(cfg, pkg, ["gfx1201"], environ=ON, say=lines.append), cfg)
        self.assertEqual(len(lines), 1)
        self.assertIn("note", lines[0])

    def test_other_backends_and_packages_without_tables_are_silent(self):
        lines = []
        pkg = fake_package(self.root)
        cfg = {"backend": "cuda"}
        self.assertIs(ht.apply(cfg, pkg, ["gfx1201"], environ=ON, say=lines.append), cfg)
        bare = self.root / "cuda-pkg"
        bare.mkdir()
        (bare / "BUILD.json").write_text(json.dumps({"version": "0.2.1"}), encoding="utf-8")
        out = ht.apply({"backend": "hip"}, bare, ["gfx1201"], environ=ON, say=lines.append)
        self.assertNotIn("env", out)
        self.assertEqual(lines, [])

class PackageContents(TempDir):
    """What each package of the release matrix carries, from the real tables of tools/hip."""

    def test_version_from_header_reads_the_rocm_tree(self):
        inc = self.root / "include" / "hipblaslt"
        inc.mkdir(parents=True)
        (inc / "hipblaslt-version.h").write_text("#define HIPBLASLT_VERSION_MAJOR 1\n#define HIPBLASLT_VERSION_MINOR 5\n"
                                                 "#define HIPBLASLT_VERSION_PATCH 0\n", encoding="utf-8")
        self.assertEqual(ht.version_from_header(self.root), 100500)
        self.assertIsNone(ht.version_from_header(self.root / "nowhere"))


if __name__ == "__main__":
    unittest.main()
