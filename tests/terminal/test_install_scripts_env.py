"""install.sh: an ARM computer is refused with the reason and the way out, and a broken download never looks like an install."""

import os
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT / "packaging" / "install.sh"


class Text(unittest.TestCase):
    def test_the_readme_one_liner_cannot_succeed_on_a_failed_download(self):
        for name in ("docs/INSTALL.md", "docs/TERMINAL_LAUNCH.md"):        # the B70 fork's README documents the source build, not the one-liner
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("install.sh | sh", text, name)                    # `curl ... | sh` runs nothing and exits 0 when curl fails
            self.assertIn("install.sh -o install.sh && sh install.sh", text, name)

    def test_arm_is_refused_with_a_reason_and_a_next_step(self):
        sh = SH.read_text(encoding="utf-8")
        self.assertIn("ARM processor", sh)
        self.assertIn("x86_64 processor and an Intel Arc card", sh)


@unittest.skipUnless(shutil.which("sh") and os.uname().sysname == "Linux" if hasattr(os, "uname") else False, "needs sh on Linux")
class InstallShFailsVisibly(unittest.TestCase):
    """A broken download of install.sh must not look like a finished install."""

    def run_sh(self, text, **env):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "install.sh"
            f.write_text(text, encoding="utf-8", newline="\n")
            home = Path(d) / "home"
            home.mkdir()
            r = subprocess.run(["sh", str(f)], capture_output=True, text=True, timeout=120,
                               env={**os.environ, "HOME": str(home), "STRATA_DATA_DIR": str(home / "data"), "STRATA_BIN_DIR": str(home / "bin"), **env})
            return r, home

    def test_the_script_is_one_function_called_on_the_last_line(self):
        text = SH.read_text(encoding="utf-8")
        self.assertIn("\nmain() {", text)
        self.assertEqual(text.rstrip().splitlines()[-1], 'main "$@"')

    def test_syntax(self):
        self.assertEqual(subprocess.run(["sh", "-n", str(SH)], capture_output=True).returncode, 0)

    def test_a_half_downloaded_script_runs_nothing_and_fails(self):
        lines = SH.read_text(encoding="utf-8").splitlines(True)
        r, home = self.run_sh("".join(lines[:int(len(lines) * 0.8)]))
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((home / "data").exists() or (home / "bin").exists())

    def test_an_unreachable_release_folder_stops_with_a_reason_and_a_failing_exit(self):
        if not shutil.which("curl") and not shutil.which("wget"):
            self.skipTest("no curl or wget")
        r, home = self.run_sh(SH.read_text(encoding="utf-8"), STRATA_FEED="http://127.0.0.1:9")
        self.assertEqual(r.returncode, 1)
        self.assertIn("could not read", r.stderr)
        self.assertFalse((home / "bin" / "strata").exists())


if __name__ == "__main__":
    unittest.main()
