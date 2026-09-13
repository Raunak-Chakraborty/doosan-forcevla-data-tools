from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_pre_conversion_audit.sh"


class PreConversionLauncherTests(unittest.TestCase):
    def _write_executable(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def _fixture(self, root: Path):
        base = root / "base_setup.bash"
        bad = root / "bad_setup.bash"
        good = root / "good_setup.bash"
        fake_python = root / "fake_python"
        capture = root / "capture.txt"

        base.write_text('export FAKE_BASE_SOURCED=1\n', encoding="utf-8")
        bad.write_text(
            'export FAKE_SELECTED_WORKSPACE=bad\nexport FAKE_ROS_CONTRACT_OK=0\n',
            encoding="utf-8",
        )
        good.write_text(
            'export FAKE_SELECTED_WORKSPACE=good\nexport FAKE_ROS_CONTRACT_OK=1\n',
            encoding="utf-8",
        )
        self._write_executable(
            fake_python,
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -u
                if [ "${1:-}" = "-c" ]; then
                    if [ "${FAKE_ROS_CONTRACT_OK:-0}" = "1" ]; then
                        echo PRECONVERSION_ROS_CONTRACT_PROBE=PASS
                        exit 0
                    fi
                    echo PRECONVERSION_ROS_TYPE_MISSING=fake >&2
                    exit 1
                fi
                printf '%s\n' "${FAKE_SELECTED_WORKSPACE:-base}" > "$FAKE_CAPTURE_FILE"
                exit 0
                """
            ),
        )
        return base, bad, good, fake_python, capture

    def _run(self, *, base: Path, fake_python: Path, capture: Path, extra_env: dict[str, str]):
        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": str(fake_python),
                "ROS_BASE_SETUP": str(base),
                "FAKE_CAPTURE_FILE": str(capture),
            }
        )
        env.update(extra_env)
        return subprocess.run(
            [str(LAUNCHER), "/fake/dataset", "/fake/output", "--workers", "auto"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_auto_selection_skips_existing_but_incomplete_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, bad, good, fake_python, capture = self._fixture(Path(tmp))
            result = self._run(
                base=base,
                fake_python=fake_python,
                capture=capture,
                extra_env={"ROS_WORKSPACE_CANDIDATES": f"{bad}:{good}"},
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertIn("Rejected ROS workspace candidate", result.stdout)
            self.assertIn(f"PRECONVERSION_ROS_WORKSPACE_SETUP={good}", result.stdout)
            self.assertIn("PRECONVERSION_RUNTIME_GATE=PASS", result.stdout)
            self.assertEqual(capture.read_text(encoding="utf-8").strip(), "good")

    def test_explicit_incomplete_workspace_fails_closed_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, bad, good, fake_python, capture = self._fixture(Path(tmp))
            result = self._run(
                base=base,
                fake_python=fake_python,
                capture=capture,
                extra_env={
                    "ROS_WORKSPACE_SETUP": str(bad),
                    "ROS_WORKSPACE_CANDIDATES": str(good),
                },
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("explicit ROS workspace does not satisfy", result.stdout)
            self.assertFalse(capture.exists())

    def test_explicit_valid_workspace_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, _bad, good, fake_python, capture = self._fixture(Path(tmp))
            result = self._run(
                base=base,
                fake_python=fake_python,
                capture=capture,
                extra_env={"ROS_WORKSPACE_SETUP": str(good)},
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertIn(f"PRECONVERSION_ROS_WORKSPACE_SETUP={good}", result.stdout)
            self.assertEqual(capture.read_text(encoding="utf-8").strip(), "good")


if __name__ == "__main__":
    unittest.main()
