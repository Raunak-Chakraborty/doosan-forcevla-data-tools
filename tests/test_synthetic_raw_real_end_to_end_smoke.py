import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.smoke.synthetic_raw_real_end_to_end import (
    run_synthetic_raw_real_end_to_end_smoke,
)
from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode
from doosan_forcevla_data.validate.validate_staged_export import validate_staged_export


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class SyntheticRawRealEndToEndSmokeTests(unittest.TestCase):
    def test_end_to_end_smoke_forcevla_succeeds_and_reports_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "smoke_output"

            report = run_synthetic_raw_real_end_to_end_smoke(
                output_root,
                frame_count=6,
                fps=30.0,
                profile="forcevla_13d",
            )

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["frame_count"], 6)
            self.assertEqual(report["profile"], "forcevla_13d")

            report_path = Path(report["paths"]["json_report"])
            self.assertTrue(report_path.is_file())
            report_from_disk = _read_json(report_path)
            self.assertEqual(report_from_disk["status"], "ok")

            raw_real_dir = Path(report["paths"]["raw_real_episode"])
            processed_dir = Path(report["paths"]["processed_episode"])
            plan_path = Path(report["paths"]["export_plan"])
            staged_dir = Path(report["paths"]["staged_export"])
            skeleton_dir = Path(report["paths"]["lerobot_skeleton"])
            real_export_dir = Path(report["paths"]["real_export_dry_run"])

            self.assertTrue(raw_real_dir.is_dir())
            self.assertTrue((raw_real_dir / "metadata.json").is_file())
            self.assertTrue(processed_dir.is_dir())
            self.assertTrue((processed_dir / "metadata_processed.json").is_file())
            self.assertTrue((processed_dir / "frames.jsonl").is_file())
            self.assertTrue(plan_path.is_file())
            self.assertTrue(staged_dir.is_dir())
            self.assertTrue((staged_dir / "metadata_staged.json").is_file())
            self.assertTrue(skeleton_dir.is_dir())
            self.assertTrue((real_export_dir / "export_attempt_report.json").is_file())

            processed_validation = validate_processed_episode(processed_dir)
            self.assertTrue(processed_validation.ok, processed_validation.errors)
            staged_validation = validate_staged_export(staged_dir)
            self.assertTrue(staged_validation.ok, staged_validation.errors)

            self.assertEqual(report["raw_real_validation"]["ok"], True)
            self.assertEqual(report["processed_validation"]["ok"], True)
            self.assertEqual(report["export_plan_validation"]["ok"], True)
            self.assertEqual(report["staged_validation"]["ok"], True)
            self.assertEqual(report["lerobot_skeleton_validation"]["ok"], True)
            self.assertEqual(report["real_export_attempt_validation"]["ok"], True)
            self.assertEqual(report["plan_summary"]["input_frame_count"], 6)
            self.assertEqual(report["plan_summary"]["exported_frame_count"], 5)
            self.assertEqual(report["staging_summary"]["observation_state_dim"], 13)
            self.assertEqual(report["staging_summary"]["action_dim"], 7)
            self.assertEqual(report["processed_summary"]["processed_frame_records"], 6)

    def test_existing_output_without_overwrite_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "smoke_output"
            output_root.mkdir()

            with self.assertRaises(FileExistsError):
                run_synthetic_raw_real_end_to_end_smoke(output_root, frame_count=4)

    def test_existing_output_with_overwrite_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "smoke_output"
            output_root.mkdir()
            stale_file = output_root / "stale.txt"
            stale_file.write_text("stale\n", encoding="utf-8")

            report = run_synthetic_raw_real_end_to_end_smoke(
                output_root,
                frame_count=4,
                overwrite=True,
                run_export_preflight=False,
            )

            self.assertEqual(report["status"], "ok")
            self.assertFalse(stale_file.exists())
            self.assertTrue(Path(report["paths"]["json_report"]).is_file())
            self.assertIn("run_export_preflight=False", report["skips"])

    def test_cli_smoke_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "smoke_output"
            env = dict(os.environ)
            src_path = Path(__file__).resolve().parents[1] / "src"
            env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "doosan_forcevla_data.smoke.synthetic_raw_real_end_to_end",
                    "--output-root",
                    str(output_root),
                    "--frames",
                    "5",
                    "--fps",
                    "30",
                    "--profile",
                    "forcevla_13d",
                    "--overwrite",
                ],
                check=False,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertIn("OK: synthetic raw-real end-to-end smoke passed", completed.stdout)
            report_path = output_root / "reports" / "synthetic_raw_real_end_to_end_report.json"
            self.assertTrue(report_path.is_file())
            self.assertEqual(_read_json(report_path)["status"], "ok")

    def test_command_context_remains_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "smoke_output"

            report = run_synthetic_raw_real_end_to_end_smoke(
                output_root,
                frame_count=5,
                include_optional_streams=True,
                run_export_preflight=False,
            )

            processed_metadata = _read_json(
                Path(report["paths"]["processed_episode"]) / "metadata_processed.json"
            )
            self.assertEqual(processed_metadata["action_label_primary"], "measured_tcp_delta")
            self.assertEqual(processed_metadata["command_context_policy"], "diagnostic only; never used as action label")
            self.assertFalse(processed_metadata["optional_debug"]["command_context"]["used_as_action_label"])

    def test_no_forbidden_ros_imports_in_smoke_module(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "doosan_forcevla_data"
            / "smoke"
            / "synthetic_raw_real_end_to_end.py"
        )
        source = source_path.read_text(encoding="utf-8")
        forbidden_imports = [
            ("import", "rclpy"),
            ("from", "rclpy"),
            ("import", "sensor_msgs"),
            ("from", "sensor_msgs"),
            ("import", "geometry_msgs"),
            ("from", "geometry_msgs"),
            ("import", "std_msgs"),
            ("from", "std_msgs"),
            ("import", "tf2_msgs"),
            ("from", "tf2_msgs"),
            ("import", "dsr_msgs2"),
            ("from", "dsr_msgs2"),
        ]
        for prefix, module in forbidden_imports:
            pattern = f"{prefix} {module}"
            self.assertNotIn(pattern, source)


if __name__ == "__main__":
    unittest.main()
