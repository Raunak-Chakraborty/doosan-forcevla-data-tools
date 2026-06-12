import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.dummy.make_synthetic_raw_real_episode import make_synthetic_raw_real_episode
from doosan_forcevla_data.inspect.inspect_raw_real_episode import inspect_raw_real_episode


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class InspectRawRealEpisodeTests(unittest.TestCase):
    def test_valid_synthetic_episode_is_ready_and_has_core_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=5, fps=25.0, include_optional_streams=False)

            report = inspect_raw_real_episode(episode)

            self.assertTrue(report["ready_for_conversion"], report["errors"])
            self.assertTrue(report["validation"]["ok"], report["validation"]["errors"])
            self.assertIn(report["status"], {"ok", "warning"})
            for key in [
                "status",
                "episode_dir",
                "schema_version",
                "episode_id",
                "task_instruction",
                "robot_type",
                "collection_method",
                "geometry_type",
                "orientation_type",
                "success",
                "failure_reason",
                "fps",
                "ready_for_conversion",
                "validation",
                "streams",
                "required_streams",
                "optional_streams",
                "timeline",
                "timestamps",
                "camera_summary",
                "robot_state_summary",
                "joint_summary",
                "wrench_summary",
                "gripper_summary",
                "command_context_summary",
                "event_summary",
                "warnings",
                "errors",
                "recommendations",
            ]:
                self.assertIn(key, report)

            self.assertEqual(report["timeline"]["primary_stream"], "robot_state_rt")
            self.assertEqual(report["timeline"]["frame_count"], 5)
            self.assertTrue(report["timeline"]["required_streams_aligned"])
            self.assertAlmostEqual(report["timestamps"]["fps"], 25.0)
            self.assertAlmostEqual(report["timestamps"]["expected_dt"], 1.0 / 25.0)
            self.assertEqual(report["streams"]["robot_state_rt"]["record_count"], 5)
            self.assertEqual(report["streams"]["joint_states"]["record_count"], 5)
            self.assertEqual(report["streams"]["external_camera"]["record_count"], 5)
            self.assertEqual(report["streams"]["wrist_camera"]["record_count"], 5)
            self.assertFalse(report["gripper_summary"]["present"])
            self.assertEqual(report["gripper_summary"]["field_used"], "none")
            self.assertTrue(report["robot_state_summary"]["has_actual_tcp_position"])
            self.assertEqual(report["wrench_summary"]["preferred_source"], "external_tcp_force")
            self.assertTrue(report["event_summary"]["has_start_event"])
            self.assertTrue(report["event_summary"]["has_end_event"])

    def test_json_output_file_is_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = root / "episode_000000"
            report_path = root / "inspection" / "raw_real_inspection.json"
            make_synthetic_raw_real_episode(episode, frame_count=4)

            report = inspect_raw_real_episode(episode, output_json=report_path)

            self.assertTrue(report_path.is_file())
            data = _read_json(report_path)
            self.assertEqual(data["ready_for_conversion"], report["ready_for_conversion"])
            self.assertEqual(data["episode_id"], report["episode_id"])
            self.assertEqual(data["timeline"]["frame_count"], 4)

    def test_cli_succeeds_for_valid_synthetic_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = root / "episode_000000"
            report_path = root / "raw_real_inspection.json"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            env = dict(os.environ)
            src_path = Path(__file__).resolve().parents[1] / "src"
            env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "doosan_forcevla_data.inspect.inspect_raw_real_episode",
                    str(episode),
                    "--json-out",
                    str(report_path),
                    "--pretty",
                ],
                check=False,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertIn("Raw-Real Episode Inspection", completed.stdout)
            self.assertIn("ready_for_conversion: yes", completed.stdout)
            self.assertTrue(report_path.is_file())
            self.assertTrue(_read_json(report_path)["ready_for_conversion"])

    def test_missing_camera_image_blocks_conversion_and_cli_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = root / "episode_000000"
            report_path = root / "bad_inspection.json"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            (episode / "streams" / "external_camera" / "frames" / "000000.ppm").unlink()

            report = inspect_raw_real_episode(episode)
            self.assertFalse(report["ready_for_conversion"])
            self.assertFalse(report["validation"]["ok"])
            self.assertEqual(report["camera_summary"]["external_camera"]["missing_files"], 1)
            self.assertTrue(any("image_path does not exist" in error for error in report["errors"]))

            env = dict(os.environ)
            src_path = Path(__file__).resolve().parents[1] / "src"
            env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "doosan_forcevla_data.inspect.inspect_raw_real_episode",
                    str(episode),
                    "--json-out",
                    str(report_path),
                ],
                check=False,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("ready_for_conversion: no", completed.stdout)
            self.assertTrue(report_path.is_file())
            self.assertFalse(_read_json(report_path)["ready_for_conversion"])

    def test_record_index_mismatch_is_reflected_in_timeline_and_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            joint_path = episode / "streams" / "joint_states.jsonl"
            joint_records = _read_jsonl(joint_path)
            _write_jsonl(joint_path, joint_records[:-1])

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertFalse(report["timeline"]["required_streams_aligned"])
            self.assertEqual(report["timeline"]["missing_by_stream"]["joint_states"], [3])
            self.assertTrue(
                any("record_index alignment with robot_state_rt failed" in error for error in report["validation"]["errors"])
            )

    def test_present_command_context_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=5, include_optional_streams=True)

            report = inspect_raw_real_episode(episode)

            self.assertTrue(report["ready_for_conversion"], report["errors"])
            self.assertTrue(report["command_context_summary"]["present"])
            self.assertEqual(report["command_context_summary"]["record_count"], 5)
            self.assertEqual(report["command_context_summary"]["policy"], "diagnostic only")
            self.assertFalse(report["command_context_summary"]["used_as_action_label"])

    def test_malformed_metadata_or_index_produces_report_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = [
                ("metadata", root / "bad_metadata", "metadata.json", "{not valid json"),
                ("index", root / "bad_index", "streams/index.json", "[]\n"),
            ]
            for case_name, episode, relative_path, contents in cases:
                with self.subTest(case=case_name):
                    make_synthetic_raw_real_episode(episode, frame_count=4)
                    target = episode / relative_path
                    target.write_text(contents, encoding="utf-8")

                    report = inspect_raw_real_episode(episode)

                    self.assertEqual(report["status"], "failed")
                    self.assertFalse(report["ready_for_conversion"])
                    self.assertFalse(report["validation"]["ok"])
                    self.assertTrue(report["errors"])

    def test_no_forbidden_ros_imports_in_inspection_module(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "doosan_forcevla_data"
            / "inspect"
            / "inspect_raw_real_episode.py"
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
            self.assertNotIn(f"{prefix} {module}", source)


if __name__ == "__main__":
    unittest.main()
