import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.raw_real_to_processed import convert_raw_real_to_processed
from doosan_forcevla_data.dummy.make_synthetic_raw_real_episode import make_synthetic_raw_real_episode
from doosan_forcevla_data.inspect.inspect_raw_real_episode import inspect_raw_real_episode
from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode
from doosan_forcevla_data.validate.validate_raw_real_episode import validate_raw_real_episode


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")



def _valid_wrench_sources_metadata() -> dict:
    return {
        "external_tcp_force": {
            "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "force_unit": "N",
            "torque_unit": "Nm",
            "frame": "base",
            "compensation": "estimated_external_tcp_force",
            "approved_for_model_state": True,
        },
        "raw_force_torque": {
            "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "force_unit": "N",
            "torque_unit": "Nm",
            "frame": "flange",
            "compensation": "raw_flange_sensor",
            "approved_for_model_state": False,
        },
    }

def _mark_non_synthetic(
    episode: Path,
    convention: str | None = "rotation_vector_degrees",
    collection_method: str = "passive_real_recorder",
    recorder_version: str = "passive_real_recorder_v0",
    strict_lab_provenance: bool = True,
    source_stamp_unit: str | None = "seconds",
    include_wrench_metadata: bool = True,
) -> None:
    metadata_path = episode / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata["collection_method"] = collection_method
    metadata["recorder_version"] = recorder_version
    metadata.pop("synthetic", None)
    metadata["source_workspace"] = {"path": "lab/offline", "verified": True}
    metadata.pop("tcp_orientation_convention_verified", None)
    if convention is None:
        metadata.pop("tcp_orientation_convention", None)
    else:
        metadata["tcp_orientation_convention"] = convention
    if strict_lab_provenance:
        metadata["lab_provenance_required"] = True
        metadata["source_workspace"] = {
            "path": "/home/ktt_rc/robotics_thesis/lab_myros2_ws/src/MyROS2",
            "git_commit": "abc1234",
            "git_remote": "https://github.com/Raunak-Chakraborty/doosan-forcevla-data-tools.git",
            "git_branch": "main",
            "verified": True,
        }
        metadata["live_graph_verification"] = {
            "exact_doosan_namespace": "/dsr01",
            "external_camera_topic": "/external_camera/color/image_raw",
            "wrist_camera_topic": "/wrist_camera/color/image_raw",
            "read_data_rt_service": "/dsr01/dsr_controller2/realtime/read_data_rt",
            "tcp_frame": "tcp_link",
            "flange_frame": "link_6",
            "tool_frame": "tool0",
            "force_torque_source": "robot_state_rt.external_tcp_force",
            "gripper_state_source": "not_available_for_this_episode",
            "time_sync_verified": True,
        }
    else:
        metadata.pop("lab_provenance_required", None)
        metadata.pop("strict_lab_provenance", None)
        metadata.pop("live_graph_verification", None)
    _write_json(metadata_path, metadata)

    recorder_report_path = episode / "recorder_report.json"
    recorder_report = _read_json(recorder_report_path)
    recorder_report["synthetic"] = False
    recorder_report["generator"] = "passive_real_recorder"
    recorder_report["generator_version"] = "passive_real_recorder_v0"
    recorder_report.pop("tcp_orientation_convention_verified", None)
    _write_json(recorder_report_path, recorder_report)

    streams_index_path = episode / "streams" / "index.json"
    streams_index = _read_json(streams_index_path)
    streams_index["synthetic"] = False
    if source_stamp_unit is None:
        streams_index.pop("timebase", None)
    else:
        streams_index["timebase"] = {"source_stamp_unit": source_stamp_unit}

    if strict_lab_provenance:
        source_names = {
            "joint_states": "/dsr01/joint_states",
            "robot_state_rt": "/dsr01/dsr_controller2/realtime/read_data_rt",
            "tf": "/tf",
            "tf_static": "/tf_static",
            "external_camera": "/external_camera/color/image_raw",
            "wrist_camera": "/wrist_camera/color/image_raw",
            "command_context": "/doosan_teleop/cmd_vel_6d",
            "gripper_state": "/gripper_state",
        }
        for stream_name, entry in streams_index.get("streams", {}).items():
            if isinstance(entry, dict):
                entry["verified"] = True
                entry["source_name"] = source_names.get(stream_name, f"/verified/{stream_name}")
    robot_entry = streams_index["streams"]["robot_state_rt"]
    if include_wrench_metadata:
        robot_entry["wrench_sources"] = _valid_wrench_sources_metadata()
    else:
        robot_entry.pop("wrench_sources", None)
    _write_json(streams_index_path, streams_index)


def _truncate_to_one_aligned_record(episode: Path) -> None:
    streams_index_path = episode / "streams" / "index.json"
    streams_index = _read_json(streams_index_path)
    for stream_name in ["joint_states", "robot_state_rt", "tf", "external_camera", "wrist_camera"]:
        streams_index["streams"][stream_name]["record_count"] = 1
    _write_json(streams_index_path, streams_index)

    for relative_path in [
        "streams/joint_states.jsonl",
        "streams/robot_state_rt.jsonl",
        "streams/tf.jsonl",
    ]:
        path = episode / relative_path
        _write_jsonl(path, _read_jsonl(path)[:1])

    for stream_name in ["external_camera", "wrist_camera"]:
        path = episode / "streams" / stream_name / "index.jsonl"
        _write_jsonl(path, _read_jsonl(path)[:1])


class RawRealToProcessedTests(unittest.TestCase):
    def test_synthetic_raw_real_episode_converts_and_validates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=6)
            output = convert_raw_real_to_processed(raw_episode, processed_episode)

            self.assertEqual(output, processed_episode)
            self.assertTrue((processed_episode / "metadata_processed.json").is_file())
            self.assertTrue((processed_episode / "frames.jsonl").is_file())

            result = validate_processed_episode(processed_episode)
            self.assertTrue(result.ok, result.errors)

            metadata = _read_json(processed_episode / "metadata_processed.json")
            self.assertEqual(metadata["frame_count"], 6)
            self.assertEqual(metadata["model_state_dim"], MODEL_STATE_DIM)
            self.assertEqual(metadata["action_dim"], ACTION_DIM)
            self.assertEqual(metadata["action_label_primary"], "measured_tcp_delta")

            frames = _read_jsonl(processed_episode / "frames.jsonl")
            self.assertEqual(len(frames), 6)
            has_nonzero_translation = False
            has_nonzero_rotation = False
            for idx, frame in enumerate(frames):
                self.assertEqual(frame["frame_index"], idx)
                self.assertEqual(len(frame["model_state"]), MODEL_STATE_DIM)
                self.assertEqual(len(frame["measured_action"]), ACTION_DIM)
                self.assertTrue(all(math.isfinite(value) for value in frame["model_state"]))
                self.assertTrue(all(math.isfinite(value) for value in frame["measured_action"]))
                self.assertTrue(frame["external_rgb_path"])
                self.assertTrue(frame["tcp_rgb_path"])
                self.assertTrue((raw_episode / frame["external_rgb_path"]).is_file())
                self.assertTrue((raw_episode / frame["tcp_rgb_path"]).is_file())

                if idx < len(frames) - 1:
                    self.assertFalse(frame["action_is_terminal_padding"])
                    if any(abs(value) > 1e-12 for value in frame["measured_action"][:3]):
                        has_nonzero_translation = True
                    if any(abs(value) > 1e-12 for value in frame["measured_action"][3:6]):
                        has_nonzero_rotation = True

            self.assertTrue(frames[-1]["action_is_terminal_padding"])
            self.assertEqual(frames[-1]["measured_action"], [0.0] * ACTION_DIM)
            self.assertTrue(has_nonzero_translation)
            self.assertTrue(has_nonzero_rotation)

    def test_two_frame_synthetic_raw_real_episode_converts_with_one_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=2)
            convert_raw_real_to_processed(raw_episode, processed_episode)

            result = validate_processed_episode(processed_episode)
            self.assertTrue(result.ok, result.errors)
            frames = _read_jsonl(processed_episode / "frames.jsonl")
            self.assertEqual(len(frames), 2)
            self.assertFalse(frames[0]["action_is_terminal_padding"])
            self.assertTrue(frames[-1]["action_is_terminal_padding"])
            self.assertTrue(any(abs(value) > 1e-12 for value in frames[0]["measured_action"][:6]))

    def test_one_aligned_record_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=2)
            _truncate_to_one_aligned_record(raw_episode)

            validation = validate_raw_real_episode(raw_episode)
            report = inspect_raw_real_episode(raw_episode)
            self.assertTrue(validation.ok, validation.errors)
            self.assertFalse(report["ready_for_conversion"])
            self.assertTrue(
                any("at least 2 records" in blocker for blocker in report["conversion_blockers"]),
                report["conversion_blockers"],
            )
            with self.assertRaisesRegex(ValueError, "requires at least 2 aligned records; got 1"):
                convert_raw_real_to_processed(raw_episode, processed_episode)

            self.assertFalse(processed_episode.exists())

    def test_one_aligned_record_fails_before_overwriting_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=2)
            _truncate_to_one_aligned_record(raw_episode)
            processed_episode.mkdir(parents=True)
            sentinel = processed_episode / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires at least 2 aligned records; got 1"):
                convert_raw_real_to_processed(raw_episode, processed_episode, overwrite=True)

            self.assertTrue(sentinel.is_file())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse((processed_episode / "metadata_processed.json").exists())
            self.assertFalse((processed_episode / "frames.jsonl").exists())

    def test_cli_one_aligned_record_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=2)
            _truncate_to_one_aligned_record(raw_episode)
            env = dict(os.environ)
            src_path = Path(__file__).resolve().parents[1] / "src"
            env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "doosan_forcevla_data.convert.raw_real_to_processed",
                    "--raw-real",
                    str(raw_episode),
                    "--output",
                    str(processed_episode),
                ],
                check=False,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("requires at least 2 aligned records; got 1", completed.stdout)
            self.assertFalse(processed_episode.exists())

    def test_optional_streams_absent_convert_with_zero_gripper_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4, include_optional_streams=False)
            convert_raw_real_to_processed(raw_episode, processed_episode)

            frames = _read_jsonl(processed_episode / "frames.jsonl")
            self.assertTrue(frames)
            for frame in frames:
                self.assertAlmostEqual(frame["model_state"][6], 0.0)
            for frame in frames[:-1]:
                self.assertAlmostEqual(frame["measured_action"][6], 0.0)

    def test_optional_gripper_stream_is_used_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=5, include_optional_streams=True)
            gripper_path = raw_episode / "streams" / "gripper_state.jsonl"
            records = _read_jsonl(gripper_path)
            for idx, record in enumerate(records):
                record["gripper_position"] = 0.10 + 0.01 * idx
            _write_jsonl(gripper_path, records)

            convert_raw_real_to_processed(raw_episode, processed_episode)

            frames = _read_jsonl(processed_episode / "frames.jsonl")
            for idx, frame in enumerate(frames):
                self.assertAlmostEqual(frame["model_state"][6], 0.10 + 0.01 * idx)
            for frame in frames[:-1]:
                self.assertAlmostEqual(frame["measured_action"][6], 0.01)

    def test_existing_output_without_overwrite_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode)
            processed_episode.mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                convert_raw_real_to_processed(raw_episode, processed_episode, overwrite=False)

    def test_existing_output_with_overwrite_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode)
            processed_episode.mkdir(parents=True)
            junk_path = processed_episode / "junk.txt"
            junk_path.write_text("junk\n", encoding="utf-8")

            convert_raw_real_to_processed(raw_episode, processed_episode, overwrite=True)

            self.assertFalse(junk_path.exists())
            result = validate_processed_episode(processed_episode)
            self.assertTrue(result.ok, result.errors)

    def test_output_equal_to_raw_root_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_episode = Path(tmpdir) / "raw_real" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode)

            with self.assertRaisesRegex(ValueError, "cannot be inside the raw-real episode"):
                convert_raw_real_to_processed(raw_episode, raw_episode, overwrite=True)

            self.assertFalse((raw_episode / "metadata_processed.json").exists())

    def test_new_output_under_raw_root_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_episode = Path(tmpdir) / "raw_real" / "episode_000000"
            output_under_raw = raw_episode / "processed_inside_raw"
            make_synthetic_raw_real_episode(raw_episode)

            with self.assertRaisesRegex(ValueError, "cannot be inside the raw-real episode"):
                convert_raw_real_to_processed(raw_episode, output_under_raw)

            self.assertFalse(output_under_raw.exists())

    def test_existing_output_under_raw_root_fails_before_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_episode = Path(tmpdir) / "raw_real" / "episode_000000"
            output_under_raw = raw_episode / "processed_inside_raw"
            make_synthetic_raw_real_episode(raw_episode)
            output_under_raw.mkdir()

            with self.assertRaisesRegex(ValueError, "cannot be inside the raw-real episode"):
                convert_raw_real_to_processed(raw_episode, output_under_raw, overwrite=True)

            self.assertTrue(output_under_raw.is_dir())
            self.assertFalse((output_under_raw / "metadata_processed.json").exists())

    def test_cli_fails_when_output_is_under_raw_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_episode = Path(tmpdir) / "raw_real" / "episode_000000"
            output_under_raw = raw_episode / "processed_inside_raw"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            env = dict(os.environ)
            src_path = Path(__file__).resolve().parents[1] / "src"
            env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "doosan_forcevla_data.convert.raw_real_to_processed",
                    "--raw-real",
                    str(raw_episode),
                    "--output",
                    str(output_under_raw),
                ],
                check=False,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("cannot be inside the raw-real episode", completed.stdout)
            self.assertFalse(output_under_raw.exists())

    def test_normal_output_sibling_directory_still_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4)

            convert_raw_real_to_processed(raw_episode, processed_episode)

            self.assertTrue((processed_episode / "metadata_processed.json").is_file())
            self.assertTrue(validate_processed_episode(processed_episode).ok)

    def test_cli_smoke_converts_and_validates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4)

            env = dict(os.environ)
            src_path = Path(__file__).resolve().parents[1] / "src"
            env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "doosan_forcevla_data.convert.raw_real_to_processed",
                    "--raw-real",
                    str(raw_episode),
                    "--output",
                    str(processed_episode),
                    "--overwrite",
                ],
                check=False,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertIn("OK: wrote processed episode:", completed.stdout)
            result = validate_processed_episode(processed_episode)
            self.assertTrue(result.ok, result.errors)

    def test_command_context_is_not_used_as_action_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=5, include_optional_streams=True)
            command_path = raw_episode / "streams" / "command_context.jsonl"
            command_records = _read_jsonl(command_path)
            for record in command_records:
                record["commanded_twist"] = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
                record["action_label"] = [9.0] * ACTION_DIM
            _write_jsonl(command_path, command_records)

            convert_raw_real_to_processed(
                raw_episode,
                processed_episode,
                include_optional_debug=True,
            )

            metadata = _read_json(processed_episode / "metadata_processed.json")
            self.assertEqual(metadata["action_label_primary"], "measured_tcp_delta")
            self.assertFalse(metadata["optional_debug"]["command_context"]["used_as_action_label"])

            frames = _read_jsonl(processed_episode / "frames.jsonl")
            for frame in frames[:-1]:
                self.assertNotEqual(frame["measured_action"], [9.0] * ACTION_DIM)
                self.assertNotEqual(frame["measured_action"][:6], [9.0, 8.0, 7.0, 6.0, 5.0, 4.0])


    def test_non_synthetic_numeric_source_stamp_without_timebase_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, source_stamp_unit=None)

            with self.assertRaisesRegex(ValueError, "source_stamp unit/timebase"):
                convert_raw_real_to_processed(raw_episode, processed_episode)

            self.assertFalse(processed_episode.exists())

    def test_non_synthetic_numeric_source_stamp_without_timebase_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            sentinel = processed_episode / "sentinel.txt"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, source_stamp_unit=None)
            processed_episode.mkdir(parents=True)
            sentinel.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source_stamp unit/timebase"):
                convert_raw_real_to_processed(raw_episode, processed_episode, overwrite=True)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")


    def test_non_synthetic_selected_wrench_missing_metadata_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, include_wrench_metadata=False)

            with self.assertRaisesRegex(ValueError, "wrench metadata"):
                convert_raw_real_to_processed(raw_episode, processed_episode)

            self.assertFalse(processed_episode.exists())

    def test_non_synthetic_selected_wrench_missing_metadata_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            sentinel = processed_episode / "sentinel.txt"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, include_wrench_metadata=False)
            processed_episode.mkdir(parents=True)
            sentinel.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "wrench metadata"):
                convert_raw_real_to_processed(raw_episode, processed_episode, overwrite=True)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_non_synthetic_without_explicit_tcp_orientation_convention_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, convention=None)

            with self.assertRaisesRegex(ValueError, "tcp_orientation_convention"):
                convert_raw_real_to_processed(raw_episode, processed_episode)

    def test_non_synthetic_without_strict_lab_provenance_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, strict_lab_provenance=False)

            with self.assertRaisesRegex(ValueError, "strict lab provenance"):
                convert_raw_real_to_processed(raw_episode, processed_episode)

            self.assertFalse(processed_episode.exists())

    def test_non_synthetic_without_strict_lab_provenance_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            sentinel = processed_episode / "sentinel.txt"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, strict_lab_provenance=False)
            processed_episode.mkdir(parents=True)
            sentinel.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "strict lab provenance"):
                convert_raw_real_to_processed(raw_episode, processed_episode, overwrite=True)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")


    def test_substring_synthetic_collection_method_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(
                raw_episode,
                convention=None,
                collection_method="non_synthetic_lab_capture",
            )

            with self.assertRaisesRegex(ValueError, "tcp_orientation_convention"):
                convert_raw_real_to_processed(raw_episode, processed_episode)

            self.assertFalse(processed_episode.exists())

    def test_substring_synthetic_collection_method_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            sentinel = processed_episode / "sentinel.txt"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(
                raw_episode,
                convention=None,
                collection_method="non_synthetic_lab_capture",
            )
            processed_episode.mkdir(parents=True)
            sentinel.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "tcp_orientation_convention"):
                convert_raw_real_to_processed(raw_episode, processed_episode, overwrite=True)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")


    def test_non_synthetic_doosan_euler_convention_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, convention="doosan_posx_euler_zyz_degrees")

            with self.assertRaises(ValueError) as context:
                convert_raw_real_to_processed(raw_episode, processed_episode)

            message = str(context.exception)
            self.assertIn("recognized but unsupported for conversion", message)
            self.assertIn("Doosan native Euler ZYZ", message)
            self.assertNotIn("produced invalid quaternion", message)
            self.assertFalse(processed_episode.exists())

    def test_non_synthetic_doosan_euler_convention_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            sentinel = processed_episode / "sentinel.txt"

            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, convention="euler_zyz_degrees")
            processed_episode.mkdir(parents=True)
            sentinel.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "recognized but unsupported"):
                convert_raw_real_to_processed(raw_episode, processed_episode, overwrite=True)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_non_synthetic_missing_robot_units_is_blocked_before_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode)

            index_path = raw_episode / "streams" / "index.json"
            index = _read_json(index_path)
            index["streams"]["robot_state_rt"].pop("units", None)
            _write_json(index_path, index)
            robot_path = raw_episode / "streams" / "robot_state_rt.jsonl"
            robot_records = _read_jsonl(robot_path)
            for record in robot_records:
                record.pop("units", None)
            _write_jsonl(robot_path, robot_records)

            validation = validate_raw_real_episode(raw_episode)
            report = inspect_raw_real_episode(raw_episode)
            self.assertFalse(validation.ok)
            self.assertFalse(report["ready_for_conversion"])
            self.assertTrue(any("tcp_position unit" in error for error in validation.errors))
            with self.assertRaisesRegex(ValueError, "tcp_position unit"):
                convert_raw_real_to_processed(raw_episode, processed_episode)
            self.assertFalse(processed_episode.exists())

    def test_non_synthetic_valid_explicit_units_and_convention_convert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, convention="rotation_vector_degrees")

            validation = validate_raw_real_episode(raw_episode)
            report = inspect_raw_real_episode(raw_episode)
            self.assertTrue(validation.ok, validation.errors)
            self.assertTrue(report["ready_for_conversion"], report["errors"])

            convert_raw_real_to_processed(raw_episode, processed_episode)

            self.assertTrue(validate_processed_episode(processed_episode).ok)

    def test_joint_states_fallback_converts_when_robot_joint_vectors_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            _mark_non_synthetic(raw_episode, convention="rotation_vector_degrees")

            robot_path = raw_episode / "streams" / "robot_state_rt.jsonl"
            robot_records = _read_jsonl(robot_path)
            for record in robot_records:
                record.pop("actual_joint_position", None)
                record.pop("actual_joint_velocity", None)
                record["units"].pop("joint_position", None)
                record["units"].pop("joint_velocity", None)
            _write_jsonl(robot_path, robot_records)

            validation = validate_raw_real_episode(raw_episode)
            report = inspect_raw_real_episode(raw_episode)
            self.assertTrue(validation.ok, validation.errors)
            self.assertTrue(report["ready_for_conversion"], report["errors"])

            convert_raw_real_to_processed(raw_episode, processed_episode)
            metadata = _read_json(processed_episode / "metadata_processed.json")
            self.assertEqual(metadata["joint_source"], ["joint_states.position; joint_states.velocity"])
            self.assertTrue(validate_processed_episode(processed_episode).ok)


if __name__ == "__main__":
    unittest.main()
