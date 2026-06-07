import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.validate.validate_raw_real_episode import validate_raw_real_episode


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _stamp(index: int, receipt: float | None = None, monotonic: float | None = None) -> dict:
    return {
        "record_index": index,
        "source_stamp": {"sec": 100 + index, "nanosec": 10},
        "receipt_stamp": 1000.0 + index if receipt is None else receipt,
        "monotonic_stamp": 2000.0 + index if monotonic is None else monotonic,
    }


def _joint_record(index: int = 0) -> dict:
    return {
        **_stamp(index),
        "joint_names": [f"joint_{idx}" for idx in range(6)],
        "position": [0.1 * idx for idx in range(6)],
        "velocity": [0.01 * idx for idx in range(6)],
    }


def _robot_state_record(index: int = 0) -> dict:
    return {
        **_stamp(index),
        "actual_tcp_position": [450.0, 0.0, 220.0, 0.0, 180.0, 0.0],
        "actual_joint_position": [1.0 * idx for idx in range(6)],
        "actual_joint_velocity": [0.1 * idx for idx in range(6)],
        "external_tcp_force": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "robot_mode": "manual",
        "robot_state": "running",
        "control_mode": "position",
    }


def _tf_record(index: int = 0, transforms: list | None = None) -> dict:
    return {**_stamp(index), "transforms": [] if transforms is None else transforms}


def _camera_record(stream_name: str, index: int = 0) -> dict:
    image_path = f"streams/{stream_name}/frames/{index:06d}.raw"
    return {
        **_stamp(index),
        "image_path": image_path,
        "width": 2,
        "height": 2,
        "channels": 3,
        "encoding": "rgb8",
        "frame_id": f"{stream_name}_optical_frame",
    }


def _stream_entry(path: str, source_name: str, source_type: str, record_count: int, required: bool = True) -> dict:
    return {
        "path": path,
        "source_name": source_name,
        "source_type": source_type,
        "required": required,
        "record_count": record_count,
        "verified": False,
    }


def _build_valid_episode(root: Path) -> Path:
    episode = root / "episode_000000"
    streams_dir = episode / "streams"

    _write_json(
        episode / "metadata.json",
        {
            "schema_version": "raw_real_v0",
            "episode_id": "episode_000000",
            "task_instruction": "Insert the peg into the hole.",
            "geometry_type": "round_peg_round_hole",
            "orientation_type": "vertical_insertion",
            "collection_method": "synthetic_raw_real_fixture",
            "action_label_primary": "measured_tcp_delta",
            "success": True,
            "failure_reason": None,
            "fps": 30,
            "robot_type": "doosan_m1013",
            "recorder_version": "test-recorder-0",
            "source_workspace": {"path": "synthetic", "verified": False},
        },
    )
    _write_json(episode / "calibration_refs.json", {"schema_version": "calibration_refs_v0"})
    _write_json(episode / "recorder_report.json", {"schema_version": "recorder_report_v0"})
    _write_jsonl(
        episode / "events.jsonl",
        [
            {"timestamp": 0.0, "event": "episode_start"},
            {"timestamp": 1.0, "event": "success"},
        ],
    )

    streams = {
        "joint_states": {
            **_stream_entry("streams/joint_states.jsonl", "/test/joint_states", "sensor_msgs/msg/JointState", 1),
            "units": {"position": "radians", "velocity": "radians_per_second"},
        },
        "robot_state_rt": _stream_entry(
            "streams/robot_state_rt.jsonl",
            "/test/realtime/read_data_rt",
            "dsr_msgs2/msg/RobotStateRt",
            1,
        ),
        "tf": _stream_entry("streams/tf.jsonl", "/tf", "tf2_msgs/msg/TFMessage", 1),
        "tf_static": _stream_entry("streams/tf_static.jsonl", "/tf_static", "tf2_msgs/msg/TFMessage", 0),
        "external_camera": _stream_entry(
            "streams/external_camera", "unknown_external_camera", "sensor_msgs/msg/Image", 1
        ),
        "wrist_camera": _stream_entry("streams/wrist_camera", "unknown_wrist_camera", "sensor_msgs/msg/Image", 1),
    }
    _write_json(episode / "streams" / "index.json", {"schema_version": "raw_real_v0", "streams": streams})

    _write_jsonl(streams_dir / "joint_states.jsonl", [_joint_record()])
    _write_jsonl(streams_dir / "robot_state_rt.jsonl", [_robot_state_record()])
    _write_jsonl(streams_dir / "tf.jsonl", [_tf_record()])
    _write_jsonl(streams_dir / "tf_static.jsonl", [])

    for stream_name in ["external_camera", "wrist_camera"]:
        image_path = episode / "streams" / stream_name / "frames" / "000000.raw"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"dummy image bytes")
        _write_jsonl(episode / "streams" / stream_name / "index.jsonl", [_camera_record(stream_name)])

    return episode


def _read_stream_index(episode: Path) -> dict:
    return json.loads((episode / "streams" / "index.json").read_text(encoding="utf-8"))


class ValidateRawRealEpisodeTests(unittest.TestCase):
    def test_valid_minimal_raw_real_episode_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)
            self.assertEqual(result.errors, [])

    def test_cli_valid_episode_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            env = dict(os.environ)
            src_path = Path(__file__).resolve().parents[1] / "src"
            env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "doosan_forcevla_data.validate.validate_raw_real_episode",
                    str(episode),
                ],
                check=False,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertIn("OK: raw real episode is valid:", completed.stdout)

    def test_missing_metadata_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            (episode / "metadata.json").unlink()

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("metadata.json" in error for error in result.errors))

    def test_missing_required_stream_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            index = _read_stream_index(episode)
            del index["streams"]["robot_state_rt"]
            _write_json(episode / "streams" / "index.json", index)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("required stream is missing: robot_state_rt" in error for error in result.errors))

    def test_camera_index_missing_image_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            (episode / "streams" / "external_camera" / "frames" / "000000.raw").unlink()

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("image_path does not exist" in error for error in result.errors))

    def test_non_monotonic_timestamp_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            first = _joint_record(0)
            second = _joint_record(1)
            second["receipt_stamp"] = first["receipt_stamp"] - 0.5
            _write_jsonl(episode / "streams" / "joint_states.jsonl", [first, second])

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("receipt_stamp must be monotonic nondecreasing" in error for error in result.errors))

    def test_non_finite_robot_state_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            record = _robot_state_record()
            record["actual_joint_position"][2] = math.inf
            _write_jsonl(episode / "streams" / "robot_state_rt.jsonl", [record])

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("actual_joint_position[2] must be a finite number" in error for error in result.errors))

    def test_wrong_joint_vector_length_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            record = _joint_record()
            record["position"] = [0.0] * 5
            _write_jsonl(episode / "streams" / "joint_states.jsonl", [record])

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("position length must be 6" in error for error in result.errors))

    def test_missing_robot_state_tcp_pose_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            record = _robot_state_record()
            del record["actual_tcp_position"]
            _write_jsonl(episode / "streams" / "robot_state_rt.jsonl", [record])

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("actual_tcp_position must be a list of length 6" in error for error in result.errors))

    def test_optional_streams_absent_only_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)
            self.assertTrue(any("optional stream command_context is absent" in warning for warning in result.warnings))
            self.assertTrue(any("optional stream gripper_state is absent" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
