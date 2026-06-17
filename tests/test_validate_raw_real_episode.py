import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.dummy.make_synthetic_raw_real_episode import make_synthetic_raw_real_episode
from doosan_forcevla_data.validate.validate_raw_real_episode import timestamp_tolerance_seconds, validate_raw_real_episode


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]




def _shift_camera_source_stamps(episode: Path, offset_sec: float) -> None:
    for stream_name in ["external_camera", "wrist_camera"]:
        index_path = episode / "streams" / stream_name / "index.jsonl"
        records = _read_jsonl(index_path)
        for record in records:
            record["source_stamp"] = float(record["source_stamp"]) + offset_sec
        _write_jsonl(index_path, records)


def _set_source_stamp_tolerance_override(episode: Path, value: object) -> None:
    streams_index_path = episode / "streams" / "index.json"
    streams_index = _read_json(streams_index_path)
    streams_index.setdefault("timebase", {})["max_camera_robot_source_stamp_offset_sec"] = value
    _write_json(streams_index_path, streams_index)

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



def _set_external_wrench_metadata_field(episode: Path, key: str, value) -> None:
    index_path = episode / "streams" / "index.json"
    index = _read_json(index_path)
    index["streams"]["robot_state_rt"]["wrench_sources"]["external_tcp_force"][key] = value
    _write_json(index_path, index)

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
        "tf_static": _stream_entry("streams/tf_static.jsonl", "/tf_static", "tf2_msgs/msg/TFMessage", 1),
        "external_camera": _stream_entry(
            "streams/external_camera", "unknown_external_camera", "sensor_msgs/msg/Image", 1
        ),
        "wrist_camera": _stream_entry("streams/wrist_camera", "unknown_wrist_camera", "sensor_msgs/msg/Image", 1),
    }
    _write_json(episode / "streams" / "index.json", {"schema_version": "raw_real_v0", "streams": streams})

    _write_jsonl(streams_dir / "joint_states.jsonl", [_joint_record()])
    _write_jsonl(streams_dir / "robot_state_rt.jsonl", [_robot_state_record()])
    _write_jsonl(streams_dir / "tf.jsonl", [_tf_record()])
    _write_jsonl(streams_dir / "tf_static.jsonl", [_tf_record()])

    for stream_name in ["external_camera", "wrist_camera"]:
        image_path = episode / "streams" / stream_name / "frames" / "000000.raw"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"dummy image bytes")
        _write_jsonl(episode / "streams" / stream_name / "index.jsonl", [_camera_record(stream_name)])

    return episode


def _read_stream_index(episode: Path) -> dict:
    return json.loads((episode / "streams" / "index.json").read_text(encoding="utf-8"))


class ValidateRawRealEpisodeTests(unittest.TestCase):

    def test_timestamp_tolerance_helper_uses_half_frame_default_and_conservative_fallback(self):
        self.assertAlmostEqual(timestamp_tolerance_seconds({"fps": 30.0}), 1.0 / 60.0)
        self.assertAlmostEqual(timestamp_tolerance_seconds(None), 0.02)
        self.assertAlmostEqual(
            timestamp_tolerance_seconds(
                {"fps": 30.0},
                {"timebase": {"max_camera_robot_source_stamp_offset_sec": 0.055}},
            ),
            0.055,
        )

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

    def test_stream_index_schema_version_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            index = _read_stream_index(episode)
            index["schema_version"] = "raw_real_future"
            _write_json(episode / "streams" / "index.json", index)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("schema_version must be 'raw_real_v0'" in error for error in result.errors))

    def test_required_stream_record_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            index = _read_stream_index(episode)
            index["streams"]["joint_states"]["record_count"] = 2
            _write_json(episode / "streams" / "index.json", index)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(
                any("record_count 2 does not match actual record count 1" in error for error in result.errors)
            )

    def test_empty_required_stream_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            _write_jsonl(episode / "streams" / "robot_state_rt.jsonl", [])

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(
                any("required stream robot_state_rt must contain at least one record" in error for error in result.errors)
            )

    def test_required_stream_record_index_alignment_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            index = _read_stream_index(episode)
            index["streams"]["robot_state_rt"]["record_count"] = 2
            _write_json(episode / "streams" / "index.json", index)
            _write_jsonl(
                episode / "streams" / "robot_state_rt.jsonl",
                [_robot_state_record(0), _robot_state_record(1)],
            )

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(
                any("joint_states: record_index alignment with robot_state_rt failed" in error for error in result.errors)
            )

    def test_camera_index_missing_image_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            (episode / "streams" / "external_camera" / "frames" / "000000.raw").unlink()

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("image_path does not exist" in error for error in result.errors))

    def test_camera_index_empty_image_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            (episode / "streams" / "external_camera" / "frames" / "000000.raw").write_bytes(b"")

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("image_path must reference a non-empty file" in error for error in result.errors))

    def test_camera_index_path_escape_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            record = _camera_record("external_camera")
            record["image_path"] = "../outside.raw"
            _write_jsonl(episode / "streams" / "external_camera" / "index.jsonl", [record])

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("image_path must be relative to episode root" in error for error in result.errors))


    def test_non_synthetic_corrupt_camera_image_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            (episode / "streams" / "external_camera" / "frames" / "000000.ppm").write_bytes(b"not an image")

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(
                any(
                    "external_camera camera record 0" in error
                    and "image_path streams/external_camera/frames/000000.ppm is not decodable" in error
                    for error in result.errors
                )
            )

    def test_non_synthetic_declared_camera_dimensions_must_match_decoded_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            index_path = episode / "streams" / "wrist_camera" / "index.jsonl"
            records = _read_jsonl(index_path)
            records[0]["width"] = 3
            _write_jsonl(index_path, records)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(
                any(
                    "wrist_camera camera record 0" in error
                    and "decoded image dimensions do not match declared metadata" in error
                    and "declared 3x2x3, decoded 2x2x3" in error
                    for error in result.errors
                )
            )

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
            record["actual_tcp_position"][2] = math.inf
            _write_jsonl(episode / "streams" / "robot_state_rt.jsonl", [record])

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("actual_tcp_position[2] must be a finite number" in error for error in result.errors))

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

    def test_empty_required_metadata_string_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            metadata["task_instruction"] = ""
            _write_json(episode / "metadata.json", metadata)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("task_instruction must be a non-empty string" in error for error in result.errors))

    def test_non_monotonic_event_timestamp_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            _write_jsonl(
                episode / "events.jsonl",
                [
                    {"timestamp": 1.0, "event": "episode_start"},
                    {"timestamp": 0.5, "event": "success"},
                ],
            )

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("event record 1: timestamp must be monotonic" in error for error in result.errors))

    def test_optional_gripper_state_invalid_value_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            index = _read_stream_index(episode)
            index["streams"]["gripper_state"] = _stream_entry(
                "streams/gripper_state.jsonl",
                "test_gripper",
                "diagnostic/gripper",
                1,
                required=False,
            )
            _write_json(episode / "streams" / "index.json", index)
            _write_jsonl(
                episode / "streams" / "gripper_state.jsonl",
                [{**_stamp(0), "gripper_width_m": -0.1}],
            )

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("gripper_width_m must be non-negative" in error for error in result.errors))

    def test_command_context_alignment_warning_not_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))
            index = _read_stream_index(episode)
            index["streams"]["command_context"] = _stream_entry(
                "streams/command_context.jsonl",
                "test_command_context",
                "diagnostic/command_context",
                2,
                required=False,
            )
            _write_json(episode / "streams" / "index.json", index)
            _write_jsonl(
                episode / "streams" / "command_context.jsonl",
                [
                    {**_stamp(0), "command_kind": "twist", "action_label": [1.0] * 7},
                    {**_stamp(1), "command_kind": "twist", "commanded_twist": [0.0] * 6},
                ],
            )

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)
            self.assertTrue(
                any("command_context: record_index differs from robot_state_rt" in warning for warning in result.warnings)
            )
            self.assertTrue(any("action-like fields are diagnostic only" in warning for warning in result.warnings))

    def test_optional_streams_absent_only_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = _build_valid_episode(Path(tmpdir))

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)
            self.assertTrue(any("optional stream command_context is absent" in warning for warning in result.warnings))
            self.assertTrue(any("optional stream gripper_state is absent" in warning for warning in result.warnings))

    def test_non_synthetic_missing_robot_state_units_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            index_path = episode / "streams" / "index.json"
            index = _read_json(index_path)
            index["streams"]["robot_state_rt"].pop("units", None)
            _write_json(index_path, index)
            robot_path = episode / "streams" / "robot_state_rt.jsonl"
            robot_records = _read_jsonl(robot_path)
            for record in robot_records:
                record.pop("units", None)
            _write_jsonl(robot_path, robot_records)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("actual_tcp_position: unsupported or missing tcp_position unit" in error for error in result.errors))

    def test_non_synthetic_unsupported_tcp_position_unit_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            robot_path = episode / "streams" / "robot_state_rt.jsonl"
            robot_records = _read_jsonl(robot_path)
            for record in robot_records:
                record["units"]["tcp_position"] = "inch"
            _write_jsonl(robot_path, robot_records)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("unsupported or missing tcp_position unit: 'inch'" in error for error in result.errors))

    def test_non_synthetic_unsupported_tcp_orientation_unit_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            robot_path = episode / "streams" / "robot_state_rt.jsonl"
            robot_records = _read_jsonl(robot_path)
            for record in robot_records:
                record["units"]["tcp_orientation"] = "turns"
            _write_jsonl(robot_path, robot_records)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("unsupported or missing tcp_orientation unit: 'turns'" in error for error in result.errors))

    def test_non_synthetic_explicit_valid_units_and_convention_passes_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, convention="rotation_vector_degrees")

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)

    def test_non_synthetic_without_strict_lab_provenance_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, strict_lab_provenance=False)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("strict lab provenance" in error for error in result.errors))


    def test_non_synthetic_numeric_source_stamp_without_timebase_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, source_stamp_unit=None)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("source_stamp unit/timebase" in error for error in result.errors))
            self.assertTrue(any("timebase.source_stamp_unit" in error for error in result.errors))

    def test_non_synthetic_unsupported_source_stamp_unit_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, source_stamp_unit="nanoseconds")

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("source_stamp unit/timebase" in error for error in result.errors))
            self.assertTrue(any("timebase.source_stamp_unit" in error for error in result.errors))


    def test_non_synthetic_invalid_selected_wrench_metadata_fails_validation(self):
        cases = [
            ("force_unit", "lbf", "force_unit"),
            ("torque_unit", "lbf_ft", "torque_unit"),
            ("frame", "unknown", "frame"),
            ("order", ["Fx", "Fy", "Fz", "Tz", "Ty", "Tx"], "order"),
            ("approved_for_model_state", False, "approved_for_model_state"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for key, value, expected_message in cases:
                with self.subTest(key=key):
                    episode = Path(tmpdir) / f"episode_{key}"
                    make_synthetic_raw_real_episode(episode, frame_count=4)
                    _mark_non_synthetic(episode)
                    _set_external_wrench_metadata_field(episode, key, value)

                    result = validate_raw_real_episode(episode)

                    self.assertFalse(result.ok)
                    self.assertTrue(any("wrench metadata" in error for error in result.errors))
                    self.assertTrue(any(expected_message in error for error in result.errors))


    def test_non_synthetic_rotation_vector_radians_passes_orientation_requirement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, convention="rotation_vector_radians")
            robot_path = episode / "streams" / "robot_state_rt.jsonl"
            robot_records = _read_jsonl(robot_path)
            for record in robot_records:
                record["units"]["tcp_orientation"] = "rad"
            _write_jsonl(robot_path, robot_records)

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)

    def test_non_synthetic_doosan_euler_conventions_fail_with_explicit_message(self):
        conventions = [
            "doosan_posx_euler_zyz_degrees",
            "doosan_robotstate_actual_tcp_position_euler_zyz_degrees",
            "euler_zyz_degrees",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, convention in enumerate(conventions):
                with self.subTest(convention=convention):
                    episode = Path(tmpdir) / f"episode_{idx:06d}"
                    make_synthetic_raw_real_episode(episode, frame_count=4)
                    _mark_non_synthetic(episode, convention=convention)

                    result = validate_raw_real_episode(episode)

                    self.assertFalse(result.ok)
                    self.assertTrue(any(convention in error for error in result.errors))
                    self.assertTrue(any("recognized but unsupported for conversion" in error for error in result.errors))
                    self.assertTrue(any("Doosan native Euler ZYZ" in error for error in result.errors))

    def test_non_synthetic_unknown_orientation_convention_lists_supported_and_unsupported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, convention="quaternion_xyzw")

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            message = "\n".join(result.errors)
            self.assertIn("unknown tcp_orientation_convention 'quaternion_xyzw'", message)
            self.assertIn("supported conversion conventions", message)
            self.assertIn("rotation_vector_degrees", message)
            self.assertIn("rotation_vector_radians", message)
            self.assertIn("recognized but unsupported Doosan/native conventions", message)
            self.assertIn("doosan_posx_euler_zyz_degrees", message)


    def test_non_synthetic_decodable_camera_images_with_matching_dimensions_pass_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, convention="rotation_vector_degrees")

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)
            self.assertFalse(any("not decodable" in error for error in result.errors))
            self.assertFalse(any("decoded image dimensions" in error for error in result.errors))

    def test_non_synthetic_missing_required_calibration_ref_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            calibration_path = episode / "calibration_refs.json"
            calibration_refs = _read_json(calibration_path)
            del calibration_refs["camera_intrinsics"]["external_camera"]
            _write_json(calibration_path, calibration_refs)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(
                any(
                    "calibration_refs.camera_intrinsics.external_camera is required" in error
                    for error in result.errors
                )
            )

    def test_non_synthetic_empty_or_unknown_calibration_ref_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            calibration_path = episode / "calibration_refs.json"
            calibration_refs = _read_json(calibration_path)
            calibration_refs["camera_intrinsics"]["external_camera"] = ""
            calibration_refs["tcp_tool_calibration"] = {"id": "unknown"}
            calibration_refs["force_torque_calibration"] = "unknown"
            _write_json(calibration_path, calibration_refs)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(
                any(
                    "calibration_refs.camera_intrinsics.external_camera must be a non-empty known" in error
                    for error in result.errors
                )
            )
            self.assertTrue(
                any("calibration_refs.tcp_tool_calibration must be a non-empty known" in error for error in result.errors)
            )
            self.assertTrue(
                any(
                    "calibration_refs.force_torque_calibration must be a non-empty known" in error
                    for error in result.errors
                )
            )

    def test_non_synthetic_verified_boolean_without_convention_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, convention=None)
            metadata_path = episode / "metadata.json"
            metadata = _read_json(metadata_path)
            metadata["tcp_orientation_convention_verified"] = True
            _write_json(metadata_path, metadata)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("tcp_orientation_convention must be one of" in error for error in result.errors))


    def test_substring_synthetic_names_do_not_bypass_non_synthetic_validation(self):
        cases = [
            ("non_synthetic_lab_capture", "passive_real_recorder_v0"),
            ("real_non_synthetic_test", "passive_real_recorder_v0"),
            ("passive_real_recorder", "non_synthetic_v1"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for case_idx, (collection_method, recorder_version) in enumerate(cases):
                with self.subTest(collection_method=collection_method, recorder_version=recorder_version):
                    episode = Path(tmpdir) / f"episode_{case_idx:06d}"
                    make_synthetic_raw_real_episode(episode, frame_count=4)
                    _mark_non_synthetic(
                        episode,
                        convention=None,
                        collection_method=collection_method,
                        recorder_version=recorder_version,
                    )

                    result = validate_raw_real_episode(episode)

                    self.assertFalse(result.ok)
                    self.assertTrue(any("tcp_orientation_convention must be one of" in error for error in result.errors))

    def test_joint_states_fallback_allows_missing_robot_state_joint_vectors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            robot_path = episode / "streams" / "robot_state_rt.jsonl"
            robot_records = _read_jsonl(robot_path)
            for record in robot_records:
                record.pop("actual_joint_position", None)
                record.pop("actual_joint_velocity", None)
                record["units"].pop("joint_position", None)
                record["units"].pop("joint_velocity", None)
            _write_jsonl(robot_path, robot_records)

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)

    def test_huge_camera_source_stamp_offset_fails_validation_even_when_indexes_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            for stream_name in ["external_camera", "wrist_camera"]:
                index_path = episode / "streams" / stream_name / "index.jsonl"
                records = _read_jsonl(index_path)
                for record in records:
                    record["source_stamp"] = float(record["source_stamp"]) + 999.0
                _write_jsonl(index_path, records)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("source_stamp differs from robot_state_rt" in error for error in result.errors))


    def test_camera_source_stamp_offset_above_half_frame_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4, fps=30.0)
            _mark_non_synthetic(episode)
            _shift_camera_source_stamps(episode, 0.05)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            message = "\n".join(result.errors)
            self.assertIn("source_stamp differs from robot_state_rt by 0.050000s", message)
            self.assertIn("allowed camera/robot source_stamp offset is 0.016667s", message)
            self.assertIn("default 0.5/fps from metadata.fps=30", message)

    def test_camera_source_stamp_tolerance_override_allows_bounded_offset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4, fps=30.0)
            _mark_non_synthetic(episode)
            _set_source_stamp_tolerance_override(episode, 0.055)
            _shift_camera_source_stamps(episode, 0.05)

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)
            self.assertTrue(any("max_camera_robot_source_stamp_offset_sec=0.055000s" in warning for warning in result.warnings))

    def test_camera_source_stamp_tolerance_override_must_be_positive_and_bounded(self):
        cases = [
            (0.0, "must be a finite positive number"),
            (0.08, "exceeds allowed maximum 0.066667s"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, (override, expected) in enumerate(cases):
                with self.subTest(override=override):
                    episode = Path(tmpdir) / f"episode_{idx:06d}"
                    make_synthetic_raw_real_episode(episode, frame_count=4, fps=30.0)
                    _mark_non_synthetic(episode)
                    _set_source_stamp_tolerance_override(episode, override)

                    result = validate_raw_real_episode(episode)

                    self.assertFalse(result.ok)
                    self.assertTrue(any(expected in error for error in result.errors))

    def test_small_camera_source_stamp_jitter_passes_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4, fps=30.0)
            _mark_non_synthetic(episode)
            for stream_name in ["external_camera", "wrist_camera"]:
                index_path = episode / "streams" / stream_name / "index.jsonl"
                records = _read_jsonl(index_path)
                for record in records:
                    record["source_stamp"] = float(record["source_stamp"]) + 0.01
                _write_jsonl(index_path, records)

            result = validate_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)


if __name__ == "__main__":
    unittest.main()
