import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.dummy.make_synthetic_raw_real_episode import make_synthetic_raw_real_episode
from doosan_forcevla_data.inspect.inspect_raw_real_episode import inspect_raw_real_episode
from doosan_forcevla_data.validate.validate_raw_real_episode import validate_raw_real_episode


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _mark_non_synthetic_with_required_units(episode: Path) -> None:
    metadata_path = episode / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata["collection_method"] = "passive_real_recorder"
    metadata["recorder_version"] = "passive_real_recorder_v0"
    metadata["tcp_orientation_convention"] = "rotation_vector_degrees"
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
    _write_json(metadata_path, metadata)

    recorder_report_path = episode / "recorder_report.json"
    recorder_report = _read_json(recorder_report_path)
    recorder_report["synthetic"] = False
    recorder_report["generator"] = "passive_real_recorder"
    recorder_report["generator_version"] = "passive_real_recorder_v0"
    recorder_report["tcp_orientation_convention"] = "rotation_vector_degrees"
    _write_json(recorder_report_path, recorder_report)

    index_path = episode / "streams" / "index.json"
    index = _read_json(index_path)
    index["synthetic"] = False
    index["timebase"] = {"source_stamp_unit": "seconds"}

    source_names = {
        "joint_states": "/joint_states",
        "robot_state_rt": "/dsr01/dsr_controller2/realtime/read_data_rt",
        "tf": "/tf",
        "tf_static": "/tf_static",
        "external_camera": "/external_camera/color/image_raw",
        "wrist_camera": "/wrist_camera/color/image_raw",
        "command_context": "/doosan_teleop/cmd_vel_6d",
        "gripper_state": "/gripper_state",
    }

    for stream_name, entry in index.get("streams", {}).items():
        if isinstance(entry, dict):
            entry["verified"] = True
            entry["source_name"] = source_names.get(stream_name, f"/verified/{stream_name}")

    _write_json(index_path, index)


class StrictLabProvenanceReadinessTests(unittest.TestCase):
    def test_strict_lab_provenance_verified_real_episode_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic_with_required_units(episode)

            result = validate_raw_real_episode(episode)
            report = inspect_raw_real_episode(episode)

            self.assertTrue(result.ok, result.errors)
            self.assertTrue(report["ready_for_conversion"], report["errors"])

    def test_strict_lab_provenance_unknown_stream_source_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic_with_required_units(episode)

            index_path = episode / "streams" / "index.json"
            index = _read_json(index_path)
            index["streams"]["external_camera"]["source_name"] = "unknown_external_camera"
            _write_json(index_path, index)

            result = validate_raw_real_episode(episode)
            report = inspect_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("stream external_camera source_name must be known" in e for e in result.errors))
            self.assertFalse(report["ready_for_conversion"])

    def test_strict_lab_provenance_unverified_stream_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic_with_required_units(episode)

            index_path = episode / "streams" / "index.json"
            index = _read_json(index_path)
            index["streams"]["wrist_camera"]["verified"] = False
            _write_json(index_path, index)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("stream wrist_camera verified must be true" in e for e in result.errors))

    def test_strict_lab_provenance_unknown_live_graph_frame_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic_with_required_units(episode)

            metadata_path = episode / "metadata.json"
            metadata = _read_json(metadata_path)
            metadata["live_graph_verification"]["tcp_frame"] = "unknown"
            _write_json(metadata_path, metadata)

            result = validate_raw_real_episode(episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("live_graph_verification.tcp_frame must be known" in e for e in result.errors))


if __name__ == "__main__":
    unittest.main()
