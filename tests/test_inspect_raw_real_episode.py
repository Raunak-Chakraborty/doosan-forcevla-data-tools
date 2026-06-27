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


def _shift_camera_source_stamps(episode: Path, offset_sec: float) -> None:
    for stream_name in ["external_camera", "wrist_camera"]:
        index_path = episode / "streams" / stream_name / "index.jsonl"
        records = _read_jsonl(index_path)
        for record in records:
            record["source_stamp"] = float(record["source_stamp"]) + offset_sec
        _write_jsonl(index_path, records)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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
    mark_gripper_real: bool = True,
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
    gripper_entry = streams_index["streams"].get("gripper_state")
    if mark_gripper_real and isinstance(gripper_entry, dict):
        gripper_entry["source_name"] = "/verified/gripper_state"
        gripper_entry["source_type"] = "verified_lab_gripper"
        gripper_entry.pop("placeholder", None)
        gripper_entry.pop("synthetic_placeholder", None)
    _write_json(streams_index_path, streams_index)

    gripper_path = episode / "streams" / "gripper_state.jsonl"
    if mark_gripper_real and gripper_path.is_file():
        gripper_records = _read_jsonl(gripper_path)
        for record in gripper_records:
            record["source_name"] = "/verified/gripper_state"
            record["source_type"] = "verified_lab_gripper"
            record.pop("placeholder", None)
        _write_jsonl(gripper_path, gripper_records)


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

    def test_non_synthetic_corrupt_camera_image_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            (episode / "streams" / "external_camera" / "frames" / "000000.ppm").write_bytes(b"not an image")

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertFalse(report["validation"]["ok"])
            self.assertTrue(
                any(
                    "external_camera camera record 0" in blocker
                    and "not decodable" in blocker
                    for blocker in report["conversion_blockers"]
                )
            )

    def test_non_synthetic_camera_dimension_mismatch_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            index_path = episode / "streams" / "wrist_camera" / "index.jsonl"
            records = _read_jsonl(index_path)
            records[0]["height"] = 3
            _write_jsonl(index_path, records)

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertFalse(report["validation"]["ok"])
            self.assertTrue(
                any(
                    "wrist_camera camera record 0" in blocker
                    and "decoded image dimensions do not match declared metadata" in blocker
                    and "declared 2x3x3, decoded 2x2x3" in blocker
                    for blocker in report["conversion_blockers"]
                )
            )

    def test_non_synthetic_missing_calibration_ref_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            calibration_path = episode / "calibration_refs.json"
            calibration_refs = _read_json(calibration_path)
            del calibration_refs["camera_extrinsics"]["wrist_camera"]
            _write_json(calibration_path, calibration_refs)

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertFalse(report["validation"]["ok"])
            self.assertTrue(
                any(
                    "calibration_refs.camera_extrinsics.wrist_camera is required" in blocker
                    for blocker in report["conversion_blockers"]
                )
            )

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

    def test_non_synthetic_missing_robot_units_blocks_readiness(self):
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

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertFalse(report["validation"]["ok"])
            self.assertTrue(any("tcp_position unit" in error for error in report["errors"]))
            self.assertTrue(any("tcp_position unit" in blocker for blocker in report["conversion_blockers"]))

    def test_non_synthetic_missing_gripper_state_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4, include_optional_streams=False)
            _mark_non_synthetic(episode)

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertFalse(report["validation"]["ok"])
            self.assertTrue(
                any(
                    "gripper_state is required for non-synthetic conversion" in blocker
                    and "gripper_pos=0.0" in blocker
                    for blocker in report["conversion_blockers"]
                )
            )

    def test_non_synthetic_unsupported_tcp_units_block_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cases = [("tcp_position", "inch"), ("tcp_orientation", "turns")]
            for unit_key, unit_value in cases:
                with self.subTest(unit=unit_key):
                    episode = Path(tmpdir) / f"episode_{unit_key}"
                    make_synthetic_raw_real_episode(episode, frame_count=4)
                    _mark_non_synthetic(episode)
                    robot_path = episode / "streams" / "robot_state_rt.jsonl"
                    robot_records = _read_jsonl(robot_path)
                    for record in robot_records:
                        record["units"][unit_key] = unit_value
                    _write_jsonl(robot_path, robot_records)

                    report = inspect_raw_real_episode(episode)

                    self.assertFalse(report["ready_for_conversion"])
                    self.assertTrue(any(unit_key in blocker for blocker in report["conversion_blockers"]))

    def test_non_synthetic_explicit_units_and_convention_are_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4, include_optional_streams=True)
            _mark_non_synthetic(episode, convention="rotation_vector_degrees")

            report = inspect_raw_real_episode(episode)

            self.assertTrue(report["validation"]["ok"], report["validation"]["errors"])
            self.assertTrue(report["ready_for_conversion"], report["errors"])
            self.assertEqual(report["conversion_readiness_errors"], [])

    def test_non_synthetic_without_strict_lab_provenance_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, strict_lab_provenance=False)

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertTrue(any("strict lab provenance" in error for error in report["errors"]))
            self.assertTrue(any("strict lab provenance" in blocker for blocker in report["conversion_blockers"]))


    def test_non_synthetic_numeric_source_stamp_without_timebase_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, source_stamp_unit=None)

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertTrue(
                any("source_stamp" in blocker and "timebase" in blocker for blocker in report["conversion_blockers"])
            )
            self.assertTrue(any("source_stamp" in error and "timebase" in error for error in report["errors"]))


    def test_non_synthetic_selected_wrench_missing_metadata_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, include_wrench_metadata=False)

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertTrue(any("wrench metadata" in error for error in report["errors"]))
            self.assertTrue(any("wrench metadata" in blocker for blocker in report["conversion_blockers"]))

    def test_non_synthetic_verified_boolean_without_convention_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, convention=None)
            metadata_path = episode / "metadata.json"
            metadata = _read_json(metadata_path)
            metadata["tcp_orientation_convention_verified"] = True
            _write_json(metadata_path, metadata)

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertTrue(any("tcp_orientation_convention" in error for error in report["errors"]))


    def test_substring_synthetic_collection_method_is_not_ready_without_real_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(
                episode,
                convention=None,
                collection_method="non_synthetic_lab_capture",
            )

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertFalse(report["robot_state_summary"]["orientation_convention_ready"])
            self.assertTrue(any("TCP orientation convention" in blocker for blocker in report["conversion_blockers"]))
            self.assertTrue(any("tcp_orientation_convention" in error for error in report["errors"]))


    def test_non_synthetic_doosan_euler_convention_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode, convention="doosan_robotstate_actual_tcp_position_euler_zyz_degrees")

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertFalse(report["robot_state_summary"]["orientation_convention_ready"])
            self.assertIn("recognized but unsupported", report["robot_state_summary"]["orientation_convention_guard"])
            self.assertTrue(any("Doosan native Euler ZYZ" in error for error in report["errors"]))
            self.assertTrue(any("recognized but unsupported" in blocker for blocker in report["conversion_blockers"]))

    def test_joint_states_fallback_can_make_non_synthetic_episode_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4, include_optional_streams=True)
            _mark_non_synthetic(episode)
            robot_path = episode / "streams" / "robot_state_rt.jsonl"
            robot_records = _read_jsonl(robot_path)
            for record in robot_records:
                record.pop("actual_joint_position", None)
                record.pop("actual_joint_velocity", None)
                record["units"].pop("joint_position", None)
                record["units"].pop("joint_velocity", None)
            _write_jsonl(robot_path, robot_records)

            report = inspect_raw_real_episode(episode)

            self.assertTrue(report["validation"]["ok"], report["validation"]["errors"])
            self.assertTrue(report["ready_for_conversion"], report["errors"])

    def test_huge_camera_source_stamp_offset_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4)
            _mark_non_synthetic(episode)
            _shift_camera_source_stamps(episode, 999.0)

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertTrue(any("source_stamp differs from robot_state_rt" in blocker for blocker in report["conversion_blockers"]))
            self.assertTrue(any("source_stamp synchronization" in recommendation for recommendation in report["recommendations"]))

    def test_camera_source_stamp_offset_above_half_frame_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4, fps=30.0)
            _mark_non_synthetic(episode)
            _shift_camera_source_stamps(episode, 0.05)

            report = inspect_raw_real_episode(episode)

            self.assertFalse(report["ready_for_conversion"])
            self.assertTrue(any("by 0.050000s" in error for error in report["errors"]))
            self.assertTrue(any("allowed camera/robot source_stamp offset is 0.016667s" in blocker for blocker in report["conversion_blockers"]))

    def test_small_camera_source_stamp_jitter_does_not_block_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode, frame_count=4, fps=30.0, include_optional_streams=True)
            _mark_non_synthetic(episode)
            _shift_camera_source_stamps(episode, 0.01)

            report = inspect_raw_real_episode(episode)

            self.assertTrue(report["validation"]["ok"], report["validation"]["errors"])
            self.assertTrue(report["ready_for_conversion"], report["errors"])

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
