import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.processed_to_lerobot_v21 import (
    VIDEO_COLUMNS,
    _rotvec_to_quat_wxyz,
    export_processed_to_lerobot_v21,
)
from doosan_forcevla_data.inspect.lerobot_v21_doctor import check_lerobot_v21_dependencies
from doosan_forcevla_data.inspect.inspect_lerobot_v21 import summarize_lerobot_v21
from doosan_forcevla_data.validate.validate_lerobot_v21 import validate_lerobot_v21


def _deps_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_ppm(path: Path, rgb: tuple[int, int, int], width: int = 16, height: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + bytes(rgb) * width * height)


def _state(frame_index: int) -> list[float]:
    return (
        [0.1 + frame_index * 0.01, 0.2, 0.3]
        + [0.0, 0.0, frame_index * 0.05]
        + [0.04]
        + [1.0 + frame_index, 2.0, 3.0, 0.1, 0.2, 0.3]
        + [0.01 * (joint + frame_index) for joint in range(6)]
        + [0.001 * (joint + frame_index) for joint in range(6)]
    )


def _make_processed_episode(root: Path, frame_count: int = 4, task: str = "synthetic task") -> None:
    metadata = {
        "source_raw_episode": str(root / "raw_source"),
        "processed_metadata_schema_version": "processed_jsonl_v1",
        "dataset_name": "doosan_peg_in_hole_v0",
        "robot_type": "Doosan M1013",
        "fps": 10.0,
        "quaternion_convention": "xyzw",
        "model_state_dim": 25,
        "action_dim": 7,
        "action_label_primary": "measured_tcp_delta",
        "frame_count": frame_count,
        "task_instruction": task,
        "geometry_type": "cartesian_tcp_pose_6d",
        "orientation_type": "rotation_vector_radians",
        "collection_method": "synthetic_test",
        "success": True,
        "failure_reason": None,
        "notes": "synthetic test fixture",
        "gripper_state_is_placeholder": True,
        "processed_camera_streams": {
            "external_camera_1": {"width": 16, "height": 16, "channels": 3},
            "external_camera_2": {"width": 16, "height": 16, "channels": 3},
            "tcp_camera": {"width": 16, "height": 16, "channels": 3},
        },
    }
    rows = []
    for idx in range(frame_count):
        for camera_idx, camera_name in enumerate(["external_camera_1", "external_camera_2", "tcp_camera"]):
            _write_ppm(
                root / "images" / camera_name / f"{idx:06d}.ppm",
                ((idx * 30 + camera_idx * 10) % 255, (idx * 20) % 255, (camera_idx * 80) % 255),
            )
        action = [0.001 * idx, 0.002 * idx, 0.003 * idx, 0.01 * idx, 0.02 * idx, 0.03 * idx, 0.0]
        terminal = idx == frame_count - 1
        if terminal:
            action = [0.0] * 7
        rows.append(
            {
                "frame_index": idx,
                "timestamp": idx * 0.1 + (0.001 if idx == 1 else 0.0),
                "external_rgb_path": f"images/external_camera_1/{idx:06d}.ppm",
                "external_camera_2_rgb_path": f"images/external_camera_2/{idx:06d}.ppm",
                "tcp_rgb_path": f"images/tcp_camera/{idx:06d}.ppm",
                "model_state": _state(idx),
                "measured_action": action,
                "action_is_terminal_padding": terminal,
            }
        )
    _write_json(root / "metadata_processed.json", metadata)
    _write_jsonl(root / "frames.jsonl", rows)


class ProcessedToLeRobotV21Tests(unittest.TestCase):
    def test_rotvec_to_quat_wxyz(self):
        quat = _rotvec_to_quat_wxyz([0.0, 0.0, math.pi / 2.0])
        self.assertAlmostEqual(quat[0], math.cos(math.pi / 4.0), places=7)
        self.assertAlmostEqual(quat[1], 0.0, places=7)
        self.assertAlmostEqual(quat[2], 0.0, places=7)
        self.assertAlmostEqual(quat[3], math.sin(math.pi / 4.0), places=7)

    def test_doctor_reports_without_forbidden_imports(self):
        report = check_lerobot_v21_dependencies()
        self.assertIn("pyarrow", report)
        self.assertFalse(report["forbidden_runtime_dependencies"]["lerobot_imported"])
        self.assertFalse(report["forbidden_runtime_dependencies"]["cv2_imported"])

    @unittest.skipUnless(_deps_available(), "requires pyarrow, ffmpeg, and ffprobe")
    def test_export_two_processed_episodes(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ep0 = root / "processed" / "episode_000000"
            ep1 = root / "processed" / "episode_000001"
            _make_processed_episode(ep0, task="task A")
            _make_processed_episode(ep1, task="task B")
            output = root / "lerobot_v21" / "doosan"

            export_processed_to_lerobot_v21([ep0, ep1], output)

            result = validate_lerobot_v21(output)
            self.assertTrue(result.ok, result.errors)
            summary = summarize_lerobot_v21(output)
            self.assertEqual(summary["total_episodes"], 2)
            self.assertEqual(summary["total_frames"], 8)
            self.assertEqual(summary["total_tasks"], 2)
            self.assertEqual(summary["total_videos"], 6)
            self.assertEqual(summary["feature_keys"].count("observation.images.center"), 1)

            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["robot_type"], "Doosan M1013")
            self.assertEqual(info["fps"], 10.0)
            self.assertEqual(info["features"]["observation.images.center"]["shape"], [16, 16, 3])
            self.assertEqual(info["total_videos"], 2 * len(VIDEO_COLUMNS))

            table = pq.read_table(output / "data" / "chunk-000" / "episode_000000.parquet")
            self.assertEqual(table.num_rows, 4)
            self.assertNotIn("observation.images.center", table.column_names)
            row0 = table.slice(0, 1).to_pylist()[0]
            self.assertEqual(row0["timestamp"], 0.0)
            self.assertTrue(all(abs(value - 0.04) < 1e-6 for value in row0["observation.state.gripper_pos"]))
            self.assertEqual(row0["episode_index"], 0)
            self.assertEqual(row0["task_index"], 0)
            row3 = table.slice(3, 1).to_pylist()[0]
            self.assertEqual(row3["action"], [0.0] * 7)

            provenance = json.loads((output / "meta" / "export_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["target_loader_reference"]["forcevla_v5_config"], "forcevla_sfp_all_trimmed_v5")
            self.assertTrue(provenance["terminal_action_policy"]["final_zero_terminal_action_retained"])
            self.assertIn("quaternion_order", provenance["guide_conflicts"][0]["topic"])

    @unittest.skipUnless(_deps_available(), "requires pyarrow, ffmpeg, and ffprobe")
    def test_refuses_nonempty_output_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = root / "processed" / "episode_000000"
            _make_processed_episode(episode)
            output = root / "existing"
            output.mkdir()
            (output / "keep.txt").write_text("do not overwrite\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_processed_to_lerobot_v21(episode, output)


if __name__ == "__main__":
    unittest.main()
