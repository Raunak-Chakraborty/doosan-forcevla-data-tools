import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import doosan_forcevla_data.convert.write_real_lerobot_dataset_export as dataset_export_module
from doosan_forcevla_data.convert.plan_lerobot_export import write_lerobot_export_plan
from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.convert.stage_lerobot_export import stage_lerobot_export
from doosan_forcevla_data.convert.write_lerobot_dataset_skeleton import write_lerobot_dataset_skeleton
from doosan_forcevla_data.convert.write_real_lerobot_dataset_export import write_real_lerobot_dataset_export
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.validate.validate_real_lerobot_dataset_export_attempt import (
    validate_real_lerobot_dataset_export_attempt,
)


class WriteRealLeRobotDatasetExportTests(unittest.TestCase):
    def _build_multi_skeleton(self, root: Path, profile: str, image_mode: str) -> Path:
        staged_dirs = []
        for episode_index in range(2):
            episode_name = f"episode_{episode_index:06d}"
            raw_episode = root / "raw" / episode_name
            processed_episode = root / "processed" / episode_name
            staged_episode = root / "staged" / profile / episode_name

            make_dummy_raw_episode(raw_episode)
            convert_raw_to_processed(raw_episode, processed_episode)
            plan_path = processed_episode / f"export_plan_{profile}.json"
            write_lerobot_export_plan(processed_episode, profile, plan_path)
            stage_lerobot_export(processed_episode, plan_path, staged_episode)
            staged_dirs.append(staged_episode)

        skeleton = root / "lerobot_multi" / profile / "doosan_peg_in_hole_v0"
        write_lerobot_dataset_skeleton(
            staged_dirs,
            skeleton,
            profile=profile,
            image_mode=image_mode,
        )
        return skeleton

    def _read_report(self, path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_multi_forcevla_13d_dry_run_writes_report_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_multi_skeleton(root, "forcevla_13d", "symlink")
            output = root / "real_lerobot_multi" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            report_path = write_real_lerobot_dataset_export(skeleton, output, mode="dry-run")

            result = validate_real_lerobot_dataset_export_attempt(output)
            self.assertTrue(result.ok, result.errors)
            report = self._read_report(report_path)
            self.assertEqual(report["mode"], "dry-run")
            self.assertEqual(report["total_episodes"], 2)
            self.assertEqual(report["total_frames"], 38)
            self.assertFalse(report["parquet_written"])
            self.assertFalse(report["videos_written"])
            self.assertFalse(report["metadata_written"])

    def test_multi_forcevla_13d_write_if_available_reports_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_multi_skeleton(root, "forcevla_13d", "symlink")
            output = root / "real_lerobot_multi" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            report_path = write_real_lerobot_dataset_export(skeleton, output, mode="write-if-available")

            result = validate_real_lerobot_dataset_export_attempt(output)
            self.assertTrue(result.ok, result.errors)
            report = self._read_report(report_path)
            self.assertEqual(report["mode"], "write-if-available")
            self.assertEqual(report["total_episodes"], 2)
            self.assertEqual(report["total_frames"], 38)
            self.assertTrue(report["metadata_written"])
            self.assertEqual(len(report["per_episode"]), 2)

            for episode_index in range(2):
                parquet_path = output / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
                if report["parquet_ready"]:
                    self.assertTrue(report["parquet_written"], report["skipped_reasons"])
                    self.assertTrue(parquet_path.is_file())
                else:
                    self.assertFalse(report["parquet_written"])

                video_paths = [
                    output / "videos" / "observation.image" / f"episode_{episode_index:06d}.mp4",
                    output / "videos" / "observation.wrist_image" / f"episode_{episode_index:06d}.mp4",
                ]
                if report["videos_written"]:
                    self.assertTrue(all(path.is_file() for path in video_paths))

    def test_dataset_export_video_ready_requires_implemented_encoder(self):
        dependencies = {
            "python": {"available": True, "version": "3", "detail": "python"},
            "pyarrow": {"available": False, "version": None, "detail": "missing"},
            "pandas": {"available": False, "version": None, "detail": "missing"},
            "lerobot": {"available": False, "version": None, "detail": "missing"},
            "cv2": {"available": False, "version": None, "detail": "missing"},
            "imageio": {"available": False, "version": None, "detail": "missing"},
            "imageio_ffmpeg": {"available": False, "version": None, "detail": "missing"},
            "PIL": {"available": True, "version": "10", "detail": "pillow"},
            "ffmpeg": {"available": True, "version": None, "detail": "system ffmpeg"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_multi_skeleton(root, "forcevla_13d", "symlink")
            output = root / "real_lerobot_multi" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            with mock.patch.object(dataset_export_module, "check_export_dependencies", return_value=dependencies):
                report_path = write_real_lerobot_dataset_export(skeleton, output, mode="write-if-available")

            report = self._read_report(report_path)

        per_episode_reasons = [
            reason
            for episode in report["per_episode"]
            for reason in episode["skipped_reasons"]
        ]
        reasons = " ".join(str(reason) for reason in report["skipped_reasons"] + per_episode_reasons)
        self.assertFalse(report["video_ready"])
        self.assertFalse(report["videos_written"])
        self.assertIn("requires one implemented video encoder: imageio_ffmpeg, imageio, or cv2", reasons)
        self.assertNotIn("PIL readiness", reasons)
        self.assertNotIn("ffmpeg with imageio", reasons)

    def test_multi_doosan_full_25d_dry_run_reports_profile_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_multi_skeleton(root, "doosan_full_25d", "copy")
            output = root / "real_lerobot_multi" / "doosan_full_25d" / "doosan_peg_in_hole_v0"

            report_path = write_real_lerobot_dataset_export(skeleton, output, mode="dry-run")

            result = validate_real_lerobot_dataset_export_attempt(output)
            self.assertTrue(result.ok, result.errors)
            report = self._read_report(report_path)
            self.assertEqual(report["profile"], "doosan_full_25d")
            self.assertEqual(report["state_dim"], 25)
            self.assertEqual(report["action_dim"], 7)
            self.assertEqual(report["total_episodes"], 2)
            self.assertFalse(report["parquet_written"])
            self.assertFalse(report["videos_written"])


if __name__ == "__main__":
    unittest.main()
