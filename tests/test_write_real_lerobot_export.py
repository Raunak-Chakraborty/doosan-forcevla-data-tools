import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.plan_lerobot_export import write_lerobot_export_plan
from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.convert.stage_lerobot_export import stage_lerobot_export
from doosan_forcevla_data.convert.write_lerobot_skeleton import write_lerobot_skeleton
from doosan_forcevla_data.convert.write_real_lerobot_export import write_real_lerobot_export
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.validate.validate_real_lerobot_export_attempt import (
    validate_real_lerobot_export_attempt,
)


class WriteRealLeRobotExportTests(unittest.TestCase):
    def _build_skeleton(self, root: Path, profile: str, image_mode: str) -> Path:
        raw_episode = root / "raw" / "episode_000000"
        processed_episode = root / "processed" / "episode_000000"
        staged_episode = root / "staged" / profile / "episode_000000"
        skeleton = root / "lerobot" / profile / "doosan_peg_in_hole_v0"

        make_dummy_raw_episode(raw_episode)
        convert_raw_to_processed(raw_episode, processed_episode)
        plan_path = processed_episode / f"export_plan_{profile}.json"
        write_lerobot_export_plan(processed_episode, profile, plan_path)
        stage_lerobot_export(processed_episode, plan_path, staged_episode)
        write_lerobot_skeleton(
            staged_episode,
            skeleton,
            episode_index=0,
            task_index=0,
            profile=profile,
            image_mode=image_mode,
        )
        return skeleton

    def _read_report(self, path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_forcevla_13d_dry_run_writes_report_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_skeleton(root, "forcevla_13d", "symlink")
            output = root / "real_lerobot" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            report_path = write_real_lerobot_export(skeleton, output, mode="dry-run")

            result = validate_real_lerobot_export_attempt(output)
            self.assertTrue(result.ok, result.errors)
            report = self._read_report(report_path)
            self.assertEqual(report["mode"], "dry-run")
            self.assertFalse(report["parquet_written"])
            self.assertFalse(report["videos_written"])
            self.assertFalse(report["metadata_written"])

    def test_forcevla_13d_write_if_available_reports_conditional_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_skeleton(root, "forcevla_13d", "symlink")
            output = root / "real_lerobot" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            report_path = write_real_lerobot_export(skeleton, output, mode="write-if-available")

            result = validate_real_lerobot_export_attempt(output)
            self.assertTrue(result.ok, result.errors)
            report = self._read_report(report_path)
            self.assertEqual(report["mode"], "write-if-available")
            self.assertTrue(report["metadata_written"])

            parquet_path = output / "data" / "chunk-000" / "episode_000000.parquet"
            if report["parquet_ready"]:
                self.assertTrue(report["parquet_written"], report["skipped_reasons"])
                self.assertTrue(parquet_path.is_file())
            else:
                self.assertFalse(report["parquet_written"])
                self.assertTrue(any("pyarrow" in reason for reason in report["skipped_reasons"]))

            video_paths = [
                output / "videos" / "observation.image" / "episode_000000.mp4",
                output / "videos" / "observation.wrist_image" / "episode_000000.mp4",
            ]
            if report["videos_written"]:
                self.assertTrue(all(path.is_file() for path in video_paths))
            else:
                reasons = " ".join(str(reason) for reason in report["skipped_reasons"])
                self.assertRegex(reasons, "video|imageio|cv2|encoding")

    def test_doosan_full_25d_dry_run_reports_profile_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_skeleton(root, "doosan_full_25d", "copy")
            output = root / "real_lerobot" / "doosan_full_25d" / "doosan_peg_in_hole_v0"

            report_path = write_real_lerobot_export(skeleton, output, mode="dry-run")

            result = validate_real_lerobot_export_attempt(output)
            self.assertTrue(result.ok, result.errors)
            report = self._read_report(report_path)
            self.assertEqual(report["profile"], "doosan_full_25d")
            self.assertEqual(report["state_dim"], 25)
            self.assertEqual(report["action_dim"], 7)
            self.assertFalse(report["parquet_written"])
            self.assertFalse(report["videos_written"])


if __name__ == "__main__":
    unittest.main()
