import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.plan_lerobot_export import write_lerobot_export_plan
from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.convert.stage_lerobot_export import stage_lerobot_export
from doosan_forcevla_data.convert.write_lerobot_skeleton import write_lerobot_skeleton
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.inspect.preflight_real_export import (
    preflight_real_export,
    write_preflight_report,
)


class PreflightRealExportTests(unittest.TestCase):
    def _build_skeleton(self, root: Path, profile: str, image_mode: str) -> Path:
        raw_episode = root / "raw" / "episode_000000"
        processed_episode = root / "processed" / "episode_000000"
        staged_episode = root / "staged" / profile / "episode_000000"
        output = root / "lerobot" / profile / "doosan_peg_in_hole_v0"

        make_dummy_raw_episode(raw_episode)
        convert_raw_to_processed(raw_episode, processed_episode)
        plan_path = processed_episode / f"export_plan_{profile}.json"
        write_lerobot_export_plan(processed_episode, profile, plan_path)
        stage_lerobot_export(processed_episode, plan_path, staged_episode)
        write_lerobot_skeleton(
            staged_episode,
            output,
            episode_index=0,
            task_index=0,
            profile=profile,
            image_mode=image_mode,
        )
        return output

    def test_forcevla_13d_symlink_preflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = self._build_skeleton(Path(tmpdir), "forcevla_13d", "symlink")

            report = preflight_real_export(output)

            self.assertTrue(report["skeleton_valid"], report["errors"])
            self.assertTrue(report["schema_valid"], report["errors"])
            self.assertEqual(report["profile"], "forcevla_13d")
            self.assertEqual(report["state_dim"], 13)
            self.assertEqual(report["action_dim"], 7)
            self.assertEqual(report["total_frames"], 19)
            self.assertTrue(report["prompt_task_compatible"])
            self.assertTrue(report["image_staging_complete"], report["errors"])
            self.assertEqual(report["image_counts"]["observation.image"], 19)
            self.assertEqual(report["image_counts"]["observation.wrist_image"], 19)
            self.assertIsInstance(report["parquet_ready"], bool)
            self.assertIsInstance(report["video_ready"], bool)

    def test_doosan_full_25d_copy_preflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = self._build_skeleton(Path(tmpdir), "doosan_full_25d", "copy")

            report = preflight_real_export(output)

            self.assertTrue(report["skeleton_valid"], report["errors"])
            self.assertTrue(report["schema_valid"], report["errors"])
            self.assertEqual(report["profile"], "doosan_full_25d")
            self.assertEqual(report["state_dim"], 25)
            self.assertEqual(report["total_frames"], 19)
            self.assertTrue(report["prompt_task_compatible"])

    def test_json_report_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = self._build_skeleton(root, "forcevla_13d", "copy")
            report_path = root / "preflight_report.json"

            report = preflight_real_export(output)
            write_preflight_report(report, report_path)

            self.assertTrue(report_path.is_file())
            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("real_export_ready", data)
            self.assertIn("video_export_ready", data)

    def test_prompt_task_mismatch_is_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = self._build_skeleton(Path(tmpdir), "forcevla_13d", "copy")
            frames_path = output / "data" / "chunk-000" / "episode_000000.jsonl"
            frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
            frames[0]["prompt"] = "Different prompt"
            frames_path.write_text(
                "".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames),
                encoding="utf-8",
            )

            report = preflight_real_export(output)

            # The helper should return a report, not raise, so callers can inspect bad exports.
            self.assertFalse(report["prompt_task_compatible"])
            self.assertFalse(report["schema_valid"])
            self.assertTrue(any("prompt must equal task" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
