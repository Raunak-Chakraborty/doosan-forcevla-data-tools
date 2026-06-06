import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.plan_lerobot_export import write_lerobot_export_plan
from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.convert.stage_lerobot_export import stage_lerobot_export
from doosan_forcevla_data.convert.write_lerobot_skeleton import write_lerobot_skeleton
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.validate.validate_lerobot_skeleton import validate_lerobot_skeleton


class WriteLeRobotSkeletonTests(unittest.TestCase):
    def test_forcevla_13d_symlink_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            staged_episode = root / "staged" / "forcevla_13d" / "episode_000000"
            output = root / "lerobot" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            make_dummy_raw_episode(raw_episode)
            convert_raw_to_processed(raw_episode, processed_episode)
            plan_path = processed_episode / "export_plan_forcevla_13d.json"
            write_lerobot_export_plan(processed_episode, "forcevla_13d", plan_path)
            stage_lerobot_export(processed_episode, plan_path, staged_episode)

            write_lerobot_skeleton(
                staged_episode,
                output,
                episode_index=0,
                task_index=0,
                profile="forcevla_13d",
                image_mode="symlink",
            )

            result = validate_lerobot_skeleton(output)
            self.assertTrue(result.ok, result.errors)

            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_frames"], 19)
            self.assertEqual(info["features"]["observation.state"]["shape"], [13])
            self.assertEqual(info["features"]["action"]["shape"], [7])

            frames = [
                json.loads(line)
                for line in (output / "data" / "chunk-000" / "episode_000000.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            first_image = output / frames[0]["observation.image"]
            first_wrist_image = output / frames[0]["observation.wrist_image"]
            self.assertTrue(first_image.exists())
            self.assertTrue(first_wrist_image.exists())
            self.assertEqual(len(frames[0]["observation.state"]), 13)
            self.assertEqual(len(frames[0]["action"]), 7)

            # If symlink creation succeeded, pathlib's symlink check is reliable on POSIX.
            if hasattr(first_image, "is_symlink"):
                self.assertTrue(first_image.is_symlink())

    def test_doosan_full_25d_copy_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            staged_episode = root / "staged" / "doosan_full_25d" / "episode_000000"
            output = root / "lerobot" / "doosan_full_25d" / "doosan_peg_in_hole_v0"

            make_dummy_raw_episode(raw_episode)
            convert_raw_to_processed(raw_episode, processed_episode)
            plan_path = processed_episode / "export_plan_doosan_full_25d.json"
            write_lerobot_export_plan(processed_episode, "doosan_full_25d", plan_path)
            stage_lerobot_export(processed_episode, plan_path, staged_episode)

            write_lerobot_skeleton(
                staged_episode,
                output,
                episode_index=0,
                task_index=0,
                profile="doosan_full_25d",
                image_mode="copy",
            )

            result = validate_lerobot_skeleton(output)
            self.assertTrue(result.ok, result.errors)

            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_frames"], 19)
            self.assertEqual(info["features"]["observation.state"]["shape"], [25])

            frames = [
                json.loads(line)
                for line in (output / "data" / "chunk-000" / "episode_000000.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            first_image = output / frames[0]["observation.image"]
            first_wrist_image = output / frames[0]["observation.wrist_image"]
            self.assertEqual(len(frames[0]["observation.state"]), 25)
            self.assertTrue(first_image.is_file())
            self.assertTrue(first_wrist_image.is_file())
            self.assertFalse(first_image.is_symlink())
            self.assertFalse(first_wrist_image.is_symlink())


if __name__ == "__main__":
    unittest.main()
