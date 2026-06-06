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
    def _build_staged_profile(self, root: Path, profile: str) -> Path:
        raw_episode = root / "raw" / "episode_000000"
        processed_episode = root / "processed" / "episode_000000"
        staged_episode = root / "staged" / profile / "episode_000000"

        make_dummy_raw_episode(raw_episode)
        convert_raw_to_processed(raw_episode, processed_episode)
        plan_path = processed_episode / f"export_plan_{profile}.json"
        write_lerobot_export_plan(processed_episode, profile, plan_path)
        stage_lerobot_export(processed_episode, plan_path, staged_episode)
        return staged_episode

    def test_forcevla_13d_symlink_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staged_episode = self._build_staged_profile(root, "forcevla_13d")
            output = root / "lerobot" / "forcevla_13d" / "doosan_peg_in_hole_v0"

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
            self.assertIn("prompt", info["features"])

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
            self.assertIn("prompt", frames[0])
            self.assertEqual(frames[0]["prompt"], frames[0]["task"])

            # If symlink creation succeeded, pathlib's symlink check is reliable on POSIX.
            if hasattr(first_image, "is_symlink"):
                self.assertTrue(first_image.is_symlink())

    def test_doosan_full_25d_copy_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staged_episode = self._build_staged_profile(root, "doosan_full_25d")
            output = root / "lerobot" / "doosan_full_25d" / "doosan_peg_in_hole_v0"

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
            self.assertIn("prompt", info["features"])

            frames = [
                json.loads(line)
                for line in (output / "data" / "chunk-000" / "episode_000000.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            first_image = output / frames[0]["observation.image"]
            first_wrist_image = output / frames[0]["observation.wrist_image"]
            self.assertEqual(len(frames[0]["observation.state"]), 25)
            self.assertEqual(frames[0]["prompt"], frames[0]["task"])
            self.assertTrue(first_image.is_file())
            self.assertTrue(first_wrist_image.is_file())
            self.assertFalse(first_image.is_symlink())
            self.assertFalse(first_wrist_image.is_symlink())

    def test_existing_output_without_overwrite_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staged_episode = self._build_staged_profile(root, "forcevla_13d")
            output = root / "lerobot" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            write_lerobot_skeleton(staged_episode, output, profile="forcevla_13d", image_mode="copy")

            with self.assertRaisesRegex(ValueError, "already exists"):
                write_lerobot_skeleton(staged_episode, output, profile="forcevla_13d", image_mode="copy")

            result = validate_lerobot_skeleton(output)
            self.assertTrue(result.ok, result.errors)

    def test_existing_output_with_overwrite_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staged_episode = self._build_staged_profile(root, "forcevla_13d")
            output = root / "lerobot" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            write_lerobot_skeleton(staged_episode, output, profile="forcevla_13d", image_mode="copy")
            stale_file = output / "stale.txt"
            stale_file.write_text("stale\n", encoding="utf-8")

            write_lerobot_skeleton(
                staged_episode,
                output,
                profile="forcevla_13d",
                image_mode="copy",
                overwrite=True,
            )

            self.assertFalse(stale_file.exists())
            result = validate_lerobot_skeleton(output)
            self.assertTrue(result.ok, result.errors)


if __name__ == "__main__":
    unittest.main()
