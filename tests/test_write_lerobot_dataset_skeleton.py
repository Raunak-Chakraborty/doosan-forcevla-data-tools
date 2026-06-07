import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.plan_lerobot_export import write_lerobot_export_plan
from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.convert.stage_lerobot_export import stage_lerobot_export
from doosan_forcevla_data.convert.write_lerobot_dataset_skeleton import write_lerobot_dataset_skeleton
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.validate.validate_lerobot_dataset_skeleton import (
    validate_lerobot_dataset_skeleton,
)


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class WriteLeRobotDatasetSkeletonTests(unittest.TestCase):
    def _build_staged_profile(self, root: Path, profile: str, episode_name: str) -> Path:
        raw_episode = root / "raw" / episode_name
        processed_episode = root / "processed" / episode_name
        staged_episode = root / "staged" / profile / episode_name

        make_dummy_raw_episode(raw_episode)
        convert_raw_to_processed(raw_episode, processed_episode)
        plan_path = processed_episode / f"export_plan_{profile}.json"
        write_lerobot_export_plan(processed_episode, profile, plan_path)
        stage_lerobot_export(processed_episode, plan_path, staged_episode)
        return staged_episode

    def test_two_forcevla_episodes_symlink_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staged_0 = self._build_staged_profile(root, "forcevla_13d", "episode_000000")
            staged_1 = self._build_staged_profile(root, "forcevla_13d", "episode_000001")
            output = root / "lerobot_multi" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            write_lerobot_dataset_skeleton(
                [staged_0, staged_1],
                output,
                profile="forcevla_13d",
                image_mode="symlink",
            )

            result = validate_lerobot_dataset_skeleton(output)
            self.assertTrue(result.ok, result.errors)

            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_frames"], 38)
            self.assertEqual(info["total_tasks"], 1)
            self.assertEqual(info["features"]["observation.state"]["shape"], [13])
            self.assertEqual(info["features"]["action"]["shape"], [7])
            self.assertEqual(info["splits"]["train"], "0:2")
            self.assertTrue(info["notes"]["multi_episode_skeleton"])

            tasks = _read_jsonl(output / "meta" / "tasks.jsonl")
            episodes = _read_jsonl(output / "meta" / "episodes.jsonl")
            stats = _read_jsonl(output / "meta" / "episodes_stats.jsonl")
            self.assertEqual(len(tasks), 1)
            self.assertEqual(len(episodes), 2)
            self.assertEqual(len(stats), 2)

            ep0 = _read_jsonl(output / "data" / "chunk-000" / "episode_000000.jsonl")
            ep1 = _read_jsonl(output / "data" / "chunk-000" / "episode_000001.jsonl")
            self.assertEqual(len(ep0), 19)
            self.assertEqual(len(ep1), 19)

            self.assertEqual(ep0[0]["episode_index"], 0)
            self.assertEqual(ep1[0]["episode_index"], 1)
            self.assertEqual(ep0[0]["index"], 0)
            self.assertEqual(ep0[-1]["index"], 18)
            self.assertEqual(ep1[0]["index"], 19)
            self.assertEqual(ep1[-1]["index"], 37)
            self.assertEqual(len(ep1[0]["observation.state"]), 13)
            self.assertEqual(len(ep1[0]["action"]), 7)
            self.assertEqual(ep1[0]["prompt"], ep1[0]["task"])

            first_image = output / ep1[0]["observation.image"]
            first_wrist_image = output / ep1[0]["observation.wrist_image"]
            self.assertTrue(first_image.exists())
            self.assertTrue(first_wrist_image.exists())
            if hasattr(first_image, "is_symlink"):
                self.assertTrue(first_image.is_symlink())

    def test_two_doosan_full_episodes_copy_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staged_0 = self._build_staged_profile(root, "doosan_full_25d", "episode_000000")
            staged_1 = self._build_staged_profile(root, "doosan_full_25d", "episode_000001")
            output = root / "lerobot_multi" / "doosan_full_25d" / "doosan_peg_in_hole_v0"

            write_lerobot_dataset_skeleton(
                [staged_0, staged_1],
                output,
                profile="doosan_full_25d",
                image_mode="copy",
            )

            result = validate_lerobot_dataset_skeleton(output)
            self.assertTrue(result.ok, result.errors)

            info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_frames"], 38)
            self.assertEqual(info["features"]["observation.state"]["shape"], [25])

            ep1 = _read_jsonl(output / "data" / "chunk-000" / "episode_000001.jsonl")
            first_image = output / ep1[0]["observation.image"]
            first_wrist_image = output / ep1[0]["observation.wrist_image"]
            self.assertTrue(first_image.is_file())
            self.assertTrue(first_wrist_image.is_file())
            self.assertFalse(first_image.is_symlink())
            self.assertFalse(first_wrist_image.is_symlink())

    def test_existing_output_without_overwrite_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staged_0 = self._build_staged_profile(root, "forcevla_13d", "episode_000000")
            staged_1 = self._build_staged_profile(root, "forcevla_13d", "episode_000001")
            output = root / "lerobot_multi" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            write_lerobot_dataset_skeleton([staged_0, staged_1], output, profile="forcevla_13d", image_mode="copy")

            with self.assertRaisesRegex(ValueError, "already exists"):
                write_lerobot_dataset_skeleton([staged_0, staged_1], output, profile="forcevla_13d", image_mode="copy")

            result = validate_lerobot_dataset_skeleton(output)
            self.assertTrue(result.ok, result.errors)

    def test_existing_output_with_overwrite_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staged_0 = self._build_staged_profile(root, "forcevla_13d", "episode_000000")
            staged_1 = self._build_staged_profile(root, "forcevla_13d", "episode_000001")
            output = root / "lerobot_multi" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            write_lerobot_dataset_skeleton([staged_0, staged_1], output, profile="forcevla_13d", image_mode="copy")
            stale = output / "stale.txt"
            stale.write_text("stale\n", encoding="utf-8")

            write_lerobot_dataset_skeleton(
                [staged_0, staged_1],
                output,
                profile="forcevla_13d",
                image_mode="copy",
                overwrite=True,
            )

            self.assertFalse(stale.exists())
            result = validate_lerobot_dataset_skeleton(output)
            self.assertTrue(result.ok, result.errors)


if __name__ == "__main__":
    unittest.main()
