import importlib.util
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.action_chunks import (
    build_future_action_chunk_from_lerobot_export,
)
from doosan_forcevla_data.convert.plan_lerobot_export import write_lerobot_export_plan
from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.convert.stage_lerobot_export import stage_lerobot_export
from doosan_forcevla_data.convert.write_lerobot_dataset_skeleton import (
    write_lerobot_dataset_skeleton,
)
from doosan_forcevla_data.convert.write_real_lerobot_dataset_export import (
    write_real_lerobot_dataset_export,
)
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.inspect.smoke_forcevla_observation_builder import (
    build_smoke_observation,
)
from doosan_forcevla_data.validate.validate_real_lerobot_dataset_export_attempt import (
    validate_real_lerobot_dataset_export_attempt,
)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@unittest.skipUnless(
    _has_module("pyarrow") and _has_module("av") and _has_module("imageio"),
    "requires pyarrow, av, and imageio",
)
class SmokeForceVLAMultiEpisodeTests(unittest.TestCase):
    def _build_two_episode_real_export(self, root: Path) -> Path:
        staged_dirs = []

        for episode_index in range(2):
            episode_name = f"episode_{episode_index:06d}"
            raw_episode = root / "raw" / episode_name
            processed_episode = root / "processed" / episode_name
            staged_episode = root / "staged" / "forcevla_13d" / episode_name

            make_dummy_raw_episode(raw_episode)
            convert_raw_to_processed(raw_episode, processed_episode)

            plan_path = processed_episode / "export_plan_forcevla_13d.json"
            write_lerobot_export_plan(
                processed_episode,
                "forcevla_13d",
                plan_path,
            )

            stage_lerobot_export(
                processed_episode,
                plan_path,
                staged_episode,
            )

            staged_dirs.append(staged_episode)

        skeleton = root / "lerobot_multi" / "forcevla_13d" / "doosan_peg_in_hole_v0"
        write_lerobot_dataset_skeleton(
            staged_dirs,
            skeleton,
            profile="forcevla_13d",
            image_mode="symlink",
        )

        real_export = root / "real_lerobot_multi" / "forcevla_13d" / "doosan_peg_in_hole_v0"
        write_real_lerobot_dataset_export(
            skeleton,
            real_export,
            mode="write-if-available",
        )

        validation = validate_real_lerobot_dataset_export_attempt(real_export)
        if not validation.ok:
            raise AssertionError(validation.errors)

        return real_export

    def test_episode_one_observation_builder_and_future_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = self._build_two_episode_real_export(root)

            observation, report = build_smoke_observation(
                dataset,
                episode_index=1,
                row_index=0,
                expected_state_dim=13,
                expected_action_dim=7,
                strict_video_shape=False,
            )

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["episode_index"], 1)
            self.assertEqual(report["row_index"], 0)
            self.assertEqual(report["parquet_rows"], 19)
            self.assertEqual(report["sample"]["episode_index"], 1)
            self.assertEqual(report["sample"]["index"], 19)
            self.assertEqual(report["forcevla_observation_summary"]["observation.state.length"], 13)
            self.assertEqual(report["forcevla_observation_summary"]["action.length"], 7)

            self.assertEqual(len(observation["observation.state"]), 13)
            self.assertEqual(len(observation["action"]), 7)
            self.assertEqual(observation["prompt"], "Insert the peg into the hole.")

            chunk_start = build_future_action_chunk_from_lerobot_export(
                dataset,
                episode_index=1,
                row_index=0,
                horizon=50,
                action_dim=7,
                pad_mode="repeat_last",
            )
            self.assertEqual(chunk_start.horizon, 50)
            self.assertEqual(chunk_start.action_dim, 7)
            self.assertEqual(chunk_start.source_action_count, 19)
            self.assertEqual(sum(chunk_start.valid_mask), 19)
            self.assertEqual(chunk_start.padded_count, 31)
            self.assertEqual(len(chunk_start.actions), 50)
            self.assertEqual(len(chunk_start.actions[0]), 7)

            chunk_last = build_future_action_chunk_from_lerobot_export(
                dataset,
                episode_index=1,
                row_index=18,
                horizon=50,
                action_dim=7,
                pad_mode="repeat_last",
            )
            self.assertEqual(sum(chunk_last.valid_mask), 1)
            self.assertEqual(chunk_last.padded_count, 49)
            self.assertEqual(len(chunk_last.actions), 50)
            self.assertEqual(len(chunk_last.actions[0]), 7)


if __name__ == "__main__":
    unittest.main()
