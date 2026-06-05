import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.plan_lerobot_export import write_lerobot_export_plan
from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.convert.stage_lerobot_export import stage_lerobot_export
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.validate.validate_staged_export import validate_staged_export


class StageLeRobotExportTests(unittest.TestCase):
    def test_stage_forcevla_and_full_profile_exports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_dummy_raw_episode(raw_episode)
            convert_raw_to_processed(raw_episode, processed_episode)

            forcevla_plan = processed_episode / "export_plan_forcevla_13d.json"
            write_lerobot_export_plan(processed_episode, "forcevla_13d", forcevla_plan)
            forcevla_staged = root / "staged" / "forcevla_13d" / "episode_000000"
            stage_lerobot_export(processed_episode, forcevla_plan, forcevla_staged)
            forcevla_validation = validate_staged_export(forcevla_staged)
            self.assertTrue(forcevla_validation.ok, forcevla_validation.errors)

            forcevla_frames = [
                json.loads(line)
                for line in (forcevla_staged / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(forcevla_frames), 19)
            self.assertEqual(len(forcevla_frames[0]["observation.state"]), 13)
            self.assertEqual(len(forcevla_frames[0]["action"]), 7)
            self.assertTrue(Path(forcevla_frames[0]["observation.image"]).is_file())
            self.assertTrue(Path(forcevla_frames[0]["observation.wrist_image"]).is_file())

            full_plan = processed_episode / "export_plan_doosan_full_25d.json"
            write_lerobot_export_plan(processed_episode, "doosan_full_25d", full_plan)
            full_staged = root / "staged" / "doosan_full_25d" / "episode_000000"
            stage_lerobot_export(processed_episode, full_plan, full_staged)
            full_validation = validate_staged_export(full_staged)
            self.assertTrue(full_validation.ok, full_validation.errors)

            full_frames = [
                json.loads(line)
                for line in (full_staged / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(full_frames), 19)
            self.assertEqual(len(full_frames[0]["observation.state"]), 25)

    def test_stage_resolves_relative_image_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_dummy_raw_episode(raw_episode)
            convert_raw_to_processed(raw_episode, processed_episode)

            frames_path = processed_episode / "frames.jsonl"
            frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
            for frame in frames:
                frame_idx = frame["frame_index"]
                frame["external_rgb_path"] = f"images/external_rgb/{frame_idx:06d}.ppm"
                frame["tcp_rgb_path"] = f"images/tcp_rgb/{frame_idx:06d}.ppm"
            frames_path.write_text(
                "".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames),
                encoding="utf-8",
            )

            plan_path = processed_episode / "export_plan_forcevla_13d.json"
            write_lerobot_export_plan(processed_episode, "forcevla_13d", plan_path)
            staged = root / "staged" / "forcevla_13d" / "episode_000000"
            stage_lerobot_export(processed_episode, plan_path, staged)

            result = validate_staged_export(staged)
            self.assertTrue(result.ok, result.errors)
            first_frame = json.loads((staged / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(Path(first_frame["observation.image"]).is_absolute())
            self.assertTrue(Path(first_frame["observation.image"]).is_file())
            self.assertTrue(Path(first_frame["observation.wrist_image"]).is_absolute())
            self.assertTrue(Path(first_frame["observation.wrist_image"]).is_file())


if __name__ == "__main__":
    unittest.main()
