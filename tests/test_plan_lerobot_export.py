import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.plan_lerobot_export import write_lerobot_export_plan
from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.validate.validate_export_plan import validate_export_plan


class PlanLeRobotExportTests(unittest.TestCase):
    def test_forcevla_and_full_profile_export_plans(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_dummy_raw_episode(raw_episode)
            convert_raw_to_processed(raw_episode, processed_episode)

            forcevla_plan_path = processed_episode / "export_plan_forcevla_13d.json"
            write_lerobot_export_plan(processed_episode, "forcevla_13d", forcevla_plan_path)
            forcevla_validation = validate_export_plan(forcevla_plan_path)
            self.assertTrue(forcevla_validation.ok, forcevla_validation.errors)

            forcevla_plan = json.loads(forcevla_plan_path.read_text(encoding="utf-8"))
            self.assertEqual(forcevla_plan["observation_state_dim"], 13)
            self.assertEqual(forcevla_plan["action_dim"], 7)
            self.assertEqual(forcevla_plan["input_frame_count"], 20)
            self.assertEqual(forcevla_plan["exported_frame_count"], 19)
            self.assertEqual(forcevla_plan["excluded_terminal_padding_frame_count"], 1)
            self.assertEqual(forcevla_plan["image_availability"]["observation.image"]["existing_count"], 19)
            self.assertEqual(
                forcevla_plan["image_availability"]["observation.wrist_image"]["existing_count"], 19
            )

            full_plan_path = processed_episode / "export_plan_doosan_full_25d.json"
            write_lerobot_export_plan(processed_episode, "doosan_full_25d", full_plan_path)
            full_validation = validate_export_plan(full_plan_path)
            self.assertTrue(full_validation.ok, full_validation.errors)

            full_plan = json.loads(full_plan_path.read_text(encoding="utf-8"))
            self.assertEqual(full_plan["observation_state_dim"], 25)
            self.assertEqual(full_plan["exported_frame_count"], 19)

    def test_export_plan_validator_accepts_generic_excluded_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "export_plan.json"
            manifest = {
                "source_processed_episode": "/tmp/example_processed_episode",
                "profile": "forcevla_13d",
                "dataset_name": "doosan_peg_in_hole_v0",
                "robot_type": "doosan_m1013",
                "fps": 30,
                "task_instruction": "Insert the peg into the hole.",
                "geometry_type": "round_peg_round_hole",
                "orientation_type": "vertical_insertion",
                "input_frame_count": 20,
                "exported_frame_count": 18,
                "excluded_terminal_padding_frame_count": 2,
                "terminal_padding_excluded": True,
                "lerobot_like_keys": {
                    "observation.image": "external_rgb_path",
                    "observation.wrist_image": "tcp_rgb_path",
                    "observation.state": "model_state[:13]",
                    "action": "measured_action",
                    "task": "task_instruction",
                },
                "observation_state_dim": 13,
                "action_dim": 7,
                "image_streams": {
                    "observation.image": {"source_key": "external_rgb_path"},
                    "observation.wrist_image": {"source_key": "tcp_rgb_path"},
                },
                "image_availability": {
                    "observation.image": {"existing_count": 18},
                    "observation.wrist_image": {"existing_count": 18},
                },
                "first_exported_record_preview": {
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "observation_state_length": 13,
                    "action_length": 7,
                    "task": "Insert the peg into the hole.",
                },
                "notes": ["dry run only"],
            }
            plan_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = validate_export_plan(plan_path)
            self.assertTrue(result.ok, result.errors)


if __name__ == "__main__":
    unittest.main()
