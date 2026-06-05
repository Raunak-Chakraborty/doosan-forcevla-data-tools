import csv
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.validate.validate_raw_episode import validate_raw_episode


class DummyRawEpisodeTests(unittest.TestCase):
    def test_create_dummy_episode_and_validate_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"
            make_dummy_raw_episode(episode_dir)

            required_paths = [
                "metadata.json",
                "robot/tcp_pose.csv",
                "robot/joint_states.csv",
                "force/wrench.csv",
                "actions/commanded_twist.csv",
                "events.csv",
                "images/external_rgb",
                "images/tcp_rgb",
            ]
            for relative_path in required_paths:
                self.assertTrue((episode_dir / relative_path).exists(), relative_path)

            result = validate_raw_episode(episode_dir)
            self.assertTrue(result.ok, result.errors)

    def test_optional_gripper_pos_is_validated_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"
            make_dummy_raw_episode(episode_dir)

            tcp_pose_path = episode_dir / "robot" / "tcp_pose.csv"
            with tcp_pose_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
            rows[0]["gripper_pos"] = "NaN"
            with tcp_pose_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            result = validate_raw_episode(episode_dir)
            self.assertFalse(result.ok)
            self.assertTrue(any("gripper_pos" in error for error in result.errors), result.errors)

    def test_dummy_joint_velocity_uses_timestamp_span(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"
            make_dummy_raw_episode(episode_dir)

            joint_states_path = episode_dir / "robot" / "joint_states.csv"
            with joint_states_path.open("r", newline="", encoding="utf-8") as handle:
                first_row = next(csv.DictReader(handle))

            expected_velocity = 0.02 / (19.0 / 30.0)
            self.assertAlmostEqual(float(first_row["joint_vel_0"]), expected_velocity)


if __name__ == "__main__":
    unittest.main()
