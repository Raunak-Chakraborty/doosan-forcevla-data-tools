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


if __name__ == "__main__":
    unittest.main()
