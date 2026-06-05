import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.inspect.inspect_processed_episode import summarize_processed_episode


class InspectProcessedEpisodeTests(unittest.TestCase):
    def test_summary_for_converted_dummy_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_dummy_raw_episode(raw_episode)
            convert_raw_to_processed(raw_episode, processed_episode)

            summary = summarize_processed_episode(processed_episode)

            self.assertEqual(summary["frame_count"], 20)
            self.assertEqual(summary["action_dim"], 7)
            self.assertEqual(summary["model_state_dim"], 25)
            self.assertGreater(summary["max_translation_step_norm"], 0.0)
            self.assertGreater(summary["max_rotation_step_norm"], 0.0)
            self.assertEqual(summary["external_rgb_files_existing"], 20)
            self.assertEqual(summary["tcp_rgb_files_existing"], 20)


if __name__ == "__main__":
    unittest.main()
