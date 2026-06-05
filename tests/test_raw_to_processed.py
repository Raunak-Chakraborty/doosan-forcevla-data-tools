import json
import math
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode


class RawToProcessedTests(unittest.TestCase):
    def test_convert_dummy_raw_episode_and_validate_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_dummy_raw_episode(raw_episode)
            output = convert_raw_to_processed(raw_episode, processed_episode)

            self.assertEqual(output, processed_episode)
            self.assertTrue((processed_episode / "metadata_processed.json").is_file())
            self.assertTrue((processed_episode / "frames.jsonl").is_file())

            result = validate_processed_episode(processed_episode)
            self.assertTrue(result.ok, result.errors)

            metadata = json.loads((processed_episode / "metadata_processed.json").read_text())
            self.assertEqual(metadata["frame_count"], 20)

            frames = [
                json.loads(line)
                for line in (processed_episode / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(frames), 20)

            has_nonzero_translation = False
            has_nonzero_rotation = False
            for idx, frame in enumerate(frames):
                model_state = frame["model_state"]
                measured_action = frame["measured_action"]
                self.assertEqual(len(model_state), MODEL_STATE_DIM)
                self.assertEqual(len(measured_action), ACTION_DIM)
                self.assertTrue(all(math.isfinite(value) for value in model_state))
                self.assertTrue(all(math.isfinite(value) for value in measured_action))

                if idx < len(frames) - 1:
                    if any(abs(value) > 1e-12 for value in measured_action[:3]):
                        has_nonzero_translation = True
                    if any(abs(value) > 1e-12 for value in measured_action[3:6]):
                        has_nonzero_rotation = True

            self.assertTrue(frames[-1]["action_is_terminal_padding"])
            self.assertEqual(frames[-1]["measured_action"], [0.0] * ACTION_DIM)
            self.assertTrue(has_nonzero_translation)
            self.assertTrue(has_nonzero_rotation)

    def test_processed_validator_accepts_zero_rotation_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"

            make_dummy_raw_episode(raw_episode)
            convert_raw_to_processed(raw_episode, processed_episode)

            frames_path = processed_episode / "frames.jsonl"
            frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
            for frame in frames[:-1]:
                frame["measured_action"][3:6] = [0.0, 0.0, 0.0]
            frames_path.write_text(
                "".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames),
                encoding="utf-8",
            )

            result = validate_processed_episode(processed_episode)
            self.assertTrue(result.ok, result.errors)


if __name__ == "__main__":
    unittest.main()
