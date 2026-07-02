import json
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.convert.raw_real_to_processed import convert_raw_real_to_processed
from doosan_forcevla_data.dummy.make_synthetic_raw_real_episode import make_synthetic_raw_real_episode
from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class ValidateProcessedEpisodeTests(unittest.TestCase):
    def test_new_processed_metadata_requires_state_and_action_layouts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            convert_raw_real_to_processed(raw_episode, processed_episode)

            metadata_path = processed_episode / "metadata_processed.json"
            metadata = _read_json(metadata_path)
            metadata.pop("model_state_layout")
            metadata.pop("measured_action_layout")
            _write_json(metadata_path, metadata)

            result = validate_processed_episode(processed_episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("model_state_layout is required" in error for error in result.errors))
            self.assertTrue(any("measured_action_layout is required" in error for error in result.errors))

    def test_terminal_action_policy_must_match_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4)
            convert_raw_real_to_processed(raw_episode, processed_episode)

            metadata_path = processed_episode / "metadata_processed.json"
            metadata = _read_json(metadata_path)
            metadata["terminal_action_policy"]["padding_count"] = 2
            _write_json(metadata_path, metadata)

            result = validate_processed_episode(processed_episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("padding_count" in error for error in result.errors))

    def test_placeholder_gripper_must_not_be_labelled_measured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_000000"
            processed_episode = root / "processed" / "episode_000000"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4, include_optional_streams=True)
            convert_raw_real_to_processed(raw_episode, processed_episode)

            metadata_path = processed_episode / "metadata_processed.json"
            metadata = _read_json(metadata_path)
            metadata["selected_streams"]["gripper_state"] = "record_index aligned measured gripper_state stream"
            _write_json(metadata_path, metadata)

            result = validate_processed_episode(processed_episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("placeholder" in error and "measured" in error for error in result.errors))

    def test_three_camera_metadata_requires_external_camera_2_frame_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_three_camera"
            processed_episode = root / "processed" / "episode_three_camera"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4, camera_layout="thesis")
            convert_raw_real_to_processed(raw_episode, processed_episode, copy_images=True)

            frames_path = processed_episode / "frames.jsonl"
            frames = _read_jsonl(frames_path)
            del frames[0]["external_camera_2_rgb_path"]
            _write_jsonl(frames_path, frames)

            result = validate_processed_episode(processed_episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("external_camera_2_rgb_path" in error for error in result.errors))

    def test_three_camera_metadata_missing_copied_external_camera_2_file_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_three_camera_missing_file"
            processed_episode = root / "processed" / "episode_three_camera_missing_file"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4, camera_layout="thesis")
            convert_raw_real_to_processed(raw_episode, processed_episode, copy_images=True)

            (processed_episode / "images" / "external_camera_2" / "000000.ppm").unlink()

            result = validate_processed_episode(processed_episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("external_camera_2_rgb_path" in error and "does not exist" in error for error in result.errors))

    def test_three_camera_metadata_detects_substituted_external_camera_2_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_episode = root / "raw_real" / "episode_three_camera_substituted"
            processed_episode = root / "processed" / "episode_three_camera_substituted"
            make_synthetic_raw_real_episode(raw_episode, frame_count=4, camera_layout="thesis")
            convert_raw_real_to_processed(raw_episode, processed_episode, copy_images=True)

            external_1 = processed_episode / "images" / "external_camera_1" / "000000.ppm"
            external_2 = processed_episode / "images" / "external_camera_2" / "000000.ppm"
            external_2.write_bytes(external_1.read_bytes())

            result = validate_processed_episode(processed_episode)

            self.assertFalse(result.ok)
            self.assertTrue(any("external_camera_2_rgb_path bytes do not match raw external_camera_2" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
