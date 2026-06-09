import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from doosan_forcevla_data.dummy.make_synthetic_raw_real_episode import (
    make_synthetic_raw_real_episode,
)
from doosan_forcevla_data.validate.validate_raw_real_episode import validate_raw_real_episode


def _jsonl_line_count(path: Path) -> int:
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


class SyntheticRawRealEpisodeTests(unittest.TestCase):
    def test_create_synthetic_raw_real_episode_and_validate_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"

            output = make_synthetic_raw_real_episode(episode_dir)

            self.assertEqual(output, episode_dir)
            required_paths = [
                "metadata.json",
                "calibration_refs.json",
                "events.jsonl",
                "recorder_report.json",
                "streams/index.json",
                "streams/joint_states.jsonl",
                "streams/robot_state_rt.jsonl",
                "streams/tf.jsonl",
                "streams/tf_static.jsonl",
                "streams/external_camera/index.jsonl",
                "streams/wrist_camera/index.jsonl",
                "streams/external_camera/frames/000000.ppm",
                "streams/wrist_camera/frames/000000.ppm",
            ]
            for relative_path in required_paths:
                self.assertTrue((episode_dir / relative_path).exists(), relative_path)

            result = validate_raw_real_episode(episode_dir)
            self.assertTrue(result.ok, result.errors)

    def test_cli_creates_valid_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(__file__).resolve().parents[1] / "src"
            env = {"PYTHONPATH": str(src_path)}

            commands = [
                ("positional", Path(tmpdir) / "episode_positional", []),
                ("output_option", Path(tmpdir) / "episode_output_option", ["--output"]),
            ]
            for style, episode_dir, output_prefix in commands:
                with self.subTest(style=style):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "doosan_forcevla_data.dummy.make_synthetic_raw_real_episode",
                            *output_prefix,
                            str(episode_dir),
                            "--frames",
                            "5",
                            "--fps",
                            "30",
                            "--overwrite",
                        ],
                        check=False,
                        env=env,
                        text=True,
                        capture_output=True,
                    )

                    self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
                    self.assertIn("Wrote synthetic raw-real episode:", completed.stdout)
                    result = validate_raw_real_episode(episode_dir)
                    self.assertTrue(result.ok, result.errors)

    def test_optional_streams_can_be_included(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"

            make_synthetic_raw_real_episode(episode_dir, include_optional_streams=True)

            self.assertTrue((episode_dir / "streams" / "command_context.jsonl").is_file())
            self.assertTrue((episode_dir / "streams" / "gripper_state.jsonl").is_file())
            result = validate_raw_real_episode(episode_dir)
            self.assertTrue(result.ok, result.errors)
            self.assertFalse(
                any("optional stream command_context is absent" in warning for warning in result.warnings)
            )
            self.assertFalse(
                any("optional stream gripper_state is absent" in warning for warning in result.warnings)
            )

    def test_optional_streams_can_be_omitted_with_warnings_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"

            make_synthetic_raw_real_episode(episode_dir, include_optional_streams=False)

            result = validate_raw_real_episode(episode_dir)
            self.assertTrue(result.ok, result.errors)
            self.assertTrue(
                any("optional stream command_context is absent" in warning for warning in result.warnings)
            )
            self.assertTrue(
                any("optional stream gripper_state is absent" in warning for warning in result.warnings)
            )

    def test_invalid_frame_count_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                make_synthetic_raw_real_episode(Path(tmpdir) / "episode_000000", frame_count=1)

    def test_invalid_fps_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                make_synthetic_raw_real_episode(Path(tmpdir) / "episode_000000", fps=0)

    def test_existing_output_without_overwrite_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode_dir)

            with self.assertRaises((FileExistsError, ValueError)):
                make_synthetic_raw_real_episode(episode_dir, overwrite=False)

    def test_existing_output_with_overwrite_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"
            make_synthetic_raw_real_episode(episode_dir)
            junk_path = episode_dir / "junk.txt"
            junk_path.write_text("junk\n", encoding="utf-8")

            make_synthetic_raw_real_episode(episode_dir, overwrite=True)

            self.assertFalse(junk_path.exists())
            result = validate_raw_real_episode(episode_dir)
            self.assertTrue(result.ok, result.errors)

    def test_frame_count_and_camera_files_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"

            make_synthetic_raw_real_episode(episode_dir, frame_count=5)

            for stream_name in ["external_camera", "wrist_camera"]:
                camera_dir = episode_dir / "streams" / stream_name
                self.assertEqual(_jsonl_line_count(camera_dir / "index.jsonl"), 5)
                ppm_files = sorted((camera_dir / "frames").iterdir())
                self.assertEqual(len(ppm_files), 5)
                self.assertTrue(all(path.suffix == ".ppm" for path in ppm_files))

    def test_stream_index_counts_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_000000"

            make_synthetic_raw_real_episode(
                episode_dir,
                frame_count=7,
                include_optional_streams=True,
            )

            index = json.loads((episode_dir / "streams" / "index.json").read_text(encoding="utf-8"))
            streams = index["streams"]
            for stream_name in [
                "joint_states",
                "robot_state_rt",
                "tf",
                "command_context",
                "gripper_state",
            ]:
                path = episode_dir / streams[stream_name]["path"]
                self.assertEqual(streams[stream_name]["record_count"], _jsonl_line_count(path))

            self.assertEqual(streams["tf_static"]["record_count"], 1)
            self.assertEqual(
                streams["tf_static"]["record_count"],
                _jsonl_line_count(episode_dir / streams["tf_static"]["path"]),
            )

            for stream_name in ["external_camera", "wrist_camera"]:
                camera_dir = episode_dir / streams[stream_name]["path"]
                index_count = _jsonl_line_count(camera_dir / "index.jsonl")
                frame_count = len(list((camera_dir / "frames").iterdir()))
                self.assertEqual(streams[stream_name]["record_count"], index_count)
                self.assertEqual(streams[stream_name]["record_count"], frame_count)


if __name__ == "__main__":
    unittest.main()
