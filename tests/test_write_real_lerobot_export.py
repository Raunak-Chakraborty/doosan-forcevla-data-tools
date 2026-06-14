import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import doosan_forcevla_data.convert.write_real_lerobot_export as real_export_module
from doosan_forcevla_data.convert.plan_lerobot_export import write_lerobot_export_plan
from doosan_forcevla_data.convert.raw_to_processed import convert_raw_to_processed
from doosan_forcevla_data.convert.stage_lerobot_export import stage_lerobot_export
from doosan_forcevla_data.convert.write_lerobot_skeleton import write_lerobot_skeleton
from doosan_forcevla_data.convert.write_real_lerobot_export import write_real_lerobot_export
from doosan_forcevla_data.dummy.make_dummy_raw_episode import make_dummy_raw_episode
from doosan_forcevla_data.validate.validate_real_lerobot_export_attempt import (
    validate_real_lerobot_export_attempt,
)


class WriteRealLeRobotExportTests(unittest.TestCase):
    def _minimal_video_frames(self, root: Path) -> list[dict[str, str]]:
        frames = []
        for frame_index in range(2):
            image_rel = f"image_staging/observation.image/episode_000000/{frame_index:06d}.png"
            wrist_rel = f"image_staging/observation.wrist_image/episode_000000/{frame_index:06d}.png"
            for relative_path in [image_rel, wrist_rel]:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fake 16x16 png placeholder")
            frames.append({"observation.image": image_rel, "observation.wrist_image": wrist_rel})
        return frames

    def _write_fake_video(self, image_paths: list[Path], output_path: Path, fps: float) -> None:
        self.assertEqual(len(image_paths), 2)
        self.assertGreater(fps, 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake mp4")

    def _build_skeleton(self, root: Path, profile: str, image_mode: str) -> Path:
        raw_episode = root / "raw" / "episode_000000"
        processed_episode = root / "processed" / "episode_000000"
        staged_episode = root / "staged" / profile / "episode_000000"
        skeleton = root / "lerobot" / profile / "doosan_peg_in_hole_v0"

        make_dummy_raw_episode(raw_episode)
        convert_raw_to_processed(raw_episode, processed_episode)
        plan_path = processed_episode / f"export_plan_{profile}.json"
        write_lerobot_export_plan(processed_episode, profile, plan_path)
        stage_lerobot_export(processed_episode, plan_path, staged_episode)
        write_lerobot_skeleton(
            staged_episode,
            skeleton,
            episode_index=0,
            task_index=0,
            profile=profile,
            image_mode=image_mode,
        )
        return skeleton

    def _read_report(self, path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_imageio_ffmpeg_backend_is_preferred_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames = self._minimal_video_frames(root)
            output = root / "output"
            dependencies = {
                "imageio_ffmpeg": {"available": True},
                "imageio": {"available": True},
                "cv2": {"available": True},
            }

            with mock.patch.object(
                real_export_module, "_encode_video_imageio_ffmpeg", side_effect=self._write_fake_video
            ) as imageio_ffmpeg_encode, mock.patch.object(
                real_export_module, "_encode_video_imageio"
            ) as imageio_encode, mock.patch.object(real_export_module, "_encode_video_cv2") as cv2_encode:
                _, video_backends, backend_errors = real_export_module._write_videos(
                    root, output, 0, frames, 30.0, dependencies
                )

            self.assertEqual(set(video_backends.values()), {"imageio_ffmpeg"})
            self.assertEqual(backend_errors, [])
            self.assertEqual(imageio_ffmpeg_encode.call_count, 2)
            imageio_encode.assert_not_called()
            cv2_encode.assert_not_called()

    def test_video_backend_falls_back_to_cv2_when_imageio_ffmpeg_unavailable_and_imageio_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames = self._minimal_video_frames(root)
            output = root / "output"
            dependencies = {
                "imageio_ffmpeg": {"available": False},
                "imageio": {"available": True},
                "cv2": {"available": True},
            }

            with mock.patch.object(real_export_module, "_encode_video_imageio_ffmpeg") as imageio_ffmpeg_encode, mock.patch.object(
                real_export_module, "_encode_video_imageio", side_effect=ValueError("imageio failed")
            ) as imageio_encode, mock.patch.object(
                real_export_module, "_encode_video_cv2", side_effect=self._write_fake_video
            ) as cv2_encode:
                _, video_backends, backend_errors = real_export_module._write_videos(
                    root, output, 0, frames, 30.0, dependencies
                )

            self.assertEqual(set(video_backends.values()), {"cv2"})
            self.assertTrue(all("imageio failed" in error for error in backend_errors))
            imageio_ffmpeg_encode.assert_not_called()
            self.assertEqual(imageio_encode.call_count, 2)
            self.assertEqual(cv2_encode.call_count, 2)

    def test_video_backend_falls_back_to_imageio_when_imageio_ffmpeg_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames = self._minimal_video_frames(root)
            output = root / "output"
            dependencies = {
                "imageio_ffmpeg": {"available": True},
                "imageio": {"available": True},
                "cv2": {"available": True},
            }

            with mock.patch.object(
                real_export_module, "_encode_video_imageio_ffmpeg", side_effect=ValueError("ffmpeg failed")
            ) as imageio_ffmpeg_encode, mock.patch.object(
                real_export_module, "_encode_video_imageio", side_effect=self._write_fake_video
            ) as imageio_encode, mock.patch.object(real_export_module, "_encode_video_cv2") as cv2_encode:
                _, video_backends, backend_errors = real_export_module._write_videos(
                    root, output, 0, frames, 30.0, dependencies
                )

            self.assertEqual(set(video_backends.values()), {"imageio"})
            self.assertTrue(all("imageio_ffmpeg failed" in error for error in backend_errors))
            self.assertEqual(imageio_ffmpeg_encode.call_count, 2)
            self.assertEqual(imageio_encode.call_count, 2)
            cv2_encode.assert_not_called()

    def test_imageio_ffmpeg_helper_uses_bundled_executable_and_deterministic_options(self):
        class FakeImageioFfmpeg:
            __version__ = "1.2.3"

            @staticmethod
            def get_ffmpeg_exe() -> str:
                return "/bundled/ffmpeg"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_paths = []
            for frame_index in range(2):
                image_path = root / f"frame_{frame_index:06d}.png"
                image_path.write_bytes(b"fake 16x16 png placeholder")
                image_paths.append(image_path)
            output = root / "episode_000000.mp4"

            with mock.patch.object(
                real_export_module.importlib, "import_module", return_value=FakeImageioFfmpeg
            ), mock.patch.object(real_export_module.subprocess, "run") as run:
                real_export_module._encode_video_imageio_ffmpeg(image_paths, output, 12.5)

            command = run.call_args.args[0]
            self.assertEqual(command[0], "/bundled/ffmpeg")
            self.assertIn("-y", command)
            self.assertIn("-framerate", command)
            self.assertEqual(command[command.index("-framerate") + 1], "12.5")
            self.assertIn("-c:v", command)
            self.assertEqual(command[command.index("-c:v") + 1], "libx264")
            self.assertIn("-pix_fmt", command)
            self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
            self.assertIn("-threads", command)
            self.assertEqual(command[command.index("-threads") + 1], "1")
            self.assertNotIn("scale", " ".join(command))

    def test_forcevla_13d_dry_run_writes_report_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_skeleton(root, "forcevla_13d", "symlink")
            output = root / "real_lerobot" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            report_path = write_real_lerobot_export(skeleton, output, mode="dry-run")

            result = validate_real_lerobot_export_attempt(output)
            self.assertTrue(result.ok, result.errors)
            report = self._read_report(report_path)
            self.assertEqual(report["mode"], "dry-run")
            self.assertFalse(report["parquet_written"])
            self.assertFalse(report["videos_written"])
            self.assertFalse(report["metadata_written"])

    def test_forcevla_13d_write_if_available_reports_conditional_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_skeleton(root, "forcevla_13d", "symlink")
            output = root / "real_lerobot" / "forcevla_13d" / "doosan_peg_in_hole_v0"

            report_path = write_real_lerobot_export(skeleton, output, mode="write-if-available")

            result = validate_real_lerobot_export_attempt(output)
            self.assertTrue(result.ok, result.errors)
            report = self._read_report(report_path)
            self.assertEqual(report["mode"], "write-if-available")
            self.assertTrue(report["metadata_written"])

            parquet_path = output / "data" / "chunk-000" / "episode_000000.parquet"
            if report["parquet_ready"]:
                self.assertTrue(report["parquet_written"], report["skipped_reasons"])
                self.assertTrue(parquet_path.is_file())
            else:
                self.assertFalse(report["parquet_written"])
                self.assertTrue(any("pyarrow" in reason for reason in report["skipped_reasons"]))

            video_paths = [
                output / "videos" / "observation.image" / "episode_000000.mp4",
                output / "videos" / "observation.wrist_image" / "episode_000000.mp4",
            ]
            if report["videos_written"]:
                self.assertTrue(all(path.is_file() for path in video_paths))
            else:
                reasons = " ".join(str(reason) for reason in report["skipped_reasons"])
                self.assertRegex(reasons, "video|imageio|cv2|encoding")

    def test_doosan_full_25d_dry_run_reports_profile_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skeleton = self._build_skeleton(root, "doosan_full_25d", "copy")
            output = root / "real_lerobot" / "doosan_full_25d" / "doosan_peg_in_hole_v0"

            report_path = write_real_lerobot_export(skeleton, output, mode="dry-run")

            result = validate_real_lerobot_export_attempt(output)
            self.assertTrue(result.ok, result.errors)
            report = self._read_report(report_path)
            self.assertEqual(report["profile"], "doosan_full_25d")
            self.assertEqual(report["state_dim"], 25)
            self.assertEqual(report["action_dim"], 7)
            self.assertFalse(report["parquet_written"])
            self.assertFalse(report["videos_written"])


if __name__ == "__main__":
    unittest.main()
