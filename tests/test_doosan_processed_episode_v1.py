from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

from doosan_forcevla_data.convert import doosan_processed_episode_v1 as module
from doosan_forcevla_data.ingest.doosan_raw_v1 import ImageRecord, RecordStamp
from doosan_forcevla_data.validate.validate_doosan_processed_episode_v1 import (
    validate_doosan_processed_episode_v1,
)


@dataclass(frozen=True)
class _Calibration:
    name: str

    def to_dict(self):
        return {"name": self.name}


@dataclass(frozen=True)
class _Inputs:
    tcp_calibration: _Calibration
    external_calibration: _Calibration


class _Selection:
    def __init__(self, source_index: int, timestamp_ns: int):
        self.source_indices = (source_index,)
        self.source_timestamps_ns = (timestamp_ns,)
        self.alpha = None


class _Decision:
    def __init__(self, source_index: int, timestamp_ns: int):
        self.selection = _Selection(source_index, timestamp_ns)


class _SourcePlan:
    def __init__(self, count: int):
        self.decisions = tuple(_Decision(index * 2, index * 100 + 7) for index in range(count))


class _Plan:
    def __init__(self, count: int):
        self._external = _SourcePlan(count)

    def source_plan(self, name: str):
        if name != "external_image":
            raise KeyError(name)
        return self._external


class _Sync:
    def __init__(self, count: int):
        self.dropped_reference_count = 0
        self.plan = _Plan(count)
        self.inputs = _Inputs(_Calibration("tcp"), _Calibration("external"))


@dataclass(frozen=True)
class _State:
    open_fraction: float


@dataclass(frozen=True)
class _ForceSample:
    reference_index: int
    reference_timestamp_ns: int


@dataclass(frozen=True)
class _GripperSample:
    reference_index: int
    reference_timestamp_ns: int
    state: _State


@dataclass(frozen=True)
class _Action:
    source_reference_index: int
    target_reference_index: int
    source_reference_timestamp_ns: int
    target_reference_timestamp_ns: int
    target: float

    def to_vector(self):
        return (0.001, 0.002, 0.003, 0.01, 0.02, 0.03, self.target)


class _ForceEpisode:
    def __init__(self, count: int):
        self.samples = tuple(_ForceSample(i, 1000 + i * 10) for i in range(count))


class _GripperEpisode:
    def __init__(self, count: int):
        self.samples = tuple(
            _GripperSample(i, 1000 + i * 10, _State(0.0 if i < count - 1 else 1.0))
            for i in range(count)
        )


class _ActionEpisode:
    def __init__(self, count: int):
        self.state_count = count
        self.action_count = count - 1
        self.terminal_reference_index = count - 1
        self.actions = tuple(
            _Action(i, i + 1, 1000 + i * 10, 1000 + (i + 1) * 10, 1.0 if i + 1 == count - 1 else 0.0)
            for i in range(count - 1)
        )


class DoosanProcessedEpisodeV1Tests(unittest.TestCase):
    def test_episode_task_requires_exact_provenance_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "episode_operator.json").write_text(
                json.dumps({"task": "real_robot_demonstration"}), encoding="utf-8"
            )
            (root / "episode_validation.json").write_text(
                json.dumps({"metadata": {"custom_data": {"task": "real_robot_demonstration"}}}),
                encoding="utf-8",
            )
            self.assertEqual(module._episode_task(root), "real_robot_demonstration")
            (root / "episode_validation.json").write_text(
                json.dumps({"metadata": {"custom_data": {"task": "different"}}}), encoding="utf-8"
            )
            with self.assertRaises(module.ProcessedEpisodeError):
                module._episode_task(root)

    def test_rgb_payload_accepts_only_frozen_native_rgb8(self):
        data = bytes(640 * 480 * 3)
        record = ImageRecord(
            stamp=RecordStamp(1, 2, "camera"),
            height=480,
            width=640,
            encoding="rgb8",
            is_bigendian=0,
            step=640 * 3,
            data=memoryview(data),
        )
        self.assertEqual(module._rgb_payload(record, camera="tcp_camera"), data)

        wrong = ImageRecord(
            stamp=record.stamp,
            height=480,
            width=848,
            encoding="rgb8",
            is_bigendian=0,
            step=848 * 3,
            data=memoryview(bytes(848 * 480 * 3)),
        )
        with self.assertRaises(module.ProcessedEpisodeError):
            module._rgb_payload(wrong, camera="tcp_camera")

    def test_build_rows_excludes_terminal_and_uses_exact_patch_layers(self):
        count = 4
        force_episode = _ForceEpisode(count)
        gripper_episode = _GripperEpisode(count)
        actions = _ActionEpisode(count)
        states = [tuple(float(i + j) for j in range(25)) for i in range(count)]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "episode_operator.json").write_text(
                json.dumps({"task": "real_robot_demonstration"}), encoding="utf-8"
            )
            (root / "episode_validation.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "custom_data": {
                                "task": "real_robot_demonstration",
                                "episode_index": 10,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(module, "build_doosan_sync_plan", return_value=_Sync(count)),
                mock.patch.object(module, "build_doosan_force_proprio_episode", return_value=force_episode),
                mock.patch.object(module, "build_doosan_gripper_episode", return_value=gripper_episode),
                mock.patch.object(module, "build_doosan_measured_action_episode", return_value=actions),
                mock.patch.object(module, "assemble_forcevla_v2_observation_states", return_value=states),
            ):
                rows, metadata = module.build_processed_rows(root)

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["reference_index"] for row in rows], [0, 1, 2])
        self.assertEqual([row["action_target_reference_index"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[-1]["action_7d"][-1], 1.0)
        self.assertEqual(rows[1]["cameras"]["external_camera_2"]["source_index"], 2)
        self.assertEqual(metadata["synchronized_state_count"], 4)
        self.assertEqual(metadata["frame_count"], 3)
        self.assertEqual(metadata["excluded_terminal_reference_index"], 3)
        self.assertFalse(metadata["terminal_action_emitted"])
        self.assertEqual(metadata["physical_camera_count"], 2)
        self.assertEqual(set(metadata["cameras"]), {"tcp_camera", "external_camera_2"})
        self.assertEqual(metadata["task"], "real_robot_demonstration")


    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg and ffprobe")
    def test_materializes_exactly_two_native_resolution_videos(self):
        def image(width: int, height: int, timestamp_ns: int) -> ImageRecord:
            return ImageRecord(
                stamp=RecordStamp(timestamp_ns, timestamp_ns, "camera"),
                height=height,
                width=width,
                encoding="rgb8",
                is_bigendian=0,
                step=width * 3,
                data=memoryview(bytes([17]) * (width * height * 3)),
            )

        rows = [
            {
                "reference_index": 0,
                "cameras": {
                    "tcp_camera": {"source_index": 0, "header_timestamp_ns": 100},
                    "external_camera_2": {"source_index": 0, "header_timestamp_ns": 200},
                },
            },
            {
                "reference_index": 1,
                "cameras": {
                    "tcp_camera": {"source_index": 1, "header_timestamp_ns": 101},
                    "external_camera_2": {"source_index": 2, "header_timestamp_ns": 202},
                },
            },
        ]
        stream = [
            (module.TCP_IMAGE_TOPIC, image(640, 480, 100)),
            (module.EXTERNAL_IMAGE_TOPIC, image(848, 480, 200)),
            (module.EXTERNAL_IMAGE_TOPIC, image(848, 480, 201)),
            (module.TCP_IMAGE_TOPIC, image(640, 480, 101)),
            (module.EXTERNAL_IMAGE_TOPIC, image(848, 480, 202)),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(module, "iter_typed_messages", return_value=iter(stream)):
                reports = module._materialize_native_videos(
                    Path("/fake/raw"), rows, Path(tmpdir)
                )
            self.assertEqual(set(reports), {"tcp_camera", "external_camera_2"})
            self.assertEqual(reports["tcp_camera"].decoded_frames, 2)
            self.assertEqual((reports["tcp_camera"].width, reports["tcp_camera"].height), (640, 480))
            self.assertEqual(reports["external_camera_2"].decoded_frames, 2)
            self.assertEqual(
                (reports["external_camera_2"].width, reports["external_camera_2"].height),
                (848, 480),
            )
            self.assertFalse((Path(tmpdir) / "videos" / "external_camera_1.mp4").exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg and ffprobe")
    def test_production_processed_validator_accepts_two_native_videos(self):
        def image(width: int, height: int, timestamp_ns: int) -> ImageRecord:
            return ImageRecord(
                stamp=RecordStamp(timestamp_ns, timestamp_ns, "camera"),
                height=height,
                width=width,
                encoding="rgb8",
                is_bigendian=0,
                step=width * 3,
                data=memoryview(bytes([31]) * (width * height * 3)),
            )

        rows = []
        for i in range(2):
            rows.append(
                {
                    "frame_index": i,
                    "reference_index": i,
                    "reference_timestamp_ns": 100 + i,
                    "lerobot_timestamp": i / 30,
                    "observation_state_25d": [float(i + j) for j in range(25)],
                    "action_7d": [float(i + j) for j in range(7)],
                    "action_target_reference_index": i + 1,
                    "action_target_reference_timestamp_ns": 101 + i,
                    "cameras": {
                        "tcp_camera": {"source_index": i, "header_timestamp_ns": 100 + i},
                        "external_camera_2": {"source_index": 2 * i, "header_timestamp_ns": 200 + 2 * i},
                    },
                }
            )
        stream = [
            (module.TCP_IMAGE_TOPIC, image(640, 480, 100)),
            (module.EXTERNAL_IMAGE_TOPIC, image(848, 480, 200)),
            (module.EXTERNAL_IMAGE_TOPIC, image(848, 480, 201)),
            (module.TCP_IMAGE_TOPIC, image(640, 480, 101)),
            (module.EXTERNAL_IMAGE_TOPIC, image(848, 480, 202)),
        ]
        metadata = {
            "schema_version": module.PROCESSED_SCHEMA_ID,
            "fps": 30,
            "state_dim": 25,
            "action_dim": 7,
            "frame_count": 2,
            "measured_action_count": 2,
            "synchronized_state_count": 3,
            "excluded_terminal_reference_index": 2,
            "terminal_action_emitted": False,
            "physical_camera_count": 2,
            "cameras": module.CAMERA_SPECS,
            "synthetic_model_slot": module.SYNTHETIC_MODEL_SLOT,
            "task": "real_robot_demonstration",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with mock.patch.object(module, "iter_typed_messages", return_value=iter(stream)):
                module._materialize_native_videos(Path("/fake/raw"), rows, root)
            (root / "metadata_processed.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with (root / "frames.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            result = validate_doosan_processed_episode_v1(root)
            self.assertTrue(result.ok, result.errors)
            self.assertEqual(result.frame_count, 2)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires ffmpeg and ffprobe")
    def test_video_materialization_allows_repeated_nearest_external_frame(self):
        def image(width: int, height: int, timestamp_ns: int) -> ImageRecord:
            return ImageRecord(
                stamp=RecordStamp(timestamp_ns, timestamp_ns, "camera"),
                height=height,
                width=width,
                encoding="rgb8",
                is_bigendian=0,
                step=width * 3,
                data=memoryview(bytes([23]) * (width * height * 3)),
            )

        rows = [
            {
                "reference_index": 0,
                "cameras": {
                    "tcp_camera": {"source_index": 0, "header_timestamp_ns": 100},
                    "external_camera_2": {"source_index": 0, "header_timestamp_ns": 200},
                },
            },
            {
                "reference_index": 1,
                "cameras": {
                    "tcp_camera": {"source_index": 1, "header_timestamp_ns": 101},
                    "external_camera_2": {"source_index": 0, "header_timestamp_ns": 200},
                },
            },
        ]
        stream = [
            (module.TCP_IMAGE_TOPIC, image(640, 480, 100)),
            (module.EXTERNAL_IMAGE_TOPIC, image(848, 480, 200)),
            (module.TCP_IMAGE_TOPIC, image(640, 480, 101)),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(module, "iter_typed_messages", return_value=iter(stream)):
                reports = module._materialize_native_videos(Path("/fake/raw"), rows, Path(tmpdir))
            self.assertEqual(reports["external_camera_2"].decoded_frames, 2)

    def test_single_nearest_selection_rejects_interpolation(self):
        decision = types.SimpleNamespace(
            selection=types.SimpleNamespace(
                source_indices=(1, 2),
                source_timestamps_ns=(10, 20),
                alpha=0.5,
            )
        )
        with self.assertRaises(module.ProcessedEpisodeError):
            module._single_nearest_source_index(decision, source="external", reference_index=0)


if __name__ == "__main__":
    unittest.main()
