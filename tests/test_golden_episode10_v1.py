from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from doosan_forcevla_data.validate import golden_episode10_v1 as module


class GoldenEpisode10ConstantsTests(unittest.TestCase):
    def test_frozen_identity_constants_match_patch8_endpoint(self):
        self.assertEqual(module.GOLDEN_EPISODE_INDEX, 10)
        self.assertEqual(module.GOLDEN_TASK, "real_robot_demonstration")
        self.assertEqual(module.GOLDEN_FRAME_COUNT, 1008)
        self.assertEqual(module.GOLDEN_SYNCHRONIZED_STATE_COUNT, 1009)
        self.assertEqual(module.GOLDEN_RELEASE_SOURCE_REFERENCE_INDEX, 955)
        self.assertEqual(module.GOLDEN_RELEASE_TARGET_REFERENCE_INDEX, 956)
        self.assertEqual(module.FROZEN_FORCEVLA_COMMIT, "9b61abef116f207d587d10aaf30170b73757c3e0")
        self.assertEqual(module.FROZEN_LEROBOT_COMMIT, "e7aea92dd833f83d163820dcf2e58250307697a4")
        self.assertEqual(module.FROZEN_DLIMP_COMMIT, "5edaa4691567873d495633f2708982b42edf1972")

    def test_runtime_uses_pinned_lerobot_video_keys_api(self):
        source = inspect.getsource(
            module.validate_golden_forcevla_runtime
        )
        self.assertIn(
            "len(dataset.meta.video_keys)",
            source,
        )
        self.assertNotIn(
            "dataset.meta.total_videos",
            source,
        )


class GoldenRawIdentityTests(unittest.TestCase):
    def _descriptor(self):
        topics = tuple(
            types.SimpleNamespace(name=name, message_count=count)
            for name, count in module.GOLDEN_TOPIC_COUNTS.items()
        )
        metadata = types.SimpleNamespace(
            relative_file_paths=(module.GOLDEN_MCAP_FILENAME,),
            message_count=module.GOLDEN_MESSAGE_COUNT,
            topics=topics,
            custom_data={
                "episode_index": 10,
                "task": module.GOLDEN_TASK,
            },
        )
        return types.SimpleNamespace(
            metadata=metadata,
            operator={"task": module.GOLDEN_TASK},
            validation={
                "metadata": {
                    "custom_data": {
                        "episode_index": 10,
                        "task": module.GOLDEN_TASK,
                    }
                }
            },
        )

    def test_exact_raw_identity_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / module.GOLDEN_MCAP_FILENAME).write_bytes(b"golden")
            with (
                mock.patch.object(module, "validate_doosan_raw_v1_episode", return_value=self._descriptor()),
                mock.patch.object(module, "_sha256_file", return_value=module.GOLDEN_MCAP_SHA256),
            ):
                report = module.validate_golden_raw_episode(root)
        self.assertEqual(report["message_count"], module.GOLDEN_MESSAGE_COUNT)
        self.assertEqual(report["mcap_sha256"], module.GOLDEN_MCAP_SHA256)

    def test_topic_count_drift_is_rejected(self):
        descriptor = self._descriptor()
        topic_list = list(descriptor.metadata.topics)
        topic_list[0] = types.SimpleNamespace(
            name=topic_list[0].name,
            message_count=topic_list[0].message_count + 1,
        )
        descriptor.metadata = types.SimpleNamespace(
            relative_file_paths=descriptor.metadata.relative_file_paths,
            message_count=descriptor.metadata.message_count,
            topics=tuple(topic_list),
            custom_data=descriptor.metadata.custom_data,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / module.GOLDEN_MCAP_FILENAME).write_bytes(b"golden")
            with mock.patch.object(module, "validate_doosan_raw_v1_episode", return_value=descriptor):
                with self.assertRaisesRegex(module.GoldenEpisode10Error, "per-topic counts"):
                    module.validate_golden_raw_episode(root)


class GoldenProcessedContractTests(unittest.TestCase):
    def _rows(self):
        rows = []
        for index in range(module.GOLDEN_FRAME_COUNT):
            state_gripper = 0.0 if index < module.GOLDEN_HELD_STATE_ROW_COUNT else 1.0
            action_gripper = 0.0 if index < module.GOLDEN_HELD_ACTION_TARGET_COUNT else 1.0
            rows.append(
                {
                    "frame_index": index,
                    "reference_index": index,
                    "reference_timestamp_ns": 1_000_000_000 + index * 33_000_000,
                    "lerobot_timestamp": index / 30,
                    "observation_state_25d": [0.0] * 6 + [state_gripper] + [0.0] * 18,
                    "action_7d": [0.0] * 6 + [action_gripper],
                    "action_target_reference_index": index + 1,
                    "action_target_reference_timestamp_ns": 1_000_000_000 + (index + 1) * 33_000_000,
                    "cameras": {
                        "tcp_camera": {
                            "source_index": index,
                            "header_timestamp_ns": 1_000_000_000 + index * 33_000_000,
                        },
                        "external_camera_2": {
                            "source_index": index * 2,
                            "header_timestamp_ns": 1_000_000_010 + index * 33_000_000,
                        },
                    },
                }
            )
        return rows

    def _write_fixture(self, root: Path, rows):
        metadata = {
            "schema_version": module.PROCESSED_SCHEMA_ID,
            "source_raw_episode": "/tmp/episode_000010",
            "source_episode_index": 10,
            "task": module.GOLDEN_TASK,
            "fps": 30,
            "synchronized_state_count": 1009,
            "frame_count": 1008,
            "measured_action_count": 1008,
            "excluded_terminal_reference_index": 1008,
            "terminal_action_emitted": False,
            "state_dim": 25,
            "action_dim": 7,
            "physical_camera_count": 2,
            "cameras": module.CAMERA_SPECS,
            "synthetic_model_slot": module.SYNTHETIC_MODEL_SLOT,
        }
        (root / "metadata_processed.json").write_text(json.dumps(metadata), encoding="utf-8")
        with (root / "frames.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    def test_processed_contract_freezes_release_and_terminal_policy(self):
        rows = self._rows()
        digest = module._canonical_jsonl_sha256(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fixture(root, rows)
            validation = types.SimpleNamespace(ok=True, errors=(), frame_count=1008)
            with (
                mock.patch.object(module, "validate_doosan_processed_episode_v1", return_value=validation),
                mock.patch.object(module, "GOLDEN_FRAMES_CANONICAL_SHA256", digest),
            ):
                report = module.validate_golden_processed_episode(root)
        self.assertEqual(report["action_gripper_target_counts"], {"held": 955, "released": 53})
        self.assertEqual(report["processed_state_gripper_counts"], {"held": 956, "released": 52})
        self.assertEqual(report["release_source_reference_index"], 955)
        self.assertEqual(report["release_target_reference_index"], 956)

    def test_fabricated_terminal_row_is_rejected_by_count(self):
        rows = self._rows() + [self._rows()[-1]]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_fixture(root, rows)
            validation = types.SimpleNamespace(ok=True, errors=(), frame_count=len(rows))
            with mock.patch.object(module, "validate_doosan_processed_episode_v1", return_value=validation):
                with self.assertRaisesRegex(module.GoldenEpisode10Error, "row count"):
                    module.validate_golden_processed_episode(root)


class ParquetSemanticComparisonTests(unittest.TestCase):
    def test_parquet_comparison_requires_exact_state_and_action(self):
        processed = []
        parquet = []
        for index in range(module.GOLDEN_FRAME_COUNT):
            state = [float(index)] * 25
            action = [float(index)] * 6 + [0.0 if index < 955 else 1.0]
            processed.append({"observation_state_25d": state, "action_7d": action})
            parquet.append(
                {
                    "observation.state": list(state),
                    "action": list(action),
                    "timestamp": index / 30,
                    "frame_index": index,
                    "episode_index": 0,
                    "index": index,
                    "task_index": 0,
                }
            )
        module._compare_parquet_rows(processed, parquet)
        parquet[10]["action"][0] += 1.0
        with self.assertRaisesRegex(module.GoldenEpisode10Error, "action changed"):
            module._compare_parquet_rows(processed, parquet)


if __name__ == "__main__":
    unittest.main()
