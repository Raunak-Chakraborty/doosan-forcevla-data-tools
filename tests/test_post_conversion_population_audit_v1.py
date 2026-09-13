from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from doosan_forcevla_data.audit import post_conversion_population_v1 as audit
from doosan_forcevla_data.convert.model_state_profile_v1 import StateMode, model_state_layout
from doosan_forcevla_data.convert.orientation_representation_v1 import (
    OrientationRepresentation,
    encode_orientation,
    rotvec_to_matrix,
)
from doosan_forcevla_data.validate.validate_doosan_lerobot_v21 import LeRobotValidationResult


class PostConversionPopulationAuditV1Tests(unittest.TestCase):
    def _make_bundle(self, root: Path, *, tamper_action: bool = False) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        bundle = root / "episode_000010"
        canonical_rotvecs = [(0.2, -0.1, 0.4), (0.25, -0.08, 0.45)]
        matrices = [rotvec_to_matrix(value) for value in canonical_rotvecs]
        previous_continuous = None

        encoded_by_representation = {}
        for representation in OrientationRepresentation:
            sequence = []
            previous_continuous = None
            for matrix in matrices:
                encoded = encode_orientation(
                    matrix,
                    representation,
                    previous_continuous_rotvec_rad=previous_continuous,
                )
                sequence.append(encoded)
                if representation is OrientationRepresentation.ROTVEC_CONTINUOUS:
                    previous_continuous = encoded
            encoded_by_representation[representation] = sequence

        for spec in audit.PROFILE_SPECS:
            dataset = bundle / spec.directory_name
            (dataset / "meta").mkdir(parents=True)
            parquet = dataset / "data" / "chunk-000" / "episode_000000.parquet"
            parquet.parent.mkdir(parents=True)

            states = []
            for index in range(2):
                orientation = encoded_by_representation[spec.layout.orientation_representation][index]
                state = [
                    0.1 + index,
                    0.2 + index,
                    0.3 + index,
                    *orientation,
                    1.0,
                    *[0.01 * j for j in range(6)],
                    *[0.02 * j for j in range(6)],
                ]
                if spec.layout.include_wrench:
                    state.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                states.append(state)

            actions = [
                [0.01, 0.02, 0.03, 0.001, 0.002, 0.003, 1.0],
                [0.04, 0.05, 0.06, 0.004, 0.005, 0.006, 0.0],
            ]
            if tamper_action and spec.directory_name == "quaternion__full":
                actions[1][0] = 999.0

            table = pa.Table.from_pylist(
                [
                    {
                        "observation.state": states[index],
                        "action": actions[index],
                        "timestamp": index / 30.0,
                        "frame_index": index,
                        "episode_index": 0,
                        "index": index,
                        "task_index": 0,
                    }
                    for index in range(2)
                ]
            )
            pq.write_table(table, parquet)

            info = {
                "features": {
                    "observation.state": {
                        "dtype": "float64",
                        "shape": [spec.layout.state_dim],
                        "names": list(spec.layout.state_fields),
                    }
                }
            }
            (dataset / "meta" / "info.json").write_text(json.dumps(info))
            (dataset / "meta" / "tasks.jsonl").write_text(
                json.dumps({"task_index": 0, "task": "insert peg"}) + "\n"
            )
            (dataset / "meta" / "episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "tasks": ["insert peg"], "length": 2}) + "\n"
            )
            provenance = {
                "source_processed_episode": "/processed/episode_000010",
                "source_raw_episode": "/raw/episode_000010",
                "source_episode_index": 10,
                "task": "insert peg",
                "frame_count": 2,
                "state_dim": spec.layout.state_dim,
                "action_dim": 7,
                "row_policy": "test",
                "terminal_policy": {"terminal_action_emitted": False},
                "camera_mapping": {"same": True},
                "synthetic_right_wrist": {"same": True},
                "timestamp_policy": "regularized",
                "target_forcevla_commit": "abc",
                "target_lerobot_commit": "def",
                "target_dlimp_commit": "ghi",
            }
            if not spec.layout.is_legacy_default:
                provenance["model_state_profile"] = spec.layout.to_metadata()
            (dataset / "meta" / "export_provenance.json").write_text(
                json.dumps(provenance)
            )
            for video_key in audit.VIDEO_KEYS:
                video = dataset / "videos" / "chunk-000" / video_key / "episode_000000.mp4"
                video.parent.mkdir(parents=True)
                video.write_bytes(("same-video-" + video_key).encode())

        return bundle

    @mock.patch.object(
        audit,
        "validate_doosan_lerobot_v21",
        return_value=LeRobotValidationResult(True, (), 2),
    )
    def test_complete_bundle_passes_cross_profile_equivalence(self, _validator):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = self._make_bundle(Path(tmpdir))
            report = audit.audit_post_conversion_population(bundle, expected_bundles=1)
            self.assertEqual(report["audit_gate"], "PASS")
            self.assertEqual(report["successful_bundle_count"], 1)
            self.assertEqual(report["total_rows_per_profile"], 2)
            self.assertLessEqual(report["max_orientation_matrix_abs_error"], 1e-9)

    @mock.patch.object(
        audit,
        "validate_doosan_lerobot_v21",
        return_value=LeRobotValidationResult(True, (), 2),
    )
    def test_action_mismatch_fails_bundle(self, _validator):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = self._make_bundle(Path(tmpdir), tamper_action=True)
            report = audit.audit_post_conversion_population(bundle)
            self.assertEqual(report["audit_gate"], "FAIL")
            self.assertEqual(report["failed_bundle_count"], 1)
            self.assertIn("exact action mismatch", report["failures"][0]["error"])

    def test_discovery_accepts_population_parent_and_requires_all_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            complete = root / "complete"
            incomplete = root / "incomplete"
            for name in audit.EXPECTED_PROFILE_NAMES:
                (complete / name).mkdir(parents=True)
            (incomplete / audit.EXPECTED_PROFILE_NAMES[0]).mkdir(parents=True)
            self.assertEqual(audit.discover_bundle_roots(root), [complete])

    @mock.patch.object(
        audit,
        "validate_doosan_lerobot_v21",
        return_value=LeRobotValidationResult(True, (), 2),
    )
    def test_expected_source_file_is_fail_closed(self, _validator):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = self._make_bundle(root)
            expected = root / "expected.txt"
            expected.write_text("/processed/wrong_episode\n")
            report = audit.audit_post_conversion_population(
                bundle,
                expected_sources_file=expected,
            )
            self.assertEqual(report["audit_gate"], "FAIL")
            self.assertTrue(report["population_issues"])


if __name__ == "__main__":
    unittest.main()
