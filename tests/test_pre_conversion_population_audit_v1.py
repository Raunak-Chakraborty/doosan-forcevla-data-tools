from __future__ import annotations

import math
from unittest import mock
import unittest

from doosan_forcevla_data.audit.pre_conversion_population_v1 import (
    AuditThresholds,
    aggregate_results,
    audit_gripper_right_samples,
    audit_joint_positions,
    audit_orientation_profiles,
    available_cpu_count,
    available_logical_cpu_count,
    available_physical_core_count,
    resolve_worker_count,
    screen_force_contact,
)
from doosan_forcevla_data.convert.orientation_representation_v1 import (
    matrix_to_principal_rotvec,
    rotvec_to_matrix,
)


def _row(index, rotvec=(0.0, 0.0, 0.0), *, joint1=0.0, gripper=0.0, target=0.0, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)):
    state = [
        0.1,
        0.2,
        0.3,
        *rotvec,
        gripper,
        joint1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        *force,
        *torque,
    ]
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, target]
    return {
        "frame_index": index,
        "reference_index": index,
        "reference_timestamp_ns": index * 100_000_000,
        "observation_state_25d": state,
        "action_7d": action,
        "action_target_reference_index": index + 1,
        "action_target_reference_timestamp_ns": (index + 1) * 100_000_000,
        "lerobot_timestamp": index / 30.0,
    }


class WorkerSelectionTests(unittest.TestCase):
    def test_affinity_controls_available_logical_cpu_count(self):
        with mock.patch("os.sched_getaffinity", return_value={2, 4, 7}), mock.patch(
            "os.cpu_count", return_value=64
        ):
            self.assertEqual(available_logical_cpu_count(), 3)

    def test_physical_core_count_collapses_smt_siblings(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = {
                0: (0, 0),
                1: (0, 0),
                2: (0, 1),
                3: (0, 1),
            }
            for cpu_id, (package_id, core_id) in topology.items():
                directory = root / f"cpu{cpu_id}" / "topology"
                directory.mkdir(parents=True)
                (directory / "physical_package_id").write_text(str(package_id))
                (directory / "core_id").write_text(str(core_id))

            with mock.patch("os.sched_getaffinity", return_value={0, 1, 2, 3}):
                self.assertEqual(
                    available_physical_core_count(topology_root=root), 2
                )

    def test_physical_core_count_respects_affinity(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = {
                0: (0, 0),
                1: (0, 0),
                2: (0, 1),
                3: (0, 1),
            }
            for cpu_id, (package_id, core_id) in topology.items():
                directory = root / f"cpu{cpu_id}" / "topology"
                directory.mkdir(parents=True)
                (directory / "physical_package_id").write_text(str(package_id))
                (directory / "core_id").write_text(str(core_id))

            with mock.patch("os.sched_getaffinity", return_value={0, 1, 2}):
                self.assertEqual(
                    available_physical_core_count(topology_root=root), 2
                )

    def test_missing_topology_falls_back_to_logical_count(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "os.sched_getaffinity", return_value={2, 4, 7}
        ):
            self.assertEqual(
                available_physical_core_count(topology_root=Path(temporary)), 3
            )

    def test_auto_is_capped_at_four_physical_cores(self):
        self.assertEqual(resolve_worker_count("auto", available=2), 2)
        self.assertEqual(resolve_worker_count("auto", available=4), 4)
        self.assertEqual(resolve_worker_count("auto", available=8), 4)
        self.assertEqual(resolve_worker_count("auto", available=64), 4)
        self.assertEqual(resolve_worker_count(None, available=8), 4)

    def test_explicit_worker_count_may_raise_parallelism_within_physical_limit(self):
        self.assertEqual(resolve_worker_count("4", available=8), 4)
        self.assertEqual(resolve_worker_count("8", available=8), 8)
        with self.assertRaises(ValueError):
            resolve_worker_count(9, available=8)
        with self.assertRaises(ValueError):
            resolve_worker_count(0, available=8)

    def test_available_cpu_count_alias_is_positive(self):
        self.assertGreaterEqual(available_cpu_count(), 1)


class OrientationPopulationAuditTests(unittest.TestCase):
    def test_all_profiles_preserve_action_and_physical_rotation(self):
        rows = [_row(0, (0.1, -0.2, 0.3)), _row(1, (0.101, -0.2, 0.3))]
        result = audit_orientation_profiles(rows, AuditThresholds())
        self.assertEqual(result["action_mismatch_count"], 0)
        self.assertEqual(result["nonorientation_mismatch_count"], 0)
        self.assertEqual(result["full_vs_no_wrench_mismatch_count"], 0)
        self.assertEqual(result["physical_rotation_mismatch_count"], 0)
        self.assertEqual(result["continuous_large_jump_count"], 0)
        self.assertEqual(result["rotation6d_large_jump_count"], 0)
        self.assertEqual(result["profile_dimensions"]["quaternion/full"], 26)
        self.assertEqual(result["profile_dimensions"]["rotation6d/full"], 28)

    def test_pi_branch_is_reported_but_continuous_and_6d_remain_clean(self):
        eps = 1e-3
        first = (math.pi - eps, 0.0, 0.0)
        second = matrix_to_principal_rotvec(rotvec_to_matrix((math.pi + eps, 0.0, 0.0)))
        rows = [_row(0, first), _row(1, second)]
        result = audit_orientation_profiles(rows, AuditThresholds())
        self.assertEqual(result["principal_branch_event_count"], 1)
        self.assertEqual(result["quaternion_large_jump_count"], 1)
        self.assertEqual(result["continuous_large_jump_count"], 0)
        self.assertEqual(result["rotation6d_large_jump_count"], 0)
        self.assertEqual(result["issues"], [])


class JointAuditTests(unittest.TestCase):
    def test_small_joint_motion_is_clean(self):
        rows = [_row(0, joint1=0.0), _row(1, joint1=0.01)]
        result = audit_joint_positions(rows, AuditThresholds())
        self.assertEqual(result["large_jump_count"], 0)
        self.assertEqual(result["classic_wrap_candidate_count"], 0)

    def test_classic_plus_pi_minus_pi_wrap_is_detected(self):
        rows = [_row(0, joint1=math.pi - 0.01), _row(1, joint1=-math.pi + 0.01)]
        result = audit_joint_positions(rows, AuditThresholds())
        self.assertEqual(result["large_jump_count"], 1)
        self.assertEqual(result["classic_wrap_candidate_count"], 1)
        self.assertEqual(result["issues"], ["JOINT_CLASSIC_WRAP_CANDIDATE"])


class GripperRightAuditTests(unittest.TestCase):
    def test_one_right_edge_and_one_release_is_clean(self):
        rows = [
            _row(0, gripper=0.0, target=0.0),
            _row(1, gripper=0.0, target=1.0),
        ]
        result = audit_gripper_right_samples(
            rows,
            right_edges_ns=[150_000_000],
            raw_holding_samples=[(0, True), (200_000_000, False)],
        )
        self.assertEqual(result["right_edge_count"], 1)
        self.assertEqual(result["effective_right_edge_ordinal"], 1)
        self.assertEqual(result["processed_release_action_count"], 1)
        self.assertEqual(result["issues"], [])

    def test_second_of_two_right_edges_must_be_effective(self):
        rows = [
            _row(0, gripper=0.0, target=0.0),
            _row(1, gripper=0.0, target=1.0),
        ]
        clean = audit_gripper_right_samples(
            rows,
            right_edges_ns=[50_000_000, 150_000_000],
            raw_holding_samples=[(0, True), (200_000_000, False)],
        )
        self.assertEqual(clean["effective_right_edge_ordinal"], 2)
        self.assertNotIn("DOUBLE_RIGHT_SECOND_EDGE_NOT_EFFECTIVE", clean["issues"])
        bad = audit_gripper_right_samples(
            rows,
            right_edges_ns=[50_000_000, 250_000_000],
            raw_holding_samples=[(0, True), (200_000_000, False)],
        )
        self.assertIn("DOUBLE_RIGHT_SECOND_EDGE_NOT_EFFECTIVE", bad["issues"])


class ForceScreenTests(unittest.TestCase):
    def _force_rows(self, forces):
        rows = []
        for index, force_x in enumerate(forces):
            target = 1.0 if index == len(forces) - 1 else 0.0
            rows.append(
                _row(
                    index,
                    gripper=0.0,
                    target=target,
                    force=(force_x, 0.0, 0.0),
                )
            )
        return rows

    def test_high_then_recovered_is_not_review(self):
        thresholds = AuditThresholds(baseline_sec=0.11)
        # baseline 0, then contact 20 N, then >=0.3 s below recovery threshold.
        rows = self._force_rows([0, 0, 20, 20, 0, 0, 0, 0])
        result = screen_force_contact(rows, thresholds)
        self.assertEqual(result.category, "HIGH_THEN_RECOVERED")
        self.assertFalse(result.broad_review_candidate)
        self.assertFalse(result.critical_review_candidate)

    def test_force_at_release_enters_critical_tier(self):
        thresholds = AuditThresholds(baseline_sec=0.11)
        rows = self._force_rows([0, 0, 35, 35, 35, 35, 35, 35])
        result = screen_force_contact(rows, thresholds)
        self.assertEqual(result.category, "CRITICAL_REVIEW")
        self.assertIn("HIGH_FORCE_AT_RELEASE", result.critical_review_reasons)

    def test_isolated_extreme_force_is_broad_review_not_critical(self):
        thresholds = AuditThresholds(baseline_sec=0.11)
        rows = self._force_rows([0, 0, 30, 0, 0, 0, 0, 0])
        result = screen_force_contact(rows, thresholds)
        self.assertEqual(result.category, "REVIEW")
        self.assertIn("EXTREME_FORCE", result.broad_review_reasons)
        self.assertEqual(result.critical_review_reasons, ())


class AggregateTests(unittest.TestCase):
    def test_structural_review_controls_conversion_gate_not_force_review(self):
        base = {
            "ok": True,
            "episode_number": 1,
            "episode": "vertical/episode_000000",
            "processed_row_count": 10,
            "dropped_reference_count": 0,
            "orientation": {
                "principal_branch_event_count": 2,
                "quaternion_large_jump_count": 2,
                "continuous_large_jump_count": 0,
                "rotation6d_large_jump_count": 0,
            },
            "joints": {"classic_wrap_candidate_count": 0},
            "gripper_right": {
                "right_edge_count": 1,
                "effective_right_edge_ordinal": 1,
            },
            "force": {
                "category": "CRITICAL_REVIEW",
                "critical_review_reasons": ("HIGH_FORCE_AT_RELEASE",),
                "broad_review_reasons": ("PERSISTENT_HIGH_FORCE",),
                "tail_median_delta_force_n": 40.0,
                "longest_high_force_sec": 1.0,
                "max_delta_force_n": 45.0,
                "max_delta_torque_nm": 2.0,
            },
            "structural_issues": [],
        }
        report = aggregate_results(
            [base],
            dataset_root=".",
            workers=1,
            thresholds=AuditThresholds(),
        )
        self.assertEqual(report["conversion_gate"], "PASS")
        self.assertEqual(report["critical_force_review_count"], 1)
        changed = dict(base)
        changed["structural_issues"] = ["JOINT_CLASSIC_WRAP_CANDIDATE"]
        changed["joints"] = {"classic_wrap_candidate_count": 1}
        report = aggregate_results(
            [changed],
            dataset_root=".",
            workers=1,
            thresholds=AuditThresholds(),
        )
        self.assertEqual(report["conversion_gate"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
