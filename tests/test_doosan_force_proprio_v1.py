import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    EXPECTED_FORCE_PROCESSING_OWNER,
    EXPECTED_FORCE_SIGNAL,
    FORCE_PROVENANCE_SCHEMA,
    OBSERVATION_STATE_FIELDS,
    DoosanForceProprioState,
    ForceCompensationPolicy,
    ForceProprioError,
    build_synchronized_force_proprio_samples,
    convert_robot_state_rt,
    doosan_zyz_deg_to_matrix,
    doosan_zyz_deg_to_rotvec,
    resolve_force_compensation_provenance,
    rotation_vector_to_matrix,
)
from doosan_forcevla_data.ingest.doosan_raw_v1 import RecordStamp, RobotStateRtRecord
from doosan_forcevla_data.sync.timestamp_plan import (
    ClockDomain,
    SyncMethod,
    SyncSpec,
    TimestampTimeline,
    build_synchronization_plan,
)


def _assert_matrix_close(testcase, actual, expected, places=12):
    for row in range(3):
        for col in range(3):
            testcase.assertAlmostEqual(actual[row][col], expected[row][col], places=places)


def _write_operator(root: Path, **overrides):
    operator = {
        "schema_version": FORCE_PROVENANCE_SCHEMA,
        "controller_external_force_reset_active_at_record_start": True,
        "controller_external_force_reset_completed_before_recording": True,
        "force_guard_tare_applied_to_mcap": False,
        "offline_force_tare_performed": False,
        "pre_reset_ft_recorded": False,
        "offline_force_processing_owner": EXPECTED_FORCE_PROCESSING_OWNER,
        "recording": {
            "controller_reset_compensated": True,
            "force_signal_in_mcap": EXPECTED_FORCE_SIGNAL,
        },
    }
    for key, value in overrides.items():
        if key.startswith("recording__"):
            operator["recording"][key.split("__", 1)[1]] = value
        else:
            operator[key] = value
    (root / "episode_operator.json").write_text(json.dumps(operator), encoding="utf-8")


def _provenance(root: Path):
    _write_operator(root)
    return resolve_force_compensation_provenance(root)


def _robot_record(
    *,
    bag_timestamp_ns=100,
    tcp=(1000.0, -250.0, 500.0, 0.0, 0.0, 0.0),
    joint_position=(0.0, 90.0, -90.0, 180.0, -180.0, 45.0),
    joint_velocity=(0.0, 10.0, -10.0, 20.0, -20.0, 5.0),
    wrench=(1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
):
    return RobotStateRtRecord(
        stamp=RecordStamp(
            bag_timestamp_ns=bag_timestamp_ns,
            header_timestamp_ns=None,
            frame_id=None,
        ),
        controller_timestamp_s=123.5,
        actual_joint_position_deg=tuple(joint_position),
        actual_joint_velocity_deg_s=tuple(joint_velocity),
        actual_tcp_position_mm_deg=tuple(tcp),
        actual_tcp_velocity_mm_deg_s=(0.0,) * 6,
        external_tcp_force_base_n_nm=tuple(wrench),
        diagnostics=(),
    )


class DoosanEulerZYZTests(unittest.TestCase):
    def test_identity(self):
        matrix = doosan_zyz_deg_to_matrix((0.0, 0.0, 0.0))
        _assert_matrix_close(
            self,
            matrix,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        self.assertEqual(doosan_zyz_deg_to_rotvec((0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))

    def test_first_z_rotation_is_reference_z(self):
        matrix = doosan_zyz_deg_to_matrix((90.0, 0.0, 0.0))
        _assert_matrix_close(
            self,
            matrix,
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        )
        rotvec = doosan_zyz_deg_to_rotvec((90.0, 0.0, 0.0))
        self.assertAlmostEqual(rotvec[0], 0.0, places=12)
        self.assertAlmostEqual(rotvec[1], 0.0, places=12)
        self.assertAlmostEqual(rotvec[2], math.pi / 2.0, places=12)

    def test_middle_rotation_is_about_rotated_y(self):
        matrix = doosan_zyz_deg_to_matrix((0.0, 90.0, 0.0))
        _assert_matrix_close(
            self,
            matrix,
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        )
        rotvec = doosan_zyz_deg_to_rotvec((0.0, 90.0, 0.0))
        self.assertAlmostEqual(rotvec[0], 0.0, places=12)
        self.assertAlmostEqual(rotvec[1], math.pi / 2.0, places=12)
        self.assertAlmostEqual(rotvec[2], 0.0, places=12)

    def test_nontrivial_matrix_matches_closed_form_rz_ry_rz(self):
        a, b, c = (30.0, -40.0, 70.0)
        ar, br, cr = map(math.radians, (a, b, c))
        ca, sa = math.cos(ar), math.sin(ar)
        cb, sb = math.cos(br), math.sin(br)
        cc, sc = math.cos(cr), math.sin(cr)
        expected = (
            (ca * cb * cc - sa * sc, -ca * cb * sc - sa * cc, ca * sb),
            (sa * cb * cc + ca * sc, -sa * cb * sc + ca * cc, sa * sb),
            (-sb * cc, sb * sc, cb),
        )
        _assert_matrix_close(self, doosan_zyz_deg_to_matrix((a, b, c)), expected)

    def test_round_trip_is_stable_near_episode10_pi_orientations(self):
        episode10_eulers = (
            (50.306827545166016, -175.6098175048828, 140.52659606933594),
            (103.35005187988281, -175.40658569335938, -164.1007537841797),
            (99.5241928100586, -175.53700256347656, -168.86476135253906),
        )
        for euler in episode10_eulers:
            with self.subTest(euler=euler):
                matrix = doosan_zyz_deg_to_matrix(euler)
                rotvec = doosan_zyz_deg_to_rotvec(euler)
                self.assertTrue(all(math.isfinite(value) for value in rotvec))
                self.assertLessEqual(math.sqrt(sum(value * value for value in rotvec)), math.pi + 1e-12)
                _assert_matrix_close(self, rotation_vector_to_matrix(rotvec), matrix, places=11)


class ForceCompensationProvenanceTests(unittest.TestCase):
    def test_episode10_style_metadata_resolves_to_passthrough(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            provenance = _provenance(root)
            self.assertIs(
                provenance.policy,
                ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH,
            )
            self.assertTrue(provenance.approved_for_training)
            self.assertFalse(provenance.offline_second_tare_allowed)

    def test_reset_compensated_plus_offline_tare_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_operator(root, offline_force_tare_performed=True)
            with self.assertRaisesRegex(ForceProprioError, "double-tared"):
                resolve_force_compensation_provenance(root)

    def test_reset_compensated_plus_guard_tare_in_mcap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_operator(root, force_guard_tare_applied_to_mcap=True)
            with self.assertRaisesRegex(ForceProprioError, "double-tared"):
                resolve_force_compensation_provenance(root)

    def test_unknown_legacy_episode_is_classified_but_not_training_approved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "episode_operator.json").write_text(
                json.dumps({"schema_version": "legacy_v1"}),
                encoding="utf-8",
            )
            provenance = resolve_force_compensation_provenance(root)
            self.assertIs(
                provenance.policy,
                ForceCompensationPolicy.LEGACY_EPISODE_REQUIRES_KNOWN_TARE_POLICY,
            )
            self.assertFalse(provenance.approved_for_training)
            self.assertFalse(provenance.offline_second_tare_allowed)
            with self.assertRaisesRegex(ForceProprioError, "legacy episode"):
                convert_robot_state_rt(_robot_record(), provenance)


class ForceProprioStateTests(unittest.TestCase):
    def test_robot_state_units_and_wrench_passthrough(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provenance = _provenance(Path(tmpdir))
            raw_wrench = (1.0, -2.0, 3.0, -0.1, 0.2, -0.3)
            state = convert_robot_state_rt(
                _robot_record(wrench=raw_wrench),
                provenance,
            )

        self.assertEqual(state.tcp_position_m, (1.0, -0.25, 0.5))
        self.assertEqual(state.wrench_base_n_nm, raw_wrench)
        expected_joint_position = tuple(
            math.radians(value) for value in (0.0, 90.0, -90.0, 180.0, -180.0, 45.0)
        )
        expected_joint_velocity = tuple(
            math.radians(value) for value in (0.0, 10.0, -10.0, 20.0, -20.0, 5.0)
        )
        for actual, expected in zip(state.joint_position_rad, expected_joint_position, strict=True):
            self.assertAlmostEqual(actual, expected, places=15)
        for actual, expected in zip(state.joint_velocity_rad_s, expected_joint_velocity, strict=True):
            self.assertAlmostEqual(actual, expected, places=15)

    def test_25d_assembly_matches_frozen_forcevla_v2_order(self):
        state = DoosanForceProprioState(
            source_bag_timestamp_ns=1,
            controller_timestamp_s=2.0,
            tcp_position_m=(1.0, 2.0, 3.0),
            tcp_rotvec_rad=(4.0, 5.0, 6.0),
            joint_position_rad=(7.0, 8.0, 9.0, 10.0, 11.0, 12.0),
            joint_velocity_rad_s=(13.0, 14.0, 15.0, 16.0, 17.0, 18.0),
            wrench_base_n_nm=(19.0, 20.0, 21.0, 22.0, 23.0, 24.0),
            force_policy=ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH,
        )
        assembled = state.to_observation_state(0.5)
        self.assertEqual(
            assembled,
            (
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                0.5,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
            ),
        )
        self.assertEqual(len(OBSERVATION_STATE_FIELDS), 25)
        self.assertEqual(OBSERVATION_STATE_FIELDS[19:25], (
            "force_x_n",
            "force_y_n",
            "force_z_n",
            "torque_x_nm",
            "torque_y_nm",
            "torque_z_nm",
        ))

    def test_gripper_assembly_rejects_non_normalized_values(self):
        state = DoosanForceProprioState(
            source_bag_timestamp_ns=1,
            controller_timestamp_s=2.0,
            tcp_position_m=(0.0, 0.0, 0.0),
            tcp_rotvec_rad=(0.0, 0.0, 0.0),
            joint_position_rad=(0.0,) * 6,
            joint_velocity_rad_s=(0.0,) * 6,
            wrench_base_n_nm=(0.0,) * 6,
            force_policy=ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH,
        )
        for value in (-0.01, 1.01, float("nan")):
            with self.subTest(value=value), self.assertRaises(ForceProprioError):
                state.to_observation_state(value)

    def test_nonfinite_training_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provenance = _provenance(Path(tmpdir))
            record = _robot_record(wrench=(1.0, 2.0, float("nan"), 0.1, 0.2, 0.3))
            with self.assertRaises(ForceProprioError):
                convert_robot_state_rt(record, provenance)


class SynchronizedSelectionTests(unittest.TestCase):
    def _sync_result(self):
        reference = TimestampTimeline.from_timestamps(
            "tcp_image",
            ClockDomain.ROS,
            (100, 200),
        )
        robot = TimestampTimeline.from_timestamps(
            "robot_state_rt",
            ClockDomain.ROS,
            (90, 210),
        )
        plan = build_synchronization_plan(
            reference,
            {"robot_state_rt": robot},
            {
                "robot_state_rt": SyncSpec(
                    method=SyncMethod.NEAREST,
                    required=True,
                    max_age_ns=20,
                )
            },
        )
        return SimpleNamespace(plan=plan)

    def test_exact_patch4_selection_drives_all_robot_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provenance = _provenance(Path(tmpdir))
            records = (
                _robot_record(
                    bag_timestamp_ns=90,
                    tcp=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    wrench=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ),
                _robot_record(
                    bag_timestamp_ns=210,
                    tcp=(2000.0, 0.0, 0.0, 90.0, 0.0, 0.0),
                    wrench=(2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ),
            )
            samples = build_synchronized_force_proprio_samples(
                records,
                self._sync_result(),
                provenance,
            )

        self.assertEqual([sample.robot_state_source_index for sample in samples], [0, 1])
        self.assertEqual([sample.state.tcp_position_m[0] for sample in samples], [1.0, 2.0])
        self.assertEqual([sample.state.wrench_base_n_nm[0] for sample in samples], [1.0, 2.0])
        self.assertAlmostEqual(samples[1].state.tcp_rotvec_rad[2], math.pi / 2.0, places=12)

    def test_selected_timestamp_must_match_decoded_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provenance = _provenance(Path(tmpdir))
            records = (_robot_record(bag_timestamp_ns=91), _robot_record(bag_timestamp_ns=210))
            with self.assertRaisesRegex(ForceProprioError, "timestamp does not match"):
                build_synchronized_force_proprio_samples(
                    records,
                    self._sync_result(),
                    provenance,
                )


if __name__ == "__main__":
    unittest.main()
