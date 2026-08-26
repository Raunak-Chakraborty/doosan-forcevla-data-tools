import math
import unittest

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    DoosanForceProprioEpisode,
    DoosanForceProprioState,
    ForceCompensationPolicy,
    ForceCompensationProvenance,
    SynchronizedForceProprioSample,
    rotation_matrix_to_rotvec,
    rotation_vector_to_matrix,
)
from doosan_forcevla_data.convert.doosan_gripper_semantics_v1 import (
    DoosanGripperEpisode,
    SynchronizedGripperSample,
    convert_gripper_state,
)
from doosan_forcevla_data.convert.doosan_measured_action_v1 import (
    ACTION_DIM,
    ACTION_FIELDS,
    FORCEVLA_CONTRACT_ID,
    SEMANTICS_ID,
    DoosanMeasuredAction,
    DoosanMeasuredActionEpisode,
    MeasuredActionError,
    build_doosan_measured_action_episode,
    compute_measured_action,
    reconstruct_target_pose,
    spatial_delta_rotvec,
)
from doosan_forcevla_data.ingest.doosan_raw_v1 import GripperStateRecord, RecordStamp


def _matmul(left, right):
    return tuple(
        tuple(
            sum(left[row][k] * right[k][col] for k in range(3))
            for col in range(3)
        )
        for row in range(3)
    )


def _rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return (
        (1.0, 0.0, 0.0),
        (0.0, c, -s),
        (0.0, s, c),
    )


def _rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return (
        (c, -s, 0.0),
        (s, c, 0.0),
        (0.0, 0.0, 1.0),
    )


def _provenance():
    return ForceCompensationProvenance(
        policy=ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH,
        schema_version="test",
        source_path="test",
        controller_external_force_reset_active_at_record_start=True,
        controller_external_force_reset_completed_before_recording=True,
        force_guard_tare_applied_to_mcap=False,
        offline_force_tare_performed=False,
        pre_reset_ft_recorded=False,
        recording_controller_reset_compensated=True,
        recording_force_signal_in_mcap="test",
        offline_force_processing_owner="test",
        approved_for_training=True,
        reason="test",
    )


def _state(position, rotvec):
    return DoosanForceProprioState(
        source_bag_timestamp_ns=1,
        controller_timestamp_s=1.0,
        tcp_position_m=tuple(position),
        tcp_rotvec_rad=tuple(rotvec),
        joint_position_rad=(0.0,) * 6,
        joint_velocity_rad_s=(0.0,) * 6,
        wrench_base_n_nm=(0.0,) * 6,
        force_policy=ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH,
    )


def _force_episode(positions, rotvecs, *, reference_indices=None, timestamps=None):
    count = len(positions)
    if reference_indices is None:
        reference_indices = list(range(count))
    if timestamps is None:
        timestamps = [100 + 10 * index for index in range(count)]
    samples = tuple(
        SynchronizedForceProprioSample(
            reference_index=reference_indices[index],
            reference_timestamp_ns=timestamps[index],
            robot_state_source_index=index,
            robot_state_source_timestamp_ns=timestamps[index],
            robot_state_signed_skew_ns=0,
            state=_state(positions[index], rotvecs[index]),
        )
        for index in range(count)
    )
    return DoosanForceProprioEpisode(
        provenance=_provenance(),
        raw_robot_state_count=count,
        complete_reference_count=count,
        dropped_reference_count=0,
        samples=samples,
    )


def _gripper_episode(holding_values, *, reference_indices=None, timestamps=None):
    count = len(holding_values)
    if reference_indices is None:
        reference_indices = list(range(count))
    if timestamps is None:
        timestamps = [100 + 10 * index for index in range(count)]
    samples = []
    for index, holding in enumerate(holding_values):
        source_timestamp = 1000 + index
        record = GripperStateRecord(
            stamp=RecordStamp(
                bag_timestamp_ns=900 + index,
                header_timestamp_ns=source_timestamp,
                frame_id="egu_50_prismatic_1",
            ),
            position_m=0.013 + index * 0.001,
            holding=holding,
        )
        samples.append(
            SynchronizedGripperSample(
                reference_index=reference_indices[index],
                reference_timestamp_ns=timestamps[index],
                gripper_source_index=index,
                gripper_source_timestamp_ns=source_timestamp,
                gripper_signed_skew_ns=0,
                state=convert_gripper_state(record),
            )
        )
    return DoosanGripperEpisode(
        raw_gripper_state_count=count,
        complete_reference_count=count,
        dropped_reference_count=0,
        samples=tuple(samples),
    )


class ContractTests(unittest.TestCase):
    def test_contract_identity_and_layout(self):
        self.assertEqual(SEMANTICS_ID, "doosan_measured_action_v1")
        self.assertEqual(FORCEVLA_CONTRACT_ID, "doosan_forcevla_dataset_contract_v2")
        self.assertEqual(ACTION_DIM, 7)
        self.assertEqual(len(ACTION_FIELDS), 7)
        self.assertEqual(ACTION_FIELDS[-1], "absolute_gripper_target_open_fraction")

    def test_absolute_gripper_target_is_not_delta(self):
        current = _state((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        target = _state((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        action = compute_measured_action(current, target, 1.0)
        self.assertEqual(action[:6], (0.0,) * 6)
        self.assertEqual(action[6], 1.0)

    def test_nonbinary_gripper_target_rejected(self):
        state = _state((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            compute_measured_action(state, state, 0.5)


class SpatialRotationTests(unittest.TestCase):
    def test_noncommuting_case_uses_spatial_not_body_relative_rotation(self):
        r_t = _rx(math.pi / 2.0)
        r_t1 = _matmul(_rz(math.pi / 2.0), r_t)
        rv_t = rotation_matrix_to_rotvec(r_t)
        rv_t1 = rotation_matrix_to_rotvec(r_t1)

        delta = spatial_delta_rotvec(rv_t, rv_t1)
        self.assertAlmostEqual(delta[0], 0.0, places=12)
        self.assertAlmostEqual(delta[1], 0.0, places=12)
        self.assertAlmostEqual(delta[2], math.pi / 2.0, places=12)

    def test_spatial_reconstruction_recovers_target_orientation(self):
        r_t = _matmul(_rz(0.7), _rx(-0.4))
        r_t1 = _matmul(_rz(-0.9), _rx(0.8))
        rv_t = rotation_matrix_to_rotvec(r_t)
        rv_t1 = rotation_matrix_to_rotvec(r_t1)
        action = compute_measured_action(
            _state((1.0, 2.0, 3.0), rv_t),
            _state((1.2, 1.5, 3.7), rv_t1),
            1.0,
        )
        position, reconstructed_rotvec = reconstruct_target_pose(
            (1.0, 2.0, 3.0),
            rv_t,
            action,
        )
        self.assertEqual(position, (1.2, 1.5, 3.7))
        expected_matrix = rotation_vector_to_matrix(rv_t1)
        actual_matrix = rotation_vector_to_matrix(reconstructed_rotvec)
        for row in range(3):
            for col in range(3):
                self.assertAlmostEqual(actual_matrix[row][col], expected_matrix[row][col], places=12)

    def test_wraparound_near_pi_reconstructs_rotation_matrix(self):
        rv_t = (0.0, 0.0, math.radians(179.0))
        rv_t1 = (0.0, 0.0, math.radians(-179.0))
        delta = spatial_delta_rotvec(rv_t, rv_t1)
        self.assertAlmostEqual(delta[2], math.radians(2.0), places=12)
        _, reconstructed = reconstruct_target_pose((0.0, 0.0, 0.0), rv_t, (*((0.0,) * 3), *delta, 0.0))
        expected = rotation_vector_to_matrix(rv_t1)
        actual = rotation_vector_to_matrix(reconstructed)
        for row in range(3):
            for col in range(3):
                self.assertAlmostEqual(actual[row][col], expected[row][col], places=12)


class EpisodeConstructionTests(unittest.TestCase):
    def _valid_episodes(self):
        positions = (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.1, 0.2, 0.0),
            (0.1, 0.2, 0.3),
        )
        rotvecs = (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.1, 0.2, 0.0),
            (0.1, 0.2, 0.3),
        )
        return (
            _force_episode(positions, rotvecs),
            _gripper_episode((True, True, False, False)),
        )

    def test_n_states_emit_exactly_n_minus_one_actions_and_no_terminal_padding(self):
        force_episode, gripper_episode = self._valid_episodes()
        episode = build_doosan_measured_action_episode(force_episode, gripper_episode)
        self.assertEqual(episode.state_count, 4)
        self.assertEqual(episode.action_count, 3)
        self.assertEqual(episode.terminal_reference_index, 3)
        self.assertEqual([a.source_reference_index for a in episode.actions], [0, 1, 2])
        self.assertEqual([a.target_reference_index for a in episode.actions], [1, 2, 3])
        self.assertFalse(episode.summary_dict()["terminal_policy"]["terminal_action_emitted"])
        self.assertFalse(episode.summary_dict()["terminal_policy"]["synthetic_terminal_zero_action"])

    def test_gripper_action_is_target_state_absolute_value(self):
        force_episode, gripper_episode = self._valid_episodes()
        episode = build_doosan_measured_action_episode(force_episode, gripper_episode)
        self.assertEqual([action.gripper_target_open_fraction for action in episode.actions], [0.0, 1.0, 1.0])
        self.assertEqual(episode.release_action_source_indices, (1,))
        self.assertEqual(
            episode.gripper_target_counts,
            {"held_or_closed_target": 1, "released_or_open_target": 2},
        )

    def test_translation_actions_reconstruct_target_positions(self):
        force_episode, gripper_episode = self._valid_episodes()
        episode = build_doosan_measured_action_episode(force_episode, gripper_episode)
        for index, action in enumerate(episode.actions):
            reconstructed, _ = reconstruct_target_pose(
                force_episode.samples[index].state.tcp_position_m,
                force_episode.samples[index].state.tcp_rotvec_rad,
                action.to_vector(),
            )
            self.assertEqual(reconstructed, force_episode.samples[index + 1].state.tcp_position_m)

    def test_reference_timestamp_mismatch_is_rejected(self):
        force_episode, _ = self._valid_episodes()
        gripper_episode = _gripper_episode(
            (True, True, False, False),
            timestamps=(100, 110, 121, 130),
        )
        with self.assertRaisesRegex(MeasuredActionError, "timestamp mismatch"):
            build_doosan_measured_action_episode(force_episode, gripper_episode)

    def test_noncontiguous_reference_index_is_rejected_instead_of_bridged(self):
        positions = ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0))
        rotvecs = ((0.0, 0.0, 0.0),) * 3
        force_episode = _force_episode(
            positions,
            rotvecs,
            reference_indices=(0, 2, 3),
        )
        gripper_episode = _gripper_episode(
            (True, False, False),
            reference_indices=(0, 2, 3),
        )
        with self.assertRaisesRegex(MeasuredActionError, "contiguous"):
            build_doosan_measured_action_episode(force_episode, gripper_episode)

    def test_nonincreasing_reference_time_is_rejected(self):
        positions = ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0))
        rotvecs = ((0.0, 0.0, 0.0),) * 3
        force_episode = _force_episode(positions, rotvecs, timestamps=(100, 100, 120))
        gripper_episode = _gripper_episode((True, False, False), timestamps=(100, 100, 120))
        with self.assertRaisesRegex(MeasuredActionError, "strictly increasing|later than source"):
            build_doosan_measured_action_episode(force_episode, gripper_episode)

    def test_invalid_release_protocol_is_rejected_at_action_boundary(self):
        positions = ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0))
        rotvecs = ((0.0, 0.0, 0.0),) * 3
        force_episode = _force_episode(positions, rotvecs)
        gripper_episode = _gripper_episode((True, True, True))
        with self.assertRaises(ValueError):
            build_doosan_measured_action_episode(force_episode, gripper_episode)


class ActionObjectTests(unittest.TestCase):
    def test_action_dict_contains_explicit_source_and_target_provenance(self):
        action = DoosanMeasuredAction(
            source_reference_index=5,
            target_reference_index=6,
            source_reference_timestamp_ns=100,
            target_reference_timestamp_ns=133,
            delta_translation_base_m=(0.1, 0.2, 0.3),
            delta_rotvec_base_rad=(0.01, 0.02, 0.03),
            gripper_target_open_fraction=1.0,
        )
        payload = action.to_dict()
        self.assertEqual(payload["delta_time_ns"], 33)
        self.assertEqual(payload["action_7d"], [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 1.0])

    def test_action_requires_adjacent_reference_indices(self):
        with self.assertRaisesRegex(MeasuredActionError, "adjacent"):
            DoosanMeasuredAction(
                source_reference_index=5,
                target_reference_index=7,
                source_reference_timestamp_ns=100,
                target_reference_timestamp_ns=133,
                delta_translation_base_m=(0.0, 0.0, 0.0),
                delta_rotvec_base_rad=(0.0, 0.0, 0.0),
                gripper_target_open_fraction=0.0,
            )

    def test_episode_object_rejects_terminal_padding_count(self):
        action = DoosanMeasuredAction(
            source_reference_index=0,
            target_reference_index=1,
            source_reference_timestamp_ns=100,
            target_reference_timestamp_ns=133,
            delta_translation_base_m=(0.0, 0.0, 0.0),
            delta_rotvec_base_rad=(0.0, 0.0, 0.0),
            gripper_target_open_fraction=1.0,
        )
        with self.assertRaisesRegex(MeasuredActionError, "N-1"):
            DoosanMeasuredActionEpisode(state_count=3, actions=(action,))


if __name__ == "__main__":
    unittest.main()
