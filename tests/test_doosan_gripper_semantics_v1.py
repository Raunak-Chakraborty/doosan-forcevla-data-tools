from types import SimpleNamespace
import unittest

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    DoosanForceProprioState,
    ForceCompensationPolicy,
    SynchronizedForceProprioSample,
)
from doosan_forcevla_data.convert.doosan_gripper_semantics_v1 import (
    FORCEVLA_CONTRACT_ID,
    HELD_OPEN_FRACTION,
    RELEASED_OPEN_FRACTION,
    SEMANTICS_ID,
    DoosanGripperEpisode,
    GripperExecutionIntent,
    GripperSemanticsError,
    SynchronizedGripperSample,
    assemble_forcevla_v2_observation_states,
    build_synchronized_gripper_samples,
    convert_gripper_state,
    holding_to_open_fraction,
    release_only_execution_intent,
    require_binary_open_fraction,
    validate_release_only_episode_protocol,
)
from doosan_forcevla_data.ingest.doosan_raw_v1 import (
    GripperStateRecord,
    RecordStamp,
)
from doosan_forcevla_data.sync.timestamp_plan import (
    ClockDomain,
    SyncMethod,
    SyncSpec,
    TimestampTimeline,
    build_synchronization_plan,
)


def _gripper_record(
    *,
    bag_timestamp_ns: int,
    header_timestamp_ns: int,
    holding: bool,
    position_m: float = 0.013,
) -> GripperStateRecord:
    return GripperStateRecord(
        stamp=RecordStamp(
            bag_timestamp_ns=bag_timestamp_ns,
            header_timestamp_ns=header_timestamp_ns,
            frame_id="egu_50_prismatic_1",
        ),
        position_m=position_m,
        holding=holding,
    )


def _sync_result(reference=(100, 200), gripper=(90, 210)):
    reference_timeline = TimestampTimeline.from_timestamps(
        "tcp_image",
        ClockDomain.ROS,
        reference,
    )
    gripper_timeline = TimestampTimeline.from_timestamps(
        "gripper_state",
        ClockDomain.ROS,
        gripper,
    )
    plan = build_synchronization_plan(
        reference_timeline,
        {"gripper_state": gripper_timeline},
        {
            "gripper_state": SyncSpec(
                method=SyncMethod.NEAREST,
                required=True,
                max_age_ns=15_000_000,
            )
        },
    )
    return SimpleNamespace(plan=plan)


def _force_state(seed: float) -> DoosanForceProprioState:
    return DoosanForceProprioState(
        source_bag_timestamp_ns=int(seed + 1),
        controller_timestamp_s=seed + 2.0,
        tcp_position_m=(seed + 1.0, seed + 2.0, seed + 3.0),
        tcp_rotvec_rad=(0.1, 0.2, 0.3),
        joint_position_rad=(seed + 4.0,) * 6,
        joint_velocity_rad_s=(seed + 5.0,) * 6,
        wrench_base_n_nm=(seed + 6.0,) * 6,
        force_policy=ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH,
    )


class BinarySemanticTests(unittest.TestCase):
    def test_contract_identity_is_frozen_v2(self):
        self.assertEqual(SEMANTICS_ID, "doosan_gripper_semantics_v1")
        self.assertEqual(FORCEVLA_CONTRACT_ID, "doosan_forcevla_dataset_contract_v2")

    def test_holding_maps_to_closed_endpoint(self):
        self.assertEqual(holding_to_open_fraction(True), HELD_OPEN_FRACTION)
        self.assertEqual(HELD_OPEN_FRACTION, 0.0)

    def test_released_maps_to_open_endpoint(self):
        self.assertEqual(holding_to_open_fraction(False), RELEASED_OPEN_FRACTION)
        self.assertEqual(RELEASED_OPEN_FRACTION, 1.0)

    def test_holding_requires_bool(self):
        for value in (0, 1, None, "true"):
            with self.subTest(value=value), self.assertRaises(GripperSemanticsError):
                holding_to_open_fraction(value)

    def test_model_value_is_binary_not_continuous_position_normalization(self):
        self.assertEqual(require_binary_open_fraction(0.0), 0.0)
        self.assertEqual(require_binary_open_fraction(1.0), 1.0)
        for value in (-0.1, 0.5, 1.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(GripperSemanticsError):
                require_binary_open_fraction(value)


class ReleaseOnlyExecutionTests(unittest.TestCase):
    def test_held_target_remains_noop(self):
        self.assertIs(
            release_only_execution_intent(0.0, 0.0),
            GripperExecutionIntent.HOLD_CURRENT,
        )

    def test_held_to_released_maps_to_release(self):
        self.assertIs(
            release_only_execution_intent(0.0, 1.0),
            GripperExecutionIntent.RELEASE,
        )

    def test_released_target_remains_noop(self):
        self.assertIs(
            release_only_execution_intent(1.0, 1.0),
            GripperExecutionIntent.REMAIN_RELEASED,
        )

    def test_regrasp_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(GripperSemanticsError, "regrasp"):
            release_only_execution_intent(1.0, 0.0)


class GripperRecordTests(unittest.TestCase):
    def test_raw_position_is_preserved_but_does_not_change_semantics(self):
        held_a = convert_gripper_state(
            _gripper_record(
                bag_timestamp_ns=10,
                header_timestamp_ns=11,
                holding=True,
                position_m=0.0131755,
            )
        )
        held_b = convert_gripper_state(
            _gripper_record(
                bag_timestamp_ns=20,
                header_timestamp_ns=21,
                holding=True,
                position_m=0.015699,
            )
        )
        self.assertNotEqual(held_a.raw_position_m, held_b.raw_position_m)
        self.assertEqual(held_a.open_fraction, 0.0)
        self.assertEqual(held_b.open_fraction, 0.0)

    def test_missing_header_timestamp_is_rejected(self):
        record = GripperStateRecord(
            stamp=RecordStamp(
                bag_timestamp_ns=10,
                header_timestamp_ns=None,
                frame_id=None,
            ),
            position_m=0.013,
            holding=True,
        )
        with self.assertRaisesRegex(GripperSemanticsError, "header timestamp"):
            convert_gripper_state(record)


class SynchronizedGripperTests(unittest.TestCase):
    def test_exact_patch4_nearest_indices_drive_discrete_state(self):
        records = (
            _gripper_record(
                bag_timestamp_ns=80,
                header_timestamp_ns=90,
                holding=True,
            ),
            _gripper_record(
                bag_timestamp_ns=205,
                header_timestamp_ns=210,
                holding=False,
                position_m=0.015,
            ),
        )
        samples = build_synchronized_gripper_samples(records, _sync_result())
        self.assertEqual([sample.reference_index for sample in samples], [0, 1])
        self.assertEqual([sample.gripper_source_index for sample in samples], [0, 1])
        self.assertEqual([sample.state.open_fraction for sample in samples], [0.0, 1.0])
        self.assertEqual(
            [sample.gripper_source_timestamp_ns for sample in samples],
            [90, 210],
        )

    def test_selected_timestamp_mismatch_is_rejected(self):
        records = (
            _gripper_record(
                bag_timestamp_ns=80,
                header_timestamp_ns=91,
                holding=True,
            ),
            _gripper_record(
                bag_timestamp_ns=205,
                header_timestamp_ns=210,
                holding=False,
            ),
        )
        with self.assertRaisesRegex(GripperSemanticsError, "timestamp"):
            build_synchronized_gripper_samples(records, _sync_result())

    def test_episode_summary_reports_transition_reference(self):
        records = (
            _gripper_record(bag_timestamp_ns=80, header_timestamp_ns=90, holding=True),
            _gripper_record(bag_timestamp_ns=205, header_timestamp_ns=210, holding=False),
        )
        samples = build_synchronized_gripper_samples(records, _sync_result())
        episode = DoosanGripperEpisode(
            raw_gripper_state_count=2,
            complete_reference_count=2,
            dropped_reference_count=0,
            samples=samples,
        )
        self.assertEqual(episode.held_sample_count, 1)
        self.assertEqual(episode.released_sample_count, 1)
        self.assertEqual(episode.transition_indices, (1,))
        summary = episode.summary_dict()
        self.assertEqual(summary["state_semantics"]["raw_position_role"], "diagnostic_only")
        self.assertFalse(summary["force_semantics"]["schunk_state_contains_measured_force"])


class ReleaseOnlyProtocolTests(unittest.TestCase):
    def _episode(self, holding_values):
        samples = tuple(
            SynchronizedGripperSample(
                reference_index=index,
                reference_timestamp_ns=100 + index,
                gripper_source_index=index,
                gripper_source_timestamp_ns=200 + index,
                gripper_signed_skew_ns=0,
                state=convert_gripper_state(
                    _gripper_record(
                        bag_timestamp_ns=190 + index,
                        header_timestamp_ns=200 + index,
                        holding=holding,
                    )
                ),
            )
            for index, holding in enumerate(holding_values)
        )
        return DoosanGripperEpisode(
            raw_gripper_state_count=len(samples),
            complete_reference_count=len(samples),
            dropped_reference_count=0,
            samples=samples,
        )

    def test_valid_single_release_protocol(self):
        validate_release_only_episode_protocol(
            self._episode((True, True, False, False))
        )

    def test_no_release_is_rejected(self):
        with self.assertRaisesRegex(GripperSemanticsError, "end after the release"):
            validate_release_only_episode_protocol(
                self._episode((True, True, True))
            )

    def test_released_start_is_rejected(self):
        with self.assertRaisesRegex(GripperSemanticsError, "start with the peg held"):
            validate_release_only_episode_protocol(
                self._episode((False, False))
            )

    def test_release_then_regrasp_is_rejected(self):
        with self.assertRaises(GripperSemanticsError):
            validate_release_only_episode_protocol(
                self._episode((True, False, True, False))
            )


class ObservationAssemblyTests(unittest.TestCase):
    def _episodes(self):
        force_samples = (
            SynchronizedForceProprioSample(
                reference_index=0,
                reference_timestamp_ns=100,
                robot_state_source_index=0,
                robot_state_source_timestamp_ns=99,
                robot_state_signed_skew_ns=-1,
                state=_force_state(0.0),
            ),
            SynchronizedForceProprioSample(
                reference_index=1,
                reference_timestamp_ns=200,
                robot_state_source_index=1,
                robot_state_source_timestamp_ns=201,
                robot_state_signed_skew_ns=1,
                state=_force_state(10.0),
            ),
        )
        gripper_records = (
            _gripper_record(bag_timestamp_ns=80, header_timestamp_ns=90, holding=True),
            _gripper_record(bag_timestamp_ns=205, header_timestamp_ns=210, holding=False),
        )
        gripper_samples = build_synchronized_gripper_samples(
            gripper_records,
            _sync_result(),
        )
        force_episode = SimpleNamespace(samples=force_samples)
        gripper_episode = DoosanGripperEpisode(
            raw_gripper_state_count=2,
            complete_reference_count=2,
            dropped_reference_count=0,
            samples=gripper_samples,
        )
        return force_episode, gripper_episode

    def test_25d_assembly_places_binary_gripper_only_at_index_6(self):
        force_episode, gripper_episode = self._episodes()
        observations = assemble_forcevla_v2_observation_states(
            force_episode,
            gripper_episode,
        )
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(len(state) == 25 for state in observations))
        self.assertEqual([state[6] for state in observations], [0.0, 1.0])
        self.assertEqual(observations[0][19:25], (6.0,) * 6)
        self.assertEqual(observations[1][19:25], (16.0,) * 6)

    def test_reference_identity_mismatch_is_rejected(self):
        force_episode, gripper_episode = self._episodes()
        bad_first = SynchronizedGripperSample(
            reference_index=0,
            reference_timestamp_ns=101,
            gripper_source_index=gripper_episode.samples[0].gripper_source_index,
            gripper_source_timestamp_ns=(
                gripper_episode.samples[0].gripper_source_timestamp_ns
            ),
            gripper_signed_skew_ns=gripper_episode.samples[0].gripper_signed_skew_ns,
            state=gripper_episode.samples[0].state,
        )
        bad_episode = DoosanGripperEpisode(
            raw_gripper_state_count=2,
            complete_reference_count=2,
            dropped_reference_count=0,
            samples=(bad_first, gripper_episode.samples[1]),
        )
        with self.assertRaisesRegex(GripperSemanticsError, "timestamps"):
            assemble_forcevla_v2_observation_states(force_episode, bad_episode)


if __name__ == "__main__":
    unittest.main()
