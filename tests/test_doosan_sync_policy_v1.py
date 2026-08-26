import json
import unittest

from doosan_forcevla_data.sync.doosan_policy_v1 import (
    POLICY_ID,
    REFERENCE_TIMESTAMP_SOURCE,
    ROS_EPOCH_MAX_HEADER_BAG_OFFSET_NS,
    SOURCE_POLICIES,
    CameraCalibration,
    DoosanPolicyError,
    DoosanPolicyInputs,
    HeaderBagOffsetEvidence,
    TimestampSource,
    build_doosan_sync_plan_from_inputs,
    policy_by_key,
)
from doosan_forcevla_data.sync.timestamp_plan import (
    ClockDomain,
    SyncMethod,
)


def _calibration(topic: str) -> CameraCalibration:
    return CameraCalibration(
        topic=topic,
        frame_id="frame",
        height=480,
        width=640,
        distortion_model="plumb_bob",
        d=(0.0,) * 5,
        k=(0.0,) * 9,
        r=(0.0,) * 9,
        p=(0.0,) * 12,
        binning_x=0,
        binning_y=0,
        roi=(0, 0, 0, 0, False),
    )


def _inputs(
    *,
    reference=(100, 200, 300),
    external=(95, 205, 295),
    robot=(90, 210, 290),
    gripper=(100, 200, 300),
    joint=(100, 200, 300),
    speedl=(90, 190, 290),
    joy=(90, 190, 290),
) -> DoosanPolicyInputs:
    evidence = HeaderBagOffsetEvidence(
        count=3,
        minimum_ns=-10,
        maximum_ns=-1,
        max_absolute_ns=10,
    )

    evidence_topics = (
        "/doosan_cameras/tcp_camera/color/image_raw",
        "/doosan_cameras/external_camera_2/color/image_raw",
        "/schunk/state",
        "/dsr01/joint_states",
        "/doosan_teleop/collector_joy",
        "/doosan_cameras/tcp_camera/color/camera_info",
        "/doosan_cameras/external_camera_2/color/camera_info",
    )

    return DoosanPolicyInputs(
        reference_timestamps_ns=tuple(reference),
        source_timestamps_ns=(
            ("external_image", tuple(external)),
            ("robot_state_rt", tuple(robot)),
            ("gripper_state", tuple(gripper)),
            ("joint_state", tuple(joint)),
            ("speedl_stream", tuple(speedl)),
            ("joy", tuple(joy)),
        ),
        header_bag_evidence=tuple(
            (topic, evidence)
            for topic in evidence_topics
        ),
        tcp_calibration=_calibration(
            "/doosan_cameras/tcp_camera/color/camera_info"
        ),
        external_calibration=_calibration(
            "/doosan_cameras/external_camera_2/color/camera_info"
        ),
    )


class DoosanSynchronizationPolicyTests(unittest.TestCase):
    def test_policy_identity_and_clock_model(self):
        self.assertEqual(POLICY_ID, "doosan_sync_policy_v1")
        self.assertEqual(REFERENCE_TIMESTAMP_SOURCE, TimestampSource.HEADER)
        self.assertTrue(hasattr(ClockDomain, "ROS"))
        self.assertEqual(ClockDomain.ROS.value, "ros")

    def test_frozen_stream_policy(self):
        expected = {
            "external_image": (TimestampSource.HEADER, SyncMethod.NEAREST, True, 12_000_000),
            "robot_state_rt": (TimestampSource.BAG, SyncMethod.NEAREST, True, 16_000_000),
            "gripper_state": (TimestampSource.HEADER, SyncMethod.NEAREST, True, 15_000_000),
            "joint_state": (TimestampSource.HEADER, SyncMethod.NEAREST, False, 15_000_000),
            "speedl_stream": (TimestampSource.BAG, SyncMethod.CAUSAL_HOLD, False, 25_000_000),
            "joy": (TimestampSource.HEADER, SyncMethod.CAUSAL_HOLD, False, 25_000_000),
        }

        self.assertEqual(
            {policy.key for policy in SOURCE_POLICIES},
            set(expected),
        )

        for key, values in expected.items():
            policy = policy_by_key(key)
            self.assertEqual(
                (
                    policy.timestamp_source,
                    policy.method,
                    policy.required,
                    policy.max_age_ns,
                ),
                values,
            )

    def test_required_streams_complete_frame(self):
        result = build_doosan_sync_plan_from_inputs(_inputs())
        self.assertEqual(result.complete_reference_count, 3)
        self.assertEqual(result.dropped_reference_count, 0)

    def test_nearest_policy_uses_symmetric_selection(self):
        result = build_doosan_sync_plan_from_inputs(
            _inputs(
                reference=(100,),
                external=(89, 111),
                robot=(84, 116),
                gripper=(85, 115),
                joint=(85, 115),
                speedl=(90,),
                joy=(90,),
            )
        )
        external = result.plan.source_plan("external_image").decisions[0]
        robot = result.plan.source_plan("robot_state_rt").decisions[0]
        self.assertEqual(external.selection.source_timestamps_ns, (89,))
        self.assertEqual(robot.selection.source_timestamps_ns, (84,))

    def test_required_stale_source_drops_reference(self):
        result = build_doosan_sync_plan_from_inputs(
            _inputs(
                reference=(20_000_000,),
                external=(0,),
                robot=(20_000_000,),
                gripper=(20_000_000,),
                joint=(20_000_000,),
                speedl=(20_000_000,),
                joy=(20_000_000,),
            )
        )
        self.assertEqual(result.complete_reference_count, 0)
        self.assertEqual(result.plan.dropped_reference_indices, (0,))
        summary = result.plan.source_plan("external_image").summary
        self.assertEqual(summary.stale, 1)

    def test_optional_missing_stream_does_not_drop_frame(self):
        result = build_doosan_sync_plan_from_inputs(
            _inputs(
                joint=(),
                speedl=(),
                joy=(),
            )
        )
        self.assertEqual(result.complete_reference_count, 3)
        self.assertEqual(result.dropped_reference_count, 0)
        self.assertEqual(result.plan.source_plan("joint_state").summary.missing, 3)

    def test_causal_command_stream_never_selects_future(self):
        result = build_doosan_sync_plan_from_inputs(
            _inputs(
                reference=(100,),
                speedl=(90, 110),
                joy=(80, 120),
            )
        )

        speedl = result.plan.source_plan("speedl_stream").decisions[0]
        joy = result.plan.source_plan("joy").decisions[0]

        self.assertEqual(speedl.selection.source_timestamps_ns, (90,))
        self.assertEqual(joy.selection.source_timestamps_ns, (80,))

    def test_timestamp_input_keys_fail_closed(self):
        with self.assertRaisesRegex(DoosanPolicyError, "key mismatch"):
            DoosanPolicyInputs(
                reference_timestamps_ns=(1,),
                source_timestamps_ns=(
                    ("external_image", (1,)),
                ),
                header_bag_evidence=tuple(
                    (
                        topic,
                        HeaderBagOffsetEvidence(1, 0, 0, 0),
                    )
                    for topic in (
                        "/doosan_cameras/tcp_camera/color/image_raw",
                        "/doosan_cameras/external_camera_2/color/image_raw",
                        "/schunk/state",
                        "/dsr01/joint_states",
                        "/doosan_teleop/collector_joy",
                        "/doosan_cameras/tcp_camera/color/camera_info",
                        "/doosan_cameras/external_camera_2/color/camera_info",
                    )
                ),
                tcp_calibration=_calibration(
                    "/doosan_cameras/tcp_camera/color/camera_info"
                ),
                external_calibration=_calibration(
                    "/doosan_cameras/external_camera_2/color/camera_info"
                ),
            )

    def test_header_bag_evidence_shape(self):
        evidence = HeaderBagOffsetEvidence(
            count=4,
            minimum_ns=-40,
            maximum_ns=-10,
            max_absolute_ns=40,
        )
        self.assertEqual(
            evidence.to_dict(),
            {
                "count": 4,
                "min_ns": -40,
                "max_ns": -10,
                "max_absolute_ns": 40,
            },
        )
        self.assertEqual(ROS_EPOCH_MAX_HEADER_BAG_OFFSET_NS, 100_000_000)

    def test_report_is_json_safe_and_records_roles(self):
        result = build_doosan_sync_plan_from_inputs(_inputs())
        report = result.to_dict()

        payload = json.dumps(report, allow_nan=False)
        self.assertIn('"policy_id": "doosan_sync_policy_v1"', payload)
        self.assertEqual(report["clock_epoch"], "ros")
        self.assertEqual(
            report["joint_state_role"],
            "optional_validation_only",
        )
        self.assertIn("actual_joint_position", report["authoritative_state_fields"])
        self.assertNotIn("decisions", report["sources"]["external_image"])

    def test_decisions_are_opt_in(self):
        result = build_doosan_sync_plan_from_inputs(_inputs())
        report = result.to_dict(include_decisions=True)

        decisions = report["sources"]["external_image"]["decisions"]
        self.assertEqual(len(decisions), 3)
        self.assertEqual(decisions[0]["reference_index"], 0)


if __name__ == "__main__":
    unittest.main()
