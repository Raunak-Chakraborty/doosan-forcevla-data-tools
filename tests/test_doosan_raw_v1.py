import math
from types import SimpleNamespace
import unittest

from doosan_forcevla_data.ingest.doosan_raw_v1 import (
    CANONICAL_JOINT_NAMES,
    DOOSAN_RAW_V1_TOPIC_TYPES,
    EXTERNAL_IMAGE_TOPIC,
    GRIPPER_STATE_TOPIC,
    JOINT_STATE_TOPIC,
    JOY_TOPIC,
    RAW_CONTRACT_ID,
    ROBOT_STATE_RT_TOPIC,
    SPEEDL_STREAM_TOPIC,
    TCP_CAMERA_INFO_TOPIC,
    TCP_IMAGE_TOPIC,
    TF_STATIC_TOPIC,
    TF_TOPIC,
    CameraInfoRecord,
    GripperStateRecord,
    ImageRecord,
    JointStateRecord,
    JoyRecord,
    RobotStateRtRecord,
    SpeedlStreamRecord,
    TFMessageRecord,
    TypedDecodeError,
    decode_doosan_raw_v1_message,
    validate_doosan_raw_v1_descriptor,
)
from doosan_forcevla_data.ingest.ros2_mcap import (
    BagMessage,
    EpisodeDescriptor,
    EpisodeMetadata,
    TopicMetadata,
)


def _stamp(sec=123, nanosec=456):
    return SimpleNamespace(
        sec=sec,
        nanosec=nanosec,
    )


def _header(frame_id="", sec=123, nanosec=456):
    return SimpleNamespace(
        stamp=_stamp(sec, nanosec),
        frame_id=frame_id,
    )


def _bag(topic, type_name, message):
    return BagMessage(
        topic=topic,
        type_name=type_name,
        bag_timestamp_ns=123_000_000_456,
        message=message,
    )


def _topic_metadata():
    return tuple(
        TopicMetadata(
            name=topic,
            type_name=type_name,
            serialization_format="cdr",
            message_count=1,
        )
        for topic, type_name in DOOSAN_RAW_V1_TOPIC_TYPES.items()
    )


def _descriptor(raw_contract=RAW_CONTRACT_ID, topics=None):
    if topics is None:
        topics = _topic_metadata()

    metadata = EpisodeMetadata(
        storage_identifier="mcap",
        relative_file_paths=("episode.mcap",),
        message_count=len(topics),
        topics=tuple(topics),
        custom_data={"raw_contract": raw_contract},
    )

    return EpisodeDescriptor(
        episode_dir=__import__("pathlib").Path("/tmp/episode"),
        metadata=metadata,
        operator={},
        validation={},
    )


def _robot_message(**overrides):
    fields = {
        "time_stamp": "float64",
        "actual_joint_position": "float64[6]",
        "actual_joint_position_abs": "float64[6]",
        "actual_joint_velocity": "float64[6]",
        "actual_joint_velocity_abs": "float64[6]",
        "actual_tcp_position": "float64[6]",
        "actual_tcp_velocity": "float64[6]",
        "raw_force_torque": "float64[6]",
        "external_tcp_force": "float64[6]",
        "robot_mode": "uint8",
    }

    values = {
        "time_stamp": 10.25,
        "actual_joint_position": [1, 2, 3, 4, 5, 6],
        "actual_joint_position_abs": [1, 2, 3, 4, 5, 6],
        "actual_joint_velocity": [6, 5, 4, 3, 2, 1],
        "actual_joint_velocity_abs": [6, 5, 4, 3, 2, 1],
        "actual_tcp_position": [100, 200, 300, 10, 20, 30],
        "actual_tcp_velocity": [1, 2, 3, 4, 5, 6],
        "raw_force_torque": [0, 0, 0, 0, 0, 0],
        "external_tcp_force": [1, 2, 3, 4, 5, 6],
        "robot_mode": 1,
    }

    values.update(overrides)

    msg = SimpleNamespace(**values)
    msg.get_fields_and_field_types = lambda: fields
    return msg


def _joint_message(
    names=CANONICAL_JOINT_NAMES,
    effort=None,
):
    if effort is None:
        effort = [float("nan")] * 6

    return SimpleNamespace(
        header=_header(""),
        name=list(names),
        position=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        velocity=[1, 2, 3, 4, 5, 6],
        effort=list(effort),
    )


def _tf_transform(parent="base", child="link"):
    return SimpleNamespace(
        header=_header(parent),
        child_frame_id=child,
        transform=SimpleNamespace(
            translation=SimpleNamespace(
                x=0.1,
                y=0.2,
                z=0.3,
            ),
            rotation=SimpleNamespace(
                x=0.0,
                y=0.0,
                z=0.0,
                w=1.0,
            ),
        ),
    )


class DoosanRawV1TypedDecoderTests(unittest.TestCase):
    def test_exact_contract_accepts_expected_topics_and_tag(self):
        validate_doosan_raw_v1_descriptor(_descriptor())

    def test_contract_rejects_wrong_raw_contract_tag(self):
        with self.assertRaisesRegex(
            TypedDecodeError,
            "raw_contract",
        ):
            validate_doosan_raw_v1_descriptor(
                _descriptor(raw_contract="wrong")
            )

    def test_contract_rejects_missing_topic(self):
        topics = list(_topic_metadata())[:-1]

        with self.assertRaisesRegex(
            TypedDecodeError,
            "contract mismatch",
        ):
            validate_doosan_raw_v1_descriptor(
                _descriptor(topics=topics)
            )

    def test_robot_state_preserves_controller_and_bag_time(self):
        record = decode_doosan_raw_v1_message(
            _bag(
                ROBOT_STATE_RT_TOPIC,
                DOOSAN_RAW_V1_TOPIC_TYPES[ROBOT_STATE_RT_TOPIC],
                _robot_message(),
            )
        )

        self.assertIsInstance(record, RobotStateRtRecord)
        self.assertEqual(record.stamp.bag_timestamp_ns, 123_000_000_456)
        self.assertIsNone(record.stamp.header_timestamp_ns)
        self.assertEqual(record.controller_timestamp_s, 10.25)
        self.assertEqual(record.actual_tcp_position_mm_deg[:3], (100.0, 200.0, 300.0))
        self.assertEqual(record.external_tcp_force_base_n_nm, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertIn(("robot_mode", 1), record.diagnostics)

    def test_robot_state_rejects_nonfinite_training_candidate(self):
        with self.assertRaisesRegex(
            TypedDecodeError,
            "external_tcp_force",
        ):
            decode_doosan_raw_v1_message(
                _bag(
                    ROBOT_STATE_RT_TOPIC,
                    DOOSAN_RAW_V1_TOPIC_TYPES[ROBOT_STATE_RT_TOPIC],
                    _robot_message(
                        external_tcp_force=[
                            1,
                            2,
                            float("nan"),
                            4,
                            5,
                            6,
                        ]
                    ),
                )
            )

    def test_joint_state_reorders_by_name_and_drops_all_nan_effort(self):
        reversed_names = tuple(reversed(CANONICAL_JOINT_NAMES))

        msg = _joint_message(names=reversed_names)
        msg.position = [6, 5, 4, 3, 2, 1]
        msg.velocity = [60, 50, 40, 30, 20, 10]

        record = decode_doosan_raw_v1_message(
            _bag(
                JOINT_STATE_TOPIC,
                DOOSAN_RAW_V1_TOPIC_TYPES[JOINT_STATE_TOPIC],
                msg,
            )
        )

        self.assertIsInstance(record, JointStateRecord)
        self.assertEqual(record.names, CANONICAL_JOINT_NAMES)
        self.assertEqual(record.position, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertEqual(record.velocity, (10.0, 20.0, 30.0, 40.0, 50.0, 60.0))
        self.assertIsNone(record.effort)
        self.assertEqual(record.effort_status, "unavailable_all_nonfinite")
        self.assertIsNotNone(record.stamp.header_timestamp_ns)

    def test_joint_state_rejects_missing_joint_instead_of_zero_filling(self):
        with self.assertRaisesRegex(
            TypedDecodeError,
            "expected joints",
        ):
            decode_doosan_raw_v1_message(
                _bag(
                    JOINT_STATE_TOPIC,
                    DOOSAN_RAW_V1_TOPIC_TYPES[JOINT_STATE_TOPIC],
                    _joint_message(
                        names=CANONICAL_JOINT_NAMES[:-1]
                    ),
                )
            )

    def test_joint_state_rejects_mixed_effort(self):
        with self.assertRaisesRegex(
            TypedDecodeError,
            "mixed",
        ):
            decode_doosan_raw_v1_message(
                _bag(
                    JOINT_STATE_TOPIC,
                    DOOSAN_RAW_V1_TOPIC_TYPES[JOINT_STATE_TOPIC],
                    _joint_message(
                        effort=[
                            float("nan"),
                            1,
                            float("nan"),
                            2,
                            float("nan"),
                            3,
                        ]
                    ),
                )
            )

    def test_speedl_time_is_payload_not_header(self):
        msg = SimpleNamespace(
            vel=[1, 2, 3, 4, 5, 6],
            acc=[10, 20],
            time=0.1,
        )

        record = decode_doosan_raw_v1_message(
            _bag(
                SPEEDL_STREAM_TOPIC,
                DOOSAN_RAW_V1_TOPIC_TYPES[SPEEDL_STREAM_TOPIC],
                msg,
            )
        )

        self.assertIsInstance(record, SpeedlStreamRecord)
        self.assertIsNone(record.stamp.header_timestamp_ns)
        self.assertEqual(record.command_time_s, 0.1)

    def test_joy_and_gripper_preserve_header_timestamp(self):
        joy = decode_doosan_raw_v1_message(
            _bag(
                JOY_TOPIC,
                DOOSAN_RAW_V1_TOPIC_TYPES[JOY_TOPIC],
                SimpleNamespace(
                    header=_header(""),
                    axes=[0, 0, 0, 0, 0, 0],
                    buttons=[0, 1],
                ),
            )
        )

        gripper = decode_doosan_raw_v1_message(
            _bag(
                GRIPPER_STATE_TOPIC,
                DOOSAN_RAW_V1_TOPIC_TYPES[GRIPPER_STATE_TOPIC],
                SimpleNamespace(
                    header=_header(""),
                    position=0.013,
                    holding=True,
                ),
            )
        )

        self.assertIsInstance(joy, JoyRecord)
        self.assertIsInstance(gripper, GripperStateRecord)
        self.assertIsNotNone(joy.stamp.header_timestamp_ns)
        self.assertIsNotNone(gripper.stamp.header_timestamp_ns)
        self.assertEqual(joy.buttons, (0, 1))
        self.assertAlmostEqual(gripper.position_m, 0.013)

    def test_image_profile_is_topic_specific_and_payload_zero_copy_compatible(self):
        payload = bytearray(480 * 640 * 3)

        record = decode_doosan_raw_v1_message(
            _bag(
                TCP_IMAGE_TOPIC,
                DOOSAN_RAW_V1_TOPIC_TYPES[TCP_IMAGE_TOPIC],
                SimpleNamespace(
                    header=_header("tcp_camera_color_optical_frame"),
                    height=480,
                    width=640,
                    encoding="rgb8",
                    is_bigendian=0,
                    step=1920,
                    data=payload,
                ),
            )
        )

        self.assertIsInstance(record, ImageRecord)
        self.assertEqual(record.data.nbytes, 921600)
        self.assertTrue(record.data.readonly)

        with self.assertRaisesRegex(
            TypedDecodeError,
            "profile mismatch",
        ):
            decode_doosan_raw_v1_message(
                _bag(
                    EXTERNAL_IMAGE_TOPIC,
                    DOOSAN_RAW_V1_TOPIC_TYPES[EXTERNAL_IMAGE_TOPIC],
                    SimpleNamespace(
                        header=_header("external_camera_2_color_optical_frame"),
                        height=480,
                        width=640,
                        encoding="rgb8",
                        is_bigendian=0,
                        step=1920,
                        data=payload,
                    ),
                )
            )

    def test_camera_info_validates_profile_and_calibration_finiteness(self):
        roi = SimpleNamespace(
            x_offset=0,
            y_offset=0,
            height=0,
            width=0,
            do_rectify=False,
        )

        msg = SimpleNamespace(
            header=_header("tcp_camera_color_optical_frame"),
            height=480,
            width=640,
            distortion_model="plumb_bob",
            d=[0] * 5,
            k=[0] * 9,
            r=[0] * 9,
            p=[0] * 12,
            binning_x=0,
            binning_y=0,
            roi=roi,
        )

        record = decode_doosan_raw_v1_message(
            _bag(
                TCP_CAMERA_INFO_TOPIC,
                DOOSAN_RAW_V1_TOPIC_TYPES[TCP_CAMERA_INFO_TOPIC],
                msg,
            )
        )

        self.assertIsInstance(record, CameraInfoRecord)

        msg.k[0] = float("nan")

        with self.assertRaisesRegex(
            TypedDecodeError,
            "non-finite",
        ):
            decode_doosan_raw_v1_message(
                _bag(
                    TCP_CAMERA_INFO_TOPIC,
                    DOOSAN_RAW_V1_TOPIC_TYPES[TCP_CAMERA_INFO_TOPIC],
                    msg,
                )
            )

    def test_tf_preserves_each_transform_header_timestamp(self):
        msg = SimpleNamespace(
            transforms=[
                _tf_transform("base", "link1"),
                _tf_transform("link1", "link2"),
            ]
        )

        record = decode_doosan_raw_v1_message(
            _bag(
                TF_TOPIC,
                DOOSAN_RAW_V1_TOPIC_TYPES[TF_TOPIC],
                msg,
            )
        )

        self.assertIsInstance(record, TFMessageRecord)
        self.assertFalse(record.is_static)
        self.assertIsNone(record.stamp.header_timestamp_ns)
        self.assertEqual(len(record.transforms), 2)
        self.assertEqual(
            record.transforms[0].header_timestamp_ns,
            123_000_000_456,
        )
        self.assertEqual(
            record.transforms[0].quaternion_xyzw,
            (0.0, 0.0, 0.0, 1.0),
        )

        static_record = decode_doosan_raw_v1_message(
            _bag(
                TF_STATIC_TOPIC,
                DOOSAN_RAW_V1_TOPIC_TYPES[TF_STATIC_TOPIC],
                SimpleNamespace(
                    transforms=[
                        _tf_transform("base", "sensor")
                    ]
                ),
            )
        )

        self.assertTrue(static_record.is_static)

    def test_wrong_message_type_fails_before_decode(self):
        with self.assertRaisesRegex(
            TypedDecodeError,
            "expected type",
        ):
            decode_doosan_raw_v1_message(
                _bag(
                    JOY_TOPIC,
                    "std_msgs/msg/String",
                    SimpleNamespace(),
                )
            )


if __name__ == "__main__":
    unittest.main()
