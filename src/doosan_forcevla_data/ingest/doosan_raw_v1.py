"""Typed decoders for the final two-camera Doosan raw MCAP contract.

Patch 2 deliberately performs no synchronization and no model-facing unit or
frame conversion. It converts ROS message objects into immutable Python records
while preserving acquisition timestamps and rejecting ambiguous training data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterator

from .ros2_mcap import (
    BagMessage,
    EpisodeDescriptor,
    McapIngestError,
    iter_deserialized_messages,
    load_episode_descriptor,
)


RAW_CONTRACT_ID = "doosan_two_camera_rosbag_raw_v1"

ROBOT_STATE_RT_TOPIC = "/dsr01/dsr_controller2/robot_state_rt_monitoring"
JOINT_STATE_TOPIC = "/dsr01/joint_states"
SPEEDL_STREAM_TOPIC = "/dsr01/dsr_controller2/speedl_stream"
JOY_TOPIC = "/doosan_teleop/collector_joy"
GRIPPER_STATE_TOPIC = "/schunk/state"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
TCP_IMAGE_TOPIC = "/doosan_cameras/tcp_camera/color/image_raw"
TCP_CAMERA_INFO_TOPIC = "/doosan_cameras/tcp_camera/color/camera_info"
EXTERNAL_IMAGE_TOPIC = "/doosan_cameras/external_camera_2/color/image_raw"
EXTERNAL_CAMERA_INFO_TOPIC = "/doosan_cameras/external_camera_2/color/camera_info"

CANONICAL_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)


class TypedDecodeError(McapIngestError):
    """Raised when a raw ROS message violates the typed Doosan contract."""


@dataclass(frozen=True)
class TopicContract:
    topic: str
    type_name: str
    role: str
    has_message_header: bool


DOOSAN_RAW_V1_TOPICS = (
    TopicContract(
        ROBOT_STATE_RT_TOPIC,
        "dsr_msgs2/msg/RobotStateRt",
        "robot_state_rt",
        False,
    ),
    TopicContract(
        JOINT_STATE_TOPIC,
        "sensor_msgs/msg/JointState",
        "joint_state",
        True,
    ),
    TopicContract(
        SPEEDL_STREAM_TOPIC,
        "dsr_msgs2/msg/SpeedlStream",
        "speedl_stream",
        False,
    ),
    TopicContract(
        JOY_TOPIC,
        "sensor_msgs/msg/Joy",
        "joy",
        True,
    ),
    TopicContract(
        GRIPPER_STATE_TOPIC,
        "gripper_msgs/msg/GripperState",
        "gripper_state",
        True,
    ),
    TopicContract(
        TF_TOPIC,
        "tf2_msgs/msg/TFMessage",
        "tf",
        False,
    ),
    TopicContract(
        TF_STATIC_TOPIC,
        "tf2_msgs/msg/TFMessage",
        "tf_static",
        False,
    ),
    TopicContract(
        TCP_IMAGE_TOPIC,
        "sensor_msgs/msg/Image",
        "tcp_image",
        True,
    ),
    TopicContract(
        TCP_CAMERA_INFO_TOPIC,
        "sensor_msgs/msg/CameraInfo",
        "tcp_camera_info",
        True,
    ),
    TopicContract(
        EXTERNAL_IMAGE_TOPIC,
        "sensor_msgs/msg/Image",
        "external_image",
        True,
    ),
    TopicContract(
        EXTERNAL_CAMERA_INFO_TOPIC,
        "sensor_msgs/msg/CameraInfo",
        "external_camera_info",
        True,
    ),
)

_TOPIC_CONTRACT_BY_NAME = {
    spec.topic: spec
    for spec in DOOSAN_RAW_V1_TOPICS
}

DOOSAN_RAW_V1_TOPIC_TYPES = {
    spec.topic: spec.type_name
    for spec in DOOSAN_RAW_V1_TOPICS
}


@dataclass(frozen=True)
class RecordStamp:
    bag_timestamp_ns: int
    header_timestamp_ns: int | None
    frame_id: str | None


@dataclass(frozen=True)
class RobotStateRtRecord:
    stamp: RecordStamp
    controller_timestamp_s: float
    actual_joint_position_deg: tuple[float, ...]
    actual_joint_velocity_deg_s: tuple[float, ...]
    actual_tcp_position_mm_deg: tuple[float, ...]
    actual_tcp_velocity_mm_deg_s: tuple[float, ...]
    external_tcp_force_base_n_nm: tuple[float, ...]
    diagnostics: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class JointStateRecord:
    stamp: RecordStamp
    names: tuple[str, ...]
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    effort: tuple[float, ...] | None
    effort_status: str


@dataclass(frozen=True)
class SpeedlStreamRecord:
    stamp: RecordStamp
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]
    command_time_s: float


@dataclass(frozen=True)
class JoyRecord:
    stamp: RecordStamp
    axes: tuple[float, ...]
    buttons: tuple[int, ...]


@dataclass(frozen=True)
class GripperStateRecord:
    stamp: RecordStamp
    position_m: float
    holding: bool


@dataclass(frozen=True)
class ImageRecord:
    stamp: RecordStamp
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: memoryview


@dataclass(frozen=True)
class CameraInfoRecord:
    stamp: RecordStamp
    height: int
    width: int
    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]
    binning_x: int
    binning_y: int
    roi: tuple[int, int, int, int, bool]


@dataclass(frozen=True)
class TransformRecord:
    bag_timestamp_ns: int
    header_timestamp_ns: int
    parent_frame: str
    child_frame: str
    translation_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class TFMessageRecord:
    stamp: RecordStamp
    transforms: tuple[TransformRecord, ...]
    is_static: bool


TypedRecord = (
    RobotStateRtRecord
    | JointStateRecord
    | SpeedlStreamRecord
    | JoyRecord
    | GripperStateRecord
    | ImageRecord
    | CameraInfoRecord
    | TFMessageRecord
)


@dataclass(frozen=True)
class TypedEpisodeSummary:
    raw_contract: str
    total_messages: int
    topic_counts: dict[str, int]
    record_class_counts: dict[str, int]
    header_present_counts: dict[str, int]
    header_absent_counts: dict[str, int]
    joint_effort_status_counts: dict[str, int]
    training_whitelist_nonfinite_values: int
    tf_transform_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_contract": self.raw_contract,
            "total_messages": self.total_messages,
            "topic_counts": dict(self.topic_counts),
            "record_class_counts": dict(self.record_class_counts),
            "header_present_counts": dict(self.header_present_counts),
            "header_absent_counts": dict(self.header_absent_counts),
            "joint_effort_status_counts": dict(self.joint_effort_status_counts),
            "training_whitelist_nonfinite_values": (
                self.training_whitelist_nonfinite_values
            ),
            "tf_transform_counts": dict(self.tf_transform_counts),
        }


def validate_doosan_raw_v1_descriptor(
    descriptor: EpisodeDescriptor,
) -> None:
    """Require the exact final two-camera raw topic/type contract."""

    actual = {
        item.name: item.type_name
        for item in descriptor.metadata.topics
    }

    if actual != DOOSAN_RAW_V1_TOPIC_TYPES:
        missing = sorted(
            set(DOOSAN_RAW_V1_TOPIC_TYPES) - set(actual)
        )
        extra = sorted(
            set(actual) - set(DOOSAN_RAW_V1_TOPIC_TYPES)
        )
        wrong_types = {
            topic: {
                "expected": DOOSAN_RAW_V1_TOPIC_TYPES[topic],
                "actual": actual[topic],
            }
            for topic in sorted(
                set(actual) & set(DOOSAN_RAW_V1_TOPIC_TYPES)
            )
            if actual[topic] != DOOSAN_RAW_V1_TOPIC_TYPES[topic]
        }
        raise TypedDecodeError(
            "raw topic/type contract mismatch; "
            f"missing={missing}, extra={extra}, wrong_types={wrong_types}"
        )

    raw_contract = descriptor.metadata.custom_data.get(
        "raw_contract"
    )

    if raw_contract != RAW_CONTRACT_ID:
        raise TypedDecodeError(
            "metadata custom_data.raw_contract must be "
            f"{RAW_CONTRACT_ID!r}, got {raw_contract!r}"
        )


def validate_doosan_raw_v1_episode(
    episode_dir: str | Path,
) -> EpisodeDescriptor:
    descriptor = load_episode_descriptor(episode_dir)
    validate_doosan_raw_v1_descriptor(descriptor)
    return descriptor


def _stamp_to_ns(stamp: Any, context: str) -> int:
    try:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except Exception as exc:
        raise TypedDecodeError(
            f"{context}: invalid ROS timestamp object"
        ) from exc

    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise TypedDecodeError(
            f"{context}: invalid ROS timestamp sec={sec}, nanosec={nanosec}"
        )

    return sec * 1_000_000_000 + nanosec


def _required_header_stamp(
    record: BagMessage,
) -> RecordStamp:
    header = getattr(record.message, "header", None)

    if header is None:
        raise TypedDecodeError(
            f"{record.topic}: expected ROS Header"
        )

    header_ns = _stamp_to_ns(
        getattr(header, "stamp", None),
        f"{record.topic}.header.stamp",
    )

    if header_ns == 0:
        raise TypedDecodeError(
            f"{record.topic}: zero header timestamp"
        )

    return RecordStamp(
        bag_timestamp_ns=int(record.bag_timestamp_ns),
        header_timestamp_ns=header_ns,
        frame_id=str(getattr(header, "frame_id", "")),
    )


def _bag_only_stamp(record: BagMessage) -> RecordStamp:
    return RecordStamp(
        bag_timestamp_ns=int(record.bag_timestamp_ns),
        header_timestamp_ns=None,
        frame_id=None,
    )


def _finite_scalar(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypedDecodeError(
            f"{context}: expected numeric scalar"
        ) from exc

    if not math.isfinite(result):
        raise TypedDecodeError(
            f"{context}: non-finite value {result!r}"
        )

    return result


def _finite_vector(
    value: Any,
    length: int,
    context: str,
) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypedDecodeError(
            f"{context}: expected numeric vector"
        ) from exc

    if len(result) != length:
        raise TypedDecodeError(
            f"{context}: expected length {length}, got {len(result)}"
        )

    if not all(math.isfinite(item) for item in result):
        raise TypedDecodeError(
            f"{context}: contains non-finite values"
        )

    return result


def _ros_to_immutable(value: Any) -> Any:
    if value is None or isinstance(
        value,
        (bool, int, float, str, bytes),
    ):
        return value

    if isinstance(value, memoryview):
        return bytes(value)

    if hasattr(value, "get_fields_and_field_types"):
        return tuple(
            (
                name,
                _ros_to_immutable(getattr(value, name)),
            )
            for name in value.get_fields_and_field_types()
        )

    try:
        return tuple(
            _ros_to_immutable(item)
            for item in value
        )
    except TypeError:
        return repr(value)


def _decode_robot_state_rt(
    record: BagMessage,
) -> RobotStateRtRecord:
    msg = record.message
    core_names = {
        "time_stamp",
        "actual_joint_position",
        "actual_joint_velocity",
        "actual_tcp_position",
        "actual_tcp_velocity",
        "external_tcp_force",
    }

    fields = (
        msg.get_fields_and_field_types()
        if hasattr(msg, "get_fields_and_field_types")
        else {}
    )

    diagnostics = tuple(
        (
            name,
            _ros_to_immutable(getattr(msg, name)),
        )
        for name in fields
        if name not in core_names
    )

    return RobotStateRtRecord(
        stamp=_bag_only_stamp(record),
        controller_timestamp_s=_finite_scalar(
            msg.time_stamp,
            f"{record.topic}.time_stamp",
        ),
        actual_joint_position_deg=_finite_vector(
            msg.actual_joint_position,
            6,
            f"{record.topic}.actual_joint_position",
        ),
        actual_joint_velocity_deg_s=_finite_vector(
            msg.actual_joint_velocity,
            6,
            f"{record.topic}.actual_joint_velocity",
        ),
        actual_tcp_position_mm_deg=_finite_vector(
            msg.actual_tcp_position,
            6,
            f"{record.topic}.actual_tcp_position",
        ),
        actual_tcp_velocity_mm_deg_s=_finite_vector(
            msg.actual_tcp_velocity,
            6,
            f"{record.topic}.actual_tcp_velocity",
        ),
        external_tcp_force_base_n_nm=_finite_vector(
            msg.external_tcp_force,
            6,
            f"{record.topic}.external_tcp_force",
        ),
        diagnostics=diagnostics,
    )


def _decode_joint_state(
    record: BagMessage,
) -> JointStateRecord:
    msg = record.message
    stamp = _required_header_stamp(record)

    names = tuple(str(name) for name in msg.name)

    if len(names) != len(set(names)):
        raise TypedDecodeError(
            f"{record.topic}: duplicate joint names"
        )

    if set(names) != set(CANONICAL_JOINT_NAMES):
        raise TypedDecodeError(
            f"{record.topic}: expected joints "
            f"{CANONICAL_JOINT_NAMES!r}, got {names!r}"
        )

    positions_raw = tuple(float(value) for value in msg.position)
    velocities_raw = tuple(float(value) for value in msg.velocity)

    if len(positions_raw) != len(names):
        raise TypedDecodeError(
            f"{record.topic}.position: length {len(positions_raw)} "
            f"does not match names length {len(names)}"
        )

    if len(velocities_raw) != len(names):
        raise TypedDecodeError(
            f"{record.topic}.velocity: length {len(velocities_raw)} "
            f"does not match names length {len(names)}"
        )

    name_to_index = {
        name: index
        for index, name in enumerate(names)
    }

    position = tuple(
        _finite_scalar(
            positions_raw[name_to_index[name]],
            f"{record.topic}.position[{name}]",
        )
        for name in CANONICAL_JOINT_NAMES
    )

    velocity = tuple(
        _finite_scalar(
            velocities_raw[name_to_index[name]],
            f"{record.topic}.velocity[{name}]",
        )
        for name in CANONICAL_JOINT_NAMES
    )

    effort_raw = tuple(float(value) for value in msg.effort)

    if not effort_raw:
        effort = None
        effort_status = "unavailable_empty"
    elif len(effort_raw) != len(names):
        raise TypedDecodeError(
            f"{record.topic}.effort: length {len(effort_raw)} "
            f"does not match names length {len(names)}"
        )
    elif all(math.isnan(value) for value in effort_raw):
        effort = None
        effort_status = "unavailable_all_nonfinite"
    elif all(math.isfinite(value) for value in effort_raw):
        effort = tuple(
            effort_raw[name_to_index[name]]
            for name in CANONICAL_JOINT_NAMES
        )
        effort_status = "available_finite"
    else:
        raise TypedDecodeError(
            f"{record.topic}.effort: mixed, infinite, or otherwise "
            "ambiguous finite/non-finite values"
        )

    return JointStateRecord(
        stamp=stamp,
        names=CANONICAL_JOINT_NAMES,
        position=position,
        velocity=velocity,
        effort=effort,
        effort_status=effort_status,
    )


def _decode_speedl_stream(
    record: BagMessage,
) -> SpeedlStreamRecord:
    msg = record.message

    return SpeedlStreamRecord(
        stamp=_bag_only_stamp(record),
        velocity=_finite_vector(
            msg.vel,
            6,
            f"{record.topic}.vel",
        ),
        acceleration=_finite_vector(
            msg.acc,
            2,
            f"{record.topic}.acc",
        ),
        command_time_s=_finite_scalar(
            msg.time,
            f"{record.topic}.time",
        ),
    )


def _decode_joy(record: BagMessage) -> JoyRecord:
    msg = record.message
    axes = _finite_vector(
        msg.axes,
        6,
        f"{record.topic}.axes",
    )

    buttons = tuple(int(value) for value in msg.buttons)

    if len(buttons) != 2:
        raise TypedDecodeError(
            f"{record.topic}.buttons: expected length 2, got {len(buttons)}"
        )

    return JoyRecord(
        stamp=_required_header_stamp(record),
        axes=axes,
        buttons=buttons,
    )


def _decode_gripper_state(
    record: BagMessage,
) -> GripperStateRecord:
    msg = record.message

    return GripperStateRecord(
        stamp=_required_header_stamp(record),
        position_m=_finite_scalar(
            msg.position,
            f"{record.topic}.position",
        ),
        holding=bool(msg.holding),
    )


_IMAGE_PROFILES = {
    TCP_IMAGE_TOPIC: (
        480,
        640,
        "rgb8",
        0,
        1920,
        "tcp_camera_color_optical_frame",
    ),
    EXTERNAL_IMAGE_TOPIC: (
        480,
        848,
        "rgb8",
        0,
        2544,
        "external_camera_2_color_optical_frame",
    ),
}


def _readonly_view(data: Any) -> memoryview:
    try:
        view = memoryview(data)
    except TypeError:
        view = memoryview(bytes(data))

    try:
        return view.toreadonly()
    except AttributeError:
        return view


def _decode_image(record: BagMessage) -> ImageRecord:
    msg = record.message
    stamp = _required_header_stamp(record)

    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    is_bigendian = int(msg.is_bigendian)
    step = int(msg.step)

    expected = _IMAGE_PROFILES[record.topic]
    actual = (
        height,
        width,
        encoding,
        is_bigendian,
        step,
        stamp.frame_id,
    )

    if actual != expected:
        raise TypedDecodeError(
            f"{record.topic}: image profile mismatch; "
            f"expected={expected!r}, actual={actual!r}"
        )

    data = _readonly_view(msg.data)
    expected_bytes = step * height

    if data.nbytes != expected_bytes:
        raise TypedDecodeError(
            f"{record.topic}: image payload length {data.nbytes} "
            f"!= step*height {expected_bytes}"
        )

    return ImageRecord(
        stamp=stamp,
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=is_bigendian,
        step=step,
        data=data,
    )


_CAMERA_INFO_PROFILES = {
    TCP_CAMERA_INFO_TOPIC: (
        480,
        640,
        "plumb_bob",
        5,
        9,
        9,
        12,
        "tcp_camera_color_optical_frame",
    ),
    EXTERNAL_CAMERA_INFO_TOPIC: (
        480,
        848,
        "plumb_bob",
        5,
        9,
        9,
        12,
        "external_camera_2_color_optical_frame",
    ),
}


def _decode_camera_info(
    record: BagMessage,
) -> CameraInfoRecord:
    msg = record.message
    stamp = _required_header_stamp(record)

    d = tuple(float(value) for value in msg.d)
    k = tuple(float(value) for value in msg.k)
    r = tuple(float(value) for value in msg.r)
    p = tuple(float(value) for value in msg.p)

    actual = (
        int(msg.height),
        int(msg.width),
        str(msg.distortion_model),
        len(d),
        len(k),
        len(r),
        len(p),
        stamp.frame_id,
    )

    expected = _CAMERA_INFO_PROFILES[record.topic]

    if actual != expected:
        raise TypedDecodeError(
            f"{record.topic}: CameraInfo profile mismatch; "
            f"expected={expected!r}, actual={actual!r}"
        )

    calibration = d + k + r + p

    if not all(math.isfinite(value) for value in calibration):
        raise TypedDecodeError(
            f"{record.topic}: non-finite CameraInfo calibration value"
        )

    roi = msg.roi

    return CameraInfoRecord(
        stamp=stamp,
        height=int(msg.height),
        width=int(msg.width),
        distortion_model=str(msg.distortion_model),
        d=d,
        k=k,
        r=r,
        p=p,
        binning_x=int(msg.binning_x),
        binning_y=int(msg.binning_y),
        roi=(
            int(roi.x_offset),
            int(roi.y_offset),
            int(roi.height),
            int(roi.width),
            bool(roi.do_rectify),
        ),
    )


def _decode_tf(record: BagMessage) -> TFMessageRecord:
    transforms = []

    for index, transform in enumerate(record.message.transforms):
        header_ns = _stamp_to_ns(
            transform.header.stamp,
            f"{record.topic}.transforms[{index}].header.stamp",
        )

        if header_ns == 0:
            raise TypedDecodeError(
                f"{record.topic}.transforms[{index}]: zero header timestamp"
            )

        parent = str(transform.header.frame_id)
        child = str(transform.child_frame_id)

        if not parent or not child:
            raise TypedDecodeError(
                f"{record.topic}.transforms[{index}]: empty frame id"
            )

        value = transform.transform

        translation = (
            _finite_scalar(
                value.translation.x,
                f"{record.topic}.transforms[{index}].translation.x",
            ),
            _finite_scalar(
                value.translation.y,
                f"{record.topic}.transforms[{index}].translation.y",
            ),
            _finite_scalar(
                value.translation.z,
                f"{record.topic}.transforms[{index}].translation.z",
            ),
        )

        quaternion = (
            _finite_scalar(
                value.rotation.x,
                f"{record.topic}.transforms[{index}].rotation.x",
            ),
            _finite_scalar(
                value.rotation.y,
                f"{record.topic}.transforms[{index}].rotation.y",
            ),
            _finite_scalar(
                value.rotation.z,
                f"{record.topic}.transforms[{index}].rotation.z",
            ),
            _finite_scalar(
                value.rotation.w,
                f"{record.topic}.transforms[{index}].rotation.w",
            ),
        )

        norm = math.sqrt(
            sum(component * component for component in quaternion)
        )

        if abs(norm - 1.0) > 1e-3:
            raise TypedDecodeError(
                f"{record.topic}.transforms[{index}]: "
                f"quaternion norm {norm} is not approximately 1"
            )

        transforms.append(
            TransformRecord(
                bag_timestamp_ns=int(record.bag_timestamp_ns),
                header_timestamp_ns=header_ns,
                parent_frame=parent,
                child_frame=child,
                translation_xyz=translation,
                quaternion_xyzw=quaternion,
            )
        )

    if not transforms:
        raise TypedDecodeError(
            f"{record.topic}: TFMessage contains no transforms"
        )

    return TFMessageRecord(
        stamp=_bag_only_stamp(record),
        transforms=tuple(transforms),
        is_static=(record.topic == TF_STATIC_TOPIC),
    )


def decode_doosan_raw_v1_message(
    record: BagMessage,
) -> TypedRecord:
    """Decode one Patch-1 BagMessage under the final raw-v1 contract."""

    spec = _TOPIC_CONTRACT_BY_NAME.get(record.topic)

    if spec is None:
        raise TypedDecodeError(
            f"unexpected topic {record.topic!r}"
        )

    if record.type_name != spec.type_name:
        raise TypedDecodeError(
            f"{record.topic}: expected type {spec.type_name!r}, "
            f"got {record.type_name!r}"
        )

    if spec.role == "robot_state_rt":
        return _decode_robot_state_rt(record)
    if spec.role == "joint_state":
        return _decode_joint_state(record)
    if spec.role == "speedl_stream":
        return _decode_speedl_stream(record)
    if spec.role == "joy":
        return _decode_joy(record)
    if spec.role == "gripper_state":
        return _decode_gripper_state(record)
    if spec.role in ("tcp_image", "external_image"):
        return _decode_image(record)
    if spec.role in ("tcp_camera_info", "external_camera_info"):
        return _decode_camera_info(record)
    if spec.role in ("tf", "tf_static"):
        return _decode_tf(record)

    raise TypedDecodeError(
        f"{record.topic}: unsupported decoder role {spec.role!r}"
    )


def iter_typed_messages(
    episode_dir: str | Path,
) -> Iterator[tuple[str, TypedRecord]]:
    """Validate the raw-v1 episode contract and stream typed records."""

    validate_doosan_raw_v1_episode(episode_dir)

    for record in iter_deserialized_messages(episode_dir):
        yield (
            record.topic,
            decode_doosan_raw_v1_message(record),
        )


def _training_whitelist_values(
    record: TypedRecord,
) -> tuple[float, ...]:
    # These are raw candidates only. Patch 5 decides final model-state sourcing,
    # units, rotation representation, and wrench reset/tare semantics.
    if isinstance(record, RobotStateRtRecord):
        return (
            record.actual_tcp_position_mm_deg
            + record.actual_joint_position_deg
            + record.actual_joint_velocity_deg_s
            + record.external_tcp_force_base_n_nm
        )

    if isinstance(record, JointStateRecord):
        return record.position + record.velocity

    if isinstance(record, GripperStateRecord):
        return (record.position_m,)

    return ()


def scan_typed_episode(
    episode_dir: str | Path,
) -> TypedEpisodeSummary:
    """Decode every message and report Patch-2 safety invariants."""

    topic_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    header_present: Counter[str] = Counter()
    header_absent: Counter[str] = Counter()
    effort_status: Counter[str] = Counter()
    tf_transform_counts: Counter[str] = Counter()

    training_nonfinite = 0

    for topic, record in iter_typed_messages(episode_dir):
        topic_counts[topic] += 1
        class_counts[type(record).__name__] += 1

        if record.stamp.header_timestamp_ns is None:
            header_absent[topic] += 1
        else:
            header_present[topic] += 1

        if isinstance(record, JointStateRecord):
            effort_status[record.effort_status] += 1

        if isinstance(record, TFMessageRecord):
            tf_transform_counts[topic] += len(record.transforms)

        training_nonfinite += sum(
            1
            for value in _training_whitelist_values(record)
            if not math.isfinite(float(value))
        )

    total = sum(topic_counts.values())

    return TypedEpisodeSummary(
        raw_contract=RAW_CONTRACT_ID,
        total_messages=total,
        topic_counts=dict(sorted(topic_counts.items())),
        record_class_counts=dict(sorted(class_counts.items())),
        header_present_counts=dict(sorted(header_present.items())),
        header_absent_counts=dict(sorted(header_absent.items())),
        joint_effort_status_counts=dict(sorted(effort_status.items())),
        training_whitelist_nonfinite_values=training_nonfinite,
        tf_transform_counts=dict(sorted(tf_transform_counts.items())),
    )


__all__ = [
    "CANONICAL_JOINT_NAMES",
    "DOOSAN_RAW_V1_TOPICS",
    "DOOSAN_RAW_V1_TOPIC_TYPES",
    "EXTERNAL_CAMERA_INFO_TOPIC",
    "EXTERNAL_IMAGE_TOPIC",
    "GRIPPER_STATE_TOPIC",
    "JOINT_STATE_TOPIC",
    "JOY_TOPIC",
    "RAW_CONTRACT_ID",
    "ROBOT_STATE_RT_TOPIC",
    "SPEEDL_STREAM_TOPIC",
    "TCP_CAMERA_INFO_TOPIC",
    "TCP_IMAGE_TOPIC",
    "TF_STATIC_TOPIC",
    "TF_TOPIC",
    "CameraInfoRecord",
    "GripperStateRecord",
    "ImageRecord",
    "JointStateRecord",
    "JoyRecord",
    "RecordStamp",
    "RobotStateRtRecord",
    "SpeedlStreamRecord",
    "TFMessageRecord",
    "TopicContract",
    "TransformRecord",
    "TypedDecodeError",
    "TypedEpisodeSummary",
    "decode_doosan_raw_v1_message",
    "iter_typed_messages",
    "scan_typed_episode",
    "validate_doosan_raw_v1_descriptor",
    "validate_doosan_raw_v1_episode",
]
