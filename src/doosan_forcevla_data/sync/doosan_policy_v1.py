"""Production Doosan synchronization policy built on the generic Patch-3 planner.

Patch 4 selects timestamp sources, association methods, freshness limits, and
required/optional stream roles for the final two-camera Doosan raw contract.

The policy deliberately separates *timestamp source* from *clock epoch*:
ROS message header stamps and rosbag record stamps are both treated as samples
of the ROS/system epoch after episode-level plausibility checks. The Doosan
controller payload timestamp remains a distinct diagnostic clock and is never
silently mixed into the ROS synchronization timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from doosan_forcevla_data.ingest.doosan_raw_v1 import (
    EXTERNAL_CAMERA_INFO_TOPIC,
    EXTERNAL_IMAGE_TOPIC,
    GRIPPER_STATE_TOPIC,
    JOINT_STATE_TOPIC,
    JOY_TOPIC,
    ROBOT_STATE_RT_TOPIC,
    SPEEDL_STREAM_TOPIC,
    TCP_CAMERA_INFO_TOPIC,
    TCP_IMAGE_TOPIC,
    CameraInfoRecord,
    iter_typed_messages,
)
from doosan_forcevla_data.sync.timestamp_plan import (
    ClockDomain,
    SyncMethod,
    SyncSpec,
    SynchronizationError,
    SynchronizationPlan,
    TimestampTimeline,
    build_synchronization_plan,
)


POLICY_ID = "doosan_sync_policy_v1"
REFERENCE_KEY = "tcp_image"
REFERENCE_TOPIC = TCP_IMAGE_TOPIC
ROS_EPOCH_MAX_HEADER_BAG_OFFSET_NS = 100_000_000


class DoosanPolicyError(ValueError):
    """Raised when an episode cannot satisfy the Doosan synchronization policy."""


class TimestampSource(str, Enum):
    """Raw timestamp field selected before values enter the common ROS epoch."""

    BAG = "bag"
    HEADER = "header"


@dataclass(frozen=True)
class StreamPolicy:
    """Doosan policy for one synchronized source stream."""

    key: str
    topic: str
    timestamp_source: TimestampSource
    method: SyncMethod
    required: bool
    max_age_ns: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise DoosanPolicyError("stream policy key must be non-empty")
        if not isinstance(self.topic, str) or not self.topic.startswith("/"):
            raise DoosanPolicyError(f"invalid ROS topic {self.topic!r}")
        if not isinstance(self.timestamp_source, TimestampSource):
            raise DoosanPolicyError("timestamp_source must be TimestampSource")
        if not isinstance(self.method, SyncMethod):
            raise DoosanPolicyError("method must be SyncMethod")
        if not isinstance(self.required, bool):
            raise DoosanPolicyError("required must be bool")
        if self.max_age_ns is not None:
            if (
                isinstance(self.max_age_ns, bool)
                or not isinstance(self.max_age_ns, int)
                or self.max_age_ns < 0
            ):
                raise DoosanPolicyError(
                    f"max_age_ns must be None or a non-negative integer, "
                    f"got {self.max_age_ns!r}"
                )

    def sync_spec(self) -> SyncSpec:
        return SyncSpec(
            method=self.method,
            required=self.required,
            max_age_ns=self.max_age_ns,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "topic": self.topic,
            "timestamp_source": self.timestamp_source.value,
            "clock_domain": ClockDomain.ROS.value,
            "method": self.method.value,
            "required": self.required,
            "max_age_ns": self.max_age_ns,
        }


REFERENCE_TIMESTAMP_SOURCE = TimestampSource.HEADER

SOURCE_POLICIES = (
    StreamPolicy(
        key="external_image",
        topic=EXTERNAL_IMAGE_TOPIC,
        timestamp_source=TimestampSource.HEADER,
        method=SyncMethod.NEAREST,
        required=True,
        max_age_ns=12_000_000,
    ),
    StreamPolicy(
        key="robot_state_rt",
        topic=ROBOT_STATE_RT_TOPIC,
        timestamp_source=TimestampSource.BAG,
        method=SyncMethod.NEAREST,
        required=True,
        max_age_ns=16_000_000,
    ),
    StreamPolicy(
        key="gripper_state",
        topic=GRIPPER_STATE_TOPIC,
        timestamp_source=TimestampSource.HEADER,
        method=SyncMethod.NEAREST,
        required=True,
        max_age_ns=15_000_000,
    ),
    StreamPolicy(
        key="joint_state",
        topic=JOINT_STATE_TOPIC,
        timestamp_source=TimestampSource.HEADER,
        method=SyncMethod.NEAREST,
        required=False,
        max_age_ns=15_000_000,
    ),
    StreamPolicy(
        key="speedl_stream",
        topic=SPEEDL_STREAM_TOPIC,
        timestamp_source=TimestampSource.BAG,
        method=SyncMethod.CAUSAL_HOLD,
        required=False,
        max_age_ns=25_000_000,
    ),
    StreamPolicy(
        key="joy",
        topic=JOY_TOPIC,
        timestamp_source=TimestampSource.HEADER,
        method=SyncMethod.CAUSAL_HOLD,
        required=False,
        max_age_ns=25_000_000,
    ),
)

_SOURCE_POLICY_BY_KEY = {policy.key: policy for policy in SOURCE_POLICIES}
_SOURCE_POLICY_BY_TOPIC = {policy.topic: policy for policy in SOURCE_POLICIES}

if len(_SOURCE_POLICY_BY_KEY) != len(SOURCE_POLICIES):
    raise RuntimeError("duplicate Doosan synchronization policy key")
if len(_SOURCE_POLICY_BY_TOPIC) != len(SOURCE_POLICIES):
    raise RuntimeError("duplicate Doosan synchronization policy topic")


@dataclass(frozen=True)
class HeaderBagOffsetEvidence:
    """Episode-level evidence that a header stamp shares the ROS/system epoch."""

    count: int
    minimum_ns: int
    maximum_ns: int
    max_absolute_ns: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise DoosanPolicyError("header/bag evidence count must be positive")
        if self.max_absolute_ns < 0:
            raise DoosanPolicyError("max_absolute_ns must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "count": self.count,
            "min_ns": self.minimum_ns,
            "max_ns": self.maximum_ns,
            "max_absolute_ns": self.max_absolute_ns,
        }


@dataclass(frozen=True)
class CameraCalibration:
    """One episode-constant CameraInfo calibration snapshot."""

    topic: str
    frame_id: str
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

    def to_dict(self) -> dict[str, object]:
        return {
            "topic": self.topic,
            "frame_id": self.frame_id,
            "height": self.height,
            "width": self.width,
            "distortion_model": self.distortion_model,
            "d": list(self.d),
            "k": list(self.k),
            "r": list(self.r),
            "p": list(self.p),
            "binning_x": self.binning_x,
            "binning_y": self.binning_y,
            "roi": list(self.roi),
        }


@dataclass(frozen=True)
class DoosanPolicyInputs:
    """Timestamp-only inputs and constant calibration metadata for Patch 4."""

    reference_timestamps_ns: tuple[int, ...]
    source_timestamps_ns: tuple[tuple[str, tuple[int, ...]], ...]
    header_bag_evidence: tuple[tuple[str, HeaderBagOffsetEvidence], ...]
    tcp_calibration: CameraCalibration
    external_calibration: CameraCalibration

    def source_timestamps(self, key: str) -> tuple[int, ...]:
        for candidate, timestamps in self.source_timestamps_ns:
            if candidate == key:
                return timestamps
        raise KeyError(key)

    def evidence_for_topic(self, topic: str) -> HeaderBagOffsetEvidence:
        for candidate, evidence in self.header_bag_evidence:
            if candidate == topic:
                return evidence
        raise KeyError(topic)

    def __post_init__(self) -> None:
        expected_keys = {policy.key for policy in SOURCE_POLICIES}
        actual_keys = {key for key, _ in self.source_timestamps_ns}
        if (
            actual_keys != expected_keys
            or len(self.source_timestamps_ns) != len(expected_keys)
        ):
            raise DoosanPolicyError(
                "source timestamp key mismatch; "
                f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}"
            )

        expected_evidence_topics = {
            TCP_IMAGE_TOPIC,
            EXTERNAL_IMAGE_TOPIC,
            GRIPPER_STATE_TOPIC,
            JOINT_STATE_TOPIC,
            JOY_TOPIC,
            TCP_CAMERA_INFO_TOPIC,
            EXTERNAL_CAMERA_INFO_TOPIC,
        }
        actual_evidence_topics = {
            topic for topic, _ in self.header_bag_evidence
        }
        if (
            actual_evidence_topics != expected_evidence_topics
            or len(self.header_bag_evidence) != len(expected_evidence_topics)
        ):
            raise DoosanPolicyError(
                "header/bag evidence topic mismatch; "
                f"expected={sorted(expected_evidence_topics)}, "
                f"actual={sorted(actual_evidence_topics)}"
            )

        if self.tcp_calibration.topic != TCP_CAMERA_INFO_TOPIC:
            raise DoosanPolicyError("TCP calibration topic mismatch")
        if self.external_calibration.topic != EXTERNAL_CAMERA_INFO_TOPIC:
            raise DoosanPolicyError("external calibration topic mismatch")

        if not self.reference_timestamps_ns:
            raise DoosanPolicyError("reference timeline must not be empty")


@dataclass(frozen=True)
class DoosanSynchronizationResult:
    """Final timestamp/index plan plus the frozen Doosan policy provenance."""

    inputs: DoosanPolicyInputs
    plan: SynchronizationPlan

    @property
    def complete_reference_count(self) -> int:
        return len(self.plan.complete_reference_indices)

    @property
    def dropped_reference_count(self) -> int:
        return len(self.plan.dropped_reference_indices)

    def to_dict(self, *, include_decisions: bool = False) -> dict[str, object]:
        source_reports: dict[str, object] = {}

        for policy in SOURCE_POLICIES:
            source_plan = self.plan.source_plan(policy.key)
            item: dict[str, object] = {
                "policy": policy.to_dict(),
                "summary": source_plan.summary.to_dict(),
            }
            if include_decisions:
                item["decisions"] = [
                    decision.to_dict() for decision in source_plan.decisions
                ]
            source_reports[policy.key] = item

        return {
            "policy_id": POLICY_ID,
            "clock_epoch": ClockDomain.ROS.value,
            "timestamp_source_is_distinct_from_clock_epoch": True,
            "reference": {
                "key": REFERENCE_KEY,
                "topic": REFERENCE_TOPIC,
                "timestamp_source": REFERENCE_TIMESTAMP_SOURCE.value,
                "clock_domain": ClockDomain.ROS.value,
                "count": len(self.inputs.reference_timestamps_ns),
            },
            "sources": source_reports,
            "complete_reference_count": self.complete_reference_count,
            "dropped_reference_count": self.dropped_reference_count,
            "dropped_reference_indices": list(self.plan.dropped_reference_indices),
            "camera_info_mode": "episode_constant_calibration_metadata",
            "camera_calibration": {
                "tcp": self.inputs.tcp_calibration.to_dict(),
                "external": self.inputs.external_calibration.to_dict(),
            },
            "header_bag_epoch_evidence": {
                topic: evidence.to_dict()
                for topic, evidence in self.inputs.header_bag_evidence
            },
            "authoritative_state_source": ROBOT_STATE_RT_TOPIC,
            "authoritative_state_fields": [
                "actual_tcp_position",
                "actual_joint_position",
                "actual_joint_velocity",
                "external_tcp_force",
            ],
            "joint_state_role": "optional_validation_only",
            "speedl_role": "optional_command_provenance_only",
            "joy_role": "optional_operator_intent_provenance_only",
            "controller_payload_timestamp_role": (
                "preserved diagnostic clock; excluded from ROS-epoch synchronization"
            ),
            "tf_role": (
                "retained by raw contract but excluded from the Patch-4 frame "
                "synchronization policy"
            ),
        }


def _timestamp_from_record(
    *,
    topic: str,
    bag_timestamp_ns: int,
    header_timestamp_ns: int | None,
    source: TimestampSource,
) -> int:
    if source is TimestampSource.BAG:
        return int(bag_timestamp_ns)

    if source is TimestampSource.HEADER:
        if header_timestamp_ns is None:
            raise DoosanPolicyError(
                f"{topic}: policy requires header timestamp but record has none"
            )
        return int(header_timestamp_ns)

    raise DoosanPolicyError(f"unsupported timestamp source {source!r}")


def _camera_calibration(topic: str, record: CameraInfoRecord) -> CameraCalibration:
    frame_id = record.stamp.frame_id
    if not isinstance(frame_id, str) or not frame_id:
        raise DoosanPolicyError(f"{topic}: CameraInfo frame_id must be non-empty")

    return CameraCalibration(
        topic=topic,
        frame_id=frame_id,
        height=int(record.height),
        width=int(record.width),
        distortion_model=str(record.distortion_model),
        d=tuple(float(value) for value in record.d),
        k=tuple(float(value) for value in record.k),
        r=tuple(float(value) for value in record.r),
        p=tuple(float(value) for value in record.p),
        binning_x=int(record.binning_x),
        binning_y=int(record.binning_y),
        roi=tuple(record.roi),
    )


def _build_offset_evidence(
    topic: str,
    offsets: list[int],
) -> HeaderBagOffsetEvidence:
    if not offsets:
        raise DoosanPolicyError(
            f"{topic}: no header/bag samples were available for epoch validation"
        )

    maximum_absolute = max(abs(value) for value in offsets)

    if maximum_absolute > ROS_EPOCH_MAX_HEADER_BAG_OFFSET_NS:
        raise DoosanPolicyError(
            f"{topic}: header/bag offset exceeds "
            f"{ROS_EPOCH_MAX_HEADER_BAG_OFFSET_NS} ns; "
            "refusing to assume a shared ROS/system epoch"
        )

    return HeaderBagOffsetEvidence(
        count=len(offsets),
        minimum_ns=min(offsets),
        maximum_ns=max(offsets),
        max_absolute_ns=maximum_absolute,
    )


def collect_doosan_policy_inputs(
    episode_dir: str | Path,
) -> DoosanPolicyInputs:
    """Read one episode once and collect timestamp-only Patch-4 policy inputs."""

    selected_times: dict[str, list[int]] = {
        REFERENCE_KEY: [],
        **{policy.key: [] for policy in SOURCE_POLICIES},
    }

    header_bag_offsets: dict[str, list[int]] = {}

    # Header-based streams whose timestamps enter the shared ROS epoch.
    for topic in (
        TCP_IMAGE_TOPIC,
        EXTERNAL_IMAGE_TOPIC,
        GRIPPER_STATE_TOPIC,
        JOINT_STATE_TOPIC,
        JOY_TOPIC,
        TCP_CAMERA_INFO_TOPIC,
        EXTERNAL_CAMERA_INFO_TOPIC,
    ):
        header_bag_offsets[topic] = []

    image_header_times = {
        TCP_IMAGE_TOPIC: [],
        EXTERNAL_IMAGE_TOPIC: [],
    }
    camera_info_header_times = {
        TCP_CAMERA_INFO_TOPIC: [],
        EXTERNAL_CAMERA_INFO_TOPIC: [],
    }

    calibrations: dict[str, CameraCalibration] = {}

    for topic, record in iter_typed_messages(episode_dir):
        bag_ns = int(record.stamp.bag_timestamp_ns)
        header_ns = record.stamp.header_timestamp_ns

        if topic in header_bag_offsets:
            if header_ns is None:
                raise DoosanPolicyError(
                    f"{topic}: expected header stamp for ROS-epoch evidence"
                )
            header_bag_offsets[topic].append(int(header_ns) - bag_ns)

        if topic == TCP_IMAGE_TOPIC:
            if header_ns is None:
                raise DoosanPolicyError(
                    f"{TCP_IMAGE_TOPIC}: reference image lacks header timestamp"
                )
            timestamp = int(header_ns)
            selected_times[REFERENCE_KEY].append(timestamp)
            image_header_times[TCP_IMAGE_TOPIC].append(timestamp)
            continue

        policy = _SOURCE_POLICY_BY_TOPIC.get(topic)
        if policy is not None:
            selected_times[policy.key].append(
                _timestamp_from_record(
                    topic=topic,
                    bag_timestamp_ns=bag_ns,
                    header_timestamp_ns=header_ns,
                    source=policy.timestamp_source,
                )
            )
            if topic == EXTERNAL_IMAGE_TOPIC:
                if header_ns is None:
                    raise DoosanPolicyError(
                        f"{EXTERNAL_IMAGE_TOPIC}: image lacks header timestamp"
                    )
                image_header_times[EXTERNAL_IMAGE_TOPIC].append(int(header_ns))
            continue

        if topic in (TCP_CAMERA_INFO_TOPIC, EXTERNAL_CAMERA_INFO_TOPIC):
            if not isinstance(record, CameraInfoRecord):
                raise DoosanPolicyError(
                    f"{topic}: expected CameraInfoRecord, got {type(record).__name__}"
                )
            if header_ns is None:
                raise DoosanPolicyError(f"{topic}: CameraInfo lacks header timestamp")

            camera_info_header_times[topic].append(int(header_ns))
            calibration = _camera_calibration(topic, record)

            previous = calibrations.get(topic)
            if previous is None:
                calibrations[topic] = calibration
            elif calibration != previous:
                raise DoosanPolicyError(
                    f"{topic}: CameraInfo calibration changed within one episode"
                )

    if (
        image_header_times[TCP_IMAGE_TOPIC]
        != camera_info_header_times[TCP_CAMERA_INFO_TOPIC]
    ):
        raise DoosanPolicyError(
            "TCP Image and CameraInfo header timestamp sequences are not identical"
        )

    if (
        image_header_times[EXTERNAL_IMAGE_TOPIC]
        != camera_info_header_times[EXTERNAL_CAMERA_INFO_TOPIC]
    ):
        raise DoosanPolicyError(
            "external Image and CameraInfo header timestamp sequences are not identical"
        )

    try:
        tcp_calibration = calibrations[TCP_CAMERA_INFO_TOPIC]
        external_calibration = calibrations[EXTERNAL_CAMERA_INFO_TOPIC]
    except KeyError as exc:
        raise DoosanPolicyError("required CameraInfo calibration is missing") from exc

    evidence = tuple(
        (
            topic,
            _build_offset_evidence(topic, offsets),
        )
        for topic, offsets in sorted(header_bag_offsets.items())
    )

    return DoosanPolicyInputs(
        reference_timestamps_ns=tuple(selected_times[REFERENCE_KEY]),
        source_timestamps_ns=tuple(
            (
                policy.key,
                tuple(selected_times[policy.key]),
            )
            for policy in SOURCE_POLICIES
        ),
        header_bag_evidence=evidence,
        tcp_calibration=tcp_calibration,
        external_calibration=external_calibration,
    )


def build_doosan_sync_plan_from_inputs(
    inputs: DoosanPolicyInputs,
) -> DoosanSynchronizationResult:
    """Build the frozen Doosan synchronization plan from timestamp-only inputs."""

    reference = TimestampTimeline.from_timestamps(
        REFERENCE_KEY,
        ClockDomain.ROS,
        inputs.reference_timestamps_ns,
    )

    sources = {
        policy.key: TimestampTimeline.from_timestamps(
            policy.key,
            ClockDomain.ROS,
            inputs.source_timestamps(policy.key),
        )
        for policy in SOURCE_POLICIES
    }

    specs = {
        policy.key: policy.sync_spec()
        for policy in SOURCE_POLICIES
    }

    try:
        plan = build_synchronization_plan(
            reference,
            sources,
            specs,
        )
    except SynchronizationError as exc:
        raise DoosanPolicyError(
            f"failed to build Doosan synchronization plan: {exc}"
        ) from exc

    return DoosanSynchronizationResult(
        inputs=inputs,
        plan=plan,
    )


def build_doosan_sync_plan(
    episode_dir: str | Path,
) -> DoosanSynchronizationResult:
    """Collect timestamp-only inputs and build the production Doosan plan."""

    return build_doosan_sync_plan_from_inputs(
        collect_doosan_policy_inputs(episode_dir)
    )


def policy_by_key(key: str) -> StreamPolicy:
    try:
        return _SOURCE_POLICY_BY_KEY[key]
    except KeyError as exc:
        raise DoosanPolicyError(f"unknown synchronization policy key {key!r}") from exc


def policy_by_topic(topic: str) -> StreamPolicy:
    try:
        return _SOURCE_POLICY_BY_TOPIC[topic]
    except KeyError as exc:
        raise DoosanPolicyError(f"topic is not synchronized by Patch 4: {topic!r}") from exc


__all__ = [
    "POLICY_ID",
    "REFERENCE_KEY",
    "REFERENCE_TOPIC",
    "REFERENCE_TIMESTAMP_SOURCE",
    "ROS_EPOCH_MAX_HEADER_BAG_OFFSET_NS",
    "SOURCE_POLICIES",
    "CameraCalibration",
    "DoosanPolicyError",
    "DoosanPolicyInputs",
    "DoosanSynchronizationResult",
    "HeaderBagOffsetEvidence",
    "StreamPolicy",
    "TimestampSource",
    "build_doosan_sync_plan",
    "build_doosan_sync_plan_from_inputs",
    "collect_doosan_policy_inputs",
    "policy_by_key",
    "policy_by_topic",
]
