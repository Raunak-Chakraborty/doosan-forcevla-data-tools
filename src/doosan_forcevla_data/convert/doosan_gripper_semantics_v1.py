"""Patch-6 SCHUNK held/released semantics for the thesis release-only protocol.

The physical episode protocol starts with the peg already held.  During the
recorded episode the operator may issue one release using the SpaceMouse RIGHT
button; no grasp/close operation is learned inside the episode.

Patch 6 therefore does *not* normalize the raw SCHUNK finger position.  The
model-facing scalar is derived only from ``GripperState.holding`` while the raw
position is retained as diagnostic provenance:

- holding=True  -> open fraction 0.0 (held / closed semantic endpoint)
- holding=False -> open fraction 1.0 (released / open semantic endpoint)

This preserves the frozen ForceVLA-v2 convention that state/action channel 6 is
an absolute gripper open fraction with 0=closed and 1=open, but restricts the
thesis dataset to the two endpoints.  Contact/load information remains the
six-dimensional RobotStateRt external TCP wrench owned by Patch 5; SCHUNK
``GripperState`` does not contain a measured force value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any, Sequence

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    DoosanForceProprioEpisode,
    OBSERVATION_STATE_DIM,
)
from doosan_forcevla_data.ingest.doosan_raw_v1 import (
    GRIPPER_STATE_TOPIC,
    GripperStateRecord,
    iter_typed_messages,
)
from doosan_forcevla_data.sync.doosan_policy_v1 import (
    DoosanSynchronizationResult,
    build_doosan_sync_plan,
)
from doosan_forcevla_data.sync.timestamp_plan import SyncMethod


SEMANTICS_ID = "doosan_gripper_semantics_v1"
FORCEVLA_CONTRACT_ID = "doosan_forcevla_dataset_contract_v2"
GRIPPER_SYNC_KEY = "gripper_state"
GRIPPER_MAX_AGE_NS = 15_000_000

HELD_OPEN_FRACTION = 0.0
RELEASED_OPEN_FRACTION = 1.0


class GripperSemanticsError(ValueError):
    """Raised when SCHUNK state/action semantics are ambiguous or inconsistent."""


class GripperExecutionIntent(str, Enum):
    """Release-only execution meaning of an absolute binary gripper target."""

    HOLD_CURRENT = "hold_current"
    RELEASE = "release"
    REMAIN_RELEASED = "remain_released"


@dataclass(frozen=True)
class DoosanGripperState:
    """One selected SCHUNK state with binary model semantics and raw provenance."""

    source_bag_timestamp_ns: int
    source_header_timestamp_ns: int
    raw_position_m: float
    holding: bool
    open_fraction: float

    def __post_init__(self) -> None:
        if self.source_bag_timestamp_ns < 0:
            raise GripperSemanticsError("source_bag_timestamp_ns must be non-negative")
        if self.source_header_timestamp_ns < 0:
            raise GripperSemanticsError("source_header_timestamp_ns must be non-negative")
        _require_finite_scalar(self.raw_position_m, "raw_position_m")
        if not isinstance(self.holding, bool):
            raise GripperSemanticsError("holding must be bool")
        expected = holding_to_open_fraction(self.holding)
        actual = require_binary_open_fraction(self.open_fraction, "open_fraction")
        if actual != expected:
            raise GripperSemanticsError(
                "open_fraction is inconsistent with holding; "
                f"holding={self.holding!r} requires {expected}, got {actual}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_bag_timestamp_ns": self.source_bag_timestamp_ns,
            "source_header_timestamp_ns": self.source_header_timestamp_ns,
            "raw_position_m_diagnostic_only": self.raw_position_m,
            "holding": self.holding,
            "open_fraction": self.open_fraction,
        }


@dataclass(frozen=True)
class SynchronizedGripperSample:
    """One Patch-4 reference frame and its exact nearest SCHUNK selection."""

    reference_index: int
    reference_timestamp_ns: int
    gripper_source_index: int
    gripper_source_timestamp_ns: int
    gripper_signed_skew_ns: int
    state: DoosanGripperState

    def __post_init__(self) -> None:
        if self.reference_index < 0:
            raise GripperSemanticsError("reference_index must be non-negative")
        if self.reference_timestamp_ns < 0:
            raise GripperSemanticsError("reference_timestamp_ns must be non-negative")
        if self.gripper_source_index < 0:
            raise GripperSemanticsError("gripper_source_index must be non-negative")
        if self.gripper_source_timestamp_ns != self.state.source_header_timestamp_ns:
            raise GripperSemanticsError(
                "selected gripper timestamp must equal the decoded state header timestamp"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_index": self.reference_index,
            "reference_timestamp_ns": self.reference_timestamp_ns,
            "gripper_source_index": self.gripper_source_index,
            "gripper_source_timestamp_ns": self.gripper_source_timestamp_ns,
            "gripper_signed_skew_ns": self.gripper_signed_skew_ns,
            "state": self.state.to_dict(),
        }


@dataclass(frozen=True)
class DoosanGripperEpisode:
    """Patch-6 binary gripper semantics for all complete Patch-4 references."""

    raw_gripper_state_count: int
    complete_reference_count: int
    dropped_reference_count: int
    samples: tuple[SynchronizedGripperSample, ...]

    def __post_init__(self) -> None:
        if self.raw_gripper_state_count < 0:
            raise GripperSemanticsError("raw_gripper_state_count must be non-negative")
        if self.complete_reference_count != len(self.samples):
            raise GripperSemanticsError(
                "complete_reference_count must equal the number of gripper samples"
            )
        if self.raw_gripper_state_count < len(self.samples):
            raise GripperSemanticsError(
                "raw_gripper_state_count cannot be smaller than selected sample count"
            )
        previous_reference_index: int | None = None
        for sample in self.samples:
            if previous_reference_index is not None and sample.reference_index <= previous_reference_index:
                raise GripperSemanticsError("gripper reference indices must be strictly increasing")
            previous_reference_index = sample.reference_index

    @property
    def held_sample_count(self) -> int:
        return sum(sample.state.holding for sample in self.samples)

    @property
    def released_sample_count(self) -> int:
        return len(self.samples) - self.held_sample_count

    @property
    def transition_indices(self) -> tuple[int, ...]:
        """Reference indices where the selected binary gripper state changes."""

        transitions: list[int] = []
        for previous, current in zip(self.samples, self.samples[1:], strict=False):
            if previous.state.holding != current.state.holding:
                transitions.append(current.reference_index)
        return tuple(transitions)

    def summary_dict(self) -> dict[str, object]:
        return {
            "semantics_id": SEMANTICS_ID,
            "forcevla_contract_id": FORCEVLA_CONTRACT_ID,
            "source_topic": GRIPPER_STATE_TOPIC,
            "raw_gripper_state_count": self.raw_gripper_state_count,
            "complete_reference_count": self.complete_reference_count,
            "dropped_reference_count": self.dropped_reference_count,
            "sample_count": len(self.samples),
            "held_sample_count": self.held_sample_count,
            "released_sample_count": self.released_sample_count,
            "transition_indices": list(self.transition_indices),
            "state_semantics": {
                "source": "GripperState.holding",
                "held_or_closed_open_fraction": HELD_OPEN_FRACTION,
                "released_or_open_open_fraction": RELEASED_OPEN_FRACTION,
                "raw_position_role": "diagnostic_only",
                "per_episode_position_normalization": False,
                "physical_endpoint_calibration_required": False,
            },
            "action_semantics": {
                "channel": 6,
                "representation": "absolute binary gripper open fraction",
                "hold_current": HELD_OPEN_FRACTION,
                "release": RELEASED_OPEN_FRACTION,
                "delta_action": False,
                "regrasp_inside_episode_supported": False,
            },
            "force_semantics": {
                "schunk_state_contains_measured_force": False,
                "contact_load_source": (
                    "Patch-5 RobotStateRt.external_tcp_force base-coordinate 6D wrench"
                ),
            },
        }


def _require_finite_scalar(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise GripperSemanticsError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GripperSemanticsError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise GripperSemanticsError(f"{context} must be finite")
    return result


def holding_to_open_fraction(holding: bool) -> float:
    """Map SCHUNK workpiece-held status to the frozen 0-closed/1-open endpoints."""

    if not isinstance(holding, bool):
        raise GripperSemanticsError("holding must be bool")
    return HELD_OPEN_FRACTION if holding else RELEASED_OPEN_FRACTION


def require_binary_open_fraction(value: Any, context: str = "gripper_open_fraction") -> float:
    """Require one of the two thesis semantic endpoints exactly."""

    result = _require_finite_scalar(value, context)
    if result not in (HELD_OPEN_FRACTION, RELEASED_OPEN_FRACTION):
        raise GripperSemanticsError(
            f"{context} must be exactly {HELD_OPEN_FRACTION} (held/closed) or "
            f"{RELEASED_OPEN_FRACTION} (released/open), got {result}"
        )
    return result


def release_only_execution_intent(
    current_open_fraction: Any,
    target_open_fraction: Any,
) -> GripperExecutionIntent:
    """Interpret one absolute binary target for the release-only thesis protocol.

    No command is needed while holding is maintained or after release has already
    occurred.  A held->released transition maps to the existing parameterless
    ``/schunk/release`` action.  Released->held is rejected because the physical
    thesis protocol starts pre-grasped and contains no autonomous regrasp step.
    """

    current = require_binary_open_fraction(current_open_fraction, "current_open_fraction")
    target = require_binary_open_fraction(target_open_fraction, "target_open_fraction")

    if current == HELD_OPEN_FRACTION and target == HELD_OPEN_FRACTION:
        return GripperExecutionIntent.HOLD_CURRENT
    if current == HELD_OPEN_FRACTION and target == RELEASED_OPEN_FRACTION:
        return GripperExecutionIntent.RELEASE
    if current == RELEASED_OPEN_FRACTION and target == RELEASED_OPEN_FRACTION:
        return GripperExecutionIntent.REMAIN_RELEASED

    raise GripperSemanticsError(
        "released->held would require a regrasp operation that is outside the "
        "release-only thesis episode contract"
    )


def convert_gripper_state(record: GripperStateRecord) -> DoosanGripperState:
    """Convert one decoded SCHUNK state without using finger position semantically."""

    if not isinstance(record, GripperStateRecord):
        raise GripperSemanticsError(
            f"expected GripperStateRecord, got {type(record).__name__}"
        )
    header_timestamp_ns = record.stamp.header_timestamp_ns
    if header_timestamp_ns is None:
        raise GripperSemanticsError("GripperStateRecord requires a ROS header timestamp")

    return DoosanGripperState(
        source_bag_timestamp_ns=int(record.stamp.bag_timestamp_ns),
        source_header_timestamp_ns=int(header_timestamp_ns),
        raw_position_m=_require_finite_scalar(record.position_m, "GripperState.position"),
        holding=record.holding,
        open_fraction=holding_to_open_fraction(record.holding),
    )


def build_synchronized_gripper_samples(
    gripper_records: Sequence[GripperStateRecord],
    sync_result: DoosanSynchronizationResult,
) -> tuple[SynchronizedGripperSample, ...]:
    """Apply Patch-6 semantics to the exact Patch-4 nearest gripper selections."""

    source_plan = sync_result.plan.source_plan(GRIPPER_SYNC_KEY)
    if source_plan.spec.method is not SyncMethod.NEAREST:
        raise GripperSemanticsError(
            "Patch-6 requires the frozen Patch-4 nearest gripper association"
        )
    if not source_plan.spec.required:
        raise GripperSemanticsError("Patch-6 requires gripper_state to remain required")
    if source_plan.spec.max_age_ns != GRIPPER_MAX_AGE_NS:
        raise GripperSemanticsError(
            "Patch-6 requires the frozen Patch-4 15 ms gripper freshness bound"
        )

    complete_reference_indices = set(sync_result.plan.complete_reference_indices)
    samples: list[SynchronizedGripperSample] = []

    for decision in source_plan.decisions:
        if decision.reference_index not in complete_reference_indices:
            continue
        selection = decision.selection
        if selection is None:
            raise GripperSemanticsError(
                "complete Patch-4 reference unexpectedly lacks a gripper selection"
            )
        if len(selection.source_indices) != 1:
            raise GripperSemanticsError(
                "nearest gripper selection must contain exactly one source index"
            )
        if len(selection.source_timestamps_ns) != 1 or len(selection.signed_skews_ns) != 1:
            raise GripperSemanticsError("nearest gripper selection provenance is malformed")
        if selection.alpha is not None:
            raise GripperSemanticsError("nearest gripper selection must not be interpolated")

        source_index = selection.source_indices[0]
        if source_index < 0 or source_index >= len(gripper_records):
            raise GripperSemanticsError(
                f"Patch-4 gripper source index out of range: {source_index}"
            )
        record = gripper_records[source_index]
        state = convert_gripper_state(record)
        selected_timestamp = selection.source_timestamps_ns[0]
        if state.source_header_timestamp_ns != selected_timestamp:
            raise GripperSemanticsError(
                "Patch-4 selected gripper timestamp does not match the decoded source record"
            )

        samples.append(
            SynchronizedGripperSample(
                reference_index=decision.reference_index,
                reference_timestamp_ns=decision.reference_timestamp_ns,
                gripper_source_index=source_index,
                gripper_source_timestamp_ns=selected_timestamp,
                gripper_signed_skew_ns=selection.signed_skews_ns[0],
                state=state,
            )
        )

    if len(samples) != len(sync_result.plan.complete_reference_indices):
        raise GripperSemanticsError(
            "selected gripper sample count does not match complete Patch-4 references"
        )
    return tuple(samples)


def _validate_release_only_holding_sequence(
    holding_values: Sequence[bool],
    context: str,
) -> None:
    """Require the thesis protocol: starts held, one release, ends released."""

    if not holding_values:
        raise GripperSemanticsError(f"{context}: holding sequence must not be empty")
    if any(not isinstance(value, bool) for value in holding_values):
        raise GripperSemanticsError(f"{context}: holding sequence must contain only bool values")
    if holding_values[0] is not True:
        raise GripperSemanticsError(f"{context}: episode must start with the peg held")
    if holding_values[-1] is not False:
        raise GripperSemanticsError(f"{context}: episode must end after the release")

    transitions = [
        index
        for index in range(1, len(holding_values))
        if holding_values[index] != holding_values[index - 1]
    ]
    if len(transitions) != 1:
        raise GripperSemanticsError(
            f"{context}: release-only episode requires exactly one holding transition, "
            f"got {len(transitions)}"
        )
    transition = transitions[0]
    if holding_values[transition - 1] is not True or holding_values[transition] is not False:
        raise GripperSemanticsError(
            f"{context}: the only allowed transition is held->released"
        )


def validate_release_only_episode_protocol(episode: DoosanGripperEpisode) -> None:
    """Validate the synchronized samples against the frozen thesis episode protocol."""

    _validate_release_only_holding_sequence(
        tuple(sample.state.holding for sample in episode.samples),
        "synchronized gripper state",
    )


def build_doosan_gripper_episode(episode_dir: str | Path) -> DoosanGripperEpisode:
    """Build and validate Patch-6 semantics from the immutable raw episode."""

    sync_result = build_doosan_sync_plan(episode_dir)
    gripper_records: list[GripperStateRecord] = []
    for topic, record in iter_typed_messages(episode_dir):
        if topic != GRIPPER_STATE_TOPIC:
            continue
        if not isinstance(record, GripperStateRecord):  # pragma: no cover - decoder contract
            raise GripperSemanticsError(
                f"{GRIPPER_STATE_TOPIC}: expected GripperStateRecord, got {type(record).__name__}"
            )
        gripper_records.append(record)

    _validate_release_only_holding_sequence(
        tuple(record.holding for record in gripper_records),
        "raw gripper state",
    )

    samples = build_synchronized_gripper_samples(gripper_records, sync_result)
    episode = DoosanGripperEpisode(
        raw_gripper_state_count=len(gripper_records),
        complete_reference_count=len(sync_result.plan.complete_reference_indices),
        dropped_reference_count=len(sync_result.plan.dropped_reference_indices),
        samples=samples,
    )
    validate_release_only_episode_protocol(episode)
    return episode


def assemble_forcevla_v2_observation_states(
    force_proprio_episode: DoosanForceProprioEpisode,
    gripper_episode: DoosanGripperEpisode,
) -> tuple[tuple[float, ...], ...]:
    """Join Patch 5 and Patch 6 by exact Patch-4 reference identity."""

    if len(force_proprio_episode.samples) != len(gripper_episode.samples):
        raise GripperSemanticsError(
            "force/proprio and gripper sample counts differ; refusing positional zip"
        )

    observations: list[tuple[float, ...]] = []
    for robot_sample, gripper_sample in zip(
        force_proprio_episode.samples,
        gripper_episode.samples,
        strict=True,
    ):
        if robot_sample.reference_index != gripper_sample.reference_index:
            raise GripperSemanticsError(
                "force/proprio and gripper reference indices differ"
            )
        if robot_sample.reference_timestamp_ns != gripper_sample.reference_timestamp_ns:
            raise GripperSemanticsError(
                "force/proprio and gripper reference timestamps differ"
            )
        gripper = require_binary_open_fraction(gripper_sample.state.open_fraction)
        state = robot_sample.state.to_observation_state(gripper)
        if len(state) != OBSERVATION_STATE_DIM:  # pragma: no cover - Patch-5 invariant
            raise GripperSemanticsError(
                f"assembled observation must have {OBSERVATION_STATE_DIM} values"
            )
        observations.append(state)

    return tuple(observations)


__all__ = [
    "FORCEVLA_CONTRACT_ID",
    "GRIPPER_MAX_AGE_NS",
    "GRIPPER_SYNC_KEY",
    "HELD_OPEN_FRACTION",
    "RELEASED_OPEN_FRACTION",
    "SEMANTICS_ID",
    "DoosanGripperEpisode",
    "DoosanGripperState",
    "GripperExecutionIntent",
    "GripperSemanticsError",
    "SynchronizedGripperSample",
    "assemble_forcevla_v2_observation_states",
    "build_doosan_gripper_episode",
    "build_synchronized_gripper_samples",
    "convert_gripper_state",
    "holding_to_open_fraction",
    "release_only_execution_intent",
    "require_binary_open_fraction",
    "validate_release_only_episode_protocol",
]
