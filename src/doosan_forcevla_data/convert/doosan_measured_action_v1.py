"""Patch-7 measured 7D actions for the Doosan ForceVLA v2 contract.

Actions are derived only from consecutive synchronized measured states produced
by Patch 5 (robot pose) and Patch 6 (binary absolute gripper semantic).  Joystick
and SpeedL streams remain auxiliary provenance and are never used as the
primary learning label.

For source state ``t`` and target state ``t+1``:

- translation: ``p[t+1] - p[t]`` in robot-base coordinates, metres
- rotation: ``Log(R[t+1] @ R[t].T)`` in robot-base/spatial coordinates, radians
- gripper: absolute Patch-6 target open fraction at ``t+1``

The terminal synchronized state has no measured successor, so Patch 7 emits no
synthetic terminal action.  A state sequence of length ``N`` therefore produces
exactly ``N-1`` measured action samples.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    DoosanForceProprioEpisode,
    DoosanForceProprioState,
    rotation_matrix_to_rotvec,
    rotation_vector_to_matrix,
)
from doosan_forcevla_data.convert.doosan_gripper_semantics_v1 import (
    DoosanGripperEpisode,
    HELD_OPEN_FRACTION,
    RELEASED_OPEN_FRACTION,
    require_binary_open_fraction,
    validate_release_only_episode_protocol,
)


SEMANTICS_ID = "doosan_measured_action_v1"
FORCEVLA_CONTRACT_ID = "doosan_forcevla_dataset_contract_v2"
ACTION_DIM = 7
ACTION_FIELDS = (
    "delta_tcp_x_base_m",
    "delta_tcp_y_base_m",
    "delta_tcp_z_base_m",
    "delta_rotvec_x_base_rad",
    "delta_rotvec_y_base_rad",
    "delta_rotvec_z_base_rad",
    "absolute_gripper_target_open_fraction",
)

if len(ACTION_FIELDS) != ACTION_DIM:  # pragma: no cover
    raise RuntimeError("Doosan Patch-7 action layout constant is inconsistent")


class MeasuredActionError(ValueError):
    """Raised when a measured action cannot be constructed unambiguously."""


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise MeasuredActionError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasuredActionError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise MeasuredActionError(f"{name} must be finite")
    return result


def _finite_vector(values: Sequence[float], expected_len: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise MeasuredActionError(f"{name} must be a list or tuple of length {expected_len}")
    if len(values) != expected_len:
        raise MeasuredActionError(
            f"{name} must have length {expected_len}, got {len(values)}"
        )
    return tuple(_finite_scalar(value, f"{name}[{index}]") for index, value in enumerate(values))


def _transpose3(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise MeasuredActionError("rotation matrix must be 3x3")
    return tuple(tuple(float(matrix[col][row]) for col in range(3)) for row in range(3))


def _matmul3(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    if len(left) != 3 or len(right) != 3:
        raise MeasuredActionError("rotation matrices must be 3x3")
    if any(len(row) != 3 for row in left) or any(len(row) != 3 for row in right):
        raise MeasuredActionError("rotation matrices must be 3x3")
    result = tuple(
        tuple(
            sum(float(left[row][k]) * float(right[k][col]) for k in range(3))
            for col in range(3)
        )
        for row in range(3)
    )
    if not all(math.isfinite(value) for row in result for value in row):
        raise MeasuredActionError("rotation matrix multiplication produced non-finite values")
    return result


def spatial_delta_rotvec(
    rotvec_t_rad: Sequence[float],
    rotvec_t1_rad: Sequence[float],
) -> tuple[float, float, float]:
    """Return ``Log(R[t+1] @ R[t].T)`` in base/spatial coordinates.

    Patch 5 stores absolute base-to-TCP rotation vectors.  The ForceVLA v2
    action contract requires the left/spatial relative transform, not the older
    body/local transform ``R[t].T @ R[t+1]``.
    """

    current = _finite_vector(rotvec_t_rad, 3, "rotvec_t_rad")
    target = _finite_vector(rotvec_t1_rad, 3, "rotvec_t1_rad")
    r_t = rotation_vector_to_matrix(current)
    r_t1 = rotation_vector_to_matrix(target)
    delta_r = _matmul3(r_t1, _transpose3(r_t))
    result = rotation_matrix_to_rotvec(delta_r)
    if not all(math.isfinite(value) for value in result):  # pragma: no cover
        raise MeasuredActionError("spatial rotation delta contains non-finite values")
    return result


@dataclass(frozen=True)
class DoosanMeasuredAction:
    """One measured transition from synchronized reference ``t`` to ``t+1``."""

    source_reference_index: int
    target_reference_index: int
    source_reference_timestamp_ns: int
    target_reference_timestamp_ns: int
    delta_translation_base_m: tuple[float, float, float]
    delta_rotvec_base_rad: tuple[float, float, float]
    gripper_target_open_fraction: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_reference_index, bool)
            or not isinstance(self.source_reference_index, int)
            or self.source_reference_index < 0
        ):
            raise MeasuredActionError("source_reference_index must be a non-negative integer")
        if (
            isinstance(self.target_reference_index, bool)
            or not isinstance(self.target_reference_index, int)
            or self.target_reference_index < 0
        ):
            raise MeasuredActionError("target_reference_index must be a non-negative integer")
        if self.target_reference_index != self.source_reference_index + 1:
            raise MeasuredActionError(
                "Patch-7 action must connect adjacent reference frames; "
                f"got {self.source_reference_index}->{self.target_reference_index}"
            )
        if (
            isinstance(self.source_reference_timestamp_ns, bool)
            or not isinstance(self.source_reference_timestamp_ns, int)
            or isinstance(self.target_reference_timestamp_ns, bool)
            or not isinstance(self.target_reference_timestamp_ns, int)
            or self.source_reference_timestamp_ns < 0
            or self.target_reference_timestamp_ns < 0
        ):
            raise MeasuredActionError("reference timestamps must be non-negative integers")
        if self.target_reference_timestamp_ns <= self.source_reference_timestamp_ns:
            raise MeasuredActionError("target reference timestamp must be later than source")
        _finite_vector(self.delta_translation_base_m, 3, "delta_translation_base_m")
        _finite_vector(self.delta_rotvec_base_rad, 3, "delta_rotvec_base_rad")
        require_binary_open_fraction(
            self.gripper_target_open_fraction,
            "gripper_target_open_fraction",
        )

    @property
    def delta_time_ns(self) -> int:
        return self.target_reference_timestamp_ns - self.source_reference_timestamp_ns

    def to_vector(self) -> tuple[float, ...]:
        vector = (
            *self.delta_translation_base_m,
            *self.delta_rotvec_base_rad,
            self.gripper_target_open_fraction,
        )
        if len(vector) != ACTION_DIM:  # pragma: no cover
            raise MeasuredActionError(f"action must have {ACTION_DIM} channels")
        if not all(math.isfinite(value) for value in vector):  # pragma: no cover
            raise MeasuredActionError("action contains non-finite values")
        return vector

    def to_dict(self) -> dict[str, object]:
        return {
            "source_reference_index": self.source_reference_index,
            "target_reference_index": self.target_reference_index,
            "source_reference_timestamp_ns": self.source_reference_timestamp_ns,
            "target_reference_timestamp_ns": self.target_reference_timestamp_ns,
            "delta_time_ns": self.delta_time_ns,
            "delta_translation_base_m": list(self.delta_translation_base_m),
            "delta_rotvec_base_rad": list(self.delta_rotvec_base_rad),
            "gripper_target_open_fraction": self.gripper_target_open_fraction,
            "action_7d": list(self.to_vector()),
        }


@dataclass(frozen=True)
class DoosanMeasuredActionEpisode:
    """Patch-7 action sequence with no fabricated terminal action."""

    state_count: int
    actions: tuple[DoosanMeasuredAction, ...]

    def __post_init__(self) -> None:
        if isinstance(self.state_count, bool) or self.state_count < 2:
            raise MeasuredActionError("state_count must be an integer >= 2")
        if len(self.actions) != self.state_count - 1:
            raise MeasuredActionError(
                "Patch-7 requires exactly N-1 measured actions for N synchronized states"
            )
        for expected_source_index, action in enumerate(self.actions):
            if action.source_reference_index != expected_source_index:
                raise MeasuredActionError(
                    "Patch-7 action source references must be exactly 0..N-2; "
                    "dropped/non-adjacent reference frames require explicit later handling"
                )
            if action.target_reference_index != expected_source_index + 1:
                raise MeasuredActionError("Patch-7 target reference sequence is inconsistent")

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def terminal_reference_index(self) -> int:
        return self.state_count - 1

    @property
    def gripper_target_counts(self) -> dict[str, int]:
        held = sum(
            action.gripper_target_open_fraction == HELD_OPEN_FRACTION
            for action in self.actions
        )
        released = self.action_count - held
        return {
            "held_or_closed_target": held,
            "released_or_open_target": released,
        }

    @property
    def release_action_source_indices(self) -> tuple[int, ...]:
        releases: list[int] = []
        previous_target = HELD_OPEN_FRACTION
        for action in self.actions:
            target = action.gripper_target_open_fraction
            if previous_target == HELD_OPEN_FRACTION and target == RELEASED_OPEN_FRACTION:
                releases.append(action.source_reference_index)
            previous_target = target
        return tuple(releases)

    def summary_dict(self) -> dict[str, object]:
        dt_values = [action.delta_time_ns for action in self.actions]
        return {
            "semantics_id": SEMANTICS_ID,
            "forcevla_contract_id": FORCEVLA_CONTRACT_ID,
            "state_count": self.state_count,
            "action_count": self.action_count,
            "action_dim": ACTION_DIM,
            "action_fields": list(ACTION_FIELDS),
            "translation_semantics": "p[t+1] - p[t], robot base coordinates, metres",
            "rotation_semantics": "Log(R[t+1] @ R[t].T), robot base/spatial coordinates, radians",
            "gripper_semantics": "absolute Patch-6 binary open target at t+1; never differenced",
            "primary_action_source": "consecutive synchronized measured state",
            "speedl_primary_action": False,
            "joy_primary_action": False,
            "terminal_policy": {
                "terminal_reference_index": self.terminal_reference_index,
                "terminal_action_emitted": False,
                "synthetic_terminal_zero_action": False,
                "training_row_policy": (
                    "Patch 8 must exclude the terminal observation or carry an explicit "
                    "invalid-action mask; Patch 7 never fabricates an action"
                ),
            },
            "gripper_target_counts": self.gripper_target_counts,
            "release_action_source_indices": list(self.release_action_source_indices),
            "delta_time_ns": {
                "minimum": min(dt_values),
                "maximum": max(dt_values),
                "mean": sum(dt_values) / len(dt_values),
            },
        }


def compute_measured_action(
    state_t: DoosanForceProprioState,
    state_t1: DoosanForceProprioState,
    gripper_target_open_fraction: float,
) -> tuple[float, ...]:
    """Compute the exact ForceVLA-v2 7D semantic action for one transition."""

    if not isinstance(state_t, DoosanForceProprioState):
        raise MeasuredActionError(
            f"state_t must be DoosanForceProprioState, got {type(state_t).__name__}"
        )
    if not isinstance(state_t1, DoosanForceProprioState):
        raise MeasuredActionError(
            f"state_t1 must be DoosanForceProprioState, got {type(state_t1).__name__}"
        )

    p_t = _finite_vector(state_t.tcp_position_m, 3, "state_t.tcp_position_m")
    p_t1 = _finite_vector(state_t1.tcp_position_m, 3, "state_t1.tcp_position_m")
    delta_translation = tuple(p_t1[index] - p_t[index] for index in range(3))
    delta_rotvec = spatial_delta_rotvec(
        state_t.tcp_rotvec_rad,
        state_t1.tcp_rotvec_rad,
    )
    gripper_target = require_binary_open_fraction(
        gripper_target_open_fraction,
        "gripper_target_open_fraction",
    )
    action = (*delta_translation, *delta_rotvec, gripper_target)
    if len(action) != ACTION_DIM or not all(math.isfinite(value) for value in action):
        raise MeasuredActionError("computed measured action must contain 7 finite values")
    return action


def build_doosan_measured_action_episode(
    force_proprio_episode: DoosanForceProprioEpisode,
    gripper_episode: DoosanGripperEpisode,
) -> DoosanMeasuredActionEpisode:
    """Build Patch-7 actions from exact Patch-5/Patch-6 synchronized samples.

    The inputs must refer to the same complete Patch-4 reference timeline.  Patch
    7 intentionally refuses to bridge a dropped reference index because doing so
    would silently change the 30 Hz action step.  Such episodes require explicit
    segmentation/row policy in a later patch rather than an implicit larger-step
    action.
    """

    if not isinstance(force_proprio_episode, DoosanForceProprioEpisode):
        raise MeasuredActionError(
            "force_proprio_episode must be a DoosanForceProprioEpisode"
        )
    if not isinstance(gripper_episode, DoosanGripperEpisode):
        raise MeasuredActionError("gripper_episode must be a DoosanGripperEpisode")
    if len(force_proprio_episode.samples) != len(gripper_episode.samples):
        raise MeasuredActionError(
            "Patch-5 and Patch-6 sample counts differ; refusing positional alignment"
        )
    if len(force_proprio_episode.samples) < 2:
        raise MeasuredActionError("at least two synchronized states are required")

    # Patch 6 already enforces the release-only episode protocol.  Re-run the
    # public validator at the action boundary so Patch 7 cannot be called with a
    # manually-constructed invalid gripper episode.
    validate_release_only_episode_protocol(gripper_episode)

    robot_samples = force_proprio_episode.samples
    gripper_samples = gripper_episode.samples

    for expected_index, (robot_sample, gripper_sample) in enumerate(
        zip(robot_samples, gripper_samples, strict=True)
    ):
        if robot_sample.reference_index != gripper_sample.reference_index:
            raise MeasuredActionError(
                "Patch-5/Patch-6 reference index mismatch at action boundary"
            )
        if robot_sample.reference_timestamp_ns != gripper_sample.reference_timestamp_ns:
            raise MeasuredActionError(
                "Patch-5/Patch-6 reference timestamp mismatch at action boundary"
            )
        if robot_sample.reference_index != expected_index:
            raise MeasuredActionError(
                "Patch-7 requires contiguous complete reference indices 0..N-1; "
                f"expected {expected_index}, got {robot_sample.reference_index}"
            )

    actions: list[DoosanMeasuredAction] = []
    for source_index in range(len(robot_samples) - 1):
        robot_t = robot_samples[source_index]
        robot_t1 = robot_samples[source_index + 1]
        gripper_t1 = gripper_samples[source_index + 1]

        if robot_t1.reference_index != robot_t.reference_index + 1:
            raise MeasuredActionError(
                "Patch-7 refuses to bridge non-adjacent synchronized reference frames"
            )
        if robot_t1.reference_timestamp_ns <= robot_t.reference_timestamp_ns:
            raise MeasuredActionError("reference timestamps must be strictly increasing")

        vector = compute_measured_action(
            robot_t.state,
            robot_t1.state,
            gripper_t1.state.open_fraction,
        )
        actions.append(
            DoosanMeasuredAction(
                source_reference_index=robot_t.reference_index,
                target_reference_index=robot_t1.reference_index,
                source_reference_timestamp_ns=robot_t.reference_timestamp_ns,
                target_reference_timestamp_ns=robot_t1.reference_timestamp_ns,
                delta_translation_base_m=(vector[0], vector[1], vector[2]),
                delta_rotvec_base_rad=(vector[3], vector[4], vector[5]),
                gripper_target_open_fraction=vector[6],
            )
        )

    return DoosanMeasuredActionEpisode(
        state_count=len(robot_samples),
        actions=tuple(actions),
    )


def reconstruct_target_pose(
    position_t_m: Sequence[float],
    rotvec_t_rad: Sequence[float],
    action: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Reconstruct the measured target pose from one Patch-7 action.

    This is primarily a validation helper for the Patch-7 acceptance criterion.
    Spatial rotation actions reconstruct as ``R[t+1] = Exp(delta) @ R[t]``.
    """

    position = _finite_vector(position_t_m, 3, "position_t_m")
    rotation = _finite_vector(rotvec_t_rad, 3, "rotvec_t_rad")
    vector = _finite_vector(action, ACTION_DIM, "action")
    target_position = tuple(position[index] + vector[index] for index in range(3))

    r_t = rotation_vector_to_matrix(rotation)
    delta_r = rotation_vector_to_matrix(vector[3:6])
    target_r = _matmul3(delta_r, r_t)
    target_rotvec = rotation_matrix_to_rotvec(target_r)
    return target_position, target_rotvec


__all__ = [
    "ACTION_DIM",
    "ACTION_FIELDS",
    "FORCEVLA_CONTRACT_ID",
    "SEMANTICS_ID",
    "DoosanMeasuredAction",
    "DoosanMeasuredActionEpisode",
    "MeasuredActionError",
    "build_doosan_measured_action_episode",
    "compute_measured_action",
    "reconstruct_target_pose",
    "spatial_delta_rotvec",
]
