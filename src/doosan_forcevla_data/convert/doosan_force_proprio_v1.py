"""Patch-5 force and proprioception semantics for the Doosan ForceVLA contract.

This module consumes the *selected* controller-native ``RobotStateRtRecord``
chosen by the frozen Patch-4 synchronization plan.  It performs only physical
semantic conversion:

- TCP translation: mm -> m
- TCP orientation: Doosan intrinsic Euler ZYZ degrees -> base-to-TCP SO(3)
  rotation vector in radians
- joint position: degrees -> radians
- joint velocity: degrees/s -> radians/s
- external TCP wrench: controller-native base-coordinate N / N m passthrough
  for explicitly reset-compensated modern episodes

No stream is re-synchronized here and no raw Euler component is interpolated.
The 25D observation vector is assembled only after a caller supplies the
normalized gripper opening defined by Patch 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from doosan_forcevla_data.ingest.doosan_raw_v1 import (
    ROBOT_STATE_RT_TOPIC,
    RobotStateRtRecord,
    iter_typed_messages,
)
from doosan_forcevla_data.sync.doosan_policy_v1 import (
    DoosanSynchronizationResult,
    build_doosan_sync_plan,
)
from doosan_forcevla_data.sync.timestamp_plan import SyncMethod


SEMANTICS_ID = "doosan_force_proprio_semantics_v1"
FORCEVLA_CONTRACT_ID = "doosan_forcevla_dataset_contract_v2"
FORCE_PROVENANCE_SCHEMA = "doosan_episode_operator_v3_force_reset"
EXPECTED_FORCE_SIGNAL = (
    "RobotStateRt.external_tcp_force with Doosan set_external_force_reset() active"
)
EXPECTED_FORCE_PROCESSING_OWNER = "doosan-forcevla-data-tools"

OBSERVATION_STATE_DIM = 25
ROBOT_SEMANTIC_DIM = 24
TCP_POSITION_DIM = 3
TCP_ROTATION_DIM = 3
JOINT_DIM = 6
WRENCH_DIM = 6

OBSERVATION_STATE_FIELDS = (
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "tcp_rotvec_x_rad",
    "tcp_rotvec_y_rad",
    "tcp_rotvec_z_rad",
    "gripper_open_fraction",
    "joint_1_position_rad",
    "joint_2_position_rad",
    "joint_3_position_rad",
    "joint_4_position_rad",
    "joint_5_position_rad",
    "joint_6_position_rad",
    "joint_1_velocity_rad_s",
    "joint_2_velocity_rad_s",
    "joint_3_velocity_rad_s",
    "joint_4_velocity_rad_s",
    "joint_5_velocity_rad_s",
    "joint_6_velocity_rad_s",
    "force_x_n",
    "force_y_n",
    "force_z_n",
    "torque_x_nm",
    "torque_y_nm",
    "torque_z_nm",
)

if len(OBSERVATION_STATE_FIELDS) != OBSERVATION_STATE_DIM:  # pragma: no cover
    raise RuntimeError("Doosan ForceVLA v2 state layout constant is inconsistent")


class ForceProprioError(ValueError):
    """Raised when force/proprio semantics are ambiguous or internally inconsistent."""


class ForceCompensationPolicy(str, Enum):
    """Episode-level policy for the raw ``external_tcp_force`` signal."""

    RESET_COMPENSATED_PASSTHROUGH = "reset_compensated_passthrough"
    LEGACY_EPISODE_REQUIRES_KNOWN_TARE_POLICY = (
        "legacy_episode_requires_known_tare_policy"
    )


@dataclass(frozen=True)
class ForceCompensationProvenance:
    """Resolved, fail-closed provenance for one episode's external TCP wrench."""

    policy: ForceCompensationPolicy
    schema_version: str | None
    source_path: str
    controller_external_force_reset_active_at_record_start: bool | None
    controller_external_force_reset_completed_before_recording: bool | None
    force_guard_tare_applied_to_mcap: bool | None
    offline_force_tare_performed: bool | None
    pre_reset_ft_recorded: bool | None
    recording_controller_reset_compensated: bool | None
    recording_force_signal_in_mcap: str | None
    offline_force_processing_owner: str | None
    approved_for_training: bool
    reason: str

    @property
    def offline_second_tare_allowed(self) -> bool:
        """Patch 5 never permits an implicit downstream second tare."""

        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.value,
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "controller_external_force_reset_active_at_record_start": (
                self.controller_external_force_reset_active_at_record_start
            ),
            "controller_external_force_reset_completed_before_recording": (
                self.controller_external_force_reset_completed_before_recording
            ),
            "force_guard_tare_applied_to_mcap": self.force_guard_tare_applied_to_mcap,
            "offline_force_tare_performed": self.offline_force_tare_performed,
            "pre_reset_ft_recorded": self.pre_reset_ft_recorded,
            "recording_controller_reset_compensated": (
                self.recording_controller_reset_compensated
            ),
            "recording_force_signal_in_mcap": self.recording_force_signal_in_mcap,
            "offline_force_processing_owner": self.offline_force_processing_owner,
            "approved_for_training": self.approved_for_training,
            "offline_second_tare_allowed": self.offline_second_tare_allowed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DoosanForceProprioState:
    """Physical robot state derived from exactly one ``RobotStateRtRecord``."""

    source_bag_timestamp_ns: int
    controller_timestamp_s: float
    tcp_position_m: tuple[float, float, float]
    tcp_rotvec_rad: tuple[float, float, float]
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    wrench_base_n_nm: tuple[float, ...]
    force_policy: ForceCompensationPolicy

    def __post_init__(self) -> None:
        if self.source_bag_timestamp_ns < 0:
            raise ForceProprioError("source_bag_timestamp_ns must be non-negative")
        _require_finite_scalar(self.controller_timestamp_s, "controller_timestamp_s")
        _require_finite_vector(self.tcp_position_m, 3, "tcp_position_m")
        _require_finite_vector(self.tcp_rotvec_rad, 3, "tcp_rotvec_rad")
        _require_finite_vector(self.joint_position_rad, 6, "joint_position_rad")
        _require_finite_vector(self.joint_velocity_rad_s, 6, "joint_velocity_rad_s")
        _require_finite_vector(self.wrench_base_n_nm, 6, "wrench_base_n_nm")
        if self.force_policy is not ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH:
            raise ForceProprioError(
                "DoosanForceProprioState is training-ready only for the explicit "
                "reset-compensated passthrough policy"
            )

    def to_observation_state(self, gripper_open_fraction: float) -> tuple[float, ...]:
        """Assemble the exact ForceVLA-v2 25D state after Patch-6 normalization.

        Patch 5 deliberately does not derive the gripper value.  The caller must
        provide an already normalized measured opening with 0=closed and 1=open.
        """

        gripper = _require_finite_scalar(
            gripper_open_fraction,
            "gripper_open_fraction",
        )
        if not 0.0 <= gripper <= 1.0:
            raise ForceProprioError(
                "gripper_open_fraction must be in [0, 1] with 0=closed and 1=open"
            )

        state = (
            *self.tcp_position_m,
            *self.tcp_rotvec_rad,
            gripper,
            *self.joint_position_rad,
            *self.joint_velocity_rad_s,
            *self.wrench_base_n_nm,
        )
        if len(state) != OBSERVATION_STATE_DIM:  # pragma: no cover
            raise ForceProprioError(
                f"observation state must have {OBSERVATION_STATE_DIM} values, got {len(state)}"
            )
        if not all(math.isfinite(value) for value in state):  # pragma: no cover
            raise ForceProprioError("observation state contains non-finite values")
        return state

    def to_dict(self) -> dict[str, object]:
        return {
            "source_bag_timestamp_ns": self.source_bag_timestamp_ns,
            "controller_timestamp_s": self.controller_timestamp_s,
            "tcp_position_m": list(self.tcp_position_m),
            "tcp_rotvec_rad": list(self.tcp_rotvec_rad),
            "joint_position_rad": list(self.joint_position_rad),
            "joint_velocity_rad_s": list(self.joint_velocity_rad_s),
            "wrench_base_n_nm": list(self.wrench_base_n_nm),
            "force_policy": self.force_policy.value,
        }


@dataclass(frozen=True)
class SynchronizedForceProprioSample:
    """One Patch-4 reference frame and its single selected RobotStateRt payload."""

    reference_index: int
    reference_timestamp_ns: int
    robot_state_source_index: int
    robot_state_source_timestamp_ns: int
    robot_state_signed_skew_ns: int
    state: DoosanForceProprioState

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_index": self.reference_index,
            "reference_timestamp_ns": self.reference_timestamp_ns,
            "robot_state_source_index": self.robot_state_source_index,
            "robot_state_source_timestamp_ns": self.robot_state_source_timestamp_ns,
            "robot_state_signed_skew_ns": self.robot_state_signed_skew_ns,
            "state": self.state.to_dict(),
        }


@dataclass(frozen=True)
class DoosanForceProprioEpisode:
    """Patch-5 semantic result for all complete Patch-4 reference frames."""

    provenance: ForceCompensationProvenance
    raw_robot_state_count: int
    complete_reference_count: int
    dropped_reference_count: int
    samples: tuple[SynchronizedForceProprioSample, ...]

    def __post_init__(self) -> None:
        if self.complete_reference_count != len(self.samples):
            raise ForceProprioError(
                "complete_reference_count must equal the number of semantic samples"
            )
        if self.raw_robot_state_count < len(self.samples):
            raise ForceProprioError(
                "raw_robot_state_count cannot be smaller than selected sample count"
            )

    def summary_dict(self) -> dict[str, object]:
        return {
            "semantics_id": SEMANTICS_ID,
            "forcevla_contract_id": FORCEVLA_CONTRACT_ID,
            "observation_state_dim": OBSERVATION_STATE_DIM,
            "observation_state_fields": list(OBSERVATION_STATE_FIELDS),
            "robot_semantic_dim_without_gripper": ROBOT_SEMANTIC_DIM,
            "raw_robot_state_count": self.raw_robot_state_count,
            "complete_reference_count": self.complete_reference_count,
            "dropped_reference_count": self.dropped_reference_count,
            "sample_count": len(self.samples),
            "force_provenance": self.provenance.to_dict(),
            "authoritative_state_source": ROBOT_STATE_RT_TOPIC,
            "joint_state_role": "optional_validation_only",
            "tcp_position": {
                "source": "RobotStateRt.actual_tcp_position[0:3]",
                "source_unit": "mm",
                "output_unit": "m",
                "frame": "base",
                "conversion": "divide_by_1000",
            },
            "tcp_orientation": {
                "source": "RobotStateRt.actual_tcp_position[3:6]",
                "source_convention": "Doosan intrinsic Euler ZYZ z-y'-z''",
                "source_unit": "deg",
                "output_convention": "absolute base-to-TCP SO(3) rotation vector",
                "output_unit": "rad",
                "conversion": "Rz(A) @ Ry(B) @ Rz(C) -> Log(R)",
                "componentwise_euler_interpolation": False,
            },
            "joint_position": {
                "source": "RobotStateRt.actual_joint_position",
                "source_unit": "deg",
                "output_unit": "rad",
            },
            "joint_velocity": {
                "source": "RobotStateRt.actual_joint_velocity",
                "source_unit": "deg/s",
                "output_unit": "rad/s",
            },
            "wrench": {
                "source": "RobotStateRt.external_tcp_force",
                "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "frame": "base coordinates",
                "units": ["N", "N", "N", "N*m", "N*m", "N*m"],
                "conversion": "passthrough",
                "same_robot_state_selection_as_tcp_and_joints": True,
            },
            "gripper": {
                "owned_by_patch": 6,
                "derived_here": False,
                "required_for_25d_assembly": True,
                "semantic": "normalized measured opening; 0=closed, 1=open",
            },
        }


def _require_finite_scalar(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ForceProprioError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ForceProprioError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise ForceProprioError(f"{context} must be finite")
    return result


def _require_finite_vector(
    values: Iterable[float],
    expected_len: int,
    context: str,
) -> tuple[float, ...]:
    try:
        result = tuple(_require_finite_scalar(value, context) for value in values)
    except TypeError as exc:
        raise ForceProprioError(f"{context} must be an iterable") from exc
    if len(result) != expected_len:
        raise ForceProprioError(
            f"{context} must have length {expected_len}, got {len(result)}"
        )
    return result


def _matmul3(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            math.fsum(left[row][k] * right[k][col] for k in range(3))
            for col in range(3)
        )
        for row in range(3)
    )


def _transpose3(
    matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(matrix[col][row]) for col in range(3)) for row in range(3))


def _determinant3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validated_rotation_matrix(
    matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ForceProprioError("rotation matrix must be 3x3")
    normalized = tuple(
        tuple(_require_finite_scalar(value, "rotation matrix") for value in row)
        for row in matrix
    )
    identity = _matmul3(_transpose3(normalized), normalized)
    max_orthogonality_error = max(
        abs(identity[row][col] - (1.0 if row == col else 0.0))
        for row in range(3)
        for col in range(3)
    )
    determinant = _determinant3(normalized)
    if max_orthogonality_error > 1e-9 or abs(determinant - 1.0) > 1e-9:
        raise ForceProprioError(
            "matrix is not a proper SO(3) rotation: "
            f"orthogonality_error={max_orthogonality_error:.3e}, det={determinant:.15g}"
        )
    return normalized


def doosan_zyz_deg_to_matrix(
    euler_zyz_deg: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Convert Doosan intrinsic ``z-y'-z''`` Euler degrees to base-to-TCP R.

    Doosan's documented moving-axis convention composes as
    ``Rz(A) @ Ry(B) @ Rz(C)`` for column vectors.
    """

    a_deg, b_deg, c_deg = _require_finite_vector(
        euler_zyz_deg,
        3,
        "Doosan Euler ZYZ",
    )
    a = math.radians(a_deg)
    b = math.radians(b_deg)
    c = math.radians(c_deg)

    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)

    rz_a = (
        (ca, -sa, 0.0),
        (sa, ca, 0.0),
        (0.0, 0.0, 1.0),
    )
    ry_b = (
        (cb, 0.0, sb),
        (0.0, 1.0, 0.0),
        (-sb, 0.0, cb),
    )
    rz_c = (
        (cc, -sc, 0.0),
        (sc, cc, 0.0),
        (0.0, 0.0, 1.0),
    )
    return _validated_rotation_matrix(_matmul3(_matmul3(rz_a, ry_b), rz_c))


def rotation_matrix_to_rotvec(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """Return the principal SO(3) logarithm as a rotation vector in radians."""

    r = _validated_rotation_matrix(matrix)
    trace = r[0][0] + r[1][1] + r[2][2]

    if trace > 0.0:
        s = 2.0 * math.sqrt(max(trace + 1.0, 0.0))
        if s <= 0.0:  # pragma: no cover - guarded by trace > 0
            raise ForceProprioError("failed to convert rotation matrix to quaternion")
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
        qw = 0.25 * s
    elif r[0][0] >= r[1][1] and r[0][0] >= r[2][2]:
        s = 2.0 * math.sqrt(max(1.0 + r[0][0] - r[1][1] - r[2][2], 0.0))
        if s <= 0.0:
            raise ForceProprioError("failed to convert rotation matrix near pi")
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
        qw = (r[2][1] - r[1][2]) / s
    elif r[1][1] >= r[2][2]:
        s = 2.0 * math.sqrt(max(1.0 + r[1][1] - r[0][0] - r[2][2], 0.0))
        if s <= 0.0:
            raise ForceProprioError("failed to convert rotation matrix near pi")
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
        qw = (r[0][2] - r[2][0]) / s
    else:
        s = 2.0 * math.sqrt(max(1.0 + r[2][2] - r[0][0] - r[1][1], 0.0))
        if s <= 0.0:
            raise ForceProprioError("failed to convert rotation matrix near pi")
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s
        qw = (r[1][0] - r[0][1]) / s

    qnorm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(qnorm) or qnorm <= 0.0:
        raise ForceProprioError("rotation matrix produced an invalid quaternion")
    qx, qy, qz, qw = (value / qnorm for value in (qx, qy, qz, qw))

    # q and -q represent the same rotation.  Force the principal angle into
    # [0, pi] by choosing a non-negative scalar component.
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    vector_norm = math.sqrt(qx * qx + qy * qy + qz * qz)
    if vector_norm < 1e-15:
        return (0.0, 0.0, 0.0)

    angle = 2.0 * math.atan2(vector_norm, max(qw, 0.0))
    if angle > math.pi and angle - math.pi < 1e-12:
        angle = math.pi
    if not 0.0 <= angle <= math.pi + 1e-12:
        raise ForceProprioError(f"principal rotation angle out of range: {angle}")
    scale = angle / vector_norm
    return (qx * scale, qy * scale, qz * scale)


def rotation_vector_to_matrix(
    rotvec_rad: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Rodrigues exponential map used for physical round-trip validation."""

    x, y, z = _require_finite_vector(rotvec_rad, 3, "rotation vector")
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-15:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    kx, ky, kz = x / angle, y / angle, z / angle
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    matrix = (
        (
            c + kx * kx * one_minus_c,
            kx * ky * one_minus_c - kz * s,
            kx * kz * one_minus_c + ky * s,
        ),
        (
            ky * kx * one_minus_c + kz * s,
            c + ky * ky * one_minus_c,
            ky * kz * one_minus_c - kx * s,
        ),
        (
            kz * kx * one_minus_c - ky * s,
            kz * ky * one_minus_c + kx * s,
            c + kz * kz * one_minus_c,
        ),
    )
    return _validated_rotation_matrix(matrix)


def doosan_zyz_deg_to_rotvec(
    euler_zyz_deg: Sequence[float],
) -> tuple[float, float, float]:
    """Convert Doosan Euler ZYZ degrees to principal base-to-TCP rotvec radians."""

    return rotation_matrix_to_rotvec(doosan_zyz_deg_to_matrix(euler_zyz_deg))


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForceProprioError(f"{path}: could not read JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ForceProprioError(f"{path}: expected a JSON object")
    return value


def _optional_bool(mapping: dict[str, Any], key: str) -> bool | None:
    value = mapping.get(key)
    return value if isinstance(value, bool) else None


def resolve_force_compensation_provenance(
    episode_dir: str | Path,
) -> ForceCompensationProvenance:
    """Resolve force processing from ``episode_operator.json`` without guessing.

    The only Patch-5 training-ready path is the exact modern controller-reset
    provenance.  Every other episode is classified as requiring an explicit,
    separately defined legacy tare policy; Patch 5 does not invent or apply one.
    Contradictory modern metadata is rejected outright.
    """

    path = Path(episode_dir) / "episode_operator.json"
    operator = _read_json_object(path)
    recording = operator.get("recording")
    if not isinstance(recording, dict):
        recording = {}

    schema_version = operator.get("schema_version")
    if not isinstance(schema_version, str):
        schema_version = None

    reset_active = _optional_bool(
        operator,
        "controller_external_force_reset_active_at_record_start",
    )
    reset_completed = _optional_bool(
        operator,
        "controller_external_force_reset_completed_before_recording",
    )
    guard_tare_applied = _optional_bool(operator, "force_guard_tare_applied_to_mcap")
    offline_tare = _optional_bool(operator, "offline_force_tare_performed")
    pre_reset_recorded = _optional_bool(operator, "pre_reset_ft_recorded")
    recording_reset = _optional_bool(recording, "controller_reset_compensated")
    force_signal = recording.get("force_signal_in_mcap")
    if not isinstance(force_signal, str):
        force_signal = None
    processing_owner = operator.get("offline_force_processing_owner")
    if not isinstance(processing_owner, str):
        processing_owner = None

    claims_reset_compensation = any(
        value is True for value in (reset_active, reset_completed, recording_reset)
    )
    if claims_reset_compensation and guard_tare_applied is True:
        raise ForceProprioError(
            f"{path}: reset-compensated recording claims force_guard_tare_applied_to_mcap=true; "
            "refusing ambiguous/double-tared force provenance"
        )
    if claims_reset_compensation and offline_tare is True:
        raise ForceProprioError(
            f"{path}: reset-compensated recording claims offline_force_tare_performed=true; "
            "refusing ambiguous/double-tared force provenance"
        )
    if claims_reset_compensation and pre_reset_recorded is True:
        raise ForceProprioError(
            f"{path}: reset-compensated recording claims pre_reset_ft_recorded=true; "
            "refusing mixed pre/post-reset force provenance"
        )

    modern_checks = (
        schema_version == FORCE_PROVENANCE_SCHEMA,
        reset_active is True,
        reset_completed is True,
        guard_tare_applied is False,
        offline_tare is False,
        pre_reset_recorded is False,
        recording_reset is True,
        force_signal == EXPECTED_FORCE_SIGNAL,
        processing_owner == EXPECTED_FORCE_PROCESSING_OWNER,
    )

    if all(modern_checks):
        return ForceCompensationProvenance(
            policy=ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH,
            schema_version=schema_version,
            source_path=str(path),
            controller_external_force_reset_active_at_record_start=reset_active,
            controller_external_force_reset_completed_before_recording=reset_completed,
            force_guard_tare_applied_to_mcap=guard_tare_applied,
            offline_force_tare_performed=offline_tare,
            pre_reset_ft_recorded=pre_reset_recorded,
            recording_controller_reset_compensated=recording_reset,
            recording_force_signal_in_mcap=force_signal,
            offline_force_processing_owner=processing_owner,
            approved_for_training=True,
            reason=(
                "controller external-force reset completed before recording; MCAP stores "
                "reset-compensated RobotStateRt.external_tcp_force; pass through exactly once"
            ),
        )

    return ForceCompensationProvenance(
        policy=ForceCompensationPolicy.LEGACY_EPISODE_REQUIRES_KNOWN_TARE_POLICY,
        schema_version=schema_version,
        source_path=str(path),
        controller_external_force_reset_active_at_record_start=reset_active,
        controller_external_force_reset_completed_before_recording=reset_completed,
        force_guard_tare_applied_to_mcap=guard_tare_applied,
        offline_force_tare_performed=offline_tare,
        pre_reset_ft_recorded=pre_reset_recorded,
        recording_controller_reset_compensated=recording_reset,
        recording_force_signal_in_mcap=force_signal,
        offline_force_processing_owner=processing_owner,
        approved_for_training=False,
        reason=(
            "episode does not satisfy the frozen modern reset-compensated provenance gate; "
            "an explicit known legacy tare policy is required before training use"
        ),
    )


def convert_robot_state_rt(
    record: RobotStateRtRecord,
    provenance: ForceCompensationProvenance,
) -> DoosanForceProprioState:
    """Convert one authoritative controller state without any re-synchronization."""

    if not isinstance(record, RobotStateRtRecord):
        raise ForceProprioError(
            f"expected RobotStateRtRecord, got {type(record).__name__}"
        )
    if provenance.policy is not ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH:
        raise ForceProprioError(
            "legacy episode requires a known explicit tare policy; Patch 5 refuses "
            "to guess or apply an implicit force offset"
        )
    if not provenance.approved_for_training:
        raise ForceProprioError("force provenance is not approved for training")

    tcp = _require_finite_vector(record.actual_tcp_position_mm_deg, 6, "actual_tcp_position")
    joint_position_deg = _require_finite_vector(
        record.actual_joint_position_deg,
        6,
        "actual_joint_position",
    )
    joint_velocity_deg_s = _require_finite_vector(
        record.actual_joint_velocity_deg_s,
        6,
        "actual_joint_velocity",
    )
    wrench = _require_finite_vector(
        record.external_tcp_force_base_n_nm,
        6,
        "external_tcp_force",
    )

    return DoosanForceProprioState(
        source_bag_timestamp_ns=int(record.stamp.bag_timestamp_ns),
        controller_timestamp_s=_require_finite_scalar(
            record.controller_timestamp_s,
            "controller_timestamp_s",
        ),
        tcp_position_m=(tcp[0] / 1000.0, tcp[1] / 1000.0, tcp[2] / 1000.0),
        tcp_rotvec_rad=doosan_zyz_deg_to_rotvec(tcp[3:6]),
        joint_position_rad=tuple(math.radians(value) for value in joint_position_deg),
        joint_velocity_rad_s=tuple(math.radians(value) for value in joint_velocity_deg_s),
        wrench_base_n_nm=wrench,
        force_policy=provenance.policy,
    )


def build_synchronized_force_proprio_samples(
    robot_records: Sequence[RobotStateRtRecord],
    sync_result: DoosanSynchronizationResult,
    provenance: ForceCompensationProvenance,
) -> tuple[SynchronizedForceProprioSample, ...]:
    """Apply Patch-5 conversion to the exact Patch-4 RobotStateRt selections."""

    if provenance.policy is not ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH:
        raise ForceProprioError(
            "episode force provenance is not training-ready reset-compensated passthrough"
        )

    robot_plan = sync_result.plan.source_plan("robot_state_rt")
    if robot_plan.spec.method is not SyncMethod.NEAREST or not robot_plan.spec.required:
        raise ForceProprioError(
            "Patch-5 requires the frozen Patch-4 required nearest RobotStateRt policy"
        )
    if len(robot_plan.decisions) != len(sync_result.plan.reference.timestamps_ns):
        raise ForceProprioError("RobotStateRt decision count does not match reference timeline")

    complete = tuple(sync_result.plan.complete_reference_indices)
    complete_set = set(complete)
    if len(complete_set) != len(complete):
        raise ForceProprioError("complete reference indices contain duplicates")

    samples: list[SynchronizedForceProprioSample] = []
    for reference_index in complete:
        if not 0 <= reference_index < len(robot_plan.decisions):
            raise ForceProprioError(
                f"complete reference index out of range: {reference_index}"
            )
        decision = robot_plan.decisions[reference_index]
        if decision.reference_index != reference_index:
            raise ForceProprioError(
                "RobotStateRt synchronization decision/reference index mismatch"
            )
        selection = decision.selection
        if selection is None:
            raise ForceProprioError(
                f"complete reference {reference_index} lacks RobotStateRt selection"
            )
        if len(selection.source_indices) != 1 or len(selection.source_timestamps_ns) != 1:
            raise ForceProprioError(
                "Patch-5 forbids interpolated or multi-record RobotStateRt selections"
            )
        if len(selection.signed_skews_ns) != 1 or selection.alpha is not None:
            raise ForceProprioError("unexpected RobotStateRt nearest-selection provenance")

        source_index = selection.source_indices[0]
        if not 0 <= source_index < len(robot_records):
            raise ForceProprioError(
                f"RobotStateRt source index {source_index} is outside decoded record range"
            )
        record = robot_records[source_index]
        selected_timestamp = selection.source_timestamps_ns[0]
        if int(record.stamp.bag_timestamp_ns) != selected_timestamp:
            raise ForceProprioError(
                "Patch-4 RobotStateRt source timestamp does not match the selected decoded record"
            )

        samples.append(
            SynchronizedForceProprioSample(
                reference_index=reference_index,
                reference_timestamp_ns=decision.reference_timestamp_ns,
                robot_state_source_index=source_index,
                robot_state_source_timestamp_ns=selected_timestamp,
                robot_state_signed_skew_ns=selection.signed_skews_ns[0],
                state=convert_robot_state_rt(record, provenance),
            )
        )

    return tuple(samples)


def build_doosan_force_proprio_episode(
    episode_dir: str | Path,
) -> DoosanForceProprioEpisode:
    """Build the complete Patch-5 semantic episode from the immutable raw MCAP."""

    provenance = resolve_force_compensation_provenance(episode_dir)
    if provenance.policy is not ForceCompensationPolicy.RESET_COMPENSATED_PASSTHROUGH:
        raise ForceProprioError(provenance.reason)

    sync_result = build_doosan_sync_plan(episode_dir)
    robot_records: list[RobotStateRtRecord] = []
    for topic, record in iter_typed_messages(episode_dir):
        if topic != ROBOT_STATE_RT_TOPIC:
            continue
        if not isinstance(record, RobotStateRtRecord):  # pragma: no cover - decoder contract
            raise ForceProprioError(
                f"{ROBOT_STATE_RT_TOPIC}: expected RobotStateRtRecord, got {type(record).__name__}"
            )
        robot_records.append(record)

    samples = build_synchronized_force_proprio_samples(
        robot_records,
        sync_result,
        provenance,
    )
    return DoosanForceProprioEpisode(
        provenance=provenance,
        raw_robot_state_count=len(robot_records),
        complete_reference_count=len(sync_result.plan.complete_reference_indices),
        dropped_reference_count=len(sync_result.plan.dropped_reference_indices),
        samples=samples,
    )


__all__ = [
    "EXPECTED_FORCE_PROCESSING_OWNER",
    "EXPECTED_FORCE_SIGNAL",
    "FORCEVLA_CONTRACT_ID",
    "FORCE_PROVENANCE_SCHEMA",
    "OBSERVATION_STATE_DIM",
    "OBSERVATION_STATE_FIELDS",
    "ROBOT_SEMANTIC_DIM",
    "SEMANTICS_ID",
    "DoosanForceProprioEpisode",
    "DoosanForceProprioState",
    "ForceCompensationPolicy",
    "ForceCompensationProvenance",
    "ForceProprioError",
    "SynchronizedForceProprioSample",
    "build_doosan_force_proprio_episode",
    "build_synchronized_force_proprio_samples",
    "convert_robot_state_rt",
    "doosan_zyz_deg_to_matrix",
    "doosan_zyz_deg_to_rotvec",
    "resolve_force_compensation_provenance",
    "rotation_matrix_to_rotvec",
    "rotation_vector_to_matrix",
]
