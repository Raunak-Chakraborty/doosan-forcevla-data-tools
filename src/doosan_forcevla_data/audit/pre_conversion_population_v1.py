"""CPU-parallel pre-conversion population audit for Doosan thesis episodes.

The audit is intentionally read-only.  It evaluates the raw episode population
before any production conversion/export is started and combines the population
checks that were validated during the thesis data freeze:

* all four model-facing orientation representations and both wrench modes;
* action/non-orientation invariance across representation profiles;
* principal-rotvec, continuous-rotvec, quaternion, and rotation6D continuity;
* numerical joint-wrap candidates on contiguous synchronized references;
* raw SCHUNK held->released semantics plus SpaceMouse RIGHT provenance;
* processed release-only gripper semantics;
* baseline-relative force/torque contact screening with a small critical-review
  tier and a broader diagnostic-review tier.

The module has no ROS imports at import time so its pure helpers remain covered
by the portable unit-test suite.  ROS/Jazzy dependencies are imported lazily in
``audit_episode`` when real MCAP episodes are processed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import traceback
from typing import Any, Iterable, Sequence

from doosan_forcevla_data.convert.doosan_processed_to_lerobot_v21 import _dataset_rows
from doosan_forcevla_data.convert.model_state_profile_v1 import StateMode, model_state_layout
from doosan_forcevla_data.convert.orientation_representation_v1 import (
    OrientationRepresentation,
    decode_orientation,
    matrix_to_principal_rotvec,
    rotvec_to_matrix,
)


AUDIT_SCHEMA = "doosan_pre_conversion_population_audit_v1"
RIGHT_BUTTON_INDEX = 1
AUTO_WORKER_CAP = 4


class PreConversionAuditError(RuntimeError):
    """Raised when an audit invariant cannot be evaluated safely."""


@dataclass(frozen=True)
class AuditThresholds:
    rotation_tolerance: float = 1e-9
    small_physical_step_rad: float = 0.1
    joint_large_jump_rad: float = math.pi
    joint_classic_wrap_residual_rad: float = 0.50
    joint_classic_wrap_small_modulo_rad: float = 0.50
    force_high_n: float = 10.0
    force_recover_n: float = 5.0
    force_extreme_n: float = 25.0
    torque_extreme_nm: float = 5.0
    force_persist_sec: float = 0.75
    force_recovery_sec: float = 0.25
    baseline_sec: float = 1.0
    tail_sec: float = 0.50
    # Deliberately stricter than the broad force-review tier.  This is the
    # short list intended for re-review of the genuinely suspicious episodes.
    critical_tail_force_n: float = 30.0
    critical_sustained_force_sec: float = 3.0

    def __post_init__(self) -> None:
        positive = {
            "rotation_tolerance": self.rotation_tolerance,
            "small_physical_step_rad": self.small_physical_step_rad,
            "joint_large_jump_rad": self.joint_large_jump_rad,
            "joint_classic_wrap_residual_rad": self.joint_classic_wrap_residual_rad,
            "joint_classic_wrap_small_modulo_rad": self.joint_classic_wrap_small_modulo_rad,
            "force_high_n": self.force_high_n,
            "force_recover_n": self.force_recover_n,
            "force_extreme_n": self.force_extreme_n,
            "torque_extreme_nm": self.torque_extreme_nm,
            "force_persist_sec": self.force_persist_sec,
            "force_recovery_sec": self.force_recovery_sec,
            "baseline_sec": self.baseline_sec,
            "tail_sec": self.tail_sec,
            "critical_tail_force_n": self.critical_tail_force_n,
            "critical_sustained_force_sec": self.critical_sustained_force_sec,
        }
        invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"thresholds must be finite and positive: {', '.join(invalid)}")
        if self.force_recover_n >= self.force_high_n:
            raise ValueError("force_recover_n must be lower than force_high_n")
        if self.critical_tail_force_n < self.force_high_n:
            raise ValueError("critical_tail_force_n cannot be below force_high_n")


@dataclass(frozen=True)
class ForceScreenResult:
    category: str
    broad_review_reasons: tuple[str, ...]
    critical_review_reasons: tuple[str, ...]
    max_delta_force_n: float
    max_delta_torque_nm: float
    longest_high_force_sec: float
    total_high_force_sec: float
    high_force_fraction: float
    terminal_recovery_sec: float
    tail_median_delta_force_n: float
    tail_max_delta_force_n: float
    max_raw_force_resultant_n: float
    max_raw_torque_resultant_nm: float
    pre_release_duration_sec: float
    release_source_reference_index: int
    release_target_reference_index: int

    @property
    def broad_review_candidate(self) -> bool:
        return bool(self.broad_review_reasons)

    @property
    def critical_review_candidate(self) -> bool:
        return bool(self.critical_review_reasons)


def _available_logical_cpu_ids() -> tuple[int, ...]:
    """Return logical CPU ids available to this process."""

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            cpu_ids = tuple(sorted(int(cpu) for cpu in get_affinity(0)))
        except (OSError, ValueError):
            cpu_ids = ()
        if cpu_ids:
            return cpu_ids
    return tuple(range(max(1, int(os.cpu_count() or 1))))


def available_logical_cpu_count() -> int:
    """Return available logical CPUs / hardware threads."""

    return len(_available_logical_cpu_ids())


def available_physical_core_count(
    *, topology_root: Path = Path("/sys/devices/system/cpu")
) -> int:
    """Return physical CPU cores available to this process on Linux.

    CPU affinity is respected first.  Linux sysfs topology then collapses SMT
    siblings that share the same ``(physical_package_id, core_id)`` pair.  If
    topology is unavailable, fall back to the affinity-aware logical CPU count.
    """

    cpu_ids = _available_logical_cpu_ids()
    physical_cores: set[tuple[int, int]] = set()

    for cpu_id in cpu_ids:
        topology = topology_root / f"cpu{cpu_id}" / "topology"
        try:
            package_id = int((topology / "physical_package_id").read_text().strip())
            core_id = int((topology / "core_id").read_text().strip())
        except (OSError, ValueError):
            return len(cpu_ids)
        physical_cores.add((package_id, core_id))

    return max(1, len(physical_cores))


def available_cpu_count() -> int:
    """Backward-compatible alias for the physical-core worker limit."""

    return available_physical_core_count()


def resolve_worker_count(requested: str | int | None, *, available: int | None = None) -> int:
    """Resolve workers with a conservative four-physical-core automatic default.

    ``auto`` deliberately caps the episode-level MCAP workload at four worker
    processes even when more physical cores are available. Explicit worker
    counts may raise parallelism up to the available physical-core count.
    """

    maximum = available_physical_core_count() if available is None else int(available)
    if maximum < 1:
        raise ValueError("available physical core count must be >= 1")
    if requested is None or requested == "auto":
        return min(maximum, AUTO_WORKER_CAP)
    if isinstance(requested, bool):
        raise ValueError("worker count must be 'auto' or a positive integer")
    try:
        value = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError("worker count must be 'auto' or a positive integer") from exc
    if value < 1:
        raise ValueError("worker count must be >= 1")
    if value > maximum:
        raise ValueError(f"requested {value} workers but only {maximum} physical CPU cores are available")
    return value


def _vec_norm(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(value) * float(value) for value in values))


def _vec_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise PreConversionAuditError("vector widths differ")
    return math.sqrt(
        math.fsum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))
    )


def _transpose3(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(matrix[col][row]) for col in range(3)) for row in range(3))


def _matmul3(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            math.fsum(float(left[row][k]) * float(right[k][col]) for k in range(3))
            for col in range(3)
        )
        for row in range(3)
    )


def _matrix_max_error(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    return max(
        abs(float(left[row][col]) - float(right[row][col]))
        for row in range(3)
        for col in range(3)
    )


def _physical_step_angle(
    previous_matrix: Sequence[Sequence[float]], current_matrix: Sequence[Sequence[float]]
) -> float:
    relative = _matmul3(current_matrix, _transpose3(previous_matrix))
    return _vec_norm(matrix_to_principal_rotvec(relative))


def _require_processed_rows(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise PreConversionAuditError("processed row sequence is empty")
    previous_reference: int | None = None
    for index, row in enumerate(rows):
        state = tuple(float(value) for value in row["observation_state_25d"])
        action = tuple(float(value) for value in row["action_7d"])
        if len(state) != 25:
            raise PreConversionAuditError(f"row {index}: expected 25D canonical state")
        if len(action) != 7:
            raise PreConversionAuditError(f"row {index}: expected 7D semantic action")
        reference = int(row["reference_index"])
        if previous_reference is not None and reference <= previous_reference:
            raise PreConversionAuditError("reference indices must be strictly increasing")
        previous_reference = reference


def audit_orientation_profiles(
    rows: Sequence[dict[str, Any]], thresholds: AuditThresholds
) -> dict[str, Any]:
    """Audit all eight orientation/state-mode profiles against canonical 25D rows."""

    _require_processed_rows(rows)
    canonical_states = [tuple(float(value) for value in row["observation_state_25d"]) for row in rows]
    canonical_actions = [tuple(float(value) for value in row["action_7d"]) for row in rows]
    references = [int(row["reference_index"]) for row in rows]

    profiles: dict[tuple[str, str], list[dict[str, Any]]] = {}
    profile_dimensions: dict[str, int] = {}
    action_mismatch_count = 0
    nonorientation_mismatch_count = 0
    full_vs_no_wrench_mismatch_count = 0
    physical_rotation_mismatch_count = 0
    max_rotation_matrix_error = 0.0

    for representation in OrientationRepresentation:
        no_layout = model_state_layout(representation, StateMode.NO_WRENCH)
        full_layout = model_state_layout(representation, StateMode.FULL)
        profile_dimensions[f"{representation.value}/no_wrench"] = no_layout.state_dim
        profile_dimensions[f"{representation.value}/full"] = full_layout.state_dim
        no_rows = _dataset_rows(rows, layout=no_layout)
        full_rows = _dataset_rows(rows, layout=full_layout)
        profiles[(representation.value, "no_wrench")] = no_rows
        profiles[(representation.value, "full")] = full_rows

        if full_layout.wrench_slice is None or full_layout.wrench_slice.stop != full_layout.state_dim:
            raise PreConversionAuditError(f"{representation.value}: wrench is not final")
        if full_layout.wrench_slice.start != full_layout.state_dim - 6:
            raise PreConversionAuditError(f"{representation.value}: wrench width is not six")
        if no_layout.wrench_slice is not None:
            raise PreConversionAuditError(f"{representation.value}: no-wrench profile exposes wrench")

        for index, (no_row, full_row, source, source_action) in enumerate(
            zip(no_rows, full_rows, canonical_states, canonical_actions, strict=True)
        ):
            no_state = tuple(float(value) for value in no_row["observation.state"])
            full_state = tuple(float(value) for value in full_row["observation.state"])
            if tuple(float(value) for value in no_row["action"]) != source_action:
                action_mismatch_count += 1
            if tuple(float(value) for value in full_row["action"]) != source_action:
                action_mismatch_count += 1
            if full_state[:-6] != no_state:
                full_vs_no_wrench_mismatch_count += 1
            if full_state[0:3] != source[0:3]:
                nonorientation_mismatch_count += 1
            if full_state[full_layout.gripper_index] != source[6]:
                nonorientation_mismatch_count += 1
            if full_state[full_layout.joint_position_slice] != source[7:13]:
                nonorientation_mismatch_count += 1
            if full_state[full_layout.joint_velocity_slice] != source[13:19]:
                nonorientation_mismatch_count += 1
            if full_state[full_layout.wrench_slice] != source[19:25]:
                nonorientation_mismatch_count += 1

            reconstructed = decode_orientation(
                full_state[full_layout.orientation_slice], representation
            )
            canonical_matrix = rotvec_to_matrix(source[3:6])
            error = _matrix_max_error(reconstructed, canonical_matrix)
            max_rotation_matrix_error = max(max_rotation_matrix_error, error)
            if error > thresholds.rotation_tolerance:
                physical_rotation_mismatch_count += 1

            if representation is OrientationRepresentation.ROTVEC_PRINCIPAL:
                if full_state != source or no_state != source[:19]:
                    nonorientation_mismatch_count += 1

    sequences: dict[str, list[tuple[float, ...]]] = {}
    for representation in OrientationRepresentation:
        layout = model_state_layout(representation, StateMode.FULL)
        sequences[representation.value] = [
            tuple(float(value) for value in row["observation.state"])[layout.orientation_slice]
            for row in profiles[(representation.value, "full")]
        ]

    event_counts = {
        "principal_branch_event_count": 0,
        "continuous_large_jump_count": 0,
        "quaternion_large_jump_count": 0,
        "rotation6d_large_jump_count": 0,
    }
    max_jumps = {
        "max_principal_adjacent_jump": 0.0,
        "max_continuous_adjacent_jump": 0.0,
        "max_quaternion_adjacent_jump": 0.0,
        "max_rotation6d_adjacent_jump": 0.0,
        "max_physical_adjacent_rotation_rad": 0.0,
    }
    contiguous_pair_count = 0

    for index in range(1, len(rows)):
        if references[index] != references[index - 1] + 1:
            continue
        contiguous_pair_count += 1
        previous_matrix = rotvec_to_matrix(canonical_states[index - 1][3:6])
        current_matrix = rotvec_to_matrix(canonical_states[index][3:6])
        physical_angle = _physical_step_angle(previous_matrix, current_matrix)
        max_jumps["max_physical_adjacent_rotation_rad"] = max(
            max_jumps["max_physical_adjacent_rotation_rad"], physical_angle
        )
        principal_jump = _vec_distance(
            sequences[OrientationRepresentation.ROTVEC_PRINCIPAL.value][index - 1],
            sequences[OrientationRepresentation.ROTVEC_PRINCIPAL.value][index],
        )
        continuous_jump = _vec_distance(
            sequences[OrientationRepresentation.ROTVEC_CONTINUOUS.value][index - 1],
            sequences[OrientationRepresentation.ROTVEC_CONTINUOUS.value][index],
        )
        quaternion_jump = _vec_distance(
            sequences[OrientationRepresentation.QUATERNION.value][index - 1],
            sequences[OrientationRepresentation.QUATERNION.value][index],
        )
        rotation6d_jump = _vec_distance(
            sequences[OrientationRepresentation.ROTATION6D.value][index - 1],
            sequences[OrientationRepresentation.ROTATION6D.value][index],
        )
        max_jumps["max_principal_adjacent_jump"] = max(
            max_jumps["max_principal_adjacent_jump"], principal_jump
        )
        max_jumps["max_continuous_adjacent_jump"] = max(
            max_jumps["max_continuous_adjacent_jump"], continuous_jump
        )
        max_jumps["max_quaternion_adjacent_jump"] = max(
            max_jumps["max_quaternion_adjacent_jump"], quaternion_jump
        )
        max_jumps["max_rotation6d_adjacent_jump"] = max(
            max_jumps["max_rotation6d_adjacent_jump"], rotation6d_jump
        )
        if physical_angle < thresholds.small_physical_step_rad:
            if principal_jump > math.pi:
                event_counts["principal_branch_event_count"] += 1
            if continuous_jump > math.pi:
                event_counts["continuous_large_jump_count"] += 1
            if quaternion_jump > 1.0:
                event_counts["quaternion_large_jump_count"] += 1
            if rotation6d_jump > 1.0:
                event_counts["rotation6d_large_jump_count"] += 1

    issues: list[str] = []
    if action_mismatch_count:
        issues.append("ORIENTATION_PROFILE_ACTION_MISMATCH")
    if nonorientation_mismatch_count:
        issues.append("ORIENTATION_PROFILE_NONORIENTATION_MISMATCH")
    if full_vs_no_wrench_mismatch_count:
        issues.append("FULL_NO_WRENCH_MISMATCH")
    if physical_rotation_mismatch_count:
        issues.append("ORIENTATION_RECONSTRUCTION_MISMATCH")
    if event_counts["continuous_large_jump_count"]:
        issues.append("CONTINUOUS_ROTVEC_LARGE_JUMP")
    if event_counts["rotation6d_large_jump_count"]:
        issues.append("ROTATION6D_LARGE_JUMP")

    return {
        "profile_dimensions": profile_dimensions,
        "contiguous_pair_count": contiguous_pair_count,
        "action_mismatch_count": action_mismatch_count,
        "nonorientation_mismatch_count": nonorientation_mismatch_count,
        "full_vs_no_wrench_mismatch_count": full_vs_no_wrench_mismatch_count,
        "physical_rotation_mismatch_count": physical_rotation_mismatch_count,
        "max_rotation_matrix_error": max_rotation_matrix_error,
        **event_counts,
        **max_jumps,
        "issues": issues,
    }


def _circular_delta(delta: float) -> float:
    return math.atan2(math.sin(delta), math.cos(delta))


def audit_joint_positions(
    rows: Sequence[dict[str, Any]], thresholds: AuditThresholds
) -> dict[str, Any]:
    """Find suspicious numerical joint wraps without changing joint semantics."""

    _require_processed_rows(rows)
    large_events: list[dict[str, Any]] = []
    classic_wrap_events: list[dict[str, Any]] = []
    max_adjacent = [0.0] * 6
    max_modulo = [0.0] * 6

    for previous, current in zip(rows, rows[1:], strict=False):
        previous_ref = int(previous["reference_index"])
        current_ref = int(current["reference_index"])
        if current_ref != previous_ref + 1:
            continue
        previous_state = tuple(float(value) for value in previous["observation_state_25d"])
        current_state = tuple(float(value) for value in current["observation_state_25d"])
        dt_s = (
            int(current["reference_timestamp_ns"])
            - int(previous["reference_timestamp_ns"])
        ) / 1e9
        for joint in range(6):
            p0 = previous_state[7 + joint]
            p1 = current_state[7 + joint]
            raw_delta = p1 - p0
            modulo_delta = _circular_delta(raw_delta)
            abs_raw = abs(raw_delta)
            abs_modulo = abs(modulo_delta)
            max_adjacent[joint] = max(max_adjacent[joint], abs_raw)
            max_modulo[joint] = max(max_modulo[joint], abs_modulo)
            if abs_raw <= thresholds.joint_large_jump_rad:
                continue
            v0 = previous_state[13 + joint]
            v1 = current_state[13 + joint]
            event = {
                "joint": joint + 1,
                "previous_reference_index": previous_ref,
                "current_reference_index": current_ref,
                "previous_position_rad": p0,
                "current_position_rad": p1,
                "raw_delta_rad": raw_delta,
                "modulo_delta_rad": modulo_delta,
                "dt_s": dt_s,
                "velocity_integral_rad": 0.5 * (v0 + v1) * dt_s,
            }
            large_events.append(event)
            if (
                abs(abs_raw - 2.0 * math.pi)
                <= thresholds.joint_classic_wrap_residual_rad
                and abs_modulo <= thresholds.joint_classic_wrap_small_modulo_rad
            ):
                classic_wrap_events.append(dict(event))

    issues = ["JOINT_CLASSIC_WRAP_CANDIDATE"] if classic_wrap_events else []
    return {
        "large_jump_count": len(large_events),
        "classic_wrap_candidate_count": len(classic_wrap_events),
        "max_adjacent_delta_rad": max_adjacent,
        "max_modulo_delta_rad": max_modulo,
        "large_events": large_events,
        "classic_wrap_events": classic_wrap_events,
        "issues": issues,
    }


def audit_gripper_right_samples(
    rows: Sequence[dict[str, Any]],
    *,
    right_edges_ns: Sequence[int],
    raw_holding_samples: Sequence[tuple[int, bool]],
) -> dict[str, Any]:
    """Audit release-only processed semantics and raw RIGHT/SCHUNK provenance."""

    _require_processed_rows(rows)
    if not raw_holding_samples:
        raise PreConversionAuditError("raw gripper stream is empty")

    raw_transitions = [
        index
        for index in range(1, len(raw_holding_samples))
        if raw_holding_samples[index][1] != raw_holding_samples[index - 1][1]
    ]
    release_transitions = [
        index
        for index in raw_transitions
        if raw_holding_samples[index - 1][1] is True
        and raw_holding_samples[index][1] is False
    ]
    release_timestamp_ns = (
        int(raw_holding_samples[release_transitions[0]][0])
        if len(release_transitions) == 1
        else None
    )
    effective_ordinal: int | None = None
    effective_timestamp_ns: int | None = None
    if release_timestamp_ns is not None:
        preceding = [
            (ordinal, int(stamp))
            for ordinal, stamp in enumerate(right_edges_ns, start=1)
            if int(stamp) <= release_timestamp_ns
        ]
        if preceding:
            effective_ordinal, effective_timestamp_ns = preceding[-1]

    release_actions: list[dict[str, int]] = []
    regrasp_count = 0
    nonbinary_count = 0
    state_transitions = 0
    previous_state_gripper: float | None = None
    for row in rows:
        state = tuple(float(value) for value in row["observation_state_25d"])
        action = tuple(float(value) for value in row["action_7d"])
        state_g = state[6]
        target_g = action[6]
        if state_g not in (0.0, 1.0) or target_g not in (0.0, 1.0):
            nonbinary_count += 1
        if state_g == 0.0 and target_g == 1.0:
            release_actions.append(
                {
                    "source_reference_index": int(row["reference_index"]),
                    "target_reference_index": int(row["action_target_reference_index"]),
                    "source_timestamp_ns": int(row["reference_timestamp_ns"]),
                    "target_timestamp_ns": int(row["action_target_reference_timestamp_ns"]),
                }
            )
        if state_g == 1.0 and target_g == 0.0:
            regrasp_count += 1
        if previous_state_gripper is not None and state_g != previous_state_gripper:
            state_transitions += 1
        previous_state_gripper = state_g

    issues: list[str] = []
    if raw_holding_samples[0][1] is not True:
        issues.append("RAW_GRIPPER_DOES_NOT_START_HELD")
    if raw_holding_samples[-1][1] is not False:
        issues.append("RAW_GRIPPER_DOES_NOT_END_RELEASED")
    if len(raw_transitions) != 1:
        issues.append(f"RAW_GRIPPER_TRANSITION_COUNT_{len(raw_transitions)}")
    if len(release_transitions) != 1:
        issues.append(f"RAW_RELEASE_TRANSITION_COUNT_{len(release_transitions)}")
    if len(right_edges_ns) not in (1, 2):
        issues.append(f"RIGHT_RISING_EDGE_COUNT_{len(right_edges_ns)}")
    if release_timestamp_ns is not None and effective_ordinal is None:
        issues.append("NO_RIGHT_EDGE_PRECEDES_RELEASE")
    if len(right_edges_ns) == 2 and effective_ordinal != 2:
        issues.append("DOUBLE_RIGHT_SECOND_EDGE_NOT_EFFECTIVE")
    if nonbinary_count:
        issues.append("PROCESSED_GRIPPER_NONBINARY")
    if float(rows[0]["observation_state_25d"][6]) != 0.0:
        issues.append("PROCESSED_EPISODE_DOES_NOT_START_HELD")
    if float(rows[-1]["action_7d"][6]) != 1.0:
        issues.append("FINAL_GRIPPER_TARGET_NOT_RELEASED")
    if len(release_actions) != 1:
        issues.append(f"PROCESSED_RELEASE_ACTION_COUNT_{len(release_actions)}")
    if regrasp_count:
        issues.append("PROCESSED_REGRASP_ACTION_PRESENT")
    if state_transitions > 1:
        issues.append("MULTIPLE_SELECTED_GRIPPER_STATE_TRANSITIONS")

    latency_ns = (
        release_timestamp_ns - effective_timestamp_ns
        if release_timestamp_ns is not None and effective_timestamp_ns is not None
        else None
    )
    return {
        "right_edge_count": len(right_edges_ns),
        "right_edges_ns": [int(value) for value in right_edges_ns],
        "effective_right_edge_ordinal": effective_ordinal,
        "effective_right_to_raw_release_ns": latency_ns,
        "raw_holding_transition_count": len(raw_transitions),
        "raw_release_transition_count": len(release_transitions),
        "processed_release_action_count": len(release_actions),
        "processed_regrasp_action_count": regrasp_count,
        "processed_nonbinary_count": nonbinary_count,
        "selected_state_transition_count": state_transitions,
        "issues": issues,
    }


def _median_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    return tuple(
        float(statistics.median(vector[column] for vector in vectors))
        for column in range(3)
    )  # type: ignore[return-value]


def screen_force_contact(
    rows: Sequence[dict[str, Any]], thresholds: AuditThresholds
) -> ForceScreenResult:
    """Screen pre-release baseline-relative wrench for manual-review priority."""

    _require_processed_rows(rows)
    release_indices = [
        index
        for index, row in enumerate(rows)
        if float(row["observation_state_25d"][6]) == 0.0
        and float(row["action_7d"][6]) == 1.0
    ]
    if len(release_indices) != 1:
        raise PreConversionAuditError(
            f"force screen requires exactly one release action, found {len(release_indices)}"
        )
    release_index = release_indices[0]
    contact_rows = rows[: release_index + 1]
    if len(contact_rows) < 3:
        raise PreConversionAuditError("too few pre-release rows for force screen")

    timestamps = [int(row["reference_timestamp_ns"]) for row in contact_rows]
    references = [int(row["reference_index"]) for row in contact_rows]
    force_vectors = [
        tuple(float(value) for value in row["observation_state_25d"][19:22])
        for row in contact_rows
    ]
    torque_vectors = [
        tuple(float(value) for value in row["observation_state_25d"][22:25])
        for row in contact_rows
    ]
    t0 = timestamps[0]
    t_end = timestamps[-1]
    baseline_limit = t0 + int(thresholds.baseline_sec * 1e9)
    baseline_indices = [index for index, stamp in enumerate(timestamps) if stamp <= baseline_limit]
    if len(baseline_indices) < 3:
        baseline_indices = list(range(min(3, len(contact_rows))))
    baseline_force = _median_vector([force_vectors[index] for index in baseline_indices])
    baseline_torque = _median_vector([torque_vectors[index] for index in baseline_indices])

    raw_force_resultant = [_vec_norm(vector) for vector in force_vectors]
    raw_torque_resultant = [_vec_norm(vector) for vector in torque_vectors]
    delta_force = [
        _vec_norm(tuple(vector[i] - baseline_force[i] for i in range(3)))
        for vector in force_vectors
    ]
    delta_torque = [
        _vec_norm(tuple(vector[i] - baseline_torque[i] for i in range(3)))
        for vector in torque_vectors
    ]

    total_high_sec = 0.0
    longest_high_sec = 0.0
    current_high_sec = 0.0
    total_valid_sec = 0.0
    for index in range(len(contact_rows) - 1):
        if references[index + 1] != references[index] + 1:
            current_high_sec = 0.0
            continue
        dt = (timestamps[index + 1] - timestamps[index]) / 1e9
        if dt <= 0:
            continue
        total_valid_sec += dt
        if delta_force[index] >= thresholds.force_high_n:
            total_high_sec += dt
            current_high_sec += dt
            longest_high_sec = max(longest_high_sec, current_high_sec)
        else:
            current_high_sec = 0.0
    high_fraction = total_high_sec / total_valid_sec if total_valid_sec > 0 else 0.0

    terminal_recovery_sec = 0.0
    for index in range(len(contact_rows) - 2, -1, -1):
        if references[index + 1] != references[index] + 1:
            break
        if delta_force[index + 1] > thresholds.force_recover_n:
            break
        dt = (timestamps[index + 1] - timestamps[index]) / 1e9
        if dt <= 0:
            break
        terminal_recovery_sec += dt

    tail_start = t_end - int(thresholds.tail_sec * 1e9)
    tail_force = [
        delta_force[index]
        for index, stamp in enumerate(timestamps)
        if stamp >= tail_start
    ] or [delta_force[-1]]

    max_force = max(delta_force)
    max_torque = max(delta_torque)
    tail_median = float(statistics.median(tail_force))
    tail_max = max(tail_force)
    meaningful_contact = max_force >= thresholds.force_high_n
    clear_recovery = terminal_recovery_sec >= thresholds.force_recovery_sec
    persistent_high = meaningful_contact and not clear_recovery and (
        longest_high_sec >= thresholds.force_persist_sec
        or tail_median >= thresholds.force_high_n
    )
    extreme_force = max_force >= thresholds.force_extreme_n
    extreme_torque = max_torque >= thresholds.torque_extreme_nm

    broad_reasons: list[str] = []
    if persistent_high:
        broad_reasons.append("PERSISTENT_HIGH_FORCE")
    if extreme_force:
        broad_reasons.append("EXTREME_FORCE")
    if extreme_torque:
        broad_reasons.append("EXTREME_TORQUE")

    critical_reasons: list[str] = []
    if persistent_high and not clear_recovery and tail_median >= thresholds.critical_tail_force_n:
        critical_reasons.append("HIGH_FORCE_AT_RELEASE")
    if (
        extreme_force
        and not clear_recovery
        and longest_high_sec >= thresholds.critical_sustained_force_sec
    ):
        critical_reasons.append("SUSTAINED_EXTREME_FORCE")
    if extreme_torque:
        critical_reasons.append("EXTREME_TORQUE")

    if critical_reasons:
        category = "CRITICAL_REVIEW"
    elif broad_reasons:
        category = "REVIEW"
    elif meaningful_contact and clear_recovery:
        category = "HIGH_THEN_RECOVERED"
    else:
        category = "NO_FORCE_FLAG"

    release_row = rows[release_index]
    return ForceScreenResult(
        category=category,
        broad_review_reasons=tuple(broad_reasons),
        critical_review_reasons=tuple(critical_reasons),
        max_delta_force_n=max_force,
        max_delta_torque_nm=max_torque,
        longest_high_force_sec=longest_high_sec,
        total_high_force_sec=total_high_sec,
        high_force_fraction=high_fraction,
        terminal_recovery_sec=terminal_recovery_sec,
        tail_median_delta_force_n=tail_median,
        tail_max_delta_force_n=tail_max,
        max_raw_force_resultant_n=max(raw_force_resultant),
        max_raw_torque_resultant_nm=max(raw_torque_resultant),
        pre_release_duration_sec=(t_end - t0) / 1e9,
        release_source_reference_index=int(release_row["reference_index"]),
        release_target_reference_index=int(release_row["action_target_reference_index"]),
    )


def audit_episode(
    episode_number: int,
    episode_dir: str | Path,
    dataset_root: str | Path,
    thresholds: AuditThresholds,
) -> dict[str, Any]:
    """Audit one real raw episode.  ROS imports are intentionally lazy."""

    episode = Path(episode_dir).resolve()
    dataset = Path(dataset_root).resolve()
    relative = str(episode.relative_to(dataset))
    try:
        from doosan_forcevla_data.convert.doosan_processed_episode_v1 import build_processed_rows
        from doosan_forcevla_data.ingest.doosan_raw_v1 import (
            GRIPPER_STATE_TOPIC,
            JOY_TOPIC,
            GripperStateRecord,
            JoyRecord,
            iter_typed_messages,
        )

        rows, metadata = build_processed_rows(episode)
        orientation = audit_orientation_profiles(rows, thresholds)
        joints = audit_joint_positions(rows, thresholds)

        right_edges_ns: list[int] = []
        raw_holding_samples: list[tuple[int, bool]] = []
        previous_right = 0
        for topic, record in iter_typed_messages(episode):
            if topic == JOY_TOPIC:
                if not isinstance(record, JoyRecord):
                    raise PreConversionAuditError("JOY topic decoded to unexpected record type")
                if len(record.buttons) <= RIGHT_BUTTON_INDEX:
                    raise PreConversionAuditError("Joy record lacks RIGHT button")
                right = int(record.buttons[RIGHT_BUTTON_INDEX])
                if right not in (0, 1):
                    raise PreConversionAuditError("RIGHT button must be binary")
                stamp = record.stamp.header_timestamp_ns
                if stamp is None:
                    raise PreConversionAuditError("Joy record lacks header timestamp")
                if previous_right == 0 and right == 1:
                    right_edges_ns.append(int(stamp))
                previous_right = right
            elif topic == GRIPPER_STATE_TOPIC:
                if not isinstance(record, GripperStateRecord):
                    raise PreConversionAuditError("gripper topic decoded to unexpected record type")
                stamp = record.stamp.header_timestamp_ns
                if stamp is None:
                    raise PreConversionAuditError("gripper record lacks header timestamp")
                raw_holding_samples.append((int(stamp), bool(record.holding)))

        gripper = audit_gripper_right_samples(
            rows,
            right_edges_ns=right_edges_ns,
            raw_holding_samples=raw_holding_samples,
        )
        force_result = screen_force_contact(rows, thresholds)
        structural_issues = [
            *orientation["issues"],
            *joints["issues"],
            *gripper["issues"],
        ]
        return {
            "ok": True,
            "episode_number": int(episode_number),
            "episode": relative,
            "processed_row_count": len(rows),
            "dropped_reference_count": int(metadata.get("dropped_reference_count", 0)),
            "orientation": orientation,
            "joints": joints,
            "gripper_right": gripper,
            "force": asdict(force_result),
            "structural_issues": structural_issues,
        }
    except Exception as exc:
        return {
            "ok": False,
            "episode_number": int(episode_number),
            "episode": relative,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _critical_sort_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    force = result["force"]
    return (
        float(force["tail_median_delta_force_n"]),
        float(force["longest_high_force_sec"]),
        float(force["max_delta_force_n"]),
        float(force["max_delta_torque_nm"]),
    )


def aggregate_results(
    results: Sequence[dict[str, Any]],
    *,
    dataset_root: str | Path,
    workers: int,
    thresholds: AuditThresholds,
) -> dict[str, Any]:
    """Aggregate deterministic population metrics and review tiers."""

    ordered = sorted(results, key=lambda item: int(item["episode_number"]))
    failures = [item for item in ordered if not item.get("ok")]
    good = [item for item in ordered if item.get("ok")]
    structural_review = [item for item in good if item["structural_issues"]]
    critical_force = [item for item in good if item["force"]["critical_review_reasons"]]
    broad_force = [item for item in good if item["force"]["broad_review_reasons"]]
    recovered = [item for item in good if item["force"]["category"] == "HIGH_THEN_RECOVERED"]
    critical_force.sort(key=_critical_sort_key, reverse=True)

    one_right = sum(item["gripper_right"]["right_edge_count"] == 1 for item in good)
    two_right = sum(item["gripper_right"]["right_edge_count"] == 2 for item in good)
    other_right = len(good) - one_right - two_right
    second_effective = sum(
        item["gripper_right"]["right_edge_count"] == 2
        and item["gripper_right"]["effective_right_edge_ordinal"] == 2
        for item in good
    )
    principal_events = sum(
        item["orientation"]["principal_branch_event_count"] for item in good
    )
    quaternion_events = sum(
        item["orientation"]["quaternion_large_jump_count"] for item in good
    )
    continuous_events = sum(
        item["orientation"]["continuous_large_jump_count"] for item in good
    )
    rotation6d_events = sum(
        item["orientation"]["rotation6d_large_jump_count"] for item in good
    )
    joint_wraps = sum(item["joints"]["classic_wrap_candidate_count"] for item in good)

    conversion_gate = "FAIL" if failures else ("REVIEW" if structural_review else "PASS")
    return {
        "audit_schema": AUDIT_SCHEMA,
        "dataset_root": str(Path(dataset_root).resolve()),
        "worker_count": int(workers),
        "available_logical_cpu_count": available_logical_cpu_count(),
        "available_physical_core_count": available_physical_core_count(),
        "available_cpu_count": available_cpu_count(),
        "thresholds": asdict(thresholds),
        "episode_count": len(ordered),
        "successful_episode_count": len(good),
        "failed_episode_count": len(failures),
        "total_processed_rows": sum(item["processed_row_count"] for item in good),
        "structural_review_episode_count": len(structural_review),
        "critical_force_review_count": len(critical_force),
        "broad_force_review_count": len(broad_force),
        "high_then_recovered_count": len(recovered),
        "principal_branch_event_count": principal_events,
        "quaternion_sign_jump_count": quaternion_events,
        "continuous_rotvec_large_jump_count": continuous_events,
        "rotation6d_large_jump_count": rotation6d_events,
        "joint_classic_wrap_candidate_count": joint_wraps,
        "one_right_edge_episode_count": one_right,
        "two_right_edge_episode_count": two_right,
        "other_right_edge_episode_count": other_right,
        "double_right_second_effective_count": second_effective,
        "conversion_gate": conversion_gate,
        "critical_force_review": critical_force,
        "broad_force_review": broad_force,
        "structural_review": structural_review,
        "high_then_recovered": recovered,
        "failures": failures,
        "episodes": good,
    }


def discover_episode_dirs(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise PreConversionAuditError(f"dataset root does not exist: {root}")
    episodes = sorted({path.parent.resolve() for path in root.rglob("episode_validation.json")})
    if not episodes:
        raise PreConversionAuditError(f"no episode_validation.json files found below {root}")
    return episodes


def write_reports(report: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "pre_conversion_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def write_episode_list(name: str, items: Iterable[dict[str, Any]]) -> None:
        (output / name).write_text(
            "".join(f"{item['episode']}\n" for item in items), encoding="utf-8"
        )

    write_episode_list("critical_force_review.txt", report["critical_force_review"])
    write_episode_list("broad_force_review.txt", report["broad_force_review"])
    write_episode_list("structural_review.txt", report["structural_review"])
    write_episode_list("high_then_recovered.txt", report["high_then_recovered"])

    fields = [
        "episode",
        "processed_row_count",
        "dropped_reference_count",
        "structural_issue_count",
        "principal_branch_events",
        "quaternion_sign_jumps",
        "continuous_large_jumps",
        "rotation6d_large_jumps",
        "joint_wrap_candidates",
        "right_edge_count",
        "effective_right_edge_ordinal",
        "force_category",
        "critical_force_reasons",
        "broad_force_reasons",
        "max_delta_force_n",
        "max_delta_torque_nm",
        "longest_high_force_sec",
        "tail_median_delta_force_n",
        "terminal_recovery_sec",
    ]
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in report["episodes"]:
            force = item["force"]
            writer.writerow(
                {
                    "episode": item["episode"],
                    "processed_row_count": item["processed_row_count"],
                    "dropped_reference_count": item["dropped_reference_count"],
                    "structural_issue_count": len(item["structural_issues"]),
                    "principal_branch_events": item["orientation"]["principal_branch_event_count"],
                    "quaternion_sign_jumps": item["orientation"]["quaternion_large_jump_count"],
                    "continuous_large_jumps": item["orientation"]["continuous_large_jump_count"],
                    "rotation6d_large_jumps": item["orientation"]["rotation6d_large_jump_count"],
                    "joint_wrap_candidates": item["joints"]["classic_wrap_candidate_count"],
                    "right_edge_count": item["gripper_right"]["right_edge_count"],
                    "effective_right_edge_ordinal": item["gripper_right"]["effective_right_edge_ordinal"],
                    "force_category": force["category"],
                    "critical_force_reasons": ";".join(force["critical_review_reasons"]),
                    "broad_force_reasons": ";".join(force["broad_review_reasons"]),
                    "max_delta_force_n": force["max_delta_force_n"],
                    "max_delta_torque_nm": force["max_delta_torque_nm"],
                    "longest_high_force_sec": force["longest_high_force_sec"],
                    "tail_median_delta_force_n": force["tail_median_delta_force_n"],
                    "terminal_recovery_sec": force["terminal_recovery_sec"],
                }
            )

    lines = [
        "============================================================",
        "DOOSAN PRE-CONVERSION POPULATION AUDIT",
        "============================================================",
        f"dataset_root={report['dataset_root']}",
        f"worker_count={report['worker_count']}",
        f"available_logical_cpu_count={report['available_logical_cpu_count']}",
        f"available_physical_core_count={report['available_physical_core_count']}",
        f"available_cpu_count={report['available_cpu_count']}",
        f"episode_count={report['episode_count']}",
        f"successful_episode_count={report['successful_episode_count']}",
        f"failed_episode_count={report['failed_episode_count']}",
        f"total_processed_rows={report['total_processed_rows']}",
        "",
        f"structural_review_episode_count={report['structural_review_episode_count']}",
        f"critical_force_review_count={report['critical_force_review_count']}",
        f"broad_force_review_count={report['broad_force_review_count']}",
        f"high_then_recovered_count={report['high_then_recovered_count']}",
        "",
        f"principal_branch_event_count={report['principal_branch_event_count']}",
        f"quaternion_sign_jump_count={report['quaternion_sign_jump_count']}",
        f"continuous_rotvec_large_jump_count={report['continuous_rotvec_large_jump_count']}",
        f"rotation6d_large_jump_count={report['rotation6d_large_jump_count']}",
        f"joint_classic_wrap_candidate_count={report['joint_classic_wrap_candidate_count']}",
        "",
        f"one_right_edge_episode_count={report['one_right_edge_episode_count']}",
        f"two_right_edge_episode_count={report['two_right_edge_episode_count']}",
        f"other_right_edge_episode_count={report['other_right_edge_episode_count']}",
        f"double_right_second_effective_count={report['double_right_second_effective_count']}",
        "",
        f"PRECONVERSION_CONVERSION_GATE={report['conversion_gate']}",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only CPU-parallel Doosan pre-conversion population audit."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--workers",
        default="auto",
        help="'auto' (default) uses up to 4 available physical CPU cores; an integer may explicitly request more, up to the available physical-core count.",
    )
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--force-high-n", type=float, default=10.0)
    parser.add_argument("--force-recover-n", type=float, default=5.0)
    parser.add_argument("--force-extreme-n", type=float, default=25.0)
    parser.add_argument("--torque-extreme-nm", type=float, default=5.0)
    parser.add_argument("--force-persist-sec", type=float, default=0.75)
    parser.add_argument("--force-recovery-sec", type=float, default=0.25)
    parser.add_argument("--baseline-sec", type=float, default=1.0)
    parser.add_argument("--tail-sec", type=float, default=0.50)
    parser.add_argument("--critical-tail-force-n", type=float, default=30.0)
    parser.add_argument("--critical-sustained-force-sec", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    thresholds = AuditThresholds(
        force_high_n=args.force_high_n,
        force_recover_n=args.force_recover_n,
        force_extreme_n=args.force_extreme_n,
        torque_extreme_nm=args.torque_extreme_nm,
        force_persist_sec=args.force_persist_sec,
        force_recovery_sec=args.force_recovery_sec,
        baseline_sec=args.baseline_sec,
        tail_sec=args.tail_sec,
        critical_tail_force_n=args.critical_tail_force_n,
        critical_sustained_force_sec=args.critical_sustained_force_sec,
    )
    try:
        workers = resolve_worker_count(args.workers)
        episodes = discover_episode_dirs(args.dataset_root)
    except (ValueError, PreConversionAuditError) as exc:
        print(f"PRECONVERSION_AUDIT_SETUP=FAIL: {exc}")
        return 2
    if args.expected_episodes is not None and len(episodes) != args.expected_episodes:
        print(
            f"PRECONVERSION_AUDIT_SETUP=FAIL: discovered {len(episodes)} episodes, "
            f"expected {args.expected_episodes}"
        )
        return 2

    jobs = [(index, path) for index, path in enumerate(episodes, start=1)]
    print(f"available_logical_cpu_count={available_logical_cpu_count()}")
    print(f"available_physical_core_count={available_physical_core_count()}")
    print(f"available_cpu_count={available_cpu_count()}")
    print(f"worker_count={workers}")
    print(f"episode_count={len(episodes)}")
    print("parallel_backend=ProcessPoolExecutor(spawn)")
    print("parallel_unit=episode")
    print()
    print("===== SEQUENTIAL PREFLIGHT =====")
    first = audit_episode(1, jobs[0][1], args.dataset_root, thresholds)
    if not first.get("ok"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "preflight_failure.json").write_text(
            json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[001/{len(episodes):03d}] PREFLIGHT FAIL {first['episode']}: "
            f"{first['error_type']}: {first['error']}"
        )
        return 1
    print(
        f"[001/{len(episodes):03d}] PREFLIGHT PASS {first['episode']} "
        f"force={first['force']['category']} structural={len(first['structural_issues'])}"
    )

    results = [first]
    completed = 1
    if len(jobs) > 1:
        print()
        print("===== PARALLEL POPULATION =====")
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            future_map = {
                executor.submit(
                    audit_episode, index, path, args.dataset_root, thresholds
                ): (index, path)
                for index, path in jobs[1:]
            }
            for future in concurrent.futures.as_completed(future_map):
                index, path = future_map[future]
                try:
                    result = future.result()
                except BaseException as exc:  # pragma: no cover - executor boundary
                    result = {
                        "ok": False,
                        "episode_number": index,
                        "episode": str(Path(path).resolve().relative_to(args.dataset_root.resolve())),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                results.append(result)
                completed += 1
                if result.get("ok"):
                    print(
                        f"[{completed:03d}/{len(episodes):03d}] PASS {result['episode']} "
                        f"force={result['force']['category']} "
                        f"structural={len(result['structural_issues'])}"
                    )
                else:
                    print(
                        f"[{completed:03d}/{len(episodes):03d}] FAIL {result['episode']} "
                        f"{result['error_type']}: {result['error']}"
                    )

    report = aggregate_results(
        results,
        dataset_root=args.dataset_root,
        workers=workers,
        thresholds=thresholds,
    )
    write_reports(report, args.output_dir)
    print()
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"), end="")
    print()
    print("===== CRITICAL FORCE REVIEW =====")
    if report["critical_force_review"]:
        for number, item in enumerate(report["critical_force_review"], start=1):
            force = item["force"]
            print(
                f"{number:02d}. {item['episode']} | "
                f"{','.join(force['critical_review_reasons'])} | "
                f"max_dF={force['max_delta_force_n']:.2f}N | "
                f"longest_high={force['longest_high_force_sec']:.2f}s | "
                f"tail={force['tail_median_delta_force_n']:.2f}N | "
                f"recovery={force['terminal_recovery_sec']:.2f}s"
            )
    else:
        print("NONE")
    print("PRECONVERSION_AUDIT_EXECUTION_GATE=" + ("PASS" if not report["failures"] else "FAIL"))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
