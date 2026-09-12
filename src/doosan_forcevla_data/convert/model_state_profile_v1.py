"""Representation-aware model-state profiles for Doosan LeRobot export.

Patch 11B keeps the canonical processed-v1 episode frozen at the validated
legacy 25D principal-rotvec state.  This module derives model-facing variants
from that canonical state at export time only.

The non-orientation ordering is invariant across profiles::

    tcp_position (3)
    orientation (3 / 4 / 6)
    gripper_open_fraction (1)
    joint_position (6)
    joint_velocity (6)
    [optional wrench (6), always final]

The semantic 7D action is intentionally outside this module and remains
unchanged for every observation representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Sequence

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    OBSERVATION_STATE_DIM,
    OBSERVATION_STATE_FIELDS,
)
from doosan_forcevla_data.convert.orientation_representation_v1 import (
    OrientationRepresentation,
    OrientationRepresentationError,
    encode_orientation,
    model_state_dimension,
    orientation_dimension,
    resolve_orientation_representation,
    rotvec_to_matrix,
)


MODEL_STATE_PROFILE_SCHEMA_ID = "doosan_model_state_profile_v1"
LEGACY_STATE_DIM = OBSERVATION_STATE_DIM
LEGACY_ORIENTATION_SLICE = slice(3, 6)
LEGACY_GRIPPER_INDEX = 6
LEGACY_JOINT_POSITION_SLICE = slice(7, 13)
LEGACY_JOINT_VELOCITY_SLICE = slice(13, 19)
LEGACY_WRENCH_SLICE = slice(19, 25)

_POSITION_FIELDS = ("tcp_x_m", "tcp_y_m", "tcp_z_m")
_GRIPPER_FIELDS = ("gripper_open_fraction",)
_JOINT_POSITION_FIELDS = tuple(
    f"joint_{index}_position_rad" for index in range(1, 7)
)
_JOINT_VELOCITY_FIELDS = tuple(
    f"joint_{index}_velocity_rad_s" for index in range(1, 7)
)
_WRENCH_FIELDS = (
    "force_x_n",
    "force_y_n",
    "force_z_n",
    "torque_x_nm",
    "torque_y_nm",
    "torque_z_nm",
)

_ORIENTATION_FIELDS = {
    OrientationRepresentation.ROTVEC_PRINCIPAL: (
        "tcp_rotvec_x_rad",
        "tcp_rotvec_y_rad",
        "tcp_rotvec_z_rad",
    ),
    OrientationRepresentation.ROTVEC_CONTINUOUS: (
        "tcp_rotvec_continuous_x_rad",
        "tcp_rotvec_continuous_y_rad",
        "tcp_rotvec_continuous_z_rad",
    ),
    OrientationRepresentation.QUATERNION: (
        "tcp_quaternion_w",
        "tcp_quaternion_x",
        "tcp_quaternion_y",
        "tcp_quaternion_z",
    ),
    OrientationRepresentation.ROTATION6D: (
        "tcp_rotation6d_col0_x",
        "tcp_rotation6d_col0_y",
        "tcp_rotation6d_col0_z",
        "tcp_rotation6d_col1_x",
        "tcp_rotation6d_col1_y",
        "tcp_rotation6d_col1_z",
    ),
}


class ModelStateProfileError(ValueError):
    """Raised when a model-facing state profile is invalid or inconsistent."""


class StateMode(str, Enum):
    """Whether the exported observation state contains the final 6D wrench."""

    NO_WRENCH = "no_wrench"
    FULL = "full"


def resolve_state_mode(value: StateMode | str) -> StateMode:
    if isinstance(value, StateMode):
        return value
    if not isinstance(value, str):
        raise ModelStateProfileError("state mode must be a string or StateMode")
    try:
        return StateMode(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in StateMode)
        raise ModelStateProfileError(
            f"unsupported state mode {value!r}; expected one of: {allowed}"
        ) from exc


@dataclass(frozen=True)
class ModelStateLayout:
    """Frozen channel layout for one model-facing observation state."""

    orientation_representation: OrientationRepresentation
    state_mode: StateMode
    state_dim: int
    state_fields: tuple[str, ...]
    orientation_slice: slice
    gripper_index: int
    joint_position_slice: slice
    joint_velocity_slice: slice
    wrench_slice: slice | None

    @property
    def include_wrench(self) -> bool:
        return self.state_mode is StateMode.FULL

    @property
    def profile_id(self) -> str:
        return (
            f"doosan_{self.state_mode.value}_"
            f"{self.orientation_representation.value}_v1"
        )

    @property
    def is_legacy_default(self) -> bool:
        return (
            self.orientation_representation
            is OrientationRepresentation.ROTVEC_PRINCIPAL
            and self.state_mode is StateMode.FULL
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_STATE_PROFILE_SCHEMA_ID,
            "profile_id": self.profile_id,
            "orientation_representation": self.orientation_representation.value,
            "state_mode": self.state_mode.value,
            "include_wrench": self.include_wrench,
            "state_dim": self.state_dim,
            "state_fields": list(self.state_fields),
            "source_processed_state": {
                "schema": "legacy_principal_rotvec_25d",
                "state_dim": LEGACY_STATE_DIM,
                "field": "observation_state_25d",
            },
            "orientation_convention": orientation_convention_metadata(
                self.orientation_representation
            ),
            "wrench_policy": (
                "final_six_channels" if self.include_wrench else "omitted"
            ),
        }


def orientation_convention_metadata(
    representation: OrientationRepresentation | str,
) -> dict[str, Any]:
    resolved = resolve_orientation_representation(representation)
    common: dict[str, Any] = {
        "physical_rotation": "R_base_tcp",
        "frame": "base_to_tcp",
    }
    if resolved is OrientationRepresentation.ROTVEC_PRINCIPAL:
        return {
            **common,
            "encoding": "principal_so3_logarithm",
            "units": "radian",
            "dimension": 3,
        }
    if resolved is OrientationRepresentation.ROTVEC_CONTINUOUS:
        return {
            **common,
            "encoding": "temporally_lifted_equivalent_rotvec",
            "units": "radian",
            "dimension": 3,
            "continuity_sequence": "exported_training_row_order",
        }
    if resolved is OrientationRepresentation.QUATERNION:
        return {
            **common,
            "encoding": "deterministic_unit_quaternion",
            "ordering": "wxyz",
            "dimension": 4,
            "sign_policy": "positive_w_then_lexicographic_vector_tie_break_at_pi",
        }
    if resolved is OrientationRepresentation.ROTATION6D:
        return {
            **common,
            "encoding": "first_two_rotation_matrix_columns",
            "flattening": ["R00", "R10", "R20", "R01", "R11", "R21"],
            "dimension": 6,
        }
    raise AssertionError(f"unhandled orientation representation: {resolved}")


def model_state_layout(
    orientation_representation: OrientationRepresentation | str = (
        OrientationRepresentation.ROTVEC_PRINCIPAL
    ),
    state_mode: StateMode | str = StateMode.FULL,
) -> ModelStateLayout:
    representation = resolve_orientation_representation(orientation_representation)
    mode = resolve_state_mode(state_mode)
    include_wrench = mode is StateMode.FULL
    orientation_width = orientation_dimension(representation)

    orientation_start = 3
    orientation_stop = orientation_start + orientation_width
    gripper_index = orientation_stop
    joint_position_start = gripper_index + 1
    joint_position_stop = joint_position_start + 6
    joint_velocity_start = joint_position_stop
    joint_velocity_stop = joint_velocity_start + 6
    wrench_slice = (
        slice(joint_velocity_stop, joint_velocity_stop + 6)
        if include_wrench
        else None
    )

    fields = (
        *_POSITION_FIELDS,
        *_ORIENTATION_FIELDS[representation],
        *_GRIPPER_FIELDS,
        *_JOINT_POSITION_FIELDS,
        *_JOINT_VELOCITY_FIELDS,
        *(_WRENCH_FIELDS if include_wrench else ()),
    )
    expected_dim = model_state_dimension(
        representation,
        include_wrench=include_wrench,
    )
    if len(fields) != expected_dim:  # pragma: no cover - module invariant
        raise RuntimeError("model-state field layout is inconsistent")

    layout = ModelStateLayout(
        orientation_representation=representation,
        state_mode=mode,
        state_dim=expected_dim,
        state_fields=tuple(fields),
        orientation_slice=slice(orientation_start, orientation_stop),
        gripper_index=gripper_index,
        joint_position_slice=slice(joint_position_start, joint_position_stop),
        joint_velocity_slice=slice(joint_velocity_start, joint_velocity_stop),
        wrench_slice=wrench_slice,
    )

    if layout.is_legacy_default and layout.state_fields != tuple(OBSERVATION_STATE_FIELDS):
        raise RuntimeError("legacy default state fields changed")
    return layout


def _finite_legacy_state(
    value: Sequence[float],
    *,
    context: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ModelStateProfileError(
            f"{context}: expected {LEGACY_STATE_DIM} finite values"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ModelStateProfileError(
            f"{context}: expected {LEGACY_STATE_DIM} finite values"
        ) from exc
    if len(items) != LEGACY_STATE_DIM:
        raise ModelStateProfileError(
            f"{context}: expected {LEGACY_STATE_DIM} values, got {len(items)}"
        )

    result: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool):
            raise ModelStateProfileError(f"{context}[{index}]: expected finite number")
        try:
            converted = float(item)
        except (TypeError, ValueError) as exc:
            raise ModelStateProfileError(
                f"{context}[{index}]: expected finite number"
            ) from exc
        if not math.isfinite(converted):
            raise ModelStateProfileError(f"{context}[{index}]: expected finite number")
        result.append(converted)
    return tuple(result)


def encode_legacy_observation_states(
    states_25d: Sequence[Sequence[float]],
    *,
    layout: ModelStateLayout,
) -> list[tuple[float, ...]]:
    """Convert canonical processed 25D states to one model-facing profile.

    Principal/full is an exact pass-through after finite/width validation so the
    historical export values remain numerically identical.  Every other
    representation reconstructs the physical ``R_base_tcp`` from the canonical
    principal rotvec and re-encodes only the orientation channels.
    """

    if not isinstance(layout, ModelStateLayout):
        raise ModelStateProfileError("layout must be ModelStateLayout")

    result: list[tuple[float, ...]] = []
    previous_continuous: tuple[float, ...] | None = None

    for index, raw_state in enumerate(states_25d):
        state = _finite_legacy_state(raw_state, context=f"state[{index}]")

        if (
            layout.orientation_representation
            is OrientationRepresentation.ROTVEC_PRINCIPAL
        ):
            # Principal-rotvec profiles are exact canonical-state projections.
            # Do not round-trip through SO(3), which would introduce needless
            # floating-point changes into an otherwise identical baseline.
            converted = state if layout.include_wrench else state[:19]
        else:
            principal_rotvec = state[LEGACY_ORIENTATION_SLICE]
            try:
                matrix = rotvec_to_matrix(principal_rotvec)
                encoded_orientation = encode_orientation(
                    matrix,
                    layout.orientation_representation,
                    previous_continuous_rotvec_rad=previous_continuous,
                )
            except OrientationRepresentationError as exc:
                raise ModelStateProfileError(
                    f"state[{index}]: invalid canonical orientation: {exc}"
                ) from exc

            if (
                layout.orientation_representation
                is OrientationRepresentation.ROTVEC_CONTINUOUS
            ):
                previous_continuous = tuple(encoded_orientation)

            converted = (
                *state[0:3],
                *encoded_orientation,
                state[LEGACY_GRIPPER_INDEX],
                *state[LEGACY_JOINT_POSITION_SLICE],
                *state[LEGACY_JOINT_VELOCITY_SLICE],
                *(
                    state[LEGACY_WRENCH_SLICE]
                    if layout.include_wrench
                    else ()
                ),
            )

        if len(converted) != layout.state_dim:  # pragma: no cover - invariant
            raise ModelStateProfileError(
                f"state[{index}]: converted width {len(converted)} != {layout.state_dim}"
            )
        if not all(math.isfinite(value) for value in converted):  # pragma: no cover
            raise ModelStateProfileError(f"state[{index}]: converted state is non-finite")
        result.append(tuple(float(value) for value in converted))

    return result


__all__ = [
    "LEGACY_STATE_DIM",
    "MODEL_STATE_PROFILE_SCHEMA_ID",
    "ModelStateLayout",
    "ModelStateProfileError",
    "StateMode",
    "encode_legacy_observation_states",
    "model_state_layout",
    "orientation_convention_metadata",
    "resolve_state_mode",
]
