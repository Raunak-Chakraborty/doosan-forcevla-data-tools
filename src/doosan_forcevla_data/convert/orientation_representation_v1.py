"""Model-facing SO(3) orientation representations for Doosan experiments.

Patch 11A introduces one authoritative, dependency-free orientation layer.
It deliberately does *not* change the existing processed-v1 state layout or
LeRobot export defaults.  Later integration can select among the representations
here without changing the underlying physical rotation.

Canonical physical input
------------------------
Every encoder consumes a proper 3x3 base-to-TCP rotation matrix ``R_base_tcp``.

Experimental representations
----------------------------
``rotvec_principal``
    Principal SO(3) logarithm, 3 values, angle in ``[0, pi]``.

``rotvec_continuous``
    A temporally lifted equivalent rotation vector, 3 values.  The first frame
    is principal.  Later frames select the ``2*pi``-equivalent vector closest
    to the previous lifted vector.  This is stateful by construction.

``quaternion``
    Deterministic unit quaternion in ``wxyz`` order, 4 values.  The sign is
    canonicalized with positive ``w`` away from the 180-degree boundary and a
    lexicographic vector-part tie break when ``w`` is numerically zero.

``rotation6d``
    First two *columns* of ``R_base_tcp``, 6 values, explicitly flattened as
    ``[R00, R10, R20, R01, R11, R21]``.

The measured 7D action contract remains independent of these observation
representations: rotational actions stay 3D spatial relative rotation vectors.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Sequence


TWO_PI = 2.0 * math.pi
_ROTATION_TOL = 1e-9
_SMALL_NORM = 1e-15
_QUATERNION_PI_TIE_TOL = 1e-12

BASE_STATE_DIM_WITHOUT_ORIENTATION_OR_WRENCH = 16
WRENCH_DIM = 6


class OrientationRepresentationError(ValueError):
    """Raised when an orientation representation is invalid or ambiguous."""


class OrientationRepresentation(str, Enum):
    """Supported model-facing absolute TCP orientation representations."""

    ROTVEC_PRINCIPAL = "rotvec_principal"
    ROTVEC_CONTINUOUS = "rotvec_continuous"
    QUATERNION = "quaternion"
    ROTATION6D = "rotation6d"


def resolve_orientation_representation(
    value: OrientationRepresentation | str,
) -> OrientationRepresentation:
    """Return a validated :class:`OrientationRepresentation` value."""

    if isinstance(value, OrientationRepresentation):
        return value
    if not isinstance(value, str):
        raise OrientationRepresentationError(
            "orientation representation must be a string or OrientationRepresentation"
        )
    try:
        return OrientationRepresentation(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in OrientationRepresentation)
        raise OrientationRepresentationError(
            f"unsupported orientation representation {value!r}; expected one of: {allowed}"
        ) from exc


def orientation_dimension(
    representation: OrientationRepresentation | str,
) -> int:
    """Return the number of scalar channels for one orientation representation."""

    resolved = resolve_orientation_representation(representation)
    if resolved in {
        OrientationRepresentation.ROTVEC_PRINCIPAL,
        OrientationRepresentation.ROTVEC_CONTINUOUS,
    }:
        return 3
    if resolved is OrientationRepresentation.QUATERNION:
        return 4
    if resolved is OrientationRepresentation.ROTATION6D:
        return 6
    raise AssertionError(f"unhandled orientation representation: {resolved}")


def model_state_dimension(
    representation: OrientationRepresentation | str,
    *,
    include_wrench: bool,
) -> int:
    """Return the planned Doosan model-facing state width.

    The non-orientation, non-wrench portion is frozen at 16 channels:
    TCP position (3) + gripper (1) + joint position (6) + joint velocity (6).
    The optional wrench contributes the final 6 channels.
    """

    if not isinstance(include_wrench, bool):
        raise OrientationRepresentationError("include_wrench must be boolean")
    return (
        BASE_STATE_DIM_WITHOUT_ORIENTATION_OR_WRENCH
        + orientation_dimension(representation)
        + (WRENCH_DIM if include_wrench else 0)
    )


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise OrientationRepresentationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OrientationRepresentationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise OrientationRepresentationError(f"{name} must be finite")
    return result


def _finite_vector(
    values: Sequence[float],
    expected_len: int,
    name: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise OrientationRepresentationError(
            f"{name} must contain {expected_len} finite numbers"
        )
    try:
        result = tuple(
            _finite_scalar(value, f"{name}[{index}]")
            for index, value in enumerate(values)
        )
    except TypeError as exc:
        raise OrientationRepresentationError(
            f"{name} must contain {expected_len} finite numbers"
        ) from exc
    if len(result) != expected_len:
        raise OrientationRepresentationError(
            f"{name} must have length {expected_len}, got {len(result)}"
        )
    return result


def _matmul3(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            math.fsum(float(left[row][k]) * float(right[k][col]) for k in range(3))
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


def validate_rotation_matrix(
    matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    """Validate and normalize a numeric 3x3 proper SO(3) matrix."""

    if isinstance(matrix, (str, bytes)):
        raise OrientationRepresentationError("rotation matrix must be 3x3")
    try:
        normalized = tuple(
            tuple(_finite_scalar(value, "rotation matrix") for value in row)
            for row in matrix
        )
    except TypeError as exc:
        raise OrientationRepresentationError("rotation matrix must be 3x3") from exc
    if len(normalized) != 3 or any(len(row) != 3 for row in normalized):
        raise OrientationRepresentationError("rotation matrix must be 3x3")
    identity = _matmul3(_transpose3(normalized), normalized)
    max_orthogonality_error = max(
        abs(identity[row][col] - (1.0 if row == col else 0.0))
        for row in range(3)
        for col in range(3)
    )
    determinant = _determinant3(normalized)
    if (
        max_orthogonality_error > _ROTATION_TOL
        or abs(determinant - 1.0) > _ROTATION_TOL
    ):
        raise OrientationRepresentationError(
            "matrix is not a proper SO(3) rotation: "
            f"orthogonality_error={max_orthogonality_error:.3e}, det={determinant:.15g}"
        )
    return normalized


def matrix_to_principal_rotvec(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """Return the principal SO(3) logarithm as a rotation vector in radians.

    This implementation intentionally preserves the existing Patch-5 numerical
    convention so the current 25D production state remains unchanged.
    """

    r = validate_rotation_matrix(matrix)
    trace = r[0][0] + r[1][1] + r[2][2]

    if trace > 0.0:
        s = 2.0 * math.sqrt(max(trace + 1.0, 0.0))
        if s <= 0.0:  # pragma: no cover - guarded by trace > 0
            raise OrientationRepresentationError(
                "failed to convert rotation matrix to quaternion"
            )
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
        qw = 0.25 * s
    elif r[0][0] >= r[1][1] and r[0][0] >= r[2][2]:
        s = 2.0 * math.sqrt(max(1.0 + r[0][0] - r[1][1] - r[2][2], 0.0))
        if s <= 0.0:
            raise OrientationRepresentationError("failed to convert rotation matrix near pi")
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
        qw = (r[2][1] - r[1][2]) / s
    elif r[1][1] >= r[2][2]:
        s = 2.0 * math.sqrt(max(1.0 + r[1][1] - r[0][0] - r[2][2], 0.0))
        if s <= 0.0:
            raise OrientationRepresentationError("failed to convert rotation matrix near pi")
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
        qw = (r[0][2] - r[2][0]) / s
    else:
        s = 2.0 * math.sqrt(max(1.0 + r[2][2] - r[0][0] - r[1][1], 0.0))
        if s <= 0.0:
            raise OrientationRepresentationError("failed to convert rotation matrix near pi")
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s
        qw = (r[1][0] - r[0][1]) / s

    qnorm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(qnorm) or qnorm <= 0.0:
        raise OrientationRepresentationError(
            "rotation matrix produced an invalid quaternion"
        )
    qx, qy, qz, qw = (value / qnorm for value in (qx, qy, qz, qw))

    # Preserve the established principal-angle convention: choose non-negative w.
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    vector_norm = math.sqrt(qx * qx + qy * qy + qz * qz)
    if vector_norm < _SMALL_NORM:
        return (0.0, 0.0, 0.0)

    angle = 2.0 * math.atan2(vector_norm, max(qw, 0.0))
    if angle > math.pi and angle - math.pi < 1e-12:
        angle = math.pi
    if not 0.0 <= angle <= math.pi + 1e-12:
        raise OrientationRepresentationError(
            f"principal rotation angle out of range: {angle}"
        )
    scale = angle / vector_norm
    return (qx * scale, qy * scale, qz * scale)


def rotvec_to_matrix(
    rotvec_rad: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Rodrigues exponential map for any finite 3D rotation vector."""

    x, y, z = _finite_vector(rotvec_rad, 3, "rotation vector")
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < _SMALL_NORM:
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
    return validate_rotation_matrix(matrix)


def _quaternion_from_matrix_raw_wxyz(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    r = validate_rotation_matrix(matrix)
    trace = r[0][0] + r[1][1] + r[2][2]

    if trace > 0.0:
        s = 2.0 * math.sqrt(max(trace + 1.0, 0.0))
        if s <= 0.0:  # pragma: no cover
            raise OrientationRepresentationError(
                "failed to convert rotation matrix to quaternion"
            )
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
        qw = 0.25 * s
    elif r[0][0] >= r[1][1] and r[0][0] >= r[2][2]:
        s = 2.0 * math.sqrt(max(1.0 + r[0][0] - r[1][1] - r[2][2], 0.0))
        if s <= 0.0:
            raise OrientationRepresentationError("failed to convert rotation matrix near pi")
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
        qw = (r[2][1] - r[1][2]) / s
    elif r[1][1] >= r[2][2]:
        s = 2.0 * math.sqrt(max(1.0 + r[1][1] - r[0][0] - r[2][2], 0.0))
        if s <= 0.0:
            raise OrientationRepresentationError("failed to convert rotation matrix near pi")
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
        qw = (r[0][2] - r[2][0]) / s
    else:
        s = 2.0 * math.sqrt(max(1.0 + r[2][2] - r[0][0] - r[1][1], 0.0))
        if s <= 0.0:
            raise OrientationRepresentationError("failed to convert rotation matrix near pi")
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s
        qw = (r[1][0] - r[0][1]) / s

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm <= 0.0:
        raise OrientationRepresentationError(
            "rotation matrix produced an invalid quaternion"
        )
    return (qw / norm, qx / norm, qy / norm, qz / norm)


def matrix_to_quaternion_wxyz(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """Return one deterministic unit quaternion in explicit ``wxyz`` order.

    ``q`` and ``-q`` encode the same rotation.  The canonical sign rule is:

    1. if ``|w| > 1e-12``, require ``w > 0``;
    2. otherwise, require the first vector component among ``x, y, z`` whose
       magnitude exceeds ``1e-12`` to be positive.

    The second rule makes the exact 180-degree boundary deterministic while
    documenting the unavoidable quaternion discontinuity there.
    """

    qw, qx, qy, qz = _quaternion_from_matrix_raw_wxyz(matrix)

    flip = False
    if qw < -_QUATERNION_PI_TIE_TOL:
        flip = True
    elif abs(qw) <= _QUATERNION_PI_TIE_TOL:
        for component in (qx, qy, qz):
            if abs(component) > _QUATERNION_PI_TIE_TOL:
                flip = component < 0.0
                break

    if flip:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz

    return (qw, qx, qy, qz)


def quaternion_wxyz_to_matrix(
    quaternion_wxyz: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Convert a finite nonzero ``wxyz`` quaternion to a proper rotation matrix."""

    qw, qx, qy, qz = _finite_vector(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= _SMALL_NORM:
        raise OrientationRepresentationError("quaternion_wxyz must be nonzero")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    matrix = (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )
    return validate_rotation_matrix(matrix)


def matrix_to_rotation6d(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float, float, float, float]:
    """Return the first two matrix columns as explicit column-major 6D data."""

    r = validate_rotation_matrix(matrix)
    return (
        r[0][0],
        r[1][0],
        r[2][0],
        r[0][1],
        r[1][1],
        r[2][1],
    )


def _dot3(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def _norm3(values: Sequence[float]) -> float:
    return math.sqrt(_dot3(values, values))


def _cross3(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def rotation6d_to_matrix(
    rotation6d: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Reconstruct a rotation matrix from the 6D two-column representation.

    Valid model inputs produced by :func:`matrix_to_rotation6d` round-trip
    exactly up to floating-point error.  For arbitrary finite 6D inputs this
    applies the standard Gram-Schmidt projection used by 6D rotation encodings.
    """

    values = _finite_vector(rotation6d, 6, "rotation6d")
    a1 = values[0:3]
    a2 = values[3:6]

    norm1 = _norm3(a1)
    if norm1 <= _SMALL_NORM:
        raise OrientationRepresentationError(
            "rotation6d first column candidate must be nonzero"
        )
    b1 = tuple(value / norm1 for value in a1)

    projection = _dot3(b1, a2)
    a2_orthogonal = tuple(
        a2[index] - projection * b1[index]
        for index in range(3)
    )
    norm2 = _norm3(a2_orthogonal)
    if norm2 <= _SMALL_NORM:
        raise OrientationRepresentationError(
            "rotation6d column candidates must not be collinear"
        )
    b2 = tuple(value / norm2 for value in a2_orthogonal)
    b3 = _cross3(b1, b2)

    matrix = (
        (b1[0], b2[0], b3[0]),
        (b1[1], b2[1], b3[1]),
        (b1[2], b2[2], b3[2]),
    )
    return validate_rotation_matrix(matrix)


def lift_continuous_rotvec(
    matrix: Sequence[Sequence[float]],
    previous_rotvec_rad: Sequence[float] | None,
) -> tuple[float, float, float]:
    """Choose the SO(3)-equivalent rotvec closest to the previous lifted value.

    The first frame (``previous_rotvec_rad is None``) is the principal rotvec.
    For non-identity rotations, all candidates ``(theta + 2*pi*k) * axis`` near
    the previous projection are considered.  At exact identity, the rotation
    axis is undefined, so a previous nonzero axis is preserved and the nearest
    integer multiple of ``2*pi`` is selected on that axis.
    """

    principal = matrix_to_principal_rotvec(matrix)
    if previous_rotvec_rad is None:
        return principal

    previous = _finite_vector(previous_rotvec_rad, 3, "previous_rotvec_rad")
    theta = _norm3(principal)

    if theta < _SMALL_NORM:
        previous_norm = _norm3(previous)
        if previous_norm < _SMALL_NORM:
            return (0.0, 0.0, 0.0)
        axis = tuple(value / previous_norm for value in previous)
        winding = int(round(previous_norm / TWO_PI))
        magnitude = TWO_PI * winding
        return tuple(magnitude * value for value in axis)

    axis = tuple(value / theta for value in principal)
    previous_projection = _dot3(previous, axis)
    center = int(round((previous_projection - theta) / TWO_PI))

    candidates: list[tuple[float, float, float]] = []
    candidate_windings: list[int] = []
    for winding in range(center - 2, center + 3):
        magnitude = theta + TWO_PI * winding
        candidates.append(tuple(magnitude * value for value in axis))
        candidate_windings.append(winding)

    def key(index: int) -> tuple[float, int, int]:
        candidate = candidates[index]
        squared_distance = math.fsum(
            (candidate[axis_index] - previous[axis_index]) ** 2
            for axis_index in range(3)
        )
        winding = candidate_windings[index]
        return (squared_distance, abs(winding), winding)

    best_index = min(range(len(candidates)), key=key)
    return candidates[best_index]


def encode_orientation(
    matrix: Sequence[Sequence[float]],
    representation: OrientationRepresentation | str,
    *,
    previous_continuous_rotvec_rad: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Encode one physical rotation using the requested representation."""

    resolved = resolve_orientation_representation(representation)
    if resolved is OrientationRepresentation.ROTVEC_PRINCIPAL:
        return matrix_to_principal_rotvec(matrix)
    if resolved is OrientationRepresentation.ROTVEC_CONTINUOUS:
        return lift_continuous_rotvec(matrix, previous_continuous_rotvec_rad)
    if resolved is OrientationRepresentation.QUATERNION:
        return matrix_to_quaternion_wxyz(matrix)
    if resolved is OrientationRepresentation.ROTATION6D:
        return matrix_to_rotation6d(matrix)
    raise AssertionError(f"unhandled orientation representation: {resolved}")


def decode_orientation(
    encoded: Sequence[float],
    representation: OrientationRepresentation | str,
) -> tuple[tuple[float, float, float], ...]:
    """Reconstruct the represented physical rotation matrix for validation."""

    resolved = resolve_orientation_representation(representation)
    if resolved in {
        OrientationRepresentation.ROTVEC_PRINCIPAL,
        OrientationRepresentation.ROTVEC_CONTINUOUS,
    }:
        return rotvec_to_matrix(encoded)
    if resolved is OrientationRepresentation.QUATERNION:
        return quaternion_wxyz_to_matrix(encoded)
    if resolved is OrientationRepresentation.ROTATION6D:
        return rotation6d_to_matrix(encoded)
    raise AssertionError(f"unhandled orientation representation: {resolved}")
