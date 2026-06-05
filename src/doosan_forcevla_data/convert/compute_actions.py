"""Pure-Python action computation from measured TCP state."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence


def _as_float_list(values: Sequence[float], expected_len: int, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of length {expected_len}")
    if len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(values)}")

    floats: list[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name}[{idx}] must be a finite float")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{idx}] must be finite")
        floats.append(number)
    return floats


def _as_optional_scalar(value: float | Sequence[float], name: str) -> float:
    if isinstance(value, (list, tuple)):
        values = _as_float_list(value, 1, name)
        return values[0]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite float or a one-element list/tuple")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def normalize_quat_xyzw(q: Sequence[float]) -> list[float]:
    """Return a unit quaternion in xyzw convention."""

    quat = _as_float_list(q, 4, "q")
    norm = math.sqrt(sum(component * component for component in quat))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("q must have non-zero finite norm")
    return [component / norm for component in quat]


def quat_conjugate_xyzw(q: Sequence[float]) -> list[float]:
    """Return the quaternion conjugate in xyzw convention."""

    x, y, z, w = _as_float_list(q, 4, "q")
    return [-x, -y, -z, w]


def quat_multiply_xyzw(q1: Sequence[float], q2: Sequence[float]) -> list[float]:
    """Multiply two quaternions in xyzw convention."""

    x1, y1, z1, w1 = _as_float_list(q1, 4, "q1")
    x2, y2, z2, w2 = _as_float_list(q2, 4, "q2")

    result = [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("quaternion multiplication produced a non-finite value")
    return result


def quat_to_rotvec_xyzw(q: Sequence[float]) -> list[float]:
    """Convert a quaternion in xyzw convention to a rotation vector."""

    x, y, z, w = normalize_quat_xyzw(q)

    # q and -q encode the same rotation. Prefer the shortest rotation vector.
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w

    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm < 1e-12:
        return [2.0 * x, 2.0 * y, 2.0 * z]

    angle = 2.0 * math.atan2(vector_norm, w)
    scale = angle / vector_norm
    result = [x * scale, y * scale, z * scale]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("rotation-vector conversion produced a non-finite value")
    return result


def relative_rotvec_xyzw(q_t: Sequence[float], q_t1: Sequence[float]) -> list[float]:
    """Return the relative rotation vector from q_t to q_t1."""

    q0 = normalize_quat_xyzw(q_t)
    q1 = normalize_quat_xyzw(q_t1)
    q_rel = quat_multiply_xyzw(quat_conjugate_xyzw(q0), q1)
    return quat_to_rotvec_xyzw(q_rel)


def compute_measured_tcp_delta_action(
    pos_t: Sequence[float],
    quat_t: Sequence[float],
    pos_t1: Sequence[float],
    quat_t1: Sequence[float],
    gripper_t: float | Sequence[float] | None = None,
    gripper_t1: float | Sequence[float] | None = None,
) -> list[float]:
    """Compute [dx, dy, dz, dRx, dRy, dRz, gripper_delta_or_zero]."""

    p0 = _as_float_list(pos_t, 3, "pos_t")
    p1 = _as_float_list(pos_t1, 3, "pos_t1")
    delta_pos = [p1[idx] - p0[idx] for idx in range(3)]
    delta_rot = relative_rotvec_xyzw(quat_t, quat_t1)

    gripper_delta = 0.0
    if gripper_t is not None and gripper_t1 is not None:
        gripper_delta = _as_optional_scalar(gripper_t1, "gripper_t1") - _as_optional_scalar(
            gripper_t, "gripper_t"
        )

    action = delta_pos + delta_rot + [gripper_delta]
    if len(action) != 7 or not all(math.isfinite(value) for value in action):
        raise ValueError("computed action must contain 7 finite floats")
    return action
