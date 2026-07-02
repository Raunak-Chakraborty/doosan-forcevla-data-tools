# Raw-Real To Processed Semantics

This document describes the Phase 1 `raw_real_v0` to processed JSONL contract. It is structural and semantic documentation only; it does not approve any episode for training.

## TCP Unit Conversion

Accepted TCP position units are `m`, `meter`, `meters`, `metre`, `metres`, `mm`, `millimeter`, `millimeters`, `millimetre`, and `millimetres`.

Processed TCP position output is always metres.

Conversion rules:

- Metres are preserved unchanged.
- Millimetres are divided by `1000.0` exactly once.
- Missing or unsupported units fail for non-synthetic conversion.
- Units are never inferred from numeric magnitude.
- Legacy synthetic fixtures may use an explicitly recorded compatibility fallback when no unit metadata exists.

Unit declarations may come from per-row `units`, stream `units`, `actual_tcp_position_units`, or metadata-level unit fields. Conflicting declarations fail explicitly.

## TCP Orientation Conversion

Accepted orientation conventions are:

- `rotation_vector_radians`
- `rotation_vector_degrees`

Processed TCP orientation output is always a rotation vector in radians.

Conversion rules:

- `rotation_vector_radians` is preserved unchanged.
- `rotation_vector_degrees` is multiplied by `pi / 180` exactly once.
- Doosan/native Euler ZYZ conventions are rejected, including `doosan_posx_euler_zyz_degrees`, `doosan_robotstate_actual_tcp_position_euler_zyz_degrees`, and `euler_zyz_degrees`.
- Euler ZYZ values must not be treated as rotation vectors.
- Degrees/radians are never inferred from magnitude.

Processed metadata records the source convention, source unit, output convention, output unit, conversion string, and whether values were preserved or converted.

## 25D Model State Layout

`model_state` is 25D with layout version `doosan_model_state_25d_v1`:

```text
0  tcp_x_m
1  tcp_y_m
2  tcp_z_m
3  tcp_rotvec_x_rad
4  tcp_rotvec_y_rad
5  tcp_rotvec_z_rad
6  gripper_position
7  tcp_force_x_n
8  tcp_force_y_n
9  tcp_force_z_n
10 tcp_torque_x_nm
11 tcp_torque_y_nm
12 tcp_torque_z_nm
13 joint_1_position_rad
14 joint_2_position_rad
15 joint_3_position_rad
16 joint_4_position_rad
17 joint_5_position_rad
18 joint_6_position_rad
19 joint_1_velocity_rad_s
20 joint_2_velocity_rad_s
21 joint_3_velocity_rad_s
22 joint_4_velocity_rad_s
23 joint_5_velocity_rad_s
24 joint_6_velocity_rad_s
```

`metadata_processed.json` includes one `model_state_layout` entry per index with name, source, unit, frame, order, conversion, provenance, and value kind.

## 7D Measured Action Layout

`measured_action` is 7D with layout version `measured_tcp_delta_7d_v1`:

```text
0 delta_tcp_x_m
1 delta_tcp_y_m
2 delta_tcp_z_m
3 delta_tcp_rotvec_x_rad
4 delta_tcp_rotvec_y_rad
5 delta_tcp_rotvec_z_rad
6 gripper_action
```

Translation deltas are `tcp_position[t+1] - tcp_position[t]` in the processed TCP position frame, currently the declared base frame.

Rotation deltas use normalized `xyzw` quaternions computed from processed TCP rotation vectors:

```text
q_rel = conjugate(q_t) * q_t1
delta_rotvec = log(q_rel)
```

This is a proper relative rotation and not component-wise rotation-vector subtraction.

`gripper_action` is the same gripper scalar delta used in `model_state[6]`: `gripper[t+1] - gripper[t]`.

## Terminal Padding

The final observation row is retained for visualization and episode completeness. Because there is no next measured pose, the final action is a 7D zero vector and `action_is_terminal_padding` is `true` only on that final row.

`terminal_action_policy` records the padding value, padding count, terminal row indices, and the rule that exporters must exclude terminal-padding rows from training exports. Current `forcevla_13d` and `doosan_full_25d` export planning excludes exactly that final padded row.

## Gripper Provenance

Processed metadata distinguishes real measured gripper state, synthetic placeholders, fallback defaults, and unsupported/absent sources.

Important fields include:

- `gripper_state_source`
- `gripper_state_provenance`
- `gripper_state_is_placeholder`
- `gripper_state_verified`
- `gripper_state_valid_for_training`
- `gripper_action_semantics`

Synthetic placeholder gripper values are not labelled measured. The numerical placeholder policy is unchanged by this Phase 1 correction.

## Wrench Provenance

Wrench values are numerically preserved from the selected raw 6D force/torque signal. This phase does not verify the Doosan wrench frame in hardware.

Processed metadata preserves available wrench declarations, including order, force units, torque units, declared frame, source name, source type, verified status, and approved-for-training status. Unverified wrench data remains unverified.

## Structural Validity Versus Training Readiness

Successful conversion and processed validation mean the episode is structurally consistent with this processed JSONL contract. They do not imply:

- real hardware verification;
- wrench-frame hardware verification;
- real gripper integration;
- calibration verification;
- final training readiness.

Those approvals are deferred to the teleop/hardware validation phase.
