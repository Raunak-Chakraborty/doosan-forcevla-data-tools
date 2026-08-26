# Doosan force + proprioception semantics v1

## Purpose

Patch 5 converts the authoritative controller state selected by the frozen
Patch-4 synchronization policy into physical ForceVLA-v2 semantics. It does not
change raw acquisition, typed decoding, synchronization, gripper calibration,
action construction, image decoding, or LeRobot export.

Semantic policy ID:

`doosan_force_proprio_semantics_v1`

Model contract:

`doosan_forcevla_dataset_contract_v2`

## Authoritative source and synchronization

The sole training-state source in this patch is:

`/dsr01/dsr_controller2/robot_state_rt_monitoring`

For every complete D405 reference frame, Patch 5 uses the exact single
`RobotStateRt` source index selected by Patch 4. The same controller record
supplies:

- TCP position and orientation
- six joint positions
- six joint velocities
- six-axis external TCP wrench

Patch 5 never independently re-synchronizes wrench or joints. `JointState`
remains optional validation-only provenance, and its unavailable effort values
are not used, repaired, or zero-filled.

## Controller-native to physical conversion

`RobotStateRt.actual_tcp_position` is:

`[x, y, z, A, B, C]`

with translation in millimetres and orientation in Doosan Euler ZYZ degrees,
all with respect to robot base coordinates.

Translation is converted deterministically:

`[x, y, z]_m = [x, y, z]_mm / 1000`

Doosan documents the orientation as intrinsic moving-axis `z-y'-z''`. Patch 5
therefore constructs the absolute base-to-TCP rotation matrix as:

`R_base_tcp = Rz(A) @ Ry(B) @ Rz(C)`

and stores the principal SO(3) logarithm:

`tcp_rotvec = Log(R_base_tcp)`

in radians. Raw Euler components are never interpolated component-wise.

Joint conversions are:

- `actual_joint_position`: degrees -> radians
- `actual_joint_velocity`: degrees/s -> radians/s

## Wrench semantics and reset provenance

`RobotStateRt.external_tcp_force` is used in this order:

`[Fx, Fy, Fz, Tx, Ty, Tz]`

The signal is already expressed with respect to robot base coordinates, with
force in newtons and moment in newton-metres.

Patch 5 exposes only two provenance classifications:

- `reset_compensated_passthrough`
- `legacy_episode_requires_known_tare_policy`

A modern training-ready episode must prove through `episode_operator.json` that:

- controller external-force reset was active at recording start
- controller reset completed before recording
- the online force-guard tare was not applied to MCAP values
- no offline force tare has already been performed
- no pre-reset F/T samples were recorded into the episode
- `recording.controller_reset_compensated` is true
- the MCAP force signal is explicitly identified as reset-compensated
  `RobotStateRt.external_tcp_force`
- offline force processing is owned by `doosan-forcevla-data-tools`

For that path the six wrench values are passed through unchanged. There is no
Patch-5 API that accepts or applies a second tare offset. Contradictory modern
metadata is rejected. An older episode that does not satisfy the modern gate is
classified as requiring a known legacy tare policy and is rejected for training
conversion until such a policy is explicitly defined; Patch 5 never guesses it.

## ForceVLA-v2 state ordering

Patch 5 defines the authoritative 25D production layout:

1. indices `0..2`: TCP position `[x,y,z]`, base frame, m
2. indices `3..5`: absolute base-to-TCP rotation vector, rad
3. index `6`: measured normalized gripper open fraction
4. indices `7..12`: joint positions J1..J6, rad
5. indices `13..18`: joint velocities J1..J6, rad/s
6. indices `19..24`: external TCP wrench `[Fx,Fy,Fz,Tx,Ty,Tz]`, base, N/Nm

Patch 5 does **not** derive index 6. Patch 6 owns the fixed physical SCHUNK
normalization with `0=closed, 1=open`. The Patch-5 state object can assemble the
exact 25D vector only when a caller supplies that already-normalized measured
value.

This intentionally does not mutate the older raw-real converter's historical
state ordering. Production MCAP integration into the processed/LeRobot path is
owned by later patches.

## Rejection conditions

Patch 5 fails closed when:

- force-reset provenance is contradictory or ambiguous
- a legacy episode has no explicitly known tare policy
- the Patch-4 RobotStateRt selection is missing, interpolated, or multi-record
- a selected source index or timestamp does not match the decoded RobotStateRt
- any training-candidate robot value is non-finite
- a required vector has the wrong dimension
- a requested gripper value is outside `[0,1]`
- a rotation matrix is not a proper SO(3) matrix

Raw MCAP data is never rewritten.
