# Doosan SCHUNK gripper semantics v1

Contract ID: `doosan_gripper_semantics_v1`

Status: Patch-6 production semantic layer for the thesis release-only gripper
protocol.

## Thesis episode protocol

Each demonstration starts with the peg already gripped. During the episode the
operator may press the SpaceMouse RIGHT button once to release the peg. No
autonomous grasp/close operation is part of the recorded skill.

The deployed ROS contract proves:

- `/schunk/state` is `gripper_msgs/msg/GripperState` with `position` and
  `holding` only;
- `holding` is the SCHUNK workpiece-gripped status bit;
- the SpaceMouse release client uses RIGHT button index `1`, requires fresh
  `holding=true`, and sends the parameterless `/schunk/release` action;
- the release action is explicitly a relative opening operation;
- `/schunk/state` contains no measured force value.

Therefore Patch 6 deliberately does **not** derive model semantics from physical
finger position or per-episode position extrema.

## Model-facing state scalar

ForceVLA v2 keeps its frozen state ordering. State index `6` remains an absolute
open-fraction semantic with the existing convention:

- `0.0` = held / closed semantic endpoint (`holding=true`)
- `1.0` = released / open semantic endpoint (`holding=false`)

Only those two exact values are emitted by Patch 6. The raw SCHUNK `position`
value is retained as diagnostic provenance and is never used for normalization.
Consequently there is no per-episode min/max normalization, finger-endpoint
calibration, clipping, or interpolation of the discrete `holding` status.

## Model-facing gripper action

The ForceVLA-v2 action direction remains unchanged:

`action[6] = absolute gripper target open fraction`

For this release-only profile the allowed semantic targets are binary:

- `0.0` = maintain the already-held grasp
- `1.0` = released/open target

Patch 6 defines deterministic execution intent for the physical protocol:

- held `0 -> 0`: no gripper command; maintain the existing grasp
- held `0 -> 1`: issue the existing parameterless `/schunk/release` action
- released `1 -> 1`: no command; remain released
- released `1 -> 0`: unsupported and rejected because it would require a new
  grasp operation that is outside this thesis episode contract

Patch 7 constructs the complete measured 7D action sequence and uses the
**target state's** binary absolute open fraction as action channel `6`. It does
not convert that channel to a delta. See `docs/doosan_measured_action_semantics_v1.md`.

## Synchronization

Patch 6 consumes the frozen Patch-4 `gripper_state` plan:

- topic: `/schunk/state`
- timestamp source: ROS Header
- method: nearest
- required: true
- maximum age: 15 ms

No new synchronization is performed. For every complete D405 reference frame,
Patch 6 uses exactly the source index/timestamp already chosen by Patch 4.

## 25D state assembly

Patch 5 owns TCP, joints and the six-dimensional external TCP wrench. Patch 6
owns only state index `6`. The final v2 observation remains:

`[tcp_position_3, tcp_rotvec_3, gripper_1, joint_position_6,
 joint_velocity_6, wrench_6]`

with dimension 25.

Patch 6 joins Patch-5 and Patch-6 samples only when their Patch-4 reference index
and reference timestamp are identical. It never zips unrelated asynchronous
records by array position.

## Force/load semantics

`GripperState` has no measured gripper-force field. The configured SCHUNK grip
force percentage is a command/configuration value, not a measured contact-force
observation.

The force-aware part of the 25D state therefore remains Patch 5's
`RobotStateRt.external_tcp_force` six-dimensional base-coordinate wrench. Patch
6 does not create, estimate, or duplicate a gripper-force channel.

## Episode-10 golden evidence

The immutable Episode-10 raw gripper stream contains 1699 messages:

- first `holding=true`
- last `holding=false`
- 1593 raw held samples
- 106 raw released samples
- exactly one raw holding transition

Under the frozen Patch-4 D405 synchronization, the same episode yields 1009
selected gripper samples:

- 956 held samples (`0.0`)
- 53 released samples (`1.0`)
- exactly one selected transition at reference index `956`

This is consistent with the independently validated exactly-once release
lifecycle: one request, one accepted goal, one successful result, followed by
`holding=false`.

## Fail-closed rules

Patch 6 rejects:

- missing/malformed Patch-4 gripper selections
- interpolation or multi-record selection for the discrete gripper state
- selected source indices outside the decoded gripper stream
- selected timestamps that do not match the decoded ROS Header timestamp
- non-finite raw position values
- non-boolean `holding` values at the semantic API boundary
- non-binary model-facing open fractions
- a released-to-held execution request in the release-only profile
- joining Patch-5/Patch-6 states with different reference identity

Raw MCAP data is never rewritten.
