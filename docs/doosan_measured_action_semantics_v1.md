# Doosan measured 7D action semantics v1

Contract ID: `doosan_measured_action_v1`

Status: Patch-7 production action layer for
`doosan_forcevla_dataset_contract_v2`.

## Primary learning label

The primary action is measured robot motion between consecutive synchronized
states. `SpeedL` and `Joy` remain auxiliary command/operator provenance and do
not define the training action.

For synchronized states `t` and `t+1`:

```text
delta_p = p[t+1] - p[t]
delta_R = R[t+1] @ R[t].T
delta_rotvec = Log(delta_R)

action[t] = [
  delta_p_x,
  delta_p_y,
  delta_p_z,
  delta_rotvec_x,
  delta_rotvec_y,
  delta_rotvec_z,
  gripper_target[t+1],
]
```

The action has exactly seven finite float channels.

## Translation convention

`p` is Patch-5 absolute TCP position in robot-base coordinates and metres.
Therefore `delta_p` is also expressed in robot-base coordinates and metres.

The action is a displacement, not a velocity. Patch 7 records the source and
target reference timestamps as provenance but does not divide translation by
`dt`.

## Rotation convention

Patch 5 stores absolute base-to-TCP rotations. Patch 7 uses the frozen
ForceVLA-v2 spatial/base-frame relative rotation:

```text
delta_R = R[t+1] @ R[t].T
```

and stores the principal SO(3) logarithm as a three-dimensional rotation vector
in radians.

The older body/local relation is explicitly not the production contract:

```text
R[t].T @ R[t+1]   # NOT Patch-7 production semantics
```

The reconstruction identity is:

```text
R[t+1] = Exp(delta_rotvec) @ R[t]
```

This convention is tested with non-commuting rotations so a body-frame
implementation cannot accidentally pass.

## Gripper channel

Action channel `6` is the Patch-6 **absolute** binary target at the target state
`t+1`:

- `0.0` = held/closed target
- `1.0` = released/open target

It is not `gripper[t+1] - gripper[t]` and ForceVLA must not delta-transform this
channel.

For the thesis release-only protocol the only supported sequence is held then
released. Episode 10 therefore has one release-bearing action: source reference
`955` targets reference `956` with channel `6 = 1.0`.

## Synchronization ownership

Patch 7 does not synchronize raw ROS streams. It consumes the exact semantic
samples already selected by Patch 4 and converted by Patches 5 and 6.

At the action boundary Patch 7 requires:

- Patch-5 and Patch-6 sample counts are identical;
- reference indices and reference timestamps match exactly for every state;
- production reference indices are contiguous `0..N-1`;
- reference timestamps are strictly increasing.

Patch 7 deliberately refuses to bridge a dropped reference frame because that
would silently change the nominal 30 Hz action step. A future dataset assembly
policy may segment/reject such episodes explicitly, but Patch 7 does not guess.

## Terminal-state policy

A sequence of `N` synchronized measured states contains only `N-1` observed
state transitions. Patch 7 therefore emits exactly `N-1` actions.

The final synchronized observation has no measured successor. Patch 7 does not
fabricate a seven-dimensional zero action and does not label that synthetic
value as demonstrated behavior.

Patch 8 must either:

- exclude the terminal observation from action-bearing training rows; or
- carry an explicit invalid-action mask.

The terminal policy must remain explicit in processed/export metadata.

## Episode-10 golden evidence

Frozen Episode 10 contains:

```text
synchronized states:             1009
measured consecutive actions:    1008
terminal reference index:        1008
terminal measured action:        none
```

Patch-6 binary state sequence:

```text
reference 0..955:    held/released scalar = 0.0
reference 956..1008: held/released scalar = 1.0
```

Therefore Patch-7 absolute gripper targets are:

```text
955 actions with target 0.0
53 actions with target 1.0
```

The unique held-to-released command-bearing transition is:

```text
source reference 955 -> target reference 956
```

## Acceptance criteria

Patch 7 is accepted only when:

- each action has exactly seven finite values;
- translation reconstructs every measured target position within numerical
  tolerance;
- `Exp(delta_rotvec) @ R[t]` reconstructs every measured target rotation;
- non-commuting rotation tests distinguish spatial from body/local convention;
- gripper action equals the Patch-6 target state, not a delta;
- actions never use SpeedL or Joy as the primary label;
- there is no episode-boundary transition;
- an `N`-state episode produces exactly `N-1` actions;
- no synthetic terminal zero action is emitted.
