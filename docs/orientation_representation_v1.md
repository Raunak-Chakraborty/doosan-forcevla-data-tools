# Doosan model-facing orientation representations v1

Patch 11A adds a dependency-free SO(3) representation layer for controlled
orientation-ablation experiments. It does **not** change the current processed-v1
25D default state, the LeRobot exporter defaults, normalization statistics, or
the 7D measured action contract.

## Canonical physical orientation

All model-facing encodings are derived from one physical rotation matrix:

`R_base_tcp`

This is the proper SO(3) base-to-TCP rotation selected by the existing Doosan
force/proprio pipeline. Representation changes must never alter that physical
rotation.

The authoritative implementation is:

`src/doosan_forcevla_data/convert/orientation_representation_v1.py`

The existing Patch-5 public helpers
`rotation_matrix_to_rotvec()` and `rotation_vector_to_matrix()` remain available
and delegate to the new implementation so current callers keep their established
principal-rotvec behavior.

## Supported representations

### `rotvec_principal`

Width: **3**

The principal SO(3) logarithm in radians. Its angle lies in `[0, pi]`.

This is the current model-state orientation convention and remains the baseline.
It has the known branch cut at `pi`, so a smooth physical trajectory can contain
a componentwise jump of approximately `2*pi`.

### `rotvec_continuous`

Width: **3**

A stateful temporal lift of the principal rotvec. The first frame is principal.
For each later frame the implementation chooses the `2*pi`-equivalent rotvec
closest in Euclidean distance to the previous lifted rotvec.

At exact identity the physical rotation has no unique axis. If a previous
nonzero lifted value exists, its axis is retained and the nearest integer
multiple of `2*pi` is chosen. This permits continuity through complete turns.

This representation is sequence-dependent and therefore requires the same
stateful rule at training-data export and live inference.

### `quaternion`

Width: **4**

Ordering is frozen as:

`[w, x, y, z]`

Because `q` and `-q` represent the same physical rotation, Patch 11A defines a
deterministic sign rule:

1. when `|w| > 1e-12`, choose the sign with positive `w`;
2. at the numerical 180-degree boundary, choose the sign for which the first
   non-negligible component among `x`, `y`, `z` is positive.

This removes arbitrary sign duplication for a single matrix but does not remove
the unavoidable quaternion discontinuity at the 180-degree boundary.

### `rotation6d`

Width: **6**

The convention is frozen as the first two **columns** of `R_base_tcp`, explicitly
flattened as:

`[R00, R10, R20, R01, R11, R21]`

Do not replace this with row-major flattening.

For validation, reconstruction uses Gram-Schmidt:

1. normalize the first 3-vector;
2. remove its projection from the second 3-vector and normalize;
3. obtain the third basis vector by the right-handed cross product.

For values produced directly from a valid rotation matrix this reconstructs the
same physical rotation up to floating-point error.

## Planned state widths

The non-orientation, non-wrench state contributes 16 channels:

- TCP position: 3
- gripper: 1
- joint position: 6
- joint velocity: 6

The optional wrench remains 6 channels.

| Orientation representation | Orientation width | No wrench | Full + wrench |
| --- | ---: | ---: | ---: |
| `rotvec_principal` | 3 | 19 | 25 |
| `rotvec_continuous` | 3 | 19 | 25 |
| `quaternion` | 4 | 20 | 26 |
| `rotation6d` | 6 | 22 | 28 |

Patch 11A provides dimension helpers for these planned profiles, but it does not
yet change the production exporter or processed schema. Patch 11B will own that
integration.

## Action invariant

Observation representation is independent of the measured action convention.
The semantic action remains 7D:

1. `delta_x`, `delta_y`, `delta_z` in base coordinates;
2. 3D spatial relative rotvec
   `Log(R[t+1] @ R[t].T)`;
3. absolute binary gripper target.

Quaternion or rotation6D observation experiments must **not** change the action
to quaternion or 6D form, and must not add another delta transform.

## Validation requirements

Patch 11A tests enforce:

- the existing principal-rotvec numerical behavior remains compatible;
- all encodings reconstruct the same physical rotation matrix;
- quaternion ordering/sign conventions are deterministic;
- `0` and `2*pi` physical rotations map to the same quaternion and rotation6D;
- rotation6D uses columns, not rows;
- continuous rotvec removes the principal `+pi/-pi` numerical jump;
- continuous rotvec remains physically equivalent through multiple full turns;
- planned state widths are exactly 19/20/22 without wrench and 25/26/28 with wrench;
- malformed rotations and degenerate encodings fail closed.

## Non-goals of Patch 11A

Patch 11A intentionally does not:

- change `OBSERVATION_STATE_DIM = 25`;
- change processed-v1 row contents;
- change LeRobot output;
- change Golden Episode-10 outputs;
- generate or reuse normalization statistics;
- modify ForceVLA or pi0.5;
- modify synchronization, gripper, joints, wrench, cameras, or actions.

Those changes, where required, belong to later integration patches after this
math layer is accepted.
