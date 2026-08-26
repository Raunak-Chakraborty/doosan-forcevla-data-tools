# Doosan synchronization policy v1

## Purpose

Patch 4 freezes the production synchronization policy for the final two-camera
Doosan raw contract. It uses the generic Patch-3 timestamp/index planner and
does not construct ForceVLA state or action vectors.

Policy ID:

`doosan_sync_policy_v1`

## Timestamp source versus clock epoch

Patch-4 evidence from Episode 10 showed that ROS message header timestamps and
rosbag record timestamps are in the same ROS/system epoch: every available
header-minus-bag offset is transport-scale (well below 100 ms), and all
timelines are strictly increasing.

They are therefore treated as two timestamp *sources* within one `ros`
`ClockDomain`.

The controller payload timestamp in `RobotStateRt.time_stamp` remains a
different diagnostic clock. Patch 4 does not estimate an offset between the
controller clock and ROS time and never silently mixes it into synchronization.

Header-based timelines are accepted only when the episode's header-minus-bag
offset stays within 100 ms. This is a fail-closed epoch plausibility check, not
a freshness threshold.

## Reference frame

The dataset frame tick is the D405 TCP camera RGB image:

`/doosan_cameras/tcp_camera/color/image_raw`

The reference timestamp is its ROS `header.stamp`, not the later rosbag record
timestamp. This better represents camera capture time.

## Frozen stream policy

| Key | Raw topic | Timestamp source | Association | Required | Max age |
| --- | --- | --- | --- | --- | --- |
| `external_image` | D435I RGB | header | nearest | yes | 12 ms |
| `robot_state_rt` | RobotStateRt | bag | nearest | yes | 16 ms |
| `gripper_state` | SCHUNK state | header | nearest | yes | 15 ms |
| `joint_state` | JointState | header | nearest | no | 15 ms |
| `speedl_stream` | SpeedL command | bag | causal hold | no | 25 ms |
| `joy` | operator Joy | header | causal hold | no | 25 ms |

Nearest is used for physical observations because conversion is offline and the
goal is the closest estimate to camera capture time. SpeedL and Joy remain
causal because they are command/operator-intent provenance: a future command
must not be associated with an earlier image.

Patch 4 intentionally does not use linear interpolation for RobotStateRt. The
raw TCP orientation is Doosan Euler ZYZ and should not be component-wise
interpolated. Patch 5 may perform representation-aware physical interpolation
later if there is a demonstrated need.

## Authoritative state source

`RobotStateRt` is the authoritative source for:

- TCP pose
- joint position
- joint velocity
- external TCP wrench

Episode-10 preflight showed that nearest `JointState` and controller-native
RobotStateRt agree extremely closely after degrees-to-radians conversion. The
maximum per-pair joint-position discrepancy was about `1.40e-4 rad`, and the
maximum velocity discrepancy was about `6.62e-3 rad/s`.

`/dsr01/joint_states` is therefore optional validation/provenance, not a
required training-state dependency. Its `effort` field remains unavailable and
is never repaired or zero-filled.

## CameraInfo handling

For both cameras in Episode 10:

- every image header timestamp exactly equals the corresponding CameraInfo
  header timestamp
- each CameraInfo topic has exactly one calibration signature across the
  episode

Patch 4 therefore validates CameraInfo pairing and calibration constancy, then
treats the calibration as episode metadata. CameraInfo is not a separate
per-frame synchronization dependency.

## Other raw streams

`/tf` and `/tf_static` remain in the immutable raw contract and typed decoder
layer but are not part of the production Patch-4 frame synchronization plan.

SpeedL and Joy are retained as optional provenance only. They do not define the
future measured 7D training action.

## Deferred

Patch 4 does not perform:

- mm-to-m or degree-to-radian conversion
- Doosan Euler to rotation-vector conversion
- wrench tare/reset logic
- SCHUNK normalized opening conversion
- physical-value interpolation
- measured 7D action construction
- image decoding/export
- ForceVLA dataset assembly

Those responsibilities remain in later roadmap patches.
