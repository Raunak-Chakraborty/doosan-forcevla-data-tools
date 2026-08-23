# Doosan raw MCAP contract v1 and typed decoding

Contract ID: `doosan_two_camera_rosbag_raw_v1`

This document defines Patch 2 of the converter. Patch 1 owns generic rosbag2
MCAP discovery and deserialization. Patch 2 makes the final Doosan acquisition
contract explicit and turns ROS message objects into typed immutable Python
records.

## Scope

Patch 2 performs no synchronization and no ForceVLA-facing geometry or unit
conversion.

It preserves raw acquisition semantics so later patches can make synchronization
and model-facing decisions independently.

The exact eleven required topics are:

1. `/dsr01/dsr_controller2/robot_state_rt_monitoring`
   - `dsr_msgs2/msg/RobotStateRt`
2. `/dsr01/joint_states`
   - `sensor_msgs/msg/JointState`
3. `/dsr01/dsr_controller2/speedl_stream`
   - `dsr_msgs2/msg/SpeedlStream`
4. `/doosan_teleop/collector_joy`
   - `sensor_msgs/msg/Joy`
5. `/schunk/state`
   - `gripper_msgs/msg/GripperState`
6. `/tf`
   - `tf2_msgs/msg/TFMessage`
7. `/tf_static`
   - `tf2_msgs/msg/TFMessage`
8. `/doosan_cameras/tcp_camera/color/image_raw`
   - `sensor_msgs/msg/Image`
9. `/doosan_cameras/tcp_camera/color/camera_info`
   - `sensor_msgs/msg/CameraInfo`
10. `/doosan_cameras/external_camera_2/color/image_raw`
    - `sensor_msgs/msg/Image`
11. `/doosan_cameras/external_camera_2/color/camera_info`
    - `sensor_msgs/msg/CameraInfo`

The metadata tag `custom_data.raw_contract` must equal
`doosan_two_camera_rosbag_raw_v1`.

Missing topics, extra topics, or topic/type substitutions fail closed.

## Timestamp contract

Every typed record preserves the rosbag write timestamp as
`bag_timestamp_ns`.

Messages with a ROS `Header` additionally preserve `header_timestamp_ns` and
`frame_id`.

This applies to:

- JointState
- Joy
- GripperState
- Image
- CameraInfo

`RobotStateRt` has no ROS Header. Its `time_stamp` payload is preserved
separately as `controller_timestamp_s`; Patch 2 does not reinterpret it as a ROS
or Unix timestamp.

`SpeedlStream` has no ROS Header. Its `time` field is preserved as
`command_time_s`; it is a command payload field, not an acquisition timestamp.

`TFMessage` itself has no header. Each contained `TransformStamped` preserves
its own header timestamp, parent frame, child frame, translation, and ROS
quaternion in XYZW order.

`/tf_static` timestamps are preserved exactly even when they substantially
precede the bag write time. Static transforms must not be treated as ordinary
freshness-gated dynamic samples merely because their header time is old.

Patch 3 will decide the synchronization clock and association policy for each
stream.

## RobotStateRt

Patch 2 explicitly extracts and validates the raw fields required by later
state/force processing:

- controller `time_stamp`
- actual joint position, degrees
- actual joint velocity, degrees/second
- actual TCP position in base coordinates, millimetres plus Doosan Euler ZYZ
  angles in degrees
- actual TCP velocity in base coordinates, millimetres/second plus angular
  degrees/second
- estimated external TCP wrench in robot base coordinates, N and Nm

Every value in those extracted candidates must be finite and every vector must
have exactly six elements.

All other RobotStateRt fields, including absolute-encoder state, raw
force/torque, torque/matrix/controller diagnostics, and reserved fields, are
retained as immutable diagnostic data without being admitted to the Patch-2
training whitelist. This preserves controller diagnostics while preventing
unrelated sentinel fields from poisoning later training vectors.

Patch 2 performs no degree-to-radian, millimetre-to-metre, Euler-to-rotation
vector, tare, or wrench-frame conversion.

## JointState

The canonical joint identity is:

`joint_1, joint_2, joint_3, joint_4, joint_5, joint_6`

The decoder validates names rather than trusting array order and emits position
and velocity in canonical J1-to-J6 order.

Missing or duplicate joints fail closed. A missing joint is never silently
replaced with zero.

Position and velocity must be finite.

Effort is handled separately:

- empty effort array -> `unavailable_empty`, typed effort is `None`
- all-NaN effort array -> `unavailable_all_nonfinite`, typed effort is `None`
- fully finite effort array -> `available_finite`, preserved canonically
- mixed finite/non-finite values, infinities, or malformed length -> fail closed

Episode 10 contains six NaNs in every JointState effort vector, so all 3362
messages classify as `unavailable_all_nonfinite`. Those NaNs are not propagated
into any whitelisted training field.

## SpeedL and Joy

SpeedL preserves the six velocity values, two acceleration values, and the
message `time` payload. All must be finite. No frame or unit reinterpretation is
performed by Patch 2.

Joy requires six finite axes and exactly two buttons. It preserves both bag and
header timestamps.

SpeedL and Joy remain auxiliary/controller-intent diagnostics. Patch 7 decides
measured action construction.

## SCHUNK state

The decoder preserves:

- bag timestamp
- ROS header timestamp
- raw ROS `position` in metres
- `holding`

Only finiteness is enforced here.

Patch 6 owns physical fixed-limit normalization, the 0-closed/1-open convention,
and inference command inversion.

## Images

Images are not decoded to NumPy arrays in Patch 2.

The ROS payload is exposed as a read-only memory view so streaming scans avoid
an unnecessary second full image copy.

The production profiles are enforced:

- D405 TCP camera: 640 x 480, `rgb8`, little-endian flag 0, step 1920,
  `tcp_camera_color_optical_frame`
- D435I external camera: 848 x 480, `rgb8`, little-endian flag 0, step 2544,
  `external_camera_2_color_optical_frame`

Payload byte count must equal `step * height`.

Patch 8 owns processed image/video output and ForceVLA image mapping.

## CameraInfo

CameraInfo preserves dimensions, distortion model, D/K/R/P arrays, binning,
ROI, frame id, and both timestamps.

The final profiles require `plumb_bob` with D/K/R/P lengths 5/9/9/12 and the
same dimensions/frame ids as their corresponding image streams.

Calibration numbers must be finite.

## TF

Each TF transform preserves:

- bag timestamp of its containing TFMessage
- TransformStamped header timestamp
- parent frame
- child frame
- translation XYZ
- quaternion XYZW
- whether the source topic is `/tf_static`

Translation/quaternion values must be finite and quaternion norm must be within
0.001 of one.

Patch 2 does not flatten the TF tree into a synchronized robot state.

## Training whitelist safety

Patch 2's raw training-candidate whitelist is intentionally narrow:

- RobotStateRt actual TCP position
- RobotStateRt actual joint position
- RobotStateRt actual joint velocity
- RobotStateRt external TCP wrench
- JointState position
- JointState velocity
- SCHUNK raw position

JointState effort is deliberately absent from this whitelist.

Patch 5 and Patch 6 decide which of these raw candidates enter the final 25D
ForceVLA state and how they are transformed.

## Upstream reference

The decoder-registry idea was reviewed against
`legalaspro/so101-ros-physical-ai`, package `rosbag_to_lerobot`.

This implementation intentionally differs in several safety-critical ways:

- dispatch is topic-role aware rather than type-only, because both cameras use
  the same ROS message types but have different required profiles
- missing JointState names fail closed instead of silently leaving zeros
- image decoding does not eagerly allocate NumPy RGB arrays
- bag and header timestamps are both preserved rather than collapsed
- TF timestamps are preserved per TransformStamped
- JointState effort NaNs are explicitly classified unavailable

Synchronization concepts from that project remain deferred to Patch 3.
