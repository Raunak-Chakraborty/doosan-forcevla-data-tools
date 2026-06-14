# Raw Recorder Schema Plan

This is a planning report only. It does not define ROS2 control code, does not assume laptop access to a live robot, and does not move the raw recorder into this offline data-pipeline repository.

Repository inspected: `/home/horus/robotics_thesis/doosan_forcevla_data_tools`

Static ROS2 report inspected: `/home/horus/robotics_thesis/lab_myros2_ws/opencode_report.txt`

Current repository head observed locally: `f2a9feb Add multi-episode ForceVLA smoke regression`

## 1. Current Data-Pipeline Capabilities

The current repository is an offline ForceVLA/Doosan data pipeline. It is already organized around raw schema, processed schema, conversion, validation, export staging, LeRobot-style export, and ForceVLA-facing smoke checks.

Current raw episode support:

| Capability | Current implementation | Notes |
| --- | --- | --- |
| Raw metadata schema | `src/doosan_forcevla_data/schema/raw_schema.py` | Defines required metadata keys: `episode_id`, `task_instruction`, `geometry_type`, `orientation_type`, `collection_method`, `action_label_primary`, `success`, `failure_reason`, and `fps`. |
| Current v0 raw layout | `RawEpisodePaths` | Expects `metadata.json`, `robot/tcp_pose.csv`, `robot/joint_states.csv`, `force/wrench.csv`, optional `actions/commanded_twist.csv`, `events.csv`, `images/external_rgb`, and `images/tcp_rgb`. |
| Dummy raw episode generator | `src/doosan_forcevla_data/dummy/make_dummy_raw_episode.py` | Creates deterministic metadata, TCP pose, joint states, wrench data, commanded twist, events, and two RGB image folders using tiny PPM images. |
| Raw validation | `src/doosan_forcevla_data/validate/validate_raw_episode.py` | Checks required files/folders, metadata keys, CSV headers, finite numeric values, positive `fps`, boolean `success`, and strictly increasing timestamps. |

Current processed episode support:

| Capability | Current implementation | Notes |
| --- | --- | --- |
| Processed schema constants | `src/doosan_forcevla_data/schema/processed_schema.py` | Defines `MODEL_STATE_DIM = 25`, `ACTION_DIM = 7`, six joints, six wrench fields, `xyzw` quaternion convention, state fields, model-state fields, and action fields. |
| Raw-to-processed conversion | `src/doosan_forcevla_data/convert/raw_to_processed.py` | Converts validated raw episodes to `metadata_processed.json` and `frames.jsonl`. |
| Processed frame content | `frames.jsonl` | Per frame: `frame_index`, `timestamp`, `external_rgb_path`, `tcp_rgb_path`, 25D `model_state`, 7D `measured_action`, and `action_is_terminal_padding`. |
| Model-state layout | `processed_schema.py` | `ee_pos(3) + ee_axis_angle(3) + gripper_pos(1) + wrench(6) + joint_pos(6) + joint_vel(6)`. |
| Processed validation | `src/doosan_forcevla_data/validate/validate_processed_episode.py` | Checks metadata keys, frame count, timestamp monotonicity, image paths, 25D model state, 7D action, and exactly one final terminal-padding frame with zero action. |

Current action support:

| Capability | Current implementation | Notes |
| --- | --- | --- |
| Measured TCP delta action | `src/doosan_forcevla_data/convert/compute_actions.py` | Pure Python computation of `[dx, dy, dz, dRx, dRy, dRz, gripper_delta_or_zero]` from consecutive measured TCP poses. |
| Quaternion utilities | `compute_actions.py` | Normalizes `xyzw` quaternions, computes relative quaternion as `conjugate(q_t) * q_t1`, and converts relative rotation to rotation vector. |
| Future action chunks | `src/doosan_forcevla_data/convert/action_chunks.py` | Builds ForceVLA/OpenPI action horizons, default horizon `50`, with `repeat_last` or `zero` padding. Can load single-step actions from LeRobot parquet when `pyarrow` is available. |

Current LeRobot and ForceVLA export support:

| Capability | Current implementation | Notes |
| --- | --- | --- |
| Export planning | `src/doosan_forcevla_data/convert/plan_lerobot_export.py` | Builds a dry-run manifest for `forcevla_13d` or `doosan_full_25d`, verifies image availability, dimensions, and terminal-frame exclusion. |
| Staged export | `src/doosan_forcevla_data/convert/stage_lerobot_export.py` | Writes `metadata_staged.json` and staged `frames.jsonl` using future-facing keys: `observation.image`, `observation.wrist_image`, `observation.state`, `action`, and `task`. |
| Single-episode skeleton export | `src/doosan_forcevla_data/convert/write_lerobot_skeleton.py` | Writes LeRobot v2.1-style metadata, JSONL placeholder records, and image staging by symlink or copy. Does not write parquet or videos. |
| Multi-episode skeleton export | `src/doosan_forcevla_data/convert/write_lerobot_dataset_skeleton.py` | Combines multiple staged episodes into one LeRobot-style dataset root with `meta/`, `data/chunk-XXX/`, task indexing, episode indexing, and global frame indexing. |
| Real single-episode export scaffold | `src/doosan_forcevla_data/convert/write_real_lerobot_export.py` | Dependency-optional local scaffold that writes an export attempt report and, in `write-if-available`, writes parquet/videos only when dependencies are present. |
| Real multi-episode export scaffold | `src/doosan_forcevla_data/convert/write_real_lerobot_dataset_export.py` | Dependency-optional multi-episode real export attempt. Writes per-episode parquet/videos when dependencies allow and reports per-episode results. |
| Export dependency check | `src/doosan_forcevla_data/inspect/check_export_dependencies.py` | Read-only dependency probe for Python, `pyarrow`, `pandas`, `lerobot`, `cv2`, `imageio`, `PIL`, and `ffmpeg`. Does not install packages. |
| Real-export preflight | `src/doosan_forcevla_data/inspect/preflight_real_export.py` | Checks skeleton schema, prompt/task compatibility, image staging, optional dependencies, parquet readiness, and video readiness. |

Current ForceVLA smoke support:

| Capability | Current implementation | Notes |
| --- | --- | --- |
| Observation builder smoke test | `src/doosan_forcevla_data/inspect/smoke_forcevla_observation_builder.py` | Reads LeRobot parquet with `pyarrow`, decodes videos with PyAV, and builds a ForceVLA-style sample with image, wrist image, state, action, and prompt. |
| Transform input smoke test | `src/doosan_forcevla_data/inspect/smoke_forcevla_transform_input.py` | Imports local ForceVLA/OpenPI modules, builds policy input, applies ForceVLA input transform, checks image/state/action shapes, and validates `Observation.from_dict`. |
| Tokenization smoke test | `src/doosan_forcevla_data/inspect/smoke_forcevla_tokenization_input.py` | Runs through ForceVLA/OpenPI model transforms up to resized images, tokenized prompt, tokenized prompt mask, and `Observation.from_dict`. No checkpoint or inference. |
| Multi-episode ForceVLA smoke regression | `tests/test_smoke_forcevla_multi_episode.py` | Builds two dummy episodes, exports a multi-episode real dataset when dependencies exist, validates episode 1 observation building, global index handling, and future action chunks. |

Config and documentation support:

| Capability | Current file | Notes |
| --- | --- | --- |
| Human-readable dataset contract | `configs/doosan_v0.yaml` | Defines `doosan_peg_in_hole_v0`, `doosan_m1013`, 30 FPS, camera streams, state fields, 25D model state, 7D measured action, and task metadata fields. |
| LeRobot export plan | `docs/lerobot_export_plan.md` | Documents `forcevla_13d`, `doosan_full_25d`, terminal-frame exclusion, staged export, skeleton writer, and image handling decisions. |
| Real schema decision | `docs/real_lerobot_schema_decision.md` | Documents final intended v2.1-style layout, parquet columns, video plan, prompt/task policy, terminal-frame policy, and open questions. |
| Real writer design | `docs/real_lerobot_writer_design.md` | Documents the full offline chain and dependency-optional real writer behavior. |
| Lab validation checklist | `docs/lab_workstation_validation_checklist.md` | Documents lab workstation validation steps for tests, dependency checks, dummy pipeline, real export attempt, and ForceVLA compatibility checks. |

## 2. What Should Stay ROS-Independent

The offline data-pipeline repository should continue to avoid `rclpy`, ROS launch files, live robot service clients, and direct ROS message imports. The inspected package has an empty `src/doosan_forcevla_data/ros/` directory and no source imports of `rclpy`, `dsr_msgs2`, `sensor_msgs`, or `geometry_msgs`.

The following modules and concepts should remain pure Python or dependency-light offline code:

| Area | Modules or concepts that should remain ROS-independent |
| --- | --- |
| Raw schema | `RawEpisodeMetadata`, `RawEpisodePaths`, future raw real-episode schema constants, schema versioning, source-topic string fields, units, and frame metadata. |
| Processed schema | `MODEL_STATE_DIM`, `ACTION_DIM`, state/action field layout, quaternion convention, ForceVLA profile definitions. |
| Dummy data | `make_dummy_raw_episode` and any future fixture generators. |
| Raw validation | Structural validation, metadata validation, timestamp validation, stream completeness validation, and raw numeric sanity checks. |
| Raw-to-processed conversion | File parsing, stream alignment, resampling, normalization, calibration application, and conversion from raw records to processed JSONL. |
| Action computation | `compute_measured_tcp_delta_action`, quaternion math, relative rotation conversion, gripper delta handling, terminal padding, and action chunk creation. |
| Export planning and staging | `plan_lerobot_export`, `stage_lerobot_export`, image path resolution, profile-specific state selection, and terminal-frame exclusion. |
| LeRobot skeleton export | Single-episode and multi-episode skeleton writers, metadata writers, task indexing, episode indexing, image staging, and validation. |
| Real export scaffolds | Dependency checks, pyarrow parquet writing, video encoding from staged images, export attempt reports, and validators. |
| ForceVLA smoke tools | Observation builder smoke checks, transform input checks, tokenization checks, and action horizon checks. |
| Calibration and normalization | Future calibration reference resolution, unit conversion, frame convention checking, normalization statistics, and schema migrations. |

The future live recorder should be a separate ROS2 package or workspace component. It can know about ROS topics, services, message classes, QoS, live clocks, camera drivers, and passive service polling. This data-pipeline repository should only consume the recorder output as files.

## 3. Proposed Raw Real-Episode Folder Schema

The future real-robot raw episode should preserve raw evidence from the recorder with explicit source names, units, frames, and timestamps. It should not write ForceVLA tensors directly and should not require LeRobot export at recording time.

Proposed folder layout:

```text
episode_YYYYMMDD_HHMMSS_<short_id>/
  metadata.json
  calibration_refs.json
  events.jsonl
  recorder_report.json
  streams/
    index.json
    robot_state_rt.jsonl
    joint_states.jsonl
    tf.jsonl
    tf_static.jsonl
    command_context.jsonl
    gripper_state.jsonl
    external_camera/
      index.jsonl
      frames/
        000000.<ext>
        000001.<ext>
      chunks/
        optional_video_or_raw_chunk_files
    wrist_camera/
      index.jsonl
      frames/
        000000.<ext>
        000001.<ext>
      chunks/
        optional_video_or_raw_chunk_files
```

Recommended file roles:

| File or folder | Required for first real schema | Purpose |
| --- | --- | --- |
| `metadata.json` | Yes | Episode-level metadata, task instruction, operator/session context, success/failure labels, schema version, recorder version, robot model, collection method, and source workspace identifiers. |
| `streams/index.json` | Yes | Manifest of streams present, stream types, source topic/service names, message types, units, frame IDs, expected rates, record counts, and file paths. |
| `streams/robot_state_rt.jsonl` | Yes if `ReadDataRt` is selected | Append-only records mirroring `dsr_msgs2/msg/RobotStateRt` fields needed for offline conversion. Records should also be parquet-ready: flat numeric arrays, no unserializable ROS objects. |
| `streams/joint_states.jsonl` | Yes | Passive `sensor_msgs/msg/JointState` records or equivalent source records with joint names, positions, velocities, efforts if meaningful, and units. |
| `streams/tf.jsonl` | Yes | Dynamic TF messages needed to reconstruct frame tree and camera/TCP relationships. |
| `streams/tf_static.jsonl` | Yes | Static TF messages captured at episode start and whenever refreshed. |
| `streams/external_camera/` | Yes | External RGB stream, image files or chunks plus `index.jsonl` with exact per-frame timestamps and metadata. |
| `streams/wrist_camera/` | Yes | Wrist/TCP RGB stream, image files or chunks plus `index.jsonl` with exact per-frame timestamps and metadata. |
| `streams/command_context.jsonl` | Optional but recommended | Commanded teleop/servo/jog/trajectory context for debugging only. Not the primary action label. |
| `streams/gripper_state.jsonl` | Required if gripper is used, optional otherwise | Gripper position/state source if available from controller state, action feedback, IO, Modbus, or a separate driver. |
| `events.jsonl` | Yes | Episode start/stop, task phase labels, contact annotations if available, aborts, errors, success/failure event, and operator annotations. |
| `calibration_refs.json` | Yes | References to camera intrinsics, extrinsics, tool/TCP calibration, force/torque calibration, and calibration version identifiers. |
| `recorder_report.json` | Yes | Recorder-side summary: stream counts, dropped frames, rate estimates, time span, warnings, errors, live ROS graph snapshot, and safety notes. |

Suggested `streams/index.json` top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Raw real-episode schema version, for example `raw_real_v0`. |
| `streams` | Object keyed by stream name. Each entry records `path`, `kind`, `required`, `source_name`, `source_type`, `units`, `frame_id`, `record_count`, `expected_rate_hz`, and `observed_rate_hz`. |
| `timebase` | Declares timestamp fields used across streams and whether clocks are ROS time, system wall time, monotonic time, controller time, or camera hardware time. |
| `alignment_policy` | Declares the current `raw_real_v0` conversion policy: converter-ready streams carry an aligned episode-level `record_index`; raw source timestamps are preserved for diagnostics. |

Suggested JSONL record style:

| Common field | Purpose |
| --- | --- |
| `record_index` | For converter-ready `raw_real_v0`, an episode-level aligned sample index shared by `robot_state_rt`, `joint_states`, `external_camera`, `wrist_camera`, and aligned optional `gripper_state`. Native per-stream counters should be stored in a separate field if needed. |
| `source_name` | Exact ROS topic or service name as observed live. |
| `source_type` | ROS message/service type string or recorder adapter type. |
| `source_stamp` | Original message header timestamp or controller timestamp when available. |
| `receipt_stamp` | Recorder receipt time. |
| `monotonic_stamp` | Recorder monotonic-clock timestamp for jitter analysis. |
| `frame_id` | ROS frame ID or explicit frame name. |
| `data` | Stream-specific flat numeric arrays, strings, and status fields. |

Current `raw_real_v0` conversion alignment policy:

- The raw recorder may capture native sensor timestamps, controller timestamps, receipt times, and native per-stream counters for auditability.
- Before an episode is converter-ready, `robot_state_rt`, `joint_states`, `external_camera`, `wrist_camera`, and aligned optional `gripper_state` records must carry the same episode-level aligned `record_index` values.
- `source_stamp` remains the original sensor/source time and is used for synchronization diagnostics.
- `receipt_stamp` and `monotonic_stamp` remain recorder/debug timing signals.
- The current converter does not perform timestamp-based alignment, resampling, or interpolation.
- Large cross-stream `source_stamp` offsets between robot_state_rt and required camera streams block conversion readiness unless a future schema introduces an explicit clock-offset model.
- A future schema version may add timestamp-based alignment, but this current version intentionally requires aligned `record_index` values for converter-required streams.

## 4. Required Raw Fields

Required fields should be enough to reconstruct processed records offline, compute measured 7D TCP delta action, validate the collection, and audit live source provenance.

Required timestamp fields:

| Field | Requirement |
| --- | --- |
| `source_stamp` | Original message header time, controller time, or camera hardware timestamp when available. Must include seconds/nanoseconds or a clearly documented numeric unit. |
| `receipt_stamp` | Recorder receipt timestamp for each record. |
| `monotonic_stamp` | Recorder monotonic time for ordering and latency diagnostics. |
| `record_index` | Episode-level aligned sample index for converter-required streams in current `raw_real_v0`; not merely a native per-stream counter. |
| `episode_time` | Optional at capture but required by processed conversion, computed offline relative to episode start. |
| `timebase_metadata` | Required in `streams/index.json` or `metadata.json`; must identify ROS time versus system time versus controller/camera hardware time. |

Required external RGB image fields:

| Field | Requirement |
| --- | --- |
| `image_path` or `chunk_path` | Path to stored image frame or chunk relative to the episode root. |
| `source_name` | Exact source topic name. Camera topics are unknown from static inspection and must be verified live. |
| `source_type` | Expected ROS type if recorded through ROS, likely `sensor_msgs/msg/Image`, but live type must be verified. |
| `source_stamp`, `receipt_stamp`, `record_index` | Required for aligned sample identity and jitter checks. `record_index` is the shared aligned sample index for current conversion; `source_stamp` remains the camera/source timestamp. |
| `frame_id` | Camera optical or camera link frame from message header. |
| `encoding` | RGB encoding, for example `rgb8` or `bgr8`, as recorded. |
| `width`, `height`, `channels` | Image dimensions. |
| `camera_role` | `external_rgb`. |
| `camera_id` | Stable camera identifier if available, such as serial number or config name. |

Required wrist/TCP RGB image fields:

| Field | Requirement |
| --- | --- |
| `image_path` or `chunk_path` | Path relative to episode root. |
| `source_name` | Exact wrist/TCP camera source topic once verified live. |
| `source_type` | Expected ROS type if recorded through ROS, likely `sensor_msgs/msg/Image`, but live type must be verified. |
| `source_stamp`, `receipt_stamp`, `record_index` | Required for aligned sample identity and jitter checks. `record_index` is the shared aligned sample index for current conversion; `source_stamp` remains the camera/source timestamp. |
| `frame_id` | Wrist/TCP camera optical or camera link frame. |
| `encoding`, `width`, `height`, `channels` | Image representation details. |
| `camera_role` | `wrist_rgb` or `tcp_rgb`. |
| `camera_id` | Stable camera identifier if available. |

Required joint position and velocity fields:

| Field | Requirement |
| --- | --- |
| `joint_names` | Ordered names from `JointState` or recorder config. |
| `position` | Six Doosan M1013 joint positions. Units must be explicit. ROS control `/joint_states` is expected in radians; Doosan RT/services comments indicate degrees. |
| `velocity` | Six joint velocities. Units must be explicit. ROS control is expected radians per second; Doosan RT/services may be degrees per second. |
| `source_name` | Exact topic or service source, likely `/dsr01/joint_states` for `JointState`, but namespace must be verified live. |
| `source_type` | Source message type, likely `sensor_msgs/msg/JointState` or `dsr_msgs2/msg/RobotStateRt`. |
| `units` | Position and velocity units. |
| `record_index`, `source_stamp`, `receipt_stamp` | Required timing/provenance. |

Required TCP pose fields:

| Field | Requirement |
| --- | --- |
| `position` | TCP translation in raw units plus normalized target units. Doosan RT pose comments indicate millimeters; processed state should use meters. |
| `orientation` | Raw representation as recorded plus normalized representation for conversion. Doosan RT comments indicate `[x, y, z, a, b, c]` with Euler angles in degrees, likely ZYZ per static report; processed path expects `xyzw` quaternion before rotation-vector conversion. Do not label native Doosan Euler values as `rotation_vector_degrees` unless live verification proves that convention and a converter path supports it. |
| `pose_frame` | Base/world/user coordinate frame for the TCP pose. |
| `tcp_frame` | Active TCP/tool frame identifier. |
| `source_name` | Exact `ReadDataRt`, `GetCurrentPosx`, or TF source. |
| `source_type` | Source interface type, for example `dsr_msgs2/msg/RobotStateRt` through `ReadDataRt`. |
| `units` | Translation units and orientation units. |
| `record_index`, `source_stamp`, `receipt_stamp` | Required timing/provenance. |

Required wrench/force-torque fields:

| Field | Requirement |
| --- | --- |
| `wrench` | Six values `[Fx, Fy, Fz, Tx, Ty, Tz]`. |
| `signal_name` | Which signal was used: `raw_force_torque`, `external_tcp_force`, `GetToolForce`, or other verified source. |
| `source_name` | Exact source topic/service. No standard `geometry_msgs/WrenchStamped` topic was found by static inspection. |
| `source_type` | Doosan RT/service type or optional RT topic type. |
| `frame_id` | Sensor/tool/TCP/base frame for the force-torque values. |
| `units` | Expected N and Nm, but must be verified and recorded. |
| `compensation` | Whether raw, gravity-compensated, tool-compensated, or external estimate. Unknown until lab validation. |
| `record_index`, `source_stamp`, `receipt_stamp` | Required timing/provenance. |

Required robot mode/state/control mode fields:

| Field | Requirement |
| --- | --- |
| `robot_mode` | From `RobotStateRt.robot_mode` if available. |
| `robot_state` | From `RobotStateRt.robot_state` or `system/get_robot_state`. |
| `control_mode` | From `RobotStateRt.control_mode` if available. |
| `error_state` | Error/disconnection records if available from `error` or `robot_disconnection` topics. |
| `source_name`, `source_type` | Exact live source names/types. |
| `record_index`, `source_stamp`, `receipt_stamp` | Required timing/provenance. |

Required task instruction fields:

| Field | Requirement |
| --- | --- |
| `task_instruction` | Natural-language instruction, for example `Insert the peg into the hole.` |
| `task_id` or `task_name` | Stable task identifier if available. |
| `geometry_type` | Example current value: `round_peg_round_hole`. |
| `orientation_type` | Example current value: `vertical_insertion`. |
| `collection_method` | Example values from config: `hand_guided`, `keyboard_teleop`, `spacemouse`, `vr_controller`, `dummy`. |

Required episode success/failure metadata:

| Field | Requirement |
| --- | --- |
| `success` | Boolean, required by current raw metadata validation. |
| `failure_reason` | Nullable string, required by current raw metadata validation. Must be non-empty when `success` is `false`; use `null` only for successful reviewed episodes. |
| `episode_start`, `episode_stop` | Events with timestamps. |
| `abort_reason` | Required when an episode is aborted. |
| `operator_notes` | Optional free-text notes, but the field should exist or be represented in events. |

Required geometry and orientation metadata:

| Field | Requirement |
| --- | --- |
| `geometry_type` | Peg/hole geometry class. |
| `orientation_type` | Insertion orientation class. |
| `object_ids` | IDs for peg, hole, fixture, or scene assets when available. |
| `workspace_layout_id` | Identifier for table/fixture setup when available. |
| `tcp_definition` | Active TCP/tool configuration reference. |

Required source topic/service names:

| Field | Requirement |
| --- | --- |
| `source_name` | Exact ROS topic or service name for every stream. |
| `source_type` | Exact ROS message/service type for every stream. |
| `source_namespace` | Resolved namespace, for example `dsr01` if verified. |
| `source_qos` | QoS profile for high-rate/image streams when available. |
| `source_role` | Semantic role such as `joint_state`, `tcp_pose`, `external_camera`, `wrist_camera`, `wrench`, or `robot_state`. |

Required units and coordinate frame metadata:

| Field | Requirement |
| --- | --- |
| `joint_units` | Radians/radians per second for processed conversion target; raw source may be degrees/degrees per second and must be recorded. |
| `tcp_translation_units` | Raw Doosan units likely millimeters; processed target meters. |
| `tcp_orientation_units` | Raw Doosan units likely degrees; processed target quaternion `xyzw` and action target rotation vector. |
| `wrench_units` | N and Nm expected, but signal-specific convention must be verified. |
| `base_frame`, `tcp_frame`, `flange_frame`, `tool_frame` | Exact frame names used in pose/wrench conversion. |
| `camera_frame_ids` | External and wrist camera frame IDs. |
| `orientation_convention` | Raw Euler convention and normalized quaternion convention. |

## 5. Optional Raw Fields

Optional fields should be captured when available because they improve auditability, debugging, and future experiments. They should not be required for the first minimal conversion unless the lab setup proves they are stable.

| Optional field | Candidate source | Use |
| --- | --- | --- |
| Flange pose | `RobotStateRt.actual_flange_position` or `GetCurrentToolFlangePosx` | Audit TCP/tool offsets and resolve TCP/flange ambiguity. |
| Flange velocity | `RobotStateRt.actual_flange_velocity` | Diagnostics and future velocity-based checks. |
| Target joint state | `RobotStateRt.target_joint_position`, `RobotStateRt.target_joint_velocity`, or trajectories | Compare measured versus target state. |
| Target TCP state | `RobotStateRt.target_tcp_position`, `RobotStateRt.target_tcp_velocity` | Debug controller target tracking; not primary action. |
| Joint torques | `RobotStateRt.actual_joint_torque`, `raw_joint_torque`, `external_joint_torque`, `GetJointTorque`, `GetExternalTorque` | Future proprioceptive/force-aware analysis. |
| Motor torques | `RobotStateRt.actual_motor_torque` | Diagnostics. |
| Gripper state | Gripper controller state, action feedback, Doosan IO, Modbus, or separate driver | Required only if gripper is active in the task or action slot. |
| Commanded teleop twist | `/doosan_teleop/cmd_vel_6d` | Debug commanded operator input; do not use as measured action label. |
| MoveIt Servo command | `/dsr01/servo_node/delta_twist_cmds` and MoveIt Servo output trajectory | Debug control context; do not use as measured action label. |
| `JogMulti` requests | Wrapped/logged `dsr_msgs2/srv/JogMulti` calls | Useful if the recorder wraps a client or logs request context; do not actively call motion services for recording. |
| Joint trajectory commands | `/dsr01/dsr_moveit_controller/joint_trajectory` or action feedback | Debug planned/servo command path. |
| Camera intrinsics identifiers | Camera calibration config, serials, calibration database IDs | Link image streams to intrinsics without embedding full calibration in every record. |
| Camera extrinsics identifiers | Hand-eye/external calibration references | Link camera frames to robot/world frames. |
| Calibration version | `calibration_refs.json` | Reproducibility and schema migrations. |
| RT diagnostic fields | `jacobian_matrix`, `gravity_torque`, `mass_matrix`, `coriolis_matrix`, `singularity`, temperatures, IO | Useful for safety diagnostics and future model variants. |

## 6. How Measured 7D TCP Delta Action Should Be Computed Offline

The primary action label should be computed offline from measured consecutive TCP poses. It should not be computed from commanded teleop input, `JogMulti` requests, MoveIt Servo twists, or commanded trajectories. Commanded streams are useful context, but they are not equivalent to executed motion.

The existing pure-Python function `compute_measured_tcp_delta_action` implements the intended core shape:

```text
[dx, dy, dz, dRx, dRy, dRz, gripper_delta_or_zero]
```

Expected output shape:

```text
[7]
```

Action components:

| Component | Meaning |
| --- | --- |
| `dx, dy, dz` | Translation delta from measured TCP position at time `t` to measured TCP position at time `t+1`, after unit normalization and frame selection. |
| `dRx, dRy, dRz` | Relative rotation represented as a rotation vector. Existing code computes `q_rel = conjugate(q_t) * q_t1` using normalized `xyzw` quaternions, then converts to rotation vector. |
| `gripper_delta_or_zero` | Gripper position delta if reliable gripper state is available; otherwise zero. |

Offline computation requirements:

| Requirement | Detail |
| --- | --- |
| Use measured TCP poses | Prefer `RobotStateRt.actual_tcp_position` or verified `GetCurrentPosx`/TF-derived measured TCP pose. |
| Normalize units first | Convert raw Doosan TCP translations from millimeters to meters before computing deltas if the raw source uses millimeters. Convert raw degrees to radians before quaternion/rotation-vector calculations. |
| Resolve orientation convention | The ROS report says Doosan task pose comments indicate `[x, y, z, a, b, c]`, mm/deg, likely Euler ZYZ. This must be verified before conversion to `xyzw` quaternion. Until then, use a blocking marker such as `doosan_posx_euler_zyz_degrees`, not `rotation_vector_degrees`. |
| Choose frame explicitly | TCP deltas should be computed in the selected base/world/task frame consistently. Do not mix base, user, flange, or tool frames. |
| Handle quaternion sign | Quaternions `q` and `-q` are equivalent; shortest-rotation handling should remain in the conversion path. |
| Align frame pairs | Current `raw_real_v0` conversion uses the shared episode-level `record_index` as the aligned frame key. It does not timestamp-resample or interpolate streams. |
| Terminal action policy | The final processed frame has no next measured pose and should use terminal zero padding, then be excluded from LeRobot/ForceVLA export as current code already does. |
| Preserve commanded context separately | Teleop twist, Servo twist, JogMulti requests, and trajectories should remain in `command_context` or optional action streams for diagnostics only. |

## 7. Proposed Validation Stages

Validation should be layered so that raw capture problems are caught before processed conversion and export problems are caught before ForceVLA smoke tests.

| Stage | Purpose | Example checks |
| --- | --- | --- |
| Raw episode structural validation | Confirm the real raw episode folder is well-formed. | Required files exist; stream manifests are readable; required streams are present; schema version is known; no empty required streams. |
| Timestamp monotonicity | Confirm each stream can be ordered and aligned. | Sequential aligned `record_index`; monotonic `source_stamp` where expected; monotonic `receipt_stamp`; no negative episode-time values; report clock regressions. |
| Stream completeness | Confirm required observation streams exist across the episode. | Joint state records, TCP pose records, wrench records, both camera streams, robot state, TF, metadata, events, and calibration refs have adequate coverage. |
| Camera frame count and timestamp checks | Confirm image streams can be aligned to robot state. | Nonzero frame counts; image files/chunks exist; dimensions/encoding stable or documented; frame timestamps within tolerance of robot_state_rt by aligned `record_index`; dropped frame counts reported; external and wrist camera rates plausible. |
| Robot state numeric sanity checks | Confirm numeric robot streams are usable. | Finite values; six joints; plausible joint positions/velocities; no impossible TCP jumps; wrench finite; robot mode/state/control mode present; units declared. |
| TCP pose availability checks | Confirm action labels can be computed. | Consecutive TCP pose availability across processed timeline; orientation conversion possible; frame and TCP/tool metadata present; no missing pose pairs except terminal frame. |
| Processed episode validation | Confirm raw-to-processed conversion output matches existing schema. | Current `validate_processed_episode` checks metadata, frames, timestamps, image paths, 25D model state, 7D action, terminal padding, and final zero action. |
| Action chunk validation | Confirm ForceVLA action horizons are valid. | `build_future_action_chunk` checks 7D finite actions, horizon length, valid mask, padding count, and padding policy. |
| LeRobot export validation | Confirm skeleton or real export layout. | Current validators check skeleton metadata, staged image paths, parquet/video attempt outputs, multi-episode metadata, task/episode/index consistency. |
| ForceVLA smoke tests | Confirm model-facing compatibility. | Observation builder, transform input, tokenization input, and multi-episode ForceVLA smoke regression should be run in the validated lab ForceVLA environment when dependencies and ForceVLA repo are available. |

Recommended validation progression:

```text
raw structural validation
-> raw stream/timestamp validation
-> raw numeric/camera validation
-> raw-to-processed conversion
-> processed validation
-> export plan validation
-> staged export validation
-> LeRobot skeleton validation
-> real parquet/video export validation
-> ForceVLA smoke tests
```

## 8. What Can Be Implemented Now On Laptop

The laptop can safely support dependency-light planning, schema, and validation work. No live robot, ROS launch, ROS control code, package installation, or lab workspace edits are needed.

Safe immediate implementation candidates:

| Candidate | Why it is safe now |
| --- | --- |
| Documentation for raw real-episode schema | Pure docs/config; no live ROS assumptions. |
| JSON/YAML schema templates | Topic maps can use placeholder source names and mark unknowns. |
| Pure-Python raw real structural validators | Validate folders, JSON/JSONL syntax, required keys, timestamps, image index entries, and finite numeric arrays from fixtures. |
| Dummy real-episode fixtures | Generate small JSONL/image fixtures that mimic the proposed real schema without ROS imports. |
| Unit conversion helpers | Convert mm to m, degrees to radians, Euler/quaternion conversions if conventions are explicitly configured. |
| Record adapters for files, not ROS | Parse `robot_state_rt.jsonl`, `joint_states.jsonl`, camera indexes, and TF JSONL records produced by a future recorder. |
| Stream alignment tests | Test aligning camera frame timestamps to robot state timestamps using synthetic data. |
| Raw-to-processed adapter skeletons | Implement adapters that consume file records once field names are known; keep live topic names configurable. |
| Validation tests | Add tests for missing streams, non-monotonic timestamps, image count mismatch, missing TCP pose, non-finite robot state, and units/frame metadata. |
| Config examples | Add example topic map, robot source selection, camera stream config, metadata defaults, and unit/frame conventions with `unknown` placeholders. |

What should not be implemented on the laptop in this repo:

| Not now | Reason |
| --- | --- |
| ROS2 recorder node | Belongs in a separate ROS2 workspace/package. Live graph details are not verified. |
| `rclpy` imports in package code | Would make offline data tools depend on ROS. |
| Doosan service clients | Live service names, safety, and read-only behavior are not verified. |
| ROS launch files | User explicitly constrained this task away from ROS launch/control work. |
| Any package install or dependency declaration changes | Not needed for schema/validator planning. |

## 9. What Must Wait For Lab Workstation

These items are blocked by the live ROS graph, exact real workspace state, and safety-reviewed lab procedure.

| Blocked item | What must be verified in lab |
| --- | --- |
| Exact Doosan namespace | Whether sources resolve as `/dsr01/...`, `/joint_states`, `/dsr01/<controller_node_name>/...`, or another namespace. |
| Exact camera topics | External and wrist/TCP camera topics, types, encoding, resolution, frame IDs, QoS, FPS, and timestamp source. |
| `/dsr01/joint_states` availability | Whether the joint state topic exists, whether velocities are populated, and whether efforts are meaningful. |
| `ReadDataRt` service availability | Exact service name, safe polling rate, side effects/load concerns, response contents, and timestamp meaning. |
| Optional RT topic publishers | Whether `/rt_topic/<field>` is enabled or should be enabled for passive streaming; static config default is disabled. |
| Exact `RobotStateRt` units and conventions | Confirm joints, TCP pose, flange pose, velocities, force/torque, robot mode/state/control mode, and controller timestamp behavior. |
| TF tree | Exact base frame, flange frame, TCP/tool frame, camera frames, and static/dynamic transform availability. |
| TCP/flange/tool choice | Whether action should use controller TCP, flange plus tool offset, or MoveIt end-effector frame. |
| Force-torque signal choice | Whether `raw_force_torque`, `external_tcp_force`, `GetToolForce`, or another signal is usable for peg-in-hole; frame and sign convention must be validated. |
| Gripper state | Model, driver, topic/action/service, state feedback, IO/Modbus mapping, position units, and open/close convention. |
| Time sync | Synchronization between robot controller, ROS host, cameras, and recorder process; timestamp latency and jitter. |
| Exact MyROS2 commit/state | Static report found no Git metadata; real branch/commit/remotes/dirty state must be captured from lab source of truth. |
| Recorder safety checks | Confirm recorder creates no command publishers or command service/action clients and only uses read-only polling if approved. |

## 10. Recommended Implementation Order

Use small branches. Keep live ROS capture separate from this offline repository.

1. Branch 1: docs/config/schema only.

   Add or update documentation and example config templates for the proposed raw real-episode schema. Include placeholder topic maps and mark all live names as unverified. Do not add package code, ROS imports, or tests that assume live topics.

2. Branch 2: pure-Python raw real validators.

   Add validators for `metadata.json`, `streams/index.json`, JSONL records, required stream presence, timestamps, source names, units, frame IDs, camera indexes, and numeric sanity. Use synthetic fixtures only.

3. Branch 3: synthetic raw real fixtures.

   Add a tiny dependency-light dummy real-episode generator or test fixtures using the proposed `streams/` layout. Keep it file-only and deterministic.

4. Branch 4: raw real to processed adapter prototype.

   Add file adapters that convert `robot_state_rt.jsonl`, `joint_states.jsonl`, camera indexes, and wrench records into the existing processed schema. Use configured source mappings and synthetic data; do not import ROS packages.

5. Branch 5: stream alignment and measured action hardening.

   Add tests for camera-to-robot timestamp alignment, TCP pose interpolation or nearest-neighbor policy, unit conversion, quaternion conversion, and measured 7D action computation from real-schema fixtures.

6. Branch 6: lab live graph inventory capture outside this repo.

   On the lab workstation only, after normal safe bringup, capture `ros2 topic list -t`, `ros2 service list -t`, `ros2 action list -t`, selected `topic info -v`, TF tree, and one-message samples from observation sources. Do not run this from the laptop.

7. Branch 7: update config templates with verified lab names.

   Once live topics/services are known, update data-pipeline config templates and validators to accept verified source names and units. Keep configs separate from any recorder implementation.

8. Branch 8: raw-to-processed adapter for verified sources.

   Complete adapters for the verified recorder output. Add tests using saved raw sample files, not a live ROS graph.

9. Branch 9: separate ROS recorder package.

   Implement in the ROS2 workspace or a separate recorder repo. It should passively subscribe/poll read-only sources, write the agreed raw schema, and never publish or call motion-control interfaces.

10. Branch 10: lab recorder dry capture and offline regression.

   Record short stationary and simple real episodes in the lab under safety procedure, then process them through this repo’s validators, conversion, LeRobot export, and ForceVLA smoke tests.

## 11. Suggested Config Files

The data-pipeline repository can own templates for expected source names, units, frames, and conversion policy. These should be examples until lab verification fills in exact values.

Suggested topic map config: `configs/raw_recorder/topic_map.example.yaml`

```yaml
schema_version: raw_recorder_topic_map_v0
namespace: unknown_until_lab_verification
topics:
  joint_states:
    source_name: /dsr01/joint_states
    source_type: sensor_msgs/msg/JointState
    required: true
    verified: false
  tf:
    source_name: /tf
    source_type: tf2_msgs/msg/TFMessage
    required: true
    verified: false
  tf_static:
    source_name: /tf_static
    source_type: tf2_msgs/msg/TFMessage
    required: true
    verified: false
  external_camera:
    source_name: unknown
    source_type: sensor_msgs/msg/Image
    required: true
    verified: false
  wrist_camera:
    source_name: unknown
    source_type: sensor_msgs/msg/Image
    required: true
    verified: false
  commanded_teleop_twist:
    source_name: /doosan_teleop/cmd_vel_6d
    source_type: geometry_msgs/msg/Twist
    required: false
    verified: false
services:
  read_data_rt:
    source_name: /dsr01/realtime/read_data_rt
    source_type: dsr_msgs2/srv/ReadDataRt
    required: true
    read_only_candidate: true
    verified: false
```

Suggested robot source selection config: `configs/raw_recorder/robot_sources.example.yaml`

```yaml
schema_version: raw_robot_sources_v0
robot_model: doosan_m1013
primary_robot_state_source:
  kind: read_data_rt
  selected: false
  reason: candidate only; lab availability and safe polling rate unknown
joint_state_source:
  kind: joint_states_topic
  selected: true
  fallback: robot_state_rt.actual_joint_position
tcp_pose_source:
  kind: robot_state_rt.actual_tcp_position
  selected: true
  fallback: aux_control/get_current_posx
wrench_source:
  kind: robot_state_rt.external_tcp_force
  selected: false
  alternatives:
    - robot_state_rt.raw_force_torque
    - aux_control/get_tool_force
  reason: frame/sign/compensation convention unknown
command_context_sources:
  - /doosan_teleop/cmd_vel_6d
  - /dsr01/servo_node/delta_twist_cmds
  - /dsr01/dsr_moveit_controller/joint_trajectory
```

Suggested camera stream config: `configs/raw_recorder/cameras.example.yaml`

```yaml
schema_version: raw_camera_streams_v0
streams:
  external_camera:
    role: observation.image
    source_name: unknown
    source_type: sensor_msgs/msg/Image
    required: true
    encoding: unknown
    expected_width: null
    expected_height: null
    expected_fps: null
    frame_id: unknown
    intrinsics_id: unknown
    extrinsics_id: unknown
  wrist_camera:
    role: observation.wrist_image
    source_name: unknown
    source_type: sensor_msgs/msg/Image
    required: true
    encoding: unknown
    expected_width: null
    expected_height: null
    expected_fps: null
    frame_id: unknown
    intrinsics_id: unknown
    extrinsics_id: unknown
storage:
  first_choice: image_frames_with_index
  allowed:
    - image_frames_with_index
    - chunked_video_with_precise_index
```

Suggested episode metadata defaults config: `configs/raw_recorder/episode_metadata_defaults.example.yaml`

```yaml
schema_version: raw_episode_metadata_defaults_v0
dataset_name: doosan_peg_in_hole_v0
robot_type: doosan_m1013
task_instruction: Insert the peg into the hole.
geometry_type: round_peg_round_hole
orientation_type: vertical_insertion
collection_method: unknown_real_collection_method
action_label_primary: measured_tcp_delta
# Keep the default explicitly non-successful until the episode is reviewed.
# For a successful reviewed episode, set:
#   success: true
#   failure_reason: null
# For a failed reviewed episode, keep success false and replace the placeholder below.
success: false
failure_reason: unannotated_episode_replace_before_training
operator_id: unknown
site: lab_workstation
calibration_version: unknown
recorder_version: unknown
lab_provenance_required: true
source_workspace:
  path: /home/horus/robotics_thesis/lab_myros2_ws
  git_commit: unknown_static_report_found_no_git_metadata
```

Suggested unit/frame conventions config: `configs/raw_recorder/unit_frame_conventions.example.yaml`

```yaml
schema_version: raw_unit_frame_conventions_v0
processed_targets:
  tcp_translation_units: meters
  joint_position_units: radians
  joint_velocity_units: radians_per_second
  orientation_quaternion_convention: xyzw
  action_rotation_representation: rotation_vector
raw_candidates:
  doosan_robot_state_rt:
    tcp_translation_units: millimeters
    tcp_orientation_units: degrees
    tcp_orientation_convention: unknown_verify_euler_order
    converter_supported_tcp_orientation_conventions:
      - rotation_vector_degrees
      - rotation_vector_radians
    recognized_but_unsupported_tcp_orientation_conventions:
      - doosan_posx_euler_zyz_degrees
      - doosan_robotstate_actual_tcp_position_euler_zyz_degrees
      - euler_zyz_degrees
    current_converter_rejects_unknown_euler_conventions: true
    notes: Do not label native Doosan Euler pose values as rotation vectors unless live verification proves that convention and a converter path supports it.
    joint_position_units: degrees
    joint_velocity_units: degrees_per_second
    force_units: N
    torque_units: Nm
frames:
  base_frame: unknown
  tcp_frame: unknown
  flange_frame: unknown
  tool_frame: unknown
  external_camera_frame: unknown
  wrist_camera_frame: unknown
conversion_policy:
  compute_action_from: measured_consecutive_tcp_pose
  record_index_policy: episode_level_aligned_sample_index_for_converter_streams
  timestamp_alignment: diagnostics_only_no_interpolation_in_raw_real_v0
  block_large_required_camera_source_stamp_offsets: true
  command_streams_are_labels: false
  terminal_padding_action: [0, 0, 0, 0, 0, 0, 0]
```

## 12. Risks And Open Questions

The ROS2/Doosan static inspection report leaves several material uncertainties. They should be treated as blockers for live recorder implementation and for final raw-to-processed source mappings.

| Risk or open question | Current status from static inspection | Required lab resolution |
| --- | --- | --- |
| Exact camera topics unknown | Static source found `/real_camera/image` as a local OpenCV example and `/rgbd_camera/image` as Gazebo/simulation. No definitive real external or wrist/TCP camera driver/topic was found. | Use live `ros2 topic list -t` and `topic info -v` after safe bringup to identify camera topics, types, encodings, frame IDs, FPS, resolution, QoS, and timestamp source. |
| Exact Doosan namespace unknown | Likely `/dsr01/joint_states`, but services may resolve as `/dsr01/motion/...` or `/dsr01/<controller_node_name>/motion/...`. | Capture live topic/service/action lists and exact launch arguments. |
| `ReadDataRt` service availability/rate unknown | Candidate source is `dsr_msgs2/srv/ReadDataRt` returning `dsr_msgs2/msg/RobotStateRt`; static report identifies it as high-value. | Verify service name, safe polling rate, response fields, controller timestamp, load concerns, and read-only safety approval. |
| Force-torque signal frame/sign convention unknown | Candidate fields include `raw_force_torque`, `external_tcp_force`, and `GetToolForce`; no standard `WrenchStamped` topic was found. | Verify signal source, units, frame, sign convention, filtering, compensation, and suitability for peg-in-hole. |
| TCP/flange/tool frame ambiguity | `RobotStateRt` has TCP and flange poses; services expose TCP/flange pose and active TCP/tool names. | Decide whether action uses controller TCP, flange plus tool offset, MoveIt end-effector, or another task frame. Record active TCP/tool config per episode. |
| Gripper state unknown | Static report found gripper command/action examples and possible IO/Modbus routes, but no definitive state source. | Identify gripper model, state topic/action/service, state units, open/close convention, and whether gripper action slot should be active. |
| Time sync unknown | Static report cannot determine synchronization between controller, ROS host, cameras, and recorder. | Measure timestamp jitter and offsets with a stationary short recording; decide alignment policy and accepted tolerances. |
| `/joint_states` effort quality unknown | Hardware code exports effort interfaces, but comments indicate effort support is TODO/ignored. | Do not rely on `JointState.effort` until verified; prefer Doosan RT torque fields if needed. |
| Optional RT topic publishers disabled by default | `dsr_controller2` can publish `/rt_topic/<field>` but default `use_rt_topic_pub` is false. | Decide in lab whether to use read-only service polling or enable passive RT topic publishing under safety procedure. |
| `RobotState` publisher unknown | Message exists, but no active publisher was found in static source. | Verify live topics and avoid assuming this message is published. |
| Real workspace Git state unknown | Static report found no Git metadata at lab workspace root, `src/MyROS2`, or `doosan-robot2`. | Capture exact real source branch/commit/remotes/dirty state from the lab source of truth before modifying the recorder workspace. |
| Recorder ownership boundary | Static report recommends a separate recorder and offline data pipeline ownership for conversion/export/model transforms. | Keep live recorder outside this repo; keep this repo responsible for schema, validation, normalization, action computation, LeRobot export, and ForceVLA transforms. |

Final integration principle:

The ROS recorder should capture raw synchronized evidence from the live system. The offline data-pipeline repository should own interpretation, validation, normalization, episode slicing, measured 7D TCP delta computation, LeRobot export, and ForceVLA model-facing transforms.

### Strict lab/source provenance readiness

For non-synthetic real-lab episodes, set `lab_provenance_required: true` or `strict_lab_provenance: true` when the episode should be treated as conversion-ready lab data. In that mode, conversion readiness requires verified source workspace metadata, concrete live graph topic/frame/source names, `time_sync_verified: true`, known stream `source_name` values, `verified: true` stream entries, and known camera `frame_id` values. Placeholder values such as `unknown`, `todo`, or empty strings are readiness errors in strict mode.
