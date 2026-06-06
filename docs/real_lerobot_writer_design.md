# Real LeRobot Writer Design

This document defines the first real LeRobot-compatible writer design for the Doosan M1013 peg-in-hole dataset tools. It is implementation-oriented, but it is not an implementation. Do not write parquet, encode videos, upload to Hugging Face, integrate any4lerobot, or add ROS in this step.

## 1. Current Pipeline Recap

The current offline pipeline is:

```text
raw episode -> processed JSONL -> export plan -> staged export JSONL -> local LeRobot skeleton -> real-export preflight -> dependency-optional real-export attempt
```

Current stages:

- Raw episode: collection-method-independent robot state, wrench, images, metadata, events, and optional action-like streams.
- Processed JSONL: fixed-shape per-frame records with `model_state`, `measured_action`, timestamps, image references, and terminal action padding.
- Export plan: dry-run JSON manifest that chooses an export profile and verifies dimensions, keys, image availability, and terminal-frame exclusion.
- Staged export JSONL: inspectable records with future-facing keys such as `observation.image`, `observation.wrist_image`, `observation.state`, `action`, and `task`.
- Local LeRobot skeleton: v2.1-style metadata, JSONL placeholder records, and controlled `image_staging/` with symlink or copy mode.
- Real-export preflight: read-only readiness report for skeleton schema, prompt/task compatibility, image staging, and parquet/video dependencies.
- Dependency-optional real-export attempt: local-only scaffold that writes an export attempt report, writes metadata in `write-if-available` mode, and writes parquet/videos only when dependencies are present.

The staged export is useful for checking record shape before committing to a parquet/video writer, but it is not training-ready LeRobot data.

## 2. First Real Export Target

First target:

```text
profile: forcevla_13d
```

Reason:

- This is the first compatibility target for ForceVLA / tshiamor-style loading.
- It keeps the state compact and close to the expected force-aware TCP state used by the target learning setup.
- It preserves the collection-method-independent primary action label: measured TCP delta from robot state.

Mapping:

- `observation.image` = external RGB camera
- `observation.wrist_image` = TCP/wrist RGB camera
- `observation.state` = 13D state
- `action` = 7D measured TCP delta
- `task` = `task_instruction`
- `prompt` = `task_instruction`

`observation.state` layout:

```text
ee_pos(3) + ee_axis_angle(3) + gripper_pos(1) + wrench(6)
```

`action` layout:

```text
dx, dy, dz, dRx, dRy, dRz, gripper_delta_or_zero
```

## 3. Future Export Target

Future target:

```text
profile: doosan_full_25d
```

Purpose:

- Future full-proprioception experiments.
- Not the first ForceVLA compatibility target.

Mapping:

- `observation.image` = external RGB camera
- `observation.wrist_image` = TCP/wrist RGB camera
- `observation.state` = full 25D `model_state`
- `action` = 7D `measured_action`
- `task` = `task_instruction`
- `prompt` = `task_instruction`

`observation.state` layout:

```text
ee_pos(3) + ee_axis_angle(3) + gripper_pos(1) + wrench(6) + joint_pos(6) + joint_vel(6)
```

## 4. LeRobot Version Target

Recommendation:

```text
target LeRobot style: v2.1 first
```

Rationale:

- Current ForceVLA/tshiamor work is closest to LeRobot v2.1-style datasets.
- Do not target v3.0 first unless later ForceVLA loading explicitly requires v3.0 metadata and layout.
- Keep the local writer focused on a known-compatible format before adding version conversion complexity.

any4lerobot policy:

- any4lerobot may be useful later as a reference for version conversion and metadata conventions.
- Do not clone, vendor, import, or integrate any4lerobot now.
- Use it only as a reference after the local export mapping and writer behavior are stable.

## 5. Proposed Output Folder Layout

Future local export folder:

```text
data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0/
meta/
  info.json
  tasks.jsonl
  episodes.jsonl
  episodes_stats.jsonl
data/
  chunk-000/
    episode_000000.parquet
videos/
  observation.image/
    episode_000000.mp4
  observation.wrist_image/
    episode_000000.mp4
```

Notes:

- This layout is the future goal, not the current staged JSONL output.
- The next implementation should start with a local dummy export only.
- No Hugging Face upload should be implemented yet.
- The first writer should be local, inspectable, and easy to delete/recreate.

## Local LeRobot Skeleton Writer

The first implemented writer step creates a local v2.1-style skeleton folder before any real parquet or video work. It targets `forcevla_13d` by default as the first ForceVLA/tshiamor compatibility profile, while the same framework also supports `doosan_full_25d` as a secondary full-proprioception profile.

The skeleton writer creates:

- `meta/info.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- `data/chunk-000/episode_000000.jsonl`
- `image_staging/observation.image/episode_000000/`
- `image_staging/observation.wrist_image/episode_000000/`

The tabular data file is intentionally a JSONL placeholder in the future LeRobot-like data path, not parquet. The writer stages images into a controlled local tree using either symlinks for dummy/local development or copies for more portable snapshots. Skeleton frame records include both `task` and `prompt`, with `prompt` currently equal to `task` for ForceVLA/OpenPI compatibility. Existing skeleton outputs are rejected unless `--overwrite` is passed. It does not encode MP4 videos, does not write parquet, does not import LeRobot, does not upload to Hugging Face, and does not create training-ready LeRobot data yet.

## 6. Image Handling Decision

Recommendation:

- First implementation should create a controlled export staging image tree or symlink tree.
- Prefer copying or symlinking referenced images into the export workspace before video encoding.
- Do not encode directly from arbitrary raw references in the first implementation.

Reason:

- The export becomes reproducible.
- The export workspace is easier to inspect.
- Broken raw references are caught before video/parquet writing.
- The eventual video writer can consume a stable local path layout.

Option A: copy images

Pros:

- Portable export folder.
- Independent of raw data paths.
- Safer for archiving and moving between machines.

Cons:

- Uses more disk space.
- Slower for large datasets.

Option B: symlink images

Pros:

- Saves disk space.
- Faster for local dummy and development runs.
- Keeps the staging step lightweight.

Cons:

- Less portable.
- Can break if raw data moves.
- Some tools or filesystems handle symlinks differently.

Recommendation by use case:

- For dummy/local export, symlink is acceptable.
- For real dataset snapshots, copy images or encode videos into final export artifacts.

## 7. Video/Parquet Dependency Plan

Do not install anything now.

Likely future dependencies:

- `pyarrow` or `pandas` for parquet if writing parquet manually.
- `imageio`, OpenCV, ffmpeg, moviepy, or LeRobot's own utilities for video encoding.
- `lerobot` package if using LeRobotDataset APIs directly.

Recommended dependency workflow:

- First check what is already installed on the laptop.
- Run `PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.check_export_dependencies` without installing anything.
- Run `PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.preflight_real_export <skeleton_dir>` on a validated skeleton.
- Run `PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_real_lerobot_export --skeleton <skeleton_dir> --output <output_dir> --mode dry-run` to inspect what would be attempted.
- Run `--mode write-if-available` only when a local attempt should write metadata and dependency-available artifacts.
- Repeat the same dependency check on the lab workstation inside the validated ForceVLA environment.
- Repeat the same real-export preflight on the lab workstation inside the validated ForceVLA environment.
- Repeat the same real-export attempt on the lab workstation before treating parquet/video compatibility as valid.
- Treat missing laptop dependencies as informational, not final blockers.
- Prefer LeRobot's official dataset creation utilities if available and compatible with the selected v2.1-style layout.
- If official utilities are unavailable or incompatible, write minimal parquet/video manually only after confirming the exact required schema.
- Do not add dependency declarations until the chosen writer path is validated locally.

## 8. Terminal-Frame Policy

Policy:

- Terminal-padded final frames should be excluded from training/export by default.
- Keep this behavior for `forcevla_13d` and `doosan_full_25d` unless later ForceVLA loading requires a different convention.

Rationale:

- The final padded action is synthetic zero action, not a measured next-step action.
- Excluding it avoids teaching the model an artificial terminal zero action.
- The current dry-run export plan and staged export already follow this behavior.

## 9. Missing Third Camera Decision

Current setup:

- One external RGB camera.
- One TCP/wrist RGB camera.

First real writer policy:

- Do not generate a fake third camera in the first real writer.
- Use `observation.image`.
- Use `observation.wrist_image`.

If ForceVLA later requires a third camera:

- Handle it in the ForceVLA transform/config layer.
- Prefer zero-filled or duplicated input there, not in the raw dataset by default.
- Keep raw and processed datasets faithful to actually recorded sensors.

## 10. Data Fields and Dtype Plan

Planned feature schema for `forcevla_13d`:

- `observation.image`: video or image stream, RGB, `uint8`
- `observation.wrist_image`: video or image stream, RGB, `uint8`
- `observation.state`: `float32`, shape `[13]`
- `action`: `float32`, shape `[7]`
- `timestamp`: `float32` or `float64`
- `frame_index`: `int64`
- `episode_index`: `int64`
- `task_index`: `int64`
- `index`: `int64`
- `task`: string or `task_index` with `tasks.jsonl`
- `prompt`: string, equal to `task` for the first ForceVLA/OpenPI compatibility check

Planned feature schema for `doosan_full_25d`:

- `observation.image`: video or image stream, RGB, `uint8`
- `observation.wrist_image`: video or image stream, RGB, `uint8`
- `observation.state`: `float32`, shape `[25]`
- `action`: `float32`, shape `[7]`
- `timestamp`: `float32` or `float64`
- `frame_index`: `int64`
- `episode_index`: `int64`
- `task_index`: `int64`
- `index`: `int64`
- `task`: string or `task_index` with `tasks.jsonl`
- `prompt`: string, equal to `task` for the first ForceVLA/OpenPI compatibility check

Implementation note:

- Keep internal JSONL floats as Python floats for now.
- Cast to `float32` only at the actual parquet/video writer boundary.

## 11. Episodes Metadata Plan

Planned episode-level metadata fields:

- `episode_index`
- `length`
- `task_index`
- `success`
- `geometry_type`
- `orientation_type`
- `collection_method`
- `source_processed_episode`
- `export_profile`

Additional metadata to preserve where useful:

- `dataset_name`
- `robot_type`
- `fps`
- `action_label_primary`
- `quaternion_convention`
- `terminal_padding_excluded`

## 12. Open Questions Before Implementation

Concrete open questions:

- Should image export use copy or symlink for the first dummy writer?
- Should the writer use LeRobot API or a manual parquet/video writer?
- Is `pyarrow` already installed on this laptop?
- Is `lerobot` installed on this laptop?
- Should dummy PPM images be converted to PNG/JPEG before video encoding?
- Should real robot export use MP4 videos or image folders first?
- Should `timestamp` be stored as `float32` for compactness or `float64` for precision?
- Should `task` be stored directly per row or only via `task_index` plus `tasks.jsonl`?
- Does the ForceVLA loader expect `prompt`, `task`, or both?
- Should parquet contain image/video reference columns, or should videos be referenced only through `info.json` metadata?
- Should the first local writer create symlinks by default and offer a copy mode later?

## 13. Next Implementation Recommendation

Next coding task:

Run the dependency check, real-export preflight, and dependency-optional real-export attempt on the lab workstation inside the validated ForceVLA environment. Use those lab results to decide whether to harden the pyarrow writer, switch to LeRobot APIs, or adjust video encoding.

The lab preflight should check:

- skeleton metadata and JSONL schema
- `task` and `prompt` compatibility fields
- image staging completeness
- dependency availability for parquet and video writing

The next implementation should:

- Preserve `forcevla_13d` as the first profile.
- Keep `doosan_full_25d` as secondary.
- Treat laptop dependency results as informational.
- Use lab ForceVLA preflight results as the compatibility source of truth.
- Avoid parquet and videos until dependencies and LeRobot schema are confirmed.

Do not implement:

- parquet writing
- video encoding
- Hugging Face upload
- ROS recording
- any4lerobot integration
- package installation
