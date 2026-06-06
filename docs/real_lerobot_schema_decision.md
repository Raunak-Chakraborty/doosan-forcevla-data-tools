# Real LeRobot Schema Decision

This document records the concrete schema direction before implementing real parquet or video export. It is implementation-oriented, but it is not an implementation. Do not write parquet, encode videos, use Hugging Face, import LeRobot, integrate any4lerobot, install packages, or add ROS in this step.

## 1. Current Status

The current output is a local LeRobot-style skeleton export. It creates v2.1-style metadata folders, a JSONL placeholder in the future data path, and a controlled `image_staging/` tree.

Current skeleton output includes:

- `meta/info.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- `data/chunk-000/episode_000000.jsonl`
- `image_staging/observation.image/episode_000000/`
- `image_staging/observation.wrist_image/episode_000000/`

Current limitations remain intentional:

- no parquet
- no videos
- no Hugging Face upload
- no LeRobot API usage
- no any4lerobot integration
- no ROS recorder

## 2. First Real Writer Target

First target:

```text
profile: forcevla_13d
```

Mapping:

- `observation.image` = external RGB camera
- `observation.wrist_image` = TCP/wrist RGB camera
- `observation.state` = 13D: `ee_pos(3) + ee_axis_angle(3) + gripper_pos(1) + wrench(6)`
- `action` = 7D measured TCP delta: `dx, dy, dz, dRx, dRy, dRz, gripper_delta_or_zero`
- `task` = `task_instruction`
- `prompt` = `task_instruction`

`prompt` is added for ForceVLA/OpenPI compatibility. `task` is kept for readability and LeRobot-style compatibility. For now, `prompt` should duplicate `task` exactly.

## 3. Secondary Writer Target

Secondary target:

```text
profile: doosan_full_25d
```

Mapping:

- `observation.image` = external RGB camera
- `observation.wrist_image` = TCP/wrist RGB camera
- `observation.state` = full 25D: `ee_pos(3) + ee_axis_angle(3) + gripper_pos(1) + wrench(6) + joint_pos(6) + joint_vel(6)`
- `action` = measured_action 7D
- `task` = `task_instruction`
- `prompt` = `task_instruction`

This target remains secondary. It is for later full-proprioception experiments. The first ForceVLA loading check should use `forcevla_13d`.

## 4. Final Intended LeRobot V2.1-Style Layout

Final intended local export layout:

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
data/chunk-000/episode_000000.parquet
videos/observation.image/episode_000000.mp4
videos/observation.wrist_image/episode_000000.mp4
```

The current skeleton uses `data/chunk-000/episode_000000.jsonl` as a placeholder until the parquet schema is confirmed.

## 5. Parquet Columns For First Real forcevla_13d Writer

Recommended parquet columns for the first real `forcevla_13d` writer:

- `observation.state`
- `action`
- `timestamp`
- `frame_index`
- `episode_index`
- `task_index`
- `index`
- `task`
- `prompt`

Column details:

- `observation.state` is 13D `float32`.
- `action` is 7D `float32`.
- `prompt` duplicates `task` for ForceVLA/OpenPI compatibility.
- `task` is kept for readability and LeRobot-style compatibility.
- Images/videos should not be embedded directly into parquet in the first writer.
- Videos should be handled separately under `videos/`.

## 6. Image/Video Plan

Recommendation:

- Use the controlled `image_staging/` tree as the video source.
- First video writer should encode MP4 from `image_staging/`, not from arbitrary raw references.
- Direct video encoding from raw references can be a later optimization.

Video targets:

```text
videos/observation.image/episode_000000.mp4
videos/observation.wrist_image/episode_000000.mp4
```

## 7. info.json Feature Plan

For the final real v2.1-style writer, planned features are:

- `observation.image`: dtype `video`, shape `[height, width, 3]` if known
- `observation.wrist_image`: dtype `video`, shape `[height, width, 3]` if known
- `observation.state`: dtype `float32`, shape `[13]` for `forcevla_13d` or `[25]` for `doosan_full_25d`
- `action`: dtype `float32`, shape `[7]`
- `timestamp`: dtype `float64`, shape `[1]`
- `frame_index`: dtype `int64`, shape `[1]`
- `episode_index`: dtype `int64`, shape `[1]`
- `task_index`: dtype `int64`, shape `[1]`
- `index`: dtype `int64`, shape `[1]`
- `task`: dtype `string`, shape `[1]`
- `prompt`: dtype `string`, shape `[1]`

## 8. Terminal Frame Policy

Terminal-padded final frames remain excluded from export/training by default. The synthetic final zero action is not a real demonstrated next-step action and should not be used as a primary learning label.

## 9. Missing Third Camera Policy

The current dataset writer should not generate a fake third camera. Use only:

- `observation.image`
- `observation.wrist_image`

If a ForceVLA transform later needs a third image, handle zero-fill or duplication in the ForceVLA transform/config layer, not in the raw dataset writer.

## 10. Laptop Versus Lab Workstation Workflow

The laptop/OpenCode environment is where most code-writing happens. The lab workstation is where real ForceVLA compatibility must be validated because it has the validated ForceVLA repository/environment.

Workflow rules:

- Run the dependency check on the laptop for local awareness.
- Run the same dependency check on the lab workstation inside the validated ForceVLA environment.
- Missing laptop dependencies do not block code development.
- ForceVLA loader, normalization-statistics, and training checks must happen on the lab workstation.

## 11. Open Questions Before Real Parquet/Video Implementation

Open questions:

- What is the exact image resolution source for real camera streams?
- Should dummy PPM images be converted before video encoding?
- Should the first real parquet writer use `pyarrow` directly or LeRobot APIs?
- Does the ForceVLA loader expect `prompt`, `task`, or both?
- Should parquet contain image/video reference struct columns or omit image columns and rely on `info.json` video metadata?
- Are `pyarrow`, `imageio`, `ffmpeg`, and `lerobot` available on the lab workstation ForceVLA environment?

## 12. Next Coding Step After This Document

Recommended next coding task:

Implement a real-export preflight command that:

- reads a skeleton export
- checks dependencies
- verifies metadata/schema choices
- reports whether parquet/video writing is possible
- does not write parquet or videos yet
