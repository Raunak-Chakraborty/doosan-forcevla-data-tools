# LeRobot / ForceVLA Export Mapping Plan

This document is a mapping plan only. Do not implement parquet export, video encoding, Hugging Face upload, or any4lerobot integration until the internal raw-to-processed pipeline is stable.

## Current Internal Processed Format

The current processed episode directory contains:

- `metadata_processed.json`
- `frames.jsonl`

`metadata_processed.json` stores episode-level fields such as dataset name, robot type, fps, task metadata, dimensions, and the source raw episode path.

`frames.jsonl` stores one JSON object per frame with image references, timestamp, `model_state`, `measured_action`, and `action_is_terminal_padding`.

`model_state` is 25D:

```text
ee_pos(3) + ee_axis_angle(3) + gripper_pos(1) + wrench(6) + joint_pos(6) + joint_vel(6)
```

`measured_action` is 7D:

```text
dx, dy, dz, dRx, dRy, dRz, gripper_delta_or_zero
```

Images are stored as references to raw episode image files. The current converter does not copy or encode images.

## Export Profile A: forcevla_13d

Purpose:

- First compatibility target for official ForceVLA / tshiamor-like loading.

Observation mapping:

- `observation.image` = `external_rgb`
- `observation.wrist_image` = `tcp_rgb`
- Optional missing third camera = zero-filled later by ForceVLA transform or export profile
- `observation.state` = 13D: `ee_pos(3) + ee_axis_angle(3) + gripper_pos(1) + wrench(6)`
- `action` = `measured_action` 7D
- `task` = `task_instruction`

Terminal frame policy:

- Exclude final terminal-padded frame from training/export by default.

## Export Profile B: doosan_full_25d

Purpose:

- Future full-proprioception experiments.

Observation mapping:

- `observation.image` = `external_rgb`
- `observation.wrist_image` = `tcp_rgb`
- `observation.state` = full 25D `model_state`
- `action` = `measured_action` 7D
- `task` = `task_instruction`

Terminal frame policy:

- Exclude final terminal-padded frame from training/export by default.

## Image Handling Decision

For now:

- Processed JSONL stores image references.

Future LeRobot export should either:

- Copy images into export staging before video encoding, or
- Encode videos directly from referenced raw images.

The current implementation sequence first produces a dry-run manifest, then staged JSONL records, then a local LeRobot-style skeleton with controlled image staging. Direct video encoding from arbitrary raw references remains a future option, not the current step.

## Dry-run Export Manifest

The dry-run export manifest is the next step before real LeRobot parquet export. It reads a validated processed JSONL episode and reports what would be exported for a selected profile without writing training-ready data.

It verifies:

- selected export profile
- planned LeRobot-like keys
- observation and action dimensions
- image availability
- terminal-frame exclusion
- first exported record shape preview

It still does not:

- write parquet
- encode videos
- copy images
- upload to Hugging Face
- import LeRobot

## Staged Export Dry Run

The staged export dry run is an inspectable JSONL representation of what the future LeRobot / ForceVLA export records would look like for a selected profile.

It writes:

- `metadata_staged.json`
- `frames.jsonl`

Each staged frame uses future-facing keys:

- `observation.image`
- `observation.wrist_image`
- `observation.state`
- `action`
- `task`

For `forcevla_13d`, `observation.state` is 13D:

```text
ee_pos(3) + ee_axis_angle(3) + gripper_pos(1) + wrench(6)
```

For `doosan_full_25d`, `observation.state` is the full 25D `model_state`.

The staged export still does not write training-ready data. It does not write parquet, encode videos, copy images, upload to Hugging Face, import LeRobot, or integrate any4lerobot. Images are referenced by path only.

## Local LeRobot Skeleton Writer

The local skeleton writer comes after staged export and before real parquet/video export.

It writes a v2.1-style folder skeleton:

- `meta/info.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- `data/chunk-000/episode_000000.jsonl`
- `image_staging/observation.image/episode_000000/`
- `image_staging/observation.wrist_image/episode_000000/`

The skeleton writer supports `forcevla_13d` as the default first target and `doosan_full_25d` as the secondary full-proprioception target. It stages images by symlink or copy, writes JSONL placeholder records instead of parquet, excludes terminal padding frames, and still does not encode videos, upload to Hugging Face, import LeRobot, or integrate any4lerobot.

## any4lerobot Usage

- Use only as a reference for LeRobot metadata and version conventions.
- Do not clone, vendor, import, or integrate it until our own export mapping is stable.

## Next Implementation After This Planning Step

After the local skeleton output is stable, decide the actual LeRobot parquet/video writer design. That decision should explicitly choose whether to continue using staged copied/symlinked images before encoding or support direct encoding from referenced raw image paths as a future option.
