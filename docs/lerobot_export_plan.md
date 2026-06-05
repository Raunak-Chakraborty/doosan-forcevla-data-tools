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

Do not decide the implementation yet. The next step should only produce a dry-run manifest.

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

## any4lerobot Usage

- Use only as a reference for LeRobot metadata and version conventions.
- Do not clone, vendor, import, or integrate it until our own export mapping is stable.

## Next Implementation After This Planning Step

After the dry-run manifest is stable, implement the first real export staging step. That should still avoid Hugging Face upload and should explicitly decide whether images are copied into staging or encoded directly from references.
