# Doosan ForceVLA Data Tools

This repository contains laptop-only dataset tools for a Doosan M1013 peg-in-hole robotics thesis. The near-term goal is to define and validate a clean dataset foundation that can later be converted into ForceVLA, tshiamor-style, or LeRobot-style imitation-learning datasets.

The tools are intentionally independent of how demonstrations are collected. A future raw episode may come from hand-guiding, keyboard or joystick teleoperation, SpaceMouse, VR controllers, shadow-mode policy logging, or another method. The processed learning target should not require any one of those collection paths.

## Dataset Layers

Layer A is the raw dataset. It stores what was observed during an episode: robot state, camera frames, force/torque measurements, metadata, events, and any optional action streams that happened to exist during collection. Raw data should preserve collection context without forcing a learning schema too early.

Layer B is the processed dataset. It converts raw episodes into per-frame tensors and labels suitable for imitation learning. The current v0 processed contract includes external and wrist camera streams, TCP state, wrench, joint state, a 25-dimensional `model_state`, and a 7-dimensional measured TCP delta action.

## Primary Action Label

The primary action label is `measured_tcp_delta`:

```text
action[t] = TCP pose delta from robot state at t to robot state at t + 1
```

This is the safest v0 label because it exists for hand-guided demonstrations and teleoperated demonstrations alike. It is computed from measured robot state, not from a specific input device or controller API.

The v0 action layout is:

```text
dx, dy, dz, dRx, dRy, dRz, gripper_delta_or_zero
```

Rotations use the relative quaternion convention `q_rel = conjugate(q_t) * q_t1`, then convert `q_rel` to a rotation vector. Quaternions are stored as `xyzw`.

## Optional Action Streams

Some collection methods may also provide action-like signals. These are optional streams, not requirements for the v0 learning label:

- `commanded_action` or `commanded_twist`, when teleoperation commands are logged
- `predicted_action`, when a policy is running in shadow mode
- `selected_action`, when a controller chooses among available streams

These streams can be useful for analysis, debugging, or future training variants, but the initial pipeline must not depend on them.

## Current Scope

This repository currently provides:

- a human-readable v0 dataset contract in `configs/doosan_v0.yaml`
- standard-library schema constants and metadata dataclasses
- pure-Python measured TCP delta action computation
- a tiny dummy raw episode generator
- a raw episode validator CLI
- a simple raw-to-processed JSONL converter
- a processed episode validator CLI
- a processed episode inspector CLI
- a dry-run LeRobot / ForceVLA export manifest planner
- a staged JSONL export dry run for planned profiles
- a local LeRobot-style skeleton writer with JSONL placeholder records and staged image references
- a dependency preflight CLI for future real parquet/video export
- a real-export preflight CLI that checks skeleton schema, image staging, prompt/task compatibility, and dependency readiness
- a dependency-optional local real-export writer scaffold with dry-run and write-if-available modes
- standard-library `unittest` tests

## Current Pipeline Status

The current v0 pipeline is:

```text
raw dummy episode -> validate raw -> convert to processed -> validate processed -> inspect processed -> plan dry-run export -> validate export plan -> stage export JSONL -> validate staged export -> write LeRobot skeleton -> validate LeRobot skeleton -> preflight real export -> attempt local real export
```

Example commands:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.dummy.make_dummy_raw_episode --output data/raw_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_raw_episode data/raw_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.raw_to_processed --raw data/raw_dummy/episode_000000 --output data/processed_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_processed_episode data/processed_dummy/episode_000000
```

The processed output is still a small human-readable JSONL manifest with image path references. The local skeleton output is closer to the planned LeRobot folder layout, but it is still not LeRobot parquet export and it is not yet ForceVLA training-ready.

## Processed Inspection Command

Inspect a processed episode with:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.inspect_processed_episode data/processed_dummy/episode_000000
```

The inspector reports timestamp span separately from nominal frame coverage. For example, a 20-frame episode at 30 FPS starting at timestamp 0 has a timestamp span of `19 / 30` seconds and nominal frame coverage of `20 / 30` seconds.

The next export target is planned as `forcevla_13d` first, using external RGB, TCP RGB, a 13D state subset, and the measured TCP delta action. The `doosan_full_25d` profile is kept for future full-proprioception experiments.

## Dry-run Export Planning

Plan the first ForceVLA-compatible profile without writing parquet or videos:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.plan_lerobot_export --processed data/processed_dummy/episode_000000 --profile forcevla_13d --output data/processed_dummy/episode_000000/export_plan_forcevla_13d.json
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_export_plan data/processed_dummy/episode_000000/export_plan_forcevla_13d.json
```

Plan the future full-proprioception profile:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.plan_lerobot_export --processed data/processed_dummy/episode_000000 --profile doosan_full_25d --output data/processed_dummy/episode_000000/export_plan_doosan_full_25d.json
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_export_plan data/processed_dummy/episode_000000/export_plan_doosan_full_25d.json
```

These export plans are dry runs only. They verify keys, dimensions, image availability, and terminal-frame exclusion, but do not write training-ready data.

## Staged Export Dry Run

Stage inspectable JSONL records for `forcevla_13d` without copying images, encoding videos, or writing parquet:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.stage_lerobot_export --processed data/processed_dummy/episode_000000 --export-plan data/processed_dummy/episode_000000/export_plan_forcevla_13d.json --output data/staged_dummy/forcevla_13d/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_staged_export data/staged_dummy/forcevla_13d/episode_000000
```

Stage the future full-proprioception profile:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.stage_lerobot_export --processed data/processed_dummy/episode_000000 --export-plan data/processed_dummy/episode_000000/export_plan_doosan_full_25d.json --output data/staged_dummy/doosan_full_25d/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_staged_export data/staged_dummy/doosan_full_25d/episode_000000
```

Staged frames are inspectable JSONL records with `observation.image`, `observation.wrist_image`, `observation.state`, `action`, and `task`. This is still not training-ready data. There is no parquet, no video encoding, and no image copying yet. The next step after this is the local LeRobot-style skeleton writer.

## Local LeRobot Skeleton Export

Write a v2.1-style local skeleton for the first target profile with symlinked staged images and JSONL placeholder rows instead of parquet:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_lerobot_skeleton --staged data/staged_dummy/forcevla_13d/episode_000000 --output data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --episode-index 0 --task-index 0 --profile forcevla_13d --image-mode symlink --overwrite
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_lerobot_skeleton data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0
```

Write the secondary full-proprioception skeleton with copied staged images:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_lerobot_skeleton --staged data/staged_dummy/doosan_full_25d/episode_000000 --output data/lerobot_dummy/doosan_full_25d/doosan_peg_in_hole_v0 --episode-index 0 --task-index 0 --profile doosan_full_25d --image-mode copy --overwrite
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_lerobot_skeleton data/lerobot_dummy/doosan_full_25d/doosan_peg_in_hole_v0
```

The skeleton output creates `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, `meta/episodes_stats.jsonl`, `data/chunk-000/episode_000000.jsonl`, and an `image_staging/` tree. Skeleton frame records include both `task` and `prompt`, with `prompt` currently equal to `task` for ForceVLA/OpenPI compatibility. Existing skeleton outputs are not overwritten unless `--overwrite` is passed. The skeleton still does not write parquet, encode videos, import LeRobot, upload to Hugging Face, or create training-ready LeRobot data.

## Export Dependency Preflight

Check optional future real-export dependencies without installing anything:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.check_export_dependencies
```

This command checks Python, `pyarrow`, `pandas`, `lerobot`, `cv2`, `imageio`, `PIL`, and `ffmpeg`. Missing laptop dependencies are not final blockers because real ForceVLA compatibility must be validated later on the lab workstation inside the validated ForceVLA environment. Run the same command on both the laptop and the lab workstation.

## Real Export Preflight

Check whether a skeleton export is structurally ready for future parquet/video work:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.preflight_real_export data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0
```

Optionally write the preflight report as JSON:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.preflight_real_export data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --output data/lerobot_dummy/forcevla_13d/preflight_report.json
```

The preflight validates the skeleton, checks schema and prompt/task compatibility, counts staged images, checks dependency availability, and reports parquet/video readiness. It does not write parquet or videos. Run it on the laptop for local awareness, then run it again on the lab workstation inside the validated ForceVLA environment.

## Local Real Export Attempt

Run a dry run that writes only `export_attempt_report.json`:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_real_lerobot_export --skeleton data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --output data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --mode dry-run
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_real_lerobot_export_attempt data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0
```

Attempt metadata, parquet, and video outputs only when dependencies are available:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_real_lerobot_export --skeleton data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --output data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --mode write-if-available
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_real_lerobot_export_attempt data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0
```

In `dry-run` mode, no metadata, parquet, videos, or images are copied. In `write-if-available` mode, metadata is written, parquet is written only if `pyarrow` is available, and videos are attempted only if video dependencies are available. Missing laptop dependencies are fine; the report says exactly what was skipped. Run the same command later on the lab workstation inside the validated ForceVLA environment.

## Real LeRobot Writer Design

The first real writer design is documented in `docs/real_lerobot_writer_design.md`. The concrete schema decision before real parquet/video work is documented in `docs/real_lerobot_schema_decision.md`. The first target remains `forcevla_13d`, with `doosan_full_25d` kept as a secondary full-proprioception target.

## Lab Workstation Validation

See `docs/lab_workstation_validation_checklist.md` for the lab workstation validation workflow using the validated ForceVLA environment.

## Limitations

- No ROS dependency is included yet.
- No integration with the lab `MyROS2` workspace is assumed.
- No real robot recorder is implemented yet.
- No LeRobot, parquet, Hugging Face, or ForceVLA training export is implemented yet.
- Dummy image files are tiny PPM placeholders, not real camera captures.
- The v0 raw validator checks structure and basic numeric validity only; it does not validate calibration, time synchronization quality, or task semantics.
- The v0 processed output uses JSONL records for inspection and testing; it is not a final training storage format.
- The export planner writes a small JSON dry-run manifest only; it does not create LeRobot parquet or videos.
- The staged export writes inspectable JSONL records only; it still does not create training-ready LeRobot data.
- The LeRobot skeleton writer creates v2.1-style folders, JSONL placeholder records, and staged images only; it still does not write parquet or encode videos.
- The real-export preflight checks readiness only; it does not write parquet or videos.
- The local real-export writer scaffold is dependency-optional; it may skip parquet or videos and report why.
- Laptop dependency availability is informative only; real ForceVLA loader and training compatibility must be checked on the lab workstation.

## Example Commands

If the package is not installed, run with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.check_export_dependencies
PYTHONPATH=src python3 -m doosan_forcevla_data.dummy.make_dummy_raw_episode --output data/raw_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_raw_episode data/raw_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.raw_to_processed --raw data/raw_dummy/episode_000000 --output data/processed_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_processed_episode data/processed_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.inspect_processed_episode data/processed_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.plan_lerobot_export --processed data/processed_dummy/episode_000000 --profile forcevla_13d --output data/processed_dummy/episode_000000/export_plan_forcevla_13d.json
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_export_plan data/processed_dummy/episode_000000/export_plan_forcevla_13d.json
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.plan_lerobot_export --processed data/processed_dummy/episode_000000 --profile doosan_full_25d --output data/processed_dummy/episode_000000/export_plan_doosan_full_25d.json
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_export_plan data/processed_dummy/episode_000000/export_plan_doosan_full_25d.json
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.stage_lerobot_export --processed data/processed_dummy/episode_000000 --export-plan data/processed_dummy/episode_000000/export_plan_forcevla_13d.json --output data/staged_dummy/forcevla_13d/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_staged_export data/staged_dummy/forcevla_13d/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.stage_lerobot_export --processed data/processed_dummy/episode_000000 --export-plan data/processed_dummy/episode_000000/export_plan_doosan_full_25d.json --output data/staged_dummy/doosan_full_25d/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_staged_export data/staged_dummy/doosan_full_25d/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_lerobot_skeleton --staged data/staged_dummy/forcevla_13d/episode_000000 --output data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --episode-index 0 --task-index 0 --profile forcevla_13d --image-mode symlink --overwrite
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_lerobot_skeleton data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.preflight_real_export data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --output data/lerobot_dummy/forcevla_13d/preflight_report.json
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_real_lerobot_export --skeleton data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --output data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --mode dry-run
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_real_lerobot_export_attempt data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_real_lerobot_export --skeleton data/lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --output data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0 --mode write-if-available
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_real_lerobot_export_attempt data/real_lerobot_dummy/forcevla_13d/doosan_peg_in_hole_v0
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.write_lerobot_skeleton --staged data/staged_dummy/doosan_full_25d/episode_000000 --output data/lerobot_dummy/doosan_full_25d/doosan_peg_in_hole_v0 --episode-index 0 --task-index 0 --profile doosan_full_25d --image-mode copy --overwrite
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_lerobot_skeleton data/lerobot_dummy/doosan_full_25d/doosan_peg_in_hole_v0
```
