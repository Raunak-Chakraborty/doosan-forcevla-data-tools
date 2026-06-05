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
- standard-library `unittest` tests

## Current Pipeline Status

The current v0 pipeline is:

```text
raw dummy episode -> validate raw -> convert to processed -> validate processed
```

Example commands:

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.dummy.make_dummy_raw_episode --output data/raw_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_raw_episode data/raw_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.raw_to_processed --raw data/raw_dummy/episode_000000 --output data/processed_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_processed_episode data/processed_dummy/episode_000000
```

The processed output is still a small human-readable JSONL manifest with image path references. It is not yet LeRobot parquet export and it is not yet ForceVLA training-ready. The next future step after this conversion layer is LeRobot export planning.

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

## Limitations

- No ROS dependency is included yet.
- No integration with the lab `MyROS2` workspace is assumed.
- No real robot recorder is implemented yet.
- No LeRobot, parquet, Hugging Face, or ForceVLA training export is implemented yet.
- Dummy image files are tiny PPM placeholders, not real camera captures.
- The v0 raw validator checks structure and basic numeric validity only; it does not validate calibration, time synchronization quality, or task semantics.
- The v0 processed output uses JSONL records for inspection and testing; it is not a final training storage format.
- The export planner writes a small JSON dry-run manifest only; it does not create LeRobot parquet or videos.

## Example Commands

If the package is not installed, run with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m doosan_forcevla_data.dummy.make_dummy_raw_episode --output data/raw_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_raw_episode data/raw_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.raw_to_processed --raw data/raw_dummy/episode_000000 --output data/processed_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_processed_episode data/processed_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.inspect_processed_episode data/processed_dummy/episode_000000
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.plan_lerobot_export --processed data/processed_dummy/episode_000000 --profile forcevla_13d --output data/processed_dummy/episode_000000/export_plan_forcevla_13d.json
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_export_plan data/processed_dummy/episode_000000/export_plan_forcevla_13d.json
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.plan_lerobot_export --processed data/processed_dummy/episode_000000 --profile doosan_full_25d --output data/processed_dummy/episode_000000/export_plan_doosan_full_25d.json
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_export_plan data/processed_dummy/episode_000000/export_plan_doosan_full_25d.json
```
