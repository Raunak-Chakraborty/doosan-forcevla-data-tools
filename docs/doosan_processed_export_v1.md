# Doosan production processed + LeRobot v2.1 export v1

Status: Patch-8 production conversion/export layer for
`doosan_forcevla_dataset_contract_v2`.

This path supersedes the older dummy/raw-real processed and three-camera export
paths for real Doosan MCAP training data.  Those older paths remain in the
repository as historical regression fixtures; they are not the production
ForceVLA contract.

## Input ownership

Patch 8 consumes the frozen semantic layers rather than recomputing them:

- Patch 4 owns synchronization;
- Patch 5 owns the 25D robot force/proprioception state;
- Patch 6 owns the binary gripper state (`0=held/closed`, `1=released/open`);
- Patch 7 owns the measured 7D action.

For `N` synchronized states, Patch 7 exposes only `N-1` measured actions.
Therefore the Patch-8 production processed episode contains exactly the
`N-1` action-bearing source observations.  The final synchronized observation
is not exported as a training row because it has no measured successor.
No zero terminal action is fabricated.

For frozen Episode 10:

```text
synchronized states:          1009
measured actions:             1008
processed training rows:      1008
excluded terminal reference:  1008
```

## Row contract

Every `frames.jsonl` row contains:

```text
observation_state_25d
  [tcp_position_3,
   tcp_rotvec_3,
   gripper_1,
   joint_position_6,
   joint_velocity_6,
   wrench_6]

action_7d
  [delta_tcp_translation_base_3,
   delta_tcp_rotation_spatial_base_3,
   absolute_gripper_target_1]
```

The semantic vectors are copied exactly from Patches 5--7.  Patch 8 does not
re-normalize, reorder, delta-transform, or tare them.

Each row also keeps the original D405 reference timestamp and the exact
Patch-4 selected source index/header timestamp for both physical cameras.

LeRobot timestamps are deliberately regularized to:

```text
timestamp = frame_index / 30
```

because the pinned LeRobot loader requires an exact `1/fps` sequence for
future-action indexing and video lookup.  Original ROS timestamps remain in
`frames.jsonl` as provenance and are not replaced in the raw MCAP.

## Two physical cameras only

The physical camera contract is exactly:

```text
D405 / tcp_camera
  640 x 480, rgb8

D435I / external_camera_2
  848 x 480, rgb8
```

Patch 8 writes two native-resolution videos and performs no crop, stretch, or
resize during conversion.

ForceVLA mapping is:

```text
observation.images.external_camera_2
  -> DoosanForcevlaInputs
  -> base_0_rgb

observation.images.tcp_camera
  -> DoosanForcevlaInputs
  -> left_wrist_0_rgb

right_wrist_0_rgb
  -> generated as an all-zero image by DoosanForcevlaInputs
  -> image_mask = false
```

`right_wrist_0_rgb` is therefore not a dataset feature, file, or physical
camera.  `external_camera_1` is not part of the production dataset.

OpenPI/ForceVLA performs its own aspect-preserving `resize_with_pad` to the
model image size.  The converter must not distort D435I to match D405.

## Production processed episode

ROS/Jazzy Python 3.12 builds:

```text
processed_episode/
  metadata_processed.json
  frames.jsonl
  videos/
    tcp_camera.mp4
    external_camera_2.mp4
```

Build:

```bash
PYTHONPATH=src /usr/bin/python3 -m \
  doosan_forcevla_data.convert.doosan_processed_episode_v1 \
  RAW_EPISODE PROCESSED_EPISODE --overwrite
```

Validate and inspect:

```bash
PYTHONPATH=src /usr/bin/python3 -m \
  doosan_forcevla_data.validate.validate_doosan_processed_episode_v1 \
  PROCESSED_EPISODE

PYTHONPATH=src /usr/bin/python3 -m \
  doosan_forcevla_data.inspect.inspect_doosan_processed_episode \
  PROCESSED_EPISODE
```

## Pinned LeRobot v2.1 export

The exporter targets the exact dependency tree frozen by the thesis:

```text
ForceVLA thesis:
9b61abef116f207d587d10aaf30170b73757c3e0

LeRobot:
e7aea92dd833f83d163820dcf2e58250307697a4

dlimp:
5edaa4691567873d495633f2708982b42edf1972
```

The export contains:

```text
lerobot_dataset/
  data/chunk-000/episode_000000.parquet

  videos/chunk-000/
    observation.images.tcp_camera/episode_000000.mp4
    observation.images.external_camera_2/episode_000000.mp4

  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
    export_provenance.json
```

The Parquet columns are exactly:

```text
observation.state       float64[25]
action                  float64[7]
timestamp               float32
frame_index             int64
episode_index           int64
index                   int64
task_index              int64
```

The two video features are declared in `meta/info.json` and are intentionally
not duplicated as Parquet columns, matching the pinned `LeRobotDataset` loader.

Run the exporter inside the frozen ForceVLA Python environment, which supplies
PyArrow:

```bash
PYTHONPATH=src:$FORCEVLA/src:$FORCEVLA/lerobot:$FORCEVLA/dlimp \
  $FORCEVLA_PY -m \
  doosan_forcevla_data.convert.doosan_processed_to_lerobot_v21 \
  PROCESSED_EPISODE LEROBOT_DATASET --overwrite
```

Validate/inspect with the same Python environment:

```bash
PYTHONPATH=src:$FORCEVLA/src:$FORCEVLA/lerobot:$FORCEVLA/dlimp \
  $FORCEVLA_PY -m \
  doosan_forcevla_data.validate.validate_doosan_lerobot_v21 \
  LEROBOT_DATASET

PYTHONPATH=src:$FORCEVLA/src:$FORCEVLA/lerobot:$FORCEVLA/dlimp \
  $FORCEVLA_PY -m \
  doosan_forcevla_data.inspect.inspect_doosan_lerobot_v21 \
  LEROBOT_DATASET
```

## Task / prompt contract

Patch 8 does not invent language text.  The task string comes from the raw
episode provenance and must match between `episode_operator.json` and
`episode_validation.json`.

For Episode 10 the exact stored task is:

```text
real_robot_demonstration
```

The LeRobot export writes that exact string to `meta/tasks.jsonl`.  During
training, the frozen ForceVLA data loader uses `PromptFromLeRobotTask` to turn
the row's `task_index` into the corresponding prompt.

Future episodes should record the intended language instruction at acquisition
time; Patch 8 will preserve it exactly rather than substitute a converter-side
prompt.

## Pinned loader behavior at the final action horizon

The dataset itself contains no synthetic terminal action.  When a 50-step
action horizon is requested near the end of an episode, the pinned LeRobot
loader clips out-of-range query indices to the final available action row and
returns an `action_is_pad` mask for those query positions.  This is loader-side
batch construction and is distinct from fabricating a demonstrated terminal
zero action in the dataset.

## Acceptance criteria

Patch 8 is accepted only when:

- production row count is exactly the measured Patch-7 action count;
- the terminal no-successor reference is absent from action-bearing rows;
- state is exactly 25D and action exactly 7D;
- exactly two physical video features exist;
- D405 remains 640x480 and D435I remains 848x480;
- both videos contain exactly one frame per processed training row;
- no physical `external_camera_1` or right-wrist video is exported;
- the exact source task reaches LeRobot task metadata;
- the pinned LeRobot v2.1 loader opens the local dataset;
- the frozen Doosan ForceVLA adapter accepts both physical image views and
  constructs the right-wrist zero/mask-false slot internally.
