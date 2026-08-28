# Golden Episode-10 end-to-end acceptance v1

Status: Patch-9 production acceptance layer.

Patch 9 does **not** define new observation, action, gripper, synchronization,
or camera semantics.  Those remain owned by Patches 4--8.  Patch 9 freezes the
known-good real Episode 10 as a deterministic acceptance target and provides
one command that exercises the complete production path across the two Python
runtimes used on the lab workstation.

## Frozen inputs

```text
Converter Patch-8 parent:
c694d284af51a02d88c770b562adfdddc9793444

ForceVLA:
9b61abef116f207d587d10aaf30170b73757c3e0

LeRobot compatibility fork:
e7aea92dd833f83d163820dcf2e58250307697a4

LeRobot URL:
https://github.com/Raunak-Chakraborty/lerobot.git

dlimp:
5edaa4691567873d495633f2708982b42edf1972

Episode-10 MCAP SHA256:
075365f21d7e6a5cbb6d42cb4ad099b0b46fa71ce4eea4f069bb9e1231936d57
```

The raw identity gate also freezes the exact 16,702-message topic-count map
stored by rosbag2.  This prevents a different recording from being accepted
merely because it happens to have the same episode directory name.

## Frozen semantic/output acceptance

The production processed episode must reproduce:

```text
synchronized states:                 1009
action-bearing rows:                 1008
terminal reference:                  1008
fabricated terminal action:          false
state dimension:                     25
semantic action dimension:           7
ForceVLA internal action dimension:  32
ForceVLA action horizon:             50
```

`frames.jsonl` is canonicalized using the same sorted compact JSON encoding as
Patch 8 and must reproduce SHA256:

```text
3ab89835fde8a8a03c780d81d1a498f04e49868fffa7b5a5b2458db0bed4629f
```

That digest freezes the exact synchronized source selections, ROS timestamps,
25D states, measured 7D actions, action targets, and camera selections for the
golden episode without depending on MP4 byte-level codec determinism.

The release-only gripper acceptance is:

```text
processed state rows:
  956 held/closed
   52 released/open

action targets:
  955 held/closed targets
   53 released/open targets

release action:
  source reference 955
  target reference 956
```

The final row is source reference 1007 with its measured action targeting the
terminal synchronized reference 1008.  Reference 1008 itself is not a training
row.

## Two physical cameras

Exactly two physical video features are allowed:

```text
D405 / tcp_camera
  640 x 480
  -> observation.images.tcp_camera
  -> ForceVLA left_wrist_0_rgb

D435i / external_camera_2
  848 x 480
  -> observation.images.external_camera_2
  -> ForceVLA base_0_rgb
```

`right_wrist_0_rgb` remains an internal all-zero ForceVLA adapter slot with
`image_mask=false`; it is not a third dataset camera.

## Runtime split

The public Patch-9 command must be started from ROS 2 Jazzy's system Python.
It performs raw MCAP ingest/synchronization and Patch-8 processed-video
materialization there.  It then launches the supplied ForceVLA Python executable
with a deliberately isolated `PYTHONPATH` for the second stage.

The frozen ForceVLA stage requires:

```text
Python       3.11
numpy        1.26.4
torch        2.12.0+cu130
torchvision  0.27.0+cu130
jax          0.5.3
datasets     4.8.5
pyarrow      24.0.0
av           17.0.1
```

ROS Python 3.12 paths are rejected inside the ForceVLA stage.  The pinned
LeRobot compatibility fork uses direct PyAV video decoding and the validated
Hugging Face `datasets.Column` compatibility path.

## One-command golden acceptance

After sourcing ROS 2 Jazzy and the lab workspace:

```bash
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
/usr/bin/python3 -m \
  doosan_forcevla_data.smoke.golden_episode10_end_to_end run \
  /home/ktt_rc/robotics_thesis/dataset/mock/episode_000010 \
  /home/ktt_rc/robotics_thesis/forcevla_lab/golden_episode10_acceptance \
  --converter-root /home/ktt_rc/robotics_thesis/forcevla_lab/doosan-forcevla-data-tools \
  --forcevla-root /home/ktt_rc/robotics_thesis/forcevla_lab/ForceVLA_thesis \
  --forcevla-python "$HOME/miniforge3/envs/forcevla_lab_tf_test/bin/python" \
  --overwrite
```

Success ends with:

```text
GOLDEN_EPISODE10_FORCEVLA_STAGE=PASS
GOLDEN_EPISODE10_ACCEPTANCE=PASS
```

The output directory contains:

```text
processed_episode_000010/
lerobot_doosan_episode10/
forcevla_acceptance.json
forcevla_stage.log
golden_episode10_acceptance.json
```

The final JSON report records the frozen source commits, exact MCAP identity,
processed-frame digest, release/terminal assertions, LeRobot provenance,
runtime package versions/module origins, loader-side terminal padding, and
ForceVLA camera/action mapping.

## What Patch 9 intentionally does not do

Patch 9 does not:

- alter Patch-4 synchronization policy;
- alter the 25D state layout;
- recompute or redefine the Patch-7 7D action;
- create a terminal zero action;
- change the binary release-only gripper semantics;
- resize the native stored camera videos;
- add a third physical camera;
- change the pinned LeRobot/ForceVLA compatibility implementation;
- create a multi-episode training dataset.

Multi-episode composition and training-readiness checks remain a later patch.
