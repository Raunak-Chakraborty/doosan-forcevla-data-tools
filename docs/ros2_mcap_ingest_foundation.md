# ROS 2 MCAP ingest foundation

## Scope

This module is Patch 1 of the production Doosan
MCAP-to-ForceVLA conversion path.

It provides only:

- episode discovery
- strict `metadata.yaml` validation
- `episode_operator.json` loading
- `episode_validation.json` loading
- ROS 2 `SequentialReader` access
- topic/type cross-checking
- generic installed-message deserialization
- streaming per-topic message counts
- bag-timestamp monotonicity checks

It does not synchronize streams or interpret message fields.

## Upstream design reference

The reader design is inspired by:

`legalaspro/so101-ros-physical-ai`

component:

`rosbag_to_lerobot/rosbag_to_lerobot/bag_reader.py`

License: Apache-2.0.

The upstream project's useful architectural idea is to keep rosbag
reading and episode discovery separate from later feature decoding and
reference-timeline synchronization.

This implementation is independent and deliberately stricter for the
Doosan production acquisition contract.

The SO101 synchronization buffers are not copied by Patch 1. Their
reference-topic / freshness concepts belong to Patch 3.

The upstream project's current LeRobot-v3 export decisions are also
outside Patch 1 and do not determine the version consumed by the
pinned local ForceVLA repository.

## Fail-closed ingest rules

A candidate episode is rejected if:

- `metadata.yaml` is missing or malformed
- rosbag storage is not `mcap`
- any listed bag file is not an `.mcap`
- an MCAP path escapes the episode directory
- a metadata-listed MCAP file is missing
- an unlisted MCAP file is present
- overall message count is zero
- topic metadata is missing or duplicated
- ROS message type syntax is invalid
- topic serialization is not `cdr`
- per-topic counts do not sum to the metadata total
- either required episode JSON sidecar is missing
- a sidecar is not a JSON object
- the rosbag reader's topic/type map differs from metadata
- an installed ROS message class cannot be resolved
- any serialized message fails deserialization
- observed counts differ from metadata
- bag timestamps regress

Patch 2 will make the final eleven-topic Doosan contract and typed
message-field decoders first-class.

## ROS Python environment

ROS imports are intentionally lazy.

Do not run the MCAP reader through the ForceVLA conda interpreter.
On the thesis workstation use the ROS Jazzy system Python:

```bash
source /opt/ros/jazzy/setup.bash
source /home/ktt_rc/robotics_thesis/lab_myros2_ws/install/setup.bash

PYTHONPATH=/home/ktt_rc/robotics_thesis/forcevla_lab/doosan-forcevla-data-tools/src:$PYTHONPATH \
/usr/bin/python3 -m \
doosan_forcevla_data.inspect.inspect_mcap_episode \
/home/ktt_rc/robotics_thesis/dataset/mock/episode_000010
```

The workspace overlay is required because Episode 10 contains custom
`dsr_msgs2` and `gripper_msgs` message definitions.

## Memory behavior

The scanner is streaming.

Each serialized record is deserialized, counted, checked, and then
released by the loop. Camera images from an entire multi-gigabyte
episode are not accumulated into a Python list.

## Explicitly deferred

Patch 1 does not alter or decide:

- D405 reference-frame synchronization
- D435I nearest-neighbor policy
- interpolation or causal-hold policy
- header-versus-bag timestamp policy
- `RobotStateRt` field semantics
- `JointState.effort` handling
- wrench compensation or tare
- SCHUNK normalization
- 25D state ordering
- 7D action construction
- relative-rotation convention
- terminal-action policy
- camera model-input mapping
- LeRobot export format

Those remain owned by later roadmap patches.
