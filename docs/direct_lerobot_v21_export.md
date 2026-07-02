# Direct LeRobot v2.1 Export

This exporter writes a local ForceVLA-compatible LeRobot v2.1 dataset directly from processed three-camera JSONL episodes.

It does not import LeRobot, ForceVLA, NumPy, Pillow, OpenCV, PyTorch, JAX, or any training stack. The only non-stdlib Python dependency is lazy `pyarrow` for Parquet. MP4 videos are encoded with system `ffmpeg` and checked with system `ffprobe`.

## Contract

- Source cameras map as `external_camera_1 -> observation.images.center`, `tcp_camera -> observation.images.left`, and `external_camera_2 -> observation.images.right`.
- Parquet stores scalar metadata plus separate state fields: `observation.state.ee_pos`, `observation.state.ee_quat`, `observation.state.gripper_pos`, `observation.state.wrench`, `observation.state.joint_pos`, `observation.state.joint_vel`, and `action`.
- `observation.state.ee_quat` is written as WXYZ because ForceVLA executable source at `9b93324` reads `w=q[0]` and `xyz=q[1:4]` in `SfpStateTransform`.
- The processed 25D state is split as TCP position, TCP rotation-vector converted to WXYZ quaternion, duplicated placeholder gripper, wrench, joint positions, and joint velocities.
- Actions are copied row-for-row from `measured_action`, including the final zero terminal action.
- LeRobot timestamps are regularized to `frame_index / fps` for ForceVLA action-horizon loading.
- MP4 decoded frame counts must exactly equal Parquet rows.

## Commands

```bash
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.lerobot_v21_doctor
PYTHONPATH=src python3 -m doosan_forcevla_data.convert.processed_to_lerobot_v21 \
  --processed /path/to/processed_episode \
  --output local_artifacts/lerobot_v21_smoke/doosan_no_contact_three_camera_v21 \
  --task "Perform a no-contact teleoperation heartbeat check."
PYTHONPATH=src python3 -m doosan_forcevla_data.validate.validate_lerobot_v21 \
  local_artifacts/lerobot_v21_smoke/doosan_no_contact_three_camera_v21
PYTHONPATH=src python3 -m doosan_forcevla_data.inspect.inspect_lerobot_v21 \
  local_artifacts/lerobot_v21_smoke/doosan_no_contact_three_camera_v21
```

Install the optional Parquet dependency with:

```bash
python3 -m pip install -e '.[lerobot-v21]'
```

Do not copy SFP-specific task strings, UR5e robot type, SFP 20 Hz fps, reference metadata defects, trailing video frames, or v5 loss weights into Doosan exports.
