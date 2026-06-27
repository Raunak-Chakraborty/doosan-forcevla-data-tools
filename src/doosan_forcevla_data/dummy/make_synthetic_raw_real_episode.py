"""Generate a tiny synthetic raw-real episode for offline schema testing.

This module writes deterministic files matching the ``raw_real_v0`` validator.
It does not import ROS packages, does not communicate with a robot, and does
not implement a live recorder.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.raw_real_schema import (
    DEFAULT_STREAM_RELATIVE_PATHS,
    OPTIONAL_STREAM_NAMES,
    RAW_REAL_SCHEMA_VERSION,
    REQUIRED_STREAM_NAMES,
    RawRealEpisodePaths,
)


DEFAULT_FRAME_COUNT = 20
DEFAULT_FPS = 30.0
DEFAULT_EPISODE_ID = "episode_raw_real_synthetic_000000"
GENERATOR_VERSION = "synthetic_raw_real_generator_v0"
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
LEGACY_CAMERA_LAYOUT = "legacy"
THESIS_CAMERA_LAYOUT = "thesis"
CAMERA_LAYOUTS = {LEGACY_CAMERA_LAYOUT, THESIS_CAMERA_LAYOUT}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _timestamp(index: int, fps: float) -> dict[str, float | int]:
    timestamp = index / fps
    return {
        "record_index": index,
        "source_stamp": timestamp,
        "receipt_stamp": timestamp + 0.001,
        "monotonic_stamp": timestamp + 10.0,
    }


def _fraction(index: int, frame_count: int) -> float:
    return index / (frame_count - 1)


def _joint_position(index: int, frame_count: int) -> list[float]:
    fraction = _fraction(index, frame_count)
    return [5.0 * joint_index + 2.0 * fraction for joint_index in range(6)]


def _joint_velocity(frame_count: int, fps: float) -> list[float]:
    timestamp_span = (frame_count - 1) / fps
    velocity = 2.0 / timestamp_span
    return [velocity for _ in range(6)]


def _tcp_position(index: int, frame_count: int) -> list[float]:
    fraction = _fraction(index, frame_count)
    return [
        450.0 + 12.0 * fraction,
        2.0 + 3.0 * fraction,
        225.0 - 18.0 * fraction,
        0.0,
        180.0,
        4.0 * fraction,
    ]


def _force_torque(index: int, frame_count: int) -> list[float]:
    fraction = _fraction(index, frame_count)
    contact_fraction = max(0.0, (fraction - 0.65) / 0.35)
    return [
        0.1 * math.sin(math.pi * fraction),
        0.05 * fraction,
        1.0 + 8.0 * contact_fraction,
        0.0,
        0.02 * contact_fraction,
        0.0,
    ]


def _write_ppm(path: Path, frame_index: int, stream_offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    red = (30 + frame_index * 9 + stream_offset) % 256
    green = (90 + frame_index * 5 + stream_offset) % 256
    blue = (150 + frame_index * 7 + stream_offset) % 256
    pixels = [
        (red, green, blue),
        (red, green // 2, blue),
        (red // 2, green, blue),
        (red, green, blue // 2),
    ]
    body = "\n".join(f"{r} {g} {b}" for r, g, b in pixels)
    path.write_text(f"P3\n2 2\n255\n{body}\n", encoding="ascii")


def _prepare_output(output: str | Path, overwrite: bool) -> Path:
    episode_dir = Path(output)
    if episode_dir.exists() or episode_dir.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {episode_dir}")
        if episode_dir.is_symlink() or not episode_dir.is_dir():
            raise ValueError(f"output path exists and is not a directory: {episode_dir}")
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=False)
    return episode_dir


def _stream_relative_path(stream_name: str) -> str:
    return DEFAULT_STREAM_RELATIVE_PATHS[stream_name]


def _camera_specs(camera_layout: str) -> list[dict[str, Any]]:
    if camera_layout == LEGACY_CAMERA_LAYOUT:
        return [
            {
                "stream_name": "external_camera",
                "source_name": "synthetic_external_camera",
                "frame_id": "external_camera_frame",
                "camera_role": "external",
                "role": "external_camera",
                "camera_id": "synthetic_external_camera",
                "model_input_key": "external_rgb_path",
                "used_for_model": True,
                "stream_offset": 0,
            },
            {
                "stream_name": "wrist_camera",
                "source_name": "synthetic_wrist_camera",
                "frame_id": "wrist_camera_frame",
                "camera_role": "wrist",
                "role": "tcp_camera",
                "camera_id": "synthetic_wrist_camera",
                "model_input_key": "tcp_rgb_path",
                "used_for_model": True,
                "stream_offset": 40,
            },
        ]
    if camera_layout == THESIS_CAMERA_LAYOUT:
        return [
            {
                "stream_name": "tcp_camera",
                "source_name": "synthetic_tcp_camera",
                "frame_id": "tcp_camera_frame",
                "camera_role": "tcp_camera",
                "role": "tcp_camera",
                "camera_id": "synthetic_tcp_camera",
                "model_input_key": "tcp_rgb_path",
                "used_for_model": True,
                "stream_offset": 40,
            },
            {
                "stream_name": "external_camera_1",
                "source_name": "synthetic_external_camera_1",
                "frame_id": "external_camera_1_frame",
                "camera_role": "external_camera",
                "role": "external_camera",
                "camera_id": "synthetic_external_camera_1",
                "external_camera_id": "1",
                "model_input_key": "external_rgb_path",
                "used_for_model": True,
                "stream_offset": 0,
            },
            {
                "stream_name": "external_camera_2",
                "source_name": "synthetic_external_camera_2",
                "frame_id": "external_camera_2_frame",
                "camera_role": "external_camera",
                "role": "external_camera",
                "camera_id": "synthetic_external_camera_2",
                "external_camera_id": "2",
                "used_for_model": False,
                "stream_offset": 80,
            },
        ]
    raise ValueError(f"camera_layout must be one of {sorted(CAMERA_LAYOUTS)!r}")


def _stream_entry(
    stream_name: str,
    *,
    kind: str,
    required: bool,
    source_name: str,
    source_type: str,
    record_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": _stream_relative_path(stream_name),
        "kind": kind,
        "required": required,
        "source_name": source_name,
        "source_type": source_type,
        "record_count": record_count,
        "verified": True,
        "synthetic": True,
    }
    if extra:
        entry.update(extra)
    return entry


def _metadata(episode_id: str, fps: float) -> dict[str, Any]:
    return {
        "schema_version": RAW_REAL_SCHEMA_VERSION,
        "episode_id": episode_id,
        "task_instruction": "Insert the peg into the hole.",
        "geometry_type": "round_peg_round_hole",
        "orientation_type": "vertical_insertion",
        "collection_method": "synthetic_raw_real",
        "action_label_primary": "measured_tcp_delta",
        "success": True,
        "failure_reason": None,
        "fps": fps,
        "robot_type": "doosan_m1013",
        "recorder_version": GENERATOR_VERSION,
        "source_workspace": {
            "path": "synthetic/offline",
            "git_commit": "synthetic",
            "branch": "synthetic",
            "dirty": False,
        },
    }


def _calibration_refs(camera_layout: str) -> dict[str, Any]:
    camera_intrinsics: dict[str, str] = {}
    camera_extrinsics: dict[str, str] = {}
    for spec in _camera_specs(camera_layout):
        stream_name = spec["stream_name"]
        camera_intrinsics[stream_name] = f"synthetic_{stream_name}_intrinsics_v0"
        camera_extrinsics[stream_name] = f"synthetic_{stream_name}_extrinsics_v0"
    return {
        "camera_intrinsics": camera_intrinsics,
        "camera_extrinsics": camera_extrinsics,
        "tcp_tool_calibration": "synthetic_tcp_tool_v0",
        "force_torque_calibration": "synthetic_force_torque_v0",
        "notes": "Synthetic placeholders for offline raw-real schema tests only.",
    }


def _events(frame_count: int, fps: float) -> list[dict[str, Any]]:
    stop_timestamp = (frame_count - 1) / fps
    contact_timestamp = int((frame_count - 1) * 0.70) / fps
    success_timestamp = int((frame_count - 1) * 0.90) / fps
    return [
        {"timestamp": 0.0, "event": "episode_start", "synthetic": True},
        {"timestamp": contact_timestamp, "event": "contact_synthetic", "synthetic": True},
        {"timestamp": success_timestamp, "event": "success", "synthetic": True},
        {"timestamp": stop_timestamp, "event": "episode_stop", "synthetic": True},
    ]


def _joint_state_records(frame_count: int, fps: float) -> list[dict[str, Any]]:
    velocity = _joint_velocity(frame_count, fps)
    return [
        {
            **_timestamp(index, fps),
            "source_name": "synthetic_joint_states",
            "source_type": "synthetic/jsonl",
            "joint_names": list(JOINT_NAMES),
            "position": _joint_position(index, frame_count),
            "velocity": list(velocity),
            "units": {"position": "deg", "velocity": "deg/s"},
        }
        for index in range(frame_count)
    ]


def _robot_state_records(frame_count: int, fps: float, *, canonical_tcp_wrench: bool) -> list[dict[str, Any]]:
    velocity = _joint_velocity(frame_count, fps)
    records: list[dict[str, Any]] = []
    for index in range(frame_count):
        force_torque = _force_torque(index, frame_count)
        record = {
                **_timestamp(index, fps),
                "source_name": "synthetic_robot_state_rt",
                "source_type": "synthetic/jsonl",
                "actual_tcp_position": _tcp_position(index, frame_count),
                "actual_joint_position": _joint_position(index, frame_count),
                "actual_joint_velocity": list(velocity),
                "robot_mode": "synthetic_auto",
                "robot_state": "synthetic_running",
                "control_mode": "synthetic_position",
                "units": {
                    "tcp_position": "mm",
                    "tcp_orientation": "deg",
                    "joint_position": "deg",
                    "joint_velocity": "deg/s",
                    "force": "N",
                    "torque": "Nm",
                },
                "frame_id": "base",
                "tcp_frame_id": "tcp_link",
            }
        if canonical_tcp_wrench:
            record["tcp_wrench"] = list(force_torque)
        else:
            record["external_tcp_force"] = list(force_torque)
            record["raw_force_torque"] = list(force_torque)
        records.append(record)
    return records


def _tf_records(frame_count: int, fps: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(frame_count):
        tcp = _tcp_position(index, frame_count)
        records.append(
            {
                **_timestamp(index, fps),
                "source_name": "synthetic_tf",
                "source_type": "synthetic/jsonl",
                "transforms": [
                    {
                        "parent_frame": "base",
                        "child_frame": "tcp_link",
                        "translation": [tcp[0] / 1000.0, tcp[1] / 1000.0, tcp[2] / 1000.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            }
        )
    return records


def _tf_static_records(camera_layout: str) -> list[dict[str, Any]]:
    camera_transforms = []
    for spec in _camera_specs(camera_layout):
        parent_frame = "tcp_link" if spec["role"] == "tcp_camera" else "base"
        translation = [0.02, 0.0, 0.04] if parent_frame == "tcp_link" else [0.6, -0.3, 0.8]
        if spec.get("external_camera_id") == "2":
            translation = [0.6, 0.3, 0.8]
        camera_transforms.append(
            {
                "parent_frame": parent_frame,
                "child_frame": spec["frame_id"],
                "translation": translation,
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        )
    return [
        {
            "record_index": 0,
            "source_stamp": 0.0,
            "receipt_stamp": 0.001,
            "monotonic_stamp": 10.0,
            "source_name": "synthetic_tf_static",
            "source_type": "synthetic/jsonl",
            "transforms": camera_transforms,
        }
    ]


def _camera_records(
    episode_dir: Path,
    stream_name: str,
    source_name: str,
    frame_id: str,
    camera_role: str,
    frame_count: int,
    fps: float,
    stream_offset: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(frame_count):
        image_relative = f"streams/{stream_name}/frames/{index:06d}.ppm"
        _write_ppm(episode_dir / image_relative, index, stream_offset)
        records.append(
            {
                **_timestamp(index, fps),
                "source_name": source_name,
                "source_type": "synthetic/image",
                "image_path": image_relative,
                "width": 2,
                "height": 2,
                "channels": 3,
                "encoding": "rgb8",
                "frame_id": frame_id,
                "camera_role": camera_role,
                "camera_id": source_name,
            }
        )
    return records


def _command_context_records(frame_count: int, fps: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(frame_count):
        fraction = _fraction(index, frame_count)
        records.append(
            {
                **_timestamp(index, fps),
                "source_name": "synthetic_command_context",
                "source_type": "synthetic/jsonl",
                "command_kind": "synthetic_twist_context",
                "commanded_twist": [0.001, 0.0001 * fraction, -0.0005, 0.0, 0.0, 0.001],
                "note": "diagnostic command context only; not action label",
            }
        )
    return records


def _gripper_state_records(frame_count: int, fps: float) -> list[dict[str, Any]]:
    return [
        {
            **_timestamp(index, fps),
            "source_name": "synthetic_constant_pending_gripper_integration",
            "source_type": "synthetic_placeholder",
            "gripper_position": 0.0,
            "gripper_width_m": 0.04,
            "gripper_command": "hold",
            "placeholder": True,
        }
        for index in range(frame_count)
    ]


def _wrench_sources_metadata(canonical_tcp_wrench: bool) -> dict[str, Any]:
    if canonical_tcp_wrench:
        return {
            "tcp_wrench": {
                "source_name": "doosan_internal_tcp_ft",
                "source_type": "doosan_internal",
                "source_service_or_topic": "synthetic/read_data_rt.tcp_wrench",
                "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "force_unit": "N",
                "torque_unit": "Nm",
                "frame": "tcp_frame",
                "compensation": "doosan_internal",
                "approved_for_model_state": True,
            }
        }
    return {
        "external_tcp_force": {
            "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "force_unit": "N",
            "torque_unit": "Nm",
            "frame": "base",
            "compensation": "estimated_external_tcp_force",
            "approved_for_model_state": True,
        },
        "raw_force_torque": {
            "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "force_unit": "N",
            "torque_unit": "Nm",
            "frame": "flange",
            "compensation": "raw_flange_sensor",
            "approved_for_model_state": False,
        },
    }


def _stream_index(frame_count: int, include_optional_streams: bool, camera_layout: str) -> dict[str, Any]:
    streams: dict[str, dict[str, Any]] = {}
    canonical_tcp_wrench = camera_layout == THESIS_CAMERA_LAYOUT

    if "joint_states" in REQUIRED_STREAM_NAMES:
        streams["joint_states"] = _stream_entry(
            "joint_states",
            kind="jsonl",
            required=True,
            source_name="synthetic_joint_states",
            source_type="synthetic/jsonl",
            record_count=frame_count,
            extra={
                "units": {"position": "deg", "velocity": "deg/s"},
                "joint_names": list(JOINT_NAMES),
            },
        )

    if "robot_state_rt" in REQUIRED_STREAM_NAMES:
        streams["robot_state_rt"] = _stream_entry(
            "robot_state_rt",
            kind="jsonl",
            required=True,
            source_name="synthetic_robot_state_rt",
            source_type="synthetic/jsonl",
            record_count=frame_count,
            extra={
                "units": {
                    "tcp_position": "mm",
                    "tcp_orientation": "deg",
                    "joint_position": "deg",
                    "joint_velocity": "deg/s",
                    "force": "N",
                    "torque": "Nm",
                },
                "frame_id": "base",
                "tcp_frame_id": "tcp_link",
                "wrench_sources": _wrench_sources_metadata(canonical_tcp_wrench),
            },
        )

    if "tf" in REQUIRED_STREAM_NAMES:
        streams["tf"] = _stream_entry(
            "tf",
            kind="jsonl",
            required=True,
            source_name="synthetic_tf",
            source_type="synthetic/jsonl",
            record_count=frame_count,
            extra={"frame_metadata": {"parent_frame": "base", "child_frame": "tcp_link"}},
        )

    if "tf_static" in REQUIRED_STREAM_NAMES:
        streams["tf_static"] = _stream_entry(
            "tf_static",
            kind="jsonl",
            required=True,
            source_name="synthetic_tf_static",
            source_type="synthetic/jsonl",
            record_count=1,
            extra={
                "frame_metadata": {
                    "static_frames": [spec["frame_id"] for spec in _camera_specs(camera_layout)]
                }
            },
        )

    for spec in _camera_specs(camera_layout):
        extra = {
            "type": "camera_rgb",
            "stream_type": "camera_rgb",
            "frame_id": spec["frame_id"],
            "role": spec["role"],
            "camera_role": spec["camera_role"],
            "camera_id": spec["camera_id"],
            "encoding": "rgb8",
            "width": 2,
            "height": 2,
            "channels": 3,
            "used_for_model": spec.get("used_for_model", False),
        }
        if spec.get("external_camera_id") is not None:
            extra["external_camera_id"] = spec["external_camera_id"]
        if spec.get("model_input_key") is not None:
            extra["model_input_key"] = spec["model_input_key"]
        streams[spec["stream_name"]] = _stream_entry(
            spec["stream_name"],
            kind="camera_images",
            required=True,
            source_name=spec["source_name"],
            source_type="synthetic/image",
            record_count=frame_count,
            extra=extra,
        )

    if include_optional_streams:
        if "command_context" in OPTIONAL_STREAM_NAMES:
            streams["command_context"] = _stream_entry(
                "command_context",
                kind="jsonl",
                required=False,
                source_name="synthetic_command_context",
                source_type="synthetic/jsonl",
                record_count=frame_count,
                extra={"label_source": False},
            )
        if "gripper_state" in OPTIONAL_STREAM_NAMES:
            streams["gripper_state"] = _stream_entry(
                "gripper_state",
                kind="jsonl",
                required=False,
                source_name="synthetic_gripper_state",
                source_type="synthetic_placeholder",
                record_count=frame_count,
                extra={
                    "units": {"gripper_width_m": "m"},
                    "source_name": "synthetic_constant_pending_gripper_integration",
                    "placeholder": True,
                },
            )

    return {
        "schema_version": RAW_REAL_SCHEMA_VERSION,
        "synthetic": True,
        "camera_layout": camera_layout,
        "model_camera_mapping": {
            "external_rgb_path": next(
                spec["stream_name"] for spec in _camera_specs(camera_layout) if spec.get("model_input_key") == "external_rgb_path"
            ),
            "tcp_rgb_path": next(
                spec["stream_name"] for spec in _camera_specs(camera_layout) if spec.get("model_input_key") == "tcp_rgb_path"
            ),
        },
        "streams": streams,
    }


def _recorder_report(
    frame_count: int,
    fps: float,
    include_optional_streams: bool,
    camera_layout: str,
    stream_record_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "generator": "make_synthetic_raw_real_episode",
        "generator_version": GENERATOR_VERSION,
        "synthetic": True,
        "frame_count": frame_count,
        "fps": fps,
        "include_optional_streams": include_optional_streams,
        "camera_layout": camera_layout,
        "stream_record_counts": stream_record_counts,
        "warnings": [],
        "notes": [
            "Offline synthetic data only.",
            "Not recorded from ROS, a Doosan controller, or a live robot.",
            "Intended for raw_real_v0 schema and validator tests.",
        ],
    }


def make_synthetic_raw_real_episode(
    output: str | Path,
    *,
    episode_id: str = DEFAULT_EPISODE_ID,
    frame_count: int = DEFAULT_FRAME_COUNT,
    fps: float = DEFAULT_FPS,
    include_optional_streams: bool = False,
    camera_layout: str = LEGACY_CAMERA_LAYOUT,
    overwrite: bool = False,
) -> Path:
    """Create a deterministic synthetic ``raw_real_v0`` episode directory."""

    if frame_count <= 1:
        raise ValueError("frame_count must be greater than 1")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if camera_layout not in CAMERA_LAYOUTS:
        raise ValueError(f"camera_layout must be one of {sorted(CAMERA_LAYOUTS)!r}")

    episode_dir = _prepare_output(output, overwrite=overwrite)
    paths = RawRealEpisodePaths(episode_dir)
    canonical_tcp_wrench = camera_layout == THESIS_CAMERA_LAYOUT

    _write_json(paths.metadata, _metadata(episode_id, float(fps)))
    _write_json(paths.calibration_refs, _calibration_refs(camera_layout))
    _write_jsonl(paths.events, _events(frame_count, float(fps)))

    _write_jsonl(paths.joint_states, _joint_state_records(frame_count, float(fps)))
    _write_jsonl(paths.robot_state_rt, _robot_state_records(frame_count, float(fps), canonical_tcp_wrench=canonical_tcp_wrench))
    _write_jsonl(paths.tf, _tf_records(frame_count, float(fps)))
    _write_jsonl(paths.tf_static, _tf_static_records(camera_layout))

    camera_record_counts: dict[str, int] = {}
    for spec in _camera_specs(camera_layout):
        camera_records = _camera_records(
            episode_dir,
            spec["stream_name"],
            spec["source_name"],
            spec["frame_id"],
            spec["camera_role"],
            frame_count,
            float(fps),
            stream_offset=int(spec["stream_offset"]),
        )
        _write_jsonl(episode_dir / DEFAULT_STREAM_RELATIVE_PATHS[spec["stream_name"]] / "index.jsonl", camera_records)
        camera_record_counts[spec["stream_name"]] = frame_count

    stream_record_counts = {
        "joint_states": frame_count,
        "robot_state_rt": frame_count,
        "tf": frame_count,
        "tf_static": 1,
    }
    stream_record_counts.update(camera_record_counts)

    if include_optional_streams:
        _write_jsonl(paths.command_context, _command_context_records(frame_count, float(fps)))
        _write_jsonl(paths.gripper_state, _gripper_state_records(frame_count, float(fps)))
        stream_record_counts["command_context"] = frame_count
        stream_record_counts["gripper_state"] = frame_count

    _write_json(paths.streams_index, _stream_index(frame_count, include_optional_streams, camera_layout))
    _write_json(
        paths.recorder_report,
        _recorder_report(frame_count, float(fps), include_optional_streams, camera_layout, stream_record_counts),
    )

    return episode_dir


def _resolve_output_arg(parser: argparse.ArgumentParser, positional: str | None, option: str | None) -> str:
    if positional is None and option is None:
        parser.error("provide an output path either positionally or with --output")
    if positional is not None and option is not None and Path(positional) != Path(option):
        parser.error("positional output and --output differ; provide only one output path")
    return option if option is not None else str(positional)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a synthetic raw_real_v0 episode.")
    parser.add_argument("output_path", nargs="?", help="Episode output directory")
    parser.add_argument("--output", dest="output_option", help="Episode output directory")
    parser.add_argument("--episode-id", default=DEFAULT_EPISODE_ID)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--include-optional-streams", action="store_true")
    parser.add_argument("--camera-layout", choices=sorted(CAMERA_LAYOUTS), default=LEGACY_CAMERA_LAYOUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args(argv)

    output = _resolve_output_arg(parser, args.output_path, args.output_option)
    try:
        episode_dir = make_synthetic_raw_real_episode(
            output,
            episode_id=args.episode_id,
            frame_count=args.frames,
            fps=args.fps,
            include_optional_streams=args.include_optional_streams,
            camera_layout=args.camera_layout,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        print(f"FAILED: could not write synthetic raw-real episode: {output}")
        print(str(exc))
        return 1

    print(f"Wrote synthetic raw-real episode: {episode_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
