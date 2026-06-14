"""Convert a validated raw_real_v0 episode into a processed JSONL episode.

This converter is intentionally offline and dependency-light. It reads files
created by a passive raw-real recorder or the synthetic raw-real generator; it
does not import ROS packages and does not communicate with a robot.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.compute_actions import compute_measured_tcp_delta_action
from doosan_forcevla_data.schema.processed_schema import (
    ACTION_DIM,
    MODEL_STATE_DIM,
    QUATERNION_CONVENTION,
)
from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode
from doosan_forcevla_data.validate.validate_raw_real_episode import (
    is_explicit_synthetic_episode,
    validate_raw_real_episode,
)


DATASET_NAME = "doosan_peg_in_hole_v0"
CONVERTER_VERSION = "raw_real_to_processed_v0"
ROTATION_VECTOR_DEGREES = "rotation_vector_degrees"
ROTATION_VECTOR_RADIANS = "rotation_vector_radians"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: could not read JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{path}: could not read JSONL: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{path}: line {line_number} is empty")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        records.append(record)
    return records


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _finite_float(value: Any, context: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _finite_vector(value: Any, expected_len: int, context: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of length {expected_len}")
    if len(value) != expected_len:
        raise ValueError(f"{context} must have length {expected_len}, got {len(value)}")
    return [_finite_float(item, f"{context}[{idx}]") for idx, item in enumerate(value)]


def _maybe_finite_vector(value: Any, expected_len: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected_len:
        return None
    if not all(_is_finite_number(item) for item in value):
        return None
    return [float(item) for item in value]


def _stream_path(raw_root: Path, streams: dict[str, Any], stream_name: str) -> Path:
    entry = streams.get(stream_name)
    if not isinstance(entry, dict):
        raise ValueError(f"streams/index.json: stream {stream_name} entry is missing or invalid")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"streams/index.json: stream {stream_name} missing path")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"streams/index.json: stream {stream_name} path must be safe and relative")
    return raw_root / relative_path


def _stream_entry(streams: dict[str, Any], stream_name: str) -> dict[str, Any]:
    entry = streams.get(stream_name)
    if not isinstance(entry, dict):
        return {}
    return entry


def _load_camera_index(raw_root: Path, streams: dict[str, Any], stream_name: str) -> list[dict[str, Any]]:
    return _read_jsonl_objects(_stream_path(raw_root, streams, stream_name) / "index.jsonl")


def _records_by_index(records: list[dict[str, Any]], stream_name: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row_number, record in enumerate(records):
        record_index = record.get("record_index")
        if isinstance(record_index, bool) or not isinstance(record_index, int):
            raise ValueError(f"{stream_name} record {row_number}: record_index must be an integer")
        if record_index in indexed:
            raise ValueError(f"{stream_name}: duplicate record_index {record_index}")
        indexed[record_index] = record
    return indexed


def _require_aligned_indexes(
    primary_indexes: set[int],
    candidate: dict[int, dict[str, Any]],
    stream_name: str,
) -> None:
    candidate_indexes = set(candidate)
    if candidate_indexes == primary_indexes:
        return
    missing = sorted(primary_indexes - candidate_indexes)
    extra = sorted(candidate_indexes - primary_indexes)
    details: list[str] = []
    if missing:
        details.append(f"missing primary record_index values {missing[:10]}")
    if extra:
        details.append(f"extra record_index values {extra[:10]}")
    raise ValueError(f"{stream_name}: record_index alignment with robot_state_rt failed: {', '.join(details)}")


def _timestamp_seconds(value: Any, context: str) -> float:
    if _is_finite_number(value):
        return float(value)
    if isinstance(value, dict):
        sec = _finite_float(value.get("sec"), f"{context}.sec")
        nanosec = _finite_float(value.get("nanosec"), f"{context}.nanosec")
        if nanosec < 0.0 or nanosec >= 1_000_000_000.0:
            raise ValueError(f"{context}.nanosec must be in [0, 1e9)")
        return sec + nanosec * 1e-9
    raise ValueError(f"{context} must be numeric or an object with sec/nanosec")


def _strictly_increasing(values: list[float]) -> bool:
    return all(values[idx] > values[idx - 1] for idx in range(1, len(values)))


def _relative_timestamps(robot_records: list[dict[str, Any]]) -> list[float]:
    source_times = [
        _timestamp_seconds(record.get("source_stamp"), f"robot_state_rt record {idx} source_stamp")
        for idx, record in enumerate(robot_records)
    ]
    if len(source_times) <= 1 or _strictly_increasing(source_times):
        first = source_times[0]
        return [time - first for time in source_times]

    monotonic_times = [
        _finite_float(record.get("monotonic_stamp"), f"robot_state_rt record {idx} monotonic_stamp")
        for idx, record in enumerate(robot_records)
    ]
    if len(monotonic_times) <= 1 or _strictly_increasing(monotonic_times):
        first = monotonic_times[0]
        return [time - first for time in monotonic_times]

    raise ValueError("robot_state_rt timestamps are not strictly increasing by source_stamp or monotonic_stamp")


def _normalized_unit(units: dict[str, Any], key: str) -> str | None:
    value = units.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace(" ", "_")


def _combined_units(record: dict[str, Any], stream_entry: dict[str, Any]) -> dict[str, Any]:
    units: dict[str, Any] = {}
    entry_units = stream_entry.get("units")
    if isinstance(entry_units, dict):
        units.update(entry_units)
    record_units = record.get("units")
    if isinstance(record_units, dict):
        units.update(record_units)
    return units


def _convert_tcp_position(values: list[float], units: dict[str, Any], synthetic: bool, context: str) -> list[float]:
    unit = _normalized_unit(units, "tcp_position")
    if unit is None and synthetic:
        unit = "mm"
    if unit in {"mm", "millimeter", "millimeters"}:
        return [value / 1000.0 for value in values]
    if unit in {"m", "meter", "meters"}:
        return values
    raise ValueError(f"{context}: unsupported or missing tcp_position unit: {unit!r}")


def _convert_tcp_orientation_rotvec(
    values: list[float],
    units: dict[str, Any],
    synthetic: bool,
    default_unit: str | None,
    context: str,
) -> list[float]:
    unit = _normalized_unit(units, "tcp_orientation")
    if unit is None and default_unit is not None:
        unit = default_unit
    if unit is None and synthetic:
        unit = "deg"
    if unit in {"deg", "degree", "degrees"}:
        return [math.radians(value) for value in values]
    if unit in {"rad", "radian", "radians"}:
        return values
    raise ValueError(f"{context}: unsupported or missing tcp_orientation unit: {unit!r}")


def _convert_joint_position(values: list[float], units: dict[str, Any], synthetic: bool, context: str) -> list[float]:
    unit = _normalized_unit(units, "joint_position") or _normalized_unit(units, "position")
    if unit is None and synthetic:
        unit = "deg"
    if unit in {"deg", "degree", "degrees"}:
        return [math.radians(value) for value in values]
    if unit in {"rad", "radian", "radians"}:
        return values
    raise ValueError(f"{context}: unsupported or missing joint position unit: {unit!r}")


def _convert_joint_velocity(values: list[float], units: dict[str, Any], synthetic: bool, context: str) -> list[float]:
    unit = _normalized_unit(units, "joint_velocity") or _normalized_unit(units, "velocity")
    if unit is None and synthetic:
        unit = "deg/s"
    if unit in {"deg/s", "deg_per_s", "degree/s", "degrees/s", "degrees_per_second"}:
        return [math.radians(value) for value in values]
    if unit in {"rad/s", "rad_per_s", "radian/s", "radians/s", "radians_per_second"}:
        return values
    raise ValueError(f"{context}: unsupported or missing joint velocity unit: {unit!r}")


def _rotvec_to_quat_xyzw(rotvec: list[float]) -> list[float]:
    rx, ry, rz = _finite_vector(rotvec, 3, "rotation vector")
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return [0.5 * rx, 0.5 * ry, 0.5 * rz, 1.0]
    scale = math.sin(angle / 2.0) / angle
    quat = [rx * scale, ry * scale, rz * scale, math.cos(angle / 2.0)]
    norm = math.sqrt(sum(component * component for component in quat))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("rotation vector produced invalid quaternion")
    return [component / norm for component in quat]


def _orientation_policy(
    metadata: dict[str, Any],
    recorder_report: dict[str, Any],
    streams_index: dict[str, Any],
) -> tuple[bool, str, str | None]:
    if is_explicit_synthetic_episode(metadata, recorder_report, streams_index):
        return True, "synthetic raw-real episode: treating actual_tcp_position[3:6] as rotation vector in degrees", "deg"

    convention = metadata.get("tcp_orientation_convention") or recorder_report.get("tcp_orientation_convention")
    if convention == ROTATION_VECTOR_DEGREES:
        return False, "tcp_orientation_convention=rotation_vector_degrees", "deg"
    if convention == ROTATION_VECTOR_RADIANS:
        return False, "tcp_orientation_convention=rotation_vector_radians", "rad"

    raise ValueError(
        "non-synthetic raw-real episode requires explicit supported TCP orientation convention; "
        "set tcp_orientation_convention='rotation_vector_degrees' or "
        "tcp_orientation_convention='rotation_vector_radians' only after lab verification"
    )


def _select_joint_vectors(
    robot_record: dict[str, Any],
    robot_entry: dict[str, Any],
    joint_record: dict[str, Any],
    joint_entry: dict[str, Any],
    synthetic: bool,
    frame_index: int,
) -> tuple[list[float], list[float], str]:
    robot_units = _combined_units(robot_record, robot_entry)
    joint_units = _combined_units(joint_record, joint_entry)

    robot_joint_pos = _maybe_finite_vector(robot_record.get("actual_joint_position"), 6)
    if robot_joint_pos is not None:
        joint_pos = _convert_joint_position(
            robot_joint_pos, robot_units, synthetic, f"robot_state_rt record {frame_index} actual_joint_position"
        )
        joint_pos_source = "robot_state_rt.actual_joint_position"
    else:
        joint_pos = _convert_joint_position(
            _finite_vector(joint_record.get("position"), 6, f"joint_states record {frame_index} position"),
            joint_units,
            synthetic,
            f"joint_states record {frame_index} position",
        )
        joint_pos_source = "joint_states.position"

    robot_joint_vel = _maybe_finite_vector(robot_record.get("actual_joint_velocity"), 6)
    if robot_joint_vel is not None:
        joint_vel = _convert_joint_velocity(
            robot_joint_vel, robot_units, synthetic, f"robot_state_rt record {frame_index} actual_joint_velocity"
        )
        joint_vel_source = "robot_state_rt.actual_joint_velocity"
    else:
        joint_vel = _convert_joint_velocity(
            _finite_vector(joint_record.get("velocity"), 6, f"joint_states record {frame_index} velocity"),
            joint_units,
            synthetic,
            f"joint_states record {frame_index} velocity",
        )
        joint_vel_source = "joint_states.velocity"

    return joint_pos, joint_vel, f"{joint_pos_source}; {joint_vel_source}"


def _select_wrench(robot_record: dict[str, Any], frame_index: int) -> tuple[list[float], str]:
    external_tcp_force = _maybe_finite_vector(robot_record.get("external_tcp_force"), 6)
    if external_tcp_force is not None:
        return external_tcp_force, "external_tcp_force"
    raw_force_torque = _maybe_finite_vector(robot_record.get("raw_force_torque"), 6)
    if raw_force_torque is not None:
        return raw_force_torque, "raw_force_torque"
    raise ValueError(
        f"robot_state_rt record {frame_index}: external_tcp_force or raw_force_torque must contain 6 finite values"
    )


def _select_gripper(gripper_record: dict[str, Any] | None, frame_index: int) -> float:
    if gripper_record is None:
        return 0.0
    if _is_finite_number(gripper_record.get("gripper_position")):
        return float(gripper_record["gripper_position"])
    if _is_finite_number(gripper_record.get("gripper_width_m")):
        return float(gripper_record["gripper_width_m"])
    raise ValueError(
        f"gripper_state record {frame_index}: expected finite gripper_position or gripper_width_m"
    )


def _resolve_raw_image(raw_root: Path, image_path_value: Any, frame_index: int, stream_name: str) -> Path:
    if not isinstance(image_path_value, str) or not image_path_value:
        raise ValueError(f"{stream_name} record {frame_index}: image_path must be a non-empty string")
    relative_path = Path(image_path_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{stream_name} record {frame_index}: image_path must be safe and relative")
    image_path = raw_root / relative_path
    if not image_path.is_file():
        raise ValueError(f"{stream_name} record {frame_index}: image file does not exist: {image_path}")
    return image_path


def _frame_image_path(
    *,
    raw_root: Path,
    output_root: Path,
    camera_record: dict[str, Any],
    stream_name: str,
    frame_index: int,
    copy_images: bool,
) -> str:
    raw_image = _resolve_raw_image(raw_root, camera_record.get("image_path"), frame_index, stream_name)
    if not copy_images:
        return str(Path(str(camera_record["image_path"])))

    suffix = raw_image.suffix
    target_relative = Path("images") / stream_name / f"{frame_index:06d}{suffix}"
    target_path = output_root / target_relative
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_image, target_path)
    return target_relative.as_posix()


def _contains_path(parent: Path, child: Path) -> bool:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    try:
        child_resolved.relative_to(parent_resolved)
    except ValueError:
        return False
    return True


def _prepare_output(raw_root: Path, output_root: Path, overwrite: bool) -> None:
    if _contains_path(raw_root, output_root):
        raise ValueError(
            f"output directory cannot be inside the raw-real episode directory: {output_root}"
        )
    if output_root.exists() or output_root.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_root}")
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError(f"output path exists and is not a directory: {output_root}")
        if _contains_path(output_root, raw_root):
            raise ValueError(f"refusing to overwrite output path that contains raw-real input: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def _build_model_state(
    *,
    robot_record: dict[str, Any],
    robot_entry: dict[str, Any],
    joint_record: dict[str, Any],
    joint_entry: dict[str, Any],
    gripper_record: dict[str, Any] | None,
    synthetic: bool,
    orientation_unit_default: str | None,
    frame_index: int,
) -> tuple[list[float], list[float], list[float], float, str, str]:
    tcp = _finite_vector(
        robot_record.get("actual_tcp_position"), 6, f"robot_state_rt record {frame_index} actual_tcp_position"
    )
    robot_units = _combined_units(robot_record, robot_entry)
    ee_pos = _convert_tcp_position(tcp[:3], robot_units, synthetic, f"robot_state_rt record {frame_index}")
    ee_axis_angle = _convert_tcp_orientation_rotvec(
        tcp[3:6],
        robot_units,
        synthetic,
        orientation_unit_default,
        f"robot_state_rt record {frame_index} actual_tcp_position[3:6]",
    )
    ee_quat = _rotvec_to_quat_xyzw(ee_axis_angle)
    gripper_pos = _select_gripper(gripper_record, frame_index)
    wrench, wrench_source = _select_wrench(robot_record, frame_index)
    joint_pos, joint_vel, joint_source = _select_joint_vectors(
        robot_record, robot_entry, joint_record, joint_entry, synthetic, frame_index
    )

    model_state = ee_pos + ee_axis_angle + [gripper_pos] + wrench + joint_pos + joint_vel
    if len(model_state) != MODEL_STATE_DIM:
        raise ValueError(f"model_state length must be {MODEL_STATE_DIM}, got {len(model_state)}")
    if not all(math.isfinite(value) for value in model_state):
        raise ValueError(f"frame {frame_index}: model_state contains a non-finite value")
    return model_state, ee_pos, ee_quat, gripper_pos, wrench_source, joint_source


def convert_raw_real_to_processed(
    raw_real_episode_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    copy_images: bool = False,
    include_optional_debug: bool = False,
) -> Path:
    """Convert a validated raw_real_v0 episode into a processed episode."""

    raw_root = Path(raw_real_episode_dir)
    output_root = Path(output_dir)

    if _contains_path(raw_root, output_root):
        raise ValueError(
            f"output directory cannot be inside the raw-real episode directory: {output_root}"
        )

    validation = validate_raw_real_episode(raw_root)
    if not validation.ok:
        message = "raw-real episode validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in validation.errors
        )
        raise ValueError(message)

    metadata = _read_json_object(raw_root / "metadata.json")
    recorder_report = _read_json_object(raw_root / "recorder_report.json")
    streams_index = _read_json_object(raw_root / "streams" / "index.json")
    streams = streams_index.get("streams")
    if not isinstance(streams, dict):
        raise ValueError(f"{raw_root / 'streams' / 'index.json'}: streams must be a JSON object")

    synthetic, orientation_note, orientation_unit_default = _orientation_policy(metadata, recorder_report, streams_index)

    robot_records = _read_jsonl_objects(_stream_path(raw_root, streams, "robot_state_rt"))
    if not robot_records:
        raise ValueError("robot_state_rt stream must contain at least one record")
    joint_records = _read_jsonl_objects(_stream_path(raw_root, streams, "joint_states"))
    external_camera_records = _load_camera_index(raw_root, streams, "external_camera")
    wrist_camera_records = _load_camera_index(raw_root, streams, "wrist_camera")

    robot_by_index = _records_by_index(robot_records, "robot_state_rt")
    joint_by_index = _records_by_index(joint_records, "joint_states")
    external_camera_by_index = _records_by_index(external_camera_records, "external_camera")
    wrist_camera_by_index = _records_by_index(wrist_camera_records, "wrist_camera")
    primary_indexes = set(robot_by_index)

    _require_aligned_indexes(primary_indexes, joint_by_index, "joint_states")
    _require_aligned_indexes(primary_indexes, external_camera_by_index, "external_camera")
    _require_aligned_indexes(primary_indexes, wrist_camera_by_index, "wrist_camera")

    gripper_by_index: dict[int, dict[str, Any]] = {}
    if "gripper_state" in streams:
        gripper_records = _read_jsonl_objects(_stream_path(raw_root, streams, "gripper_state"))
        gripper_by_index = _records_by_index(gripper_records, "gripper_state")
        _require_aligned_indexes(primary_indexes, gripper_by_index, "gripper_state")

    command_context_debug: dict[str, Any] | None = None
    if include_optional_debug and "command_context" in streams:
        command_context_records = _read_jsonl_objects(_stream_path(raw_root, streams, "command_context"))
        command_context_debug = {
            "record_count": len(command_context_records),
            "first_record": command_context_records[0] if command_context_records else None,
            "last_record": command_context_records[-1] if command_context_records else None,
            "used_as_action_label": False,
        }

    ordered_indexes = sorted(primary_indexes)
    if ordered_indexes != list(range(len(ordered_indexes))):
        raise ValueError("robot_state_rt record_index values must be contiguous from 0 for processed frame_index mapping")
    ordered_robot_records = [robot_by_index[index] for index in ordered_indexes]
    timestamps = _relative_timestamps(ordered_robot_records)

    _prepare_output(raw_root, output_root, overwrite=overwrite)

    robot_entry = _stream_entry(streams, "robot_state_rt")
    joint_entry = _stream_entry(streams, "joint_states")
    frames: list[dict[str, Any]] = []
    tcp_positions: list[list[float]] = []
    tcp_quats: list[list[float]] = []
    gripper_positions: list[float] = []
    wrench_sources: set[str] = set()
    joint_sources: set[str] = set()

    for frame_index, record_index in enumerate(ordered_indexes):
        model_state, tcp_pos, tcp_quat, gripper_pos, wrench_source, joint_source = _build_model_state(
            robot_record=robot_by_index[record_index],
            robot_entry=robot_entry,
            joint_record=joint_by_index[record_index],
            joint_entry=joint_entry,
            gripper_record=gripper_by_index.get(record_index),
            synthetic=synthetic,
            orientation_unit_default=orientation_unit_default,
            frame_index=frame_index,
        )
        tcp_positions.append(tcp_pos)
        tcp_quats.append(tcp_quat)
        gripper_positions.append(gripper_pos)
        wrench_sources.add(wrench_source)
        joint_sources.add(joint_source)

        frames.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamps[frame_index],
                "external_rgb_path": _frame_image_path(
                    raw_root=raw_root,
                    output_root=output_root,
                    camera_record=external_camera_by_index[record_index],
                    stream_name="external_camera",
                    frame_index=frame_index,
                    copy_images=copy_images,
                ),
                "tcp_rgb_path": _frame_image_path(
                    raw_root=raw_root,
                    output_root=output_root,
                    camera_record=wrist_camera_by_index[record_index],
                    stream_name="wrist_camera",
                    frame_index=frame_index,
                    copy_images=copy_images,
                ),
                "model_state": model_state,
                "measured_action": [0.0] * ACTION_DIM,
                "action_is_terminal_padding": True,
            }
        )

    for frame_index, frame in enumerate(frames):
        if frame_index == len(frames) - 1:
            frame["measured_action"] = [0.0] * ACTION_DIM
            frame["action_is_terminal_padding"] = True
            continue
        frame["measured_action"] = compute_measured_tcp_delta_action(
            tcp_positions[frame_index],
            tcp_quats[frame_index],
            tcp_positions[frame_index + 1],
            tcp_quats[frame_index + 1],
            gripper_t=gripper_positions[frame_index],
            gripper_t1=gripper_positions[frame_index + 1],
        )
        frame["action_is_terminal_padding"] = False

    processed_metadata: dict[str, Any] = {
        "source_raw_episode": str(raw_root.resolve()),
        "dataset_name": DATASET_NAME,
        "robot_type": metadata.get("robot_type"),
        "fps": metadata.get("fps"),
        "quaternion_convention": QUATERNION_CONVENTION,
        "model_state_dim": MODEL_STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_label_primary": "measured_tcp_delta",
        "frame_count": len(frames),
        "task_instruction": metadata.get("task_instruction"),
        "geometry_type": metadata.get("geometry_type"),
        "orientation_type": metadata.get("orientation_type"),
        "collection_method": metadata.get("collection_method"),
        "success": metadata.get("success"),
        "failure_reason": metadata.get("failure_reason"),
        "notes": (
            "raw_real_v0 to processed JSONL conversion. Actions are measured consecutive TCP deltas; "
            "command_context is diagnostic only and terminal action padding is applied."
        ),
        "source_schema_version": metadata.get("schema_version"),
        "source_episode_id": metadata.get("episode_id"),
        "converter_version": CONVERTER_VERSION,
        "alignment_policy": "record_index equality against robot_state_rt primary timeline",
        "selected_streams": {
            "primary_timeline": "robot_state_rt",
            "joint_states": "record_index aligned fallback only when robot_state_rt joint vectors are unavailable",
            "external_rgb_path": "external_camera.image_path",
            "tcp_rgb_path": "wrist_camera.image_path",
            "gripper_state": "record_index aligned optional stream" if gripper_by_index else "absent; gripper_pos=0.0",
        },
        "unit_conversions": {
            "tcp_position": "raw units to meters",
            "tcp_orientation": "rotation vector units to radians",
            "joint_position": "raw units to radians",
            "joint_velocity": "raw units to radians_per_second",
            "wrench": "preserved from selected raw 6D force/torque signal",
        },
        "wrench_source": sorted(wrench_sources),
        "joint_source": sorted(joint_sources),
        "orientation_conversion": orientation_note,
        "image_copy_policy": "copied into processed images/" if copy_images else "raw-real relative image paths",
        "raw_validation_warnings": validation.warnings,
        "command_context_policy": "diagnostic only; never used as action label",
    }
    if command_context_debug is not None:
        processed_metadata["optional_debug"] = {"command_context": command_context_debug}

    _write_json(output_root / "metadata_processed.json", processed_metadata)
    _write_jsonl(output_root / "frames.jsonl", frames)

    processed_validation = validate_processed_episode(output_root)
    if not processed_validation.ok:
        message = "processed episode validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in processed_validation.errors
        )
        raise ValueError(message)

    return output_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a raw_real_v0 episode to processed JSONL.")
    parser.add_argument("--raw-real", required=True, help="Raw-real episode directory")
    parser.add_argument("--output", required=True, help="Processed episode output directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into the processed episode")
    parser.add_argument(
        "--include-optional-debug",
        action="store_true",
        help="Copy optional diagnostic metadata such as command_context summary",
    )
    args = parser.parse_args(argv)

    try:
        output_dir = convert_raw_real_to_processed(
            args.raw_real,
            args.output,
            overwrite=args.overwrite,
            copy_images=args.copy_images,
            include_optional_debug=args.include_optional_debug,
        )
    except (OSError, ValueError) as exc:
        print(f"FAILED: could not convert raw-real episode: {args.raw_real}")
        print(str(exc))
        return 1

    print(f"OK: wrote processed episode: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
