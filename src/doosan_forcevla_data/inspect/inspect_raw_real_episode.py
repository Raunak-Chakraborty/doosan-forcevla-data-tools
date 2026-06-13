"""Inspect and preflight a raw_real_v0 episode before conversion.

This tool is offline and dependency-light. It reads JSON, JSONL, and local file
paths only; it does not import ROS packages and does not communicate with a
robot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.raw_real_schema import (
    DEFAULT_STREAM_RELATIVE_PATHS,
    OPTIONAL_STREAM_NAMES,
    REQUIRED_STREAM_NAMES,
)
from doosan_forcevla_data.validate.validate_raw_real_episode import (
    ROTATION_VECTOR_DEGREES,
    ROTATION_VECTOR_RADIANS,
    raw_real_conversion_readiness_errors,
    validate_raw_real_episode,
)


CAMERA_STREAM_NAMES = {"external_camera", "wrist_camera"}
JSONL_STREAM_NAMES = {
    "joint_states",
    "robot_state_rt",
    "tf",
    "tf_static",
    "command_context",
    "gripper_state",
}
CONVERTER_REQUIRED_STREAMS = ["joint_states", "external_camera", "wrist_camera"]


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _finite_vector(value: Any, expected_len: int) -> bool:
    return isinstance(value, list) and len(value) == expected_len and all(_is_finite_number(item) for item in value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _unique_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _source_stamp_seconds(value: Any) -> float | None:
    if _is_finite_number(value):
        return float(value)
    if not isinstance(value, dict):
        return None
    sec = value.get("sec")
    nanosec = value.get("nanosec")
    if not _is_finite_number(sec) or not _is_finite_number(nanosec):
        return None
    nanosec_float = float(nanosec)
    if nanosec_float < 0.0 or nanosec_float >= 1_000_000_000.0:
        return None
    return float(sec) + nanosec_float * 1e-9


def _safe_relative_path(root: Path, value: Any) -> tuple[Path | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, "path must be a non-empty string"
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None, f"path must be safe and relative: {value}"
    path = root / relative_path
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return None, f"path must stay inside episode root: {value}"
    return path, None


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{path}: could not read {label} JSON object: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{path}: {label} must be a JSON object"]
    return data, []


def _read_jsonl_objects(path: Path, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return records, [f"{path}: could not read {label} JSONL: {exc}"]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"{path}: line {line_number} is empty")
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(record, dict):
            errors.append(f"{path}: line {line_number} must be a JSON object")
            continue
        records.append(record)
    return records, errors


def _stream_entry(streams: dict[str, Any] | None, stream_name: str) -> dict[str, Any] | None:
    if not isinstance(streams, dict):
        return None
    entry = streams.get(stream_name)
    if isinstance(entry, dict):
        return entry
    return None


def _declared_count(entry: dict[str, Any] | None) -> int | None:
    if entry is None:
        return None
    value = entry.get("record_count")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _record_indexes(records: list[dict[str, Any]]) -> list[int]:
    indexes: list[int] = []
    for record in records:
        value = record.get("record_index")
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        indexes.append(value)
    return indexes


def _stream_source_stamps(records: list[dict[str, Any]]) -> list[float | None]:
    return [_source_stamp_seconds(record.get("source_stamp")) for record in records]


def _records_by_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        record_index = record.get("record_index")
        if isinstance(record_index, bool) or not isinstance(record_index, int):
            continue
        if record_index not in indexed:
            indexed[record_index] = record
    return indexed


def _default_stream_entry(stream_name: str) -> dict[str, Any]:
    return {"path": DEFAULT_STREAM_RELATIVE_PATHS.get(stream_name, f"streams/{stream_name}.jsonl")}


def _stream_records_path(root: Path, stream_name: str, entry: dict[str, Any] | None) -> tuple[Path | None, list[str]]:
    effective_entry = entry if entry is not None else _default_stream_entry(stream_name)
    path, path_error = _safe_relative_path(root, effective_entry.get("path"))
    if path is None:
        return None, [f"streams/index.json: stream {stream_name} {path_error}"]
    if stream_name in CAMERA_STREAM_NAMES:
        return path / "index.jsonl", []
    return path, []


def _summarize_stream(
    root: Path,
    stream_name: str,
    entry: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    summary: dict[str, Any] = {
        "present": False,
        "declared": entry is not None,
        "path": None,
        "index_path": None,
        "record_count": None,
        "record_index_min": None,
        "record_index_max": None,
        "record_index_count": None,
        "record_index_unique": None,
        "declared_count": _declared_count(entry),
        "declared_count_matches_actual": None,
        "first_source_stamp": None,
        "last_source_stamp": None,
        "duration_source_stamp": None,
        "read_errors": [],
    }
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    if entry is None and stream_name in OPTIONAL_STREAM_NAMES:
        return summary, records, errors

    records_path, path_errors = _stream_records_path(root, stream_name, entry)
    errors.extend(path_errors)
    if records_path is None:
        summary["read_errors"] = list(errors)
        return summary, records, errors

    if stream_name in CAMERA_STREAM_NAMES:
        stream_dir = records_path.parent
        summary["path"] = str(stream_dir)
        summary["index_path"] = str(records_path)
        summary["present"] = stream_dir.is_dir() and records_path.is_file()
    else:
        summary["path"] = str(records_path)
        summary["present"] = records_path.is_file()

    if not records_path.is_file():
        errors.append(f"{records_path}: stream {stream_name} records file is missing")
        summary["read_errors"] = list(errors)
        return summary, records, errors

    records, read_errors = _read_jsonl_objects(records_path, stream_name)
    errors.extend(read_errors)
    summary["record_count"] = len(records)
    if summary["declared_count"] is not None:
        summary["declared_count_matches_actual"] = summary["declared_count"] == len(records)

    indexes = _record_indexes(records)
    unique_indexes = sorted(set(indexes))
    summary["record_index_count"] = len(unique_indexes)
    summary["record_index_unique"] = len(indexes) == len(records) and len(indexes) == len(unique_indexes)
    if unique_indexes:
        summary["record_index_min"] = unique_indexes[0]
        summary["record_index_max"] = unique_indexes[-1]

    stamps = _stream_source_stamps(records)
    first_stamp = stamps[0] if stamps else None
    last_stamp = stamps[-1] if stamps else None
    summary["first_source_stamp"] = first_stamp
    summary["last_source_stamp"] = last_stamp
    if first_stamp is not None and last_stamp is not None:
        summary["duration_source_stamp"] = last_stamp - first_stamp

    summary["read_errors"] = list(errors)
    return summary, records, errors


def _required_streams_summary(streams: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "present": bool(streams.get(name, {}).get("present")),
            "declared": bool(streams.get(name, {}).get("declared")),
            "record_count": streams.get(name, {}).get("record_count"),
            "declared_count": streams.get(name, {}).get("declared_count"),
            "declared_count_matches_actual": streams.get(name, {}).get("declared_count_matches_actual"),
        }
        for name in REQUIRED_STREAM_NAMES
    }


def _optional_streams_summary(streams: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "present": bool(streams.get(name, {}).get("present")),
            "declared": bool(streams.get(name, {}).get("declared")),
            "record_count": streams.get(name, {}).get("record_count"),
            "declared_count": streams.get(name, {}).get("declared_count"),
        }
        for name in OPTIONAL_STREAM_NAMES
    }


def _timeline_summary(records_by_stream: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    robot_records = records_by_stream.get("robot_state_rt", [])
    robot_indexes = sorted(set(_record_indexes(robot_records)))
    primary_set = set(robot_indexes)
    missing_by_stream: dict[str, list[int]] = {}
    extra_by_stream: dict[str, list[int]] = {}

    for stream_name in CONVERTER_REQUIRED_STREAMS:
        indexes = set(_record_indexes(records_by_stream.get(stream_name, [])))
        missing_by_stream[stream_name] = sorted(primary_set - indexes)
        extra_by_stream[stream_name] = sorted(indexes - primary_set)

    required_aligned = bool(primary_set) and all(
        not missing_by_stream[stream_name] and not extra_by_stream[stream_name]
        for stream_name in CONVERTER_REQUIRED_STREAMS
    )
    return {
        "primary_stream": "robot_state_rt",
        "frame_count": len(robot_indexes),
        "record_index_min": robot_indexes[0] if robot_indexes else None,
        "record_index_max": robot_indexes[-1] if robot_indexes else None,
        "required_streams_aligned": required_aligned,
        "missing_by_stream": missing_by_stream,
        "extra_by_stream": extra_by_stream,
    }


def _offset_summary(
    robot_records: list[dict[str, Any]],
    stream_records: list[dict[str, Any]],
) -> dict[str, Any]:
    robot_by_index = _records_by_index(robot_records)
    stream_by_index = _records_by_index(stream_records)
    offsets: list[float] = []
    for record_index in sorted(set(robot_by_index) & set(stream_by_index)):
        robot_stamp = _source_stamp_seconds(robot_by_index[record_index].get("source_stamp"))
        stream_stamp = _source_stamp_seconds(stream_by_index[record_index].get("source_stamp"))
        if robot_stamp is None or stream_stamp is None:
            continue
        offsets.append(stream_stamp - robot_stamp)
    return {
        "count": len(offsets),
        "min": min(offsets) if offsets else None,
        "max": max(offsets) if offsets else None,
        "mean": _mean(offsets),
        "max_abs": max((abs(value) for value in offsets), default=None),
    }


def _timestamp_summary(
    metadata: dict[str, Any] | None,
    records_by_stream: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    fps_value = metadata.get("fps") if isinstance(metadata, dict) else None
    fps = float(fps_value) if _is_finite_number(fps_value) and float(fps_value) > 0.0 else None
    robot_records = records_by_stream.get("robot_state_rt", [])
    robot_stamps = [stamp for stamp in _stream_source_stamps(robot_records) if stamp is not None]
    dts = [robot_stamps[idx] - robot_stamps[idx - 1] for idx in range(1, len(robot_stamps))]

    return {
        "fps": fps,
        "expected_dt": 1.0 / fps if fps else None,
        "robot_state_rt_duration": robot_stamps[-1] - robot_stamps[0] if len(robot_stamps) >= 2 else None,
        "robot_state_rt_dt_min": min(dts) if dts else None,
        "robot_state_rt_dt_max": max(dts) if dts else None,
        "robot_state_rt_dt_mean": _mean(dts),
        "camera_source_stamp_offset_summary": {
            name: _offset_summary(robot_records, records_by_stream.get(name, [])) for name in sorted(CAMERA_STREAM_NAMES)
        },
    }


def _camera_summary(root: Path, records_by_stream: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for stream_name in sorted(CAMERA_STREAM_NAMES):
        records = records_by_stream.get(stream_name, [])
        image_paths = [record.get("image_path") for record in records if isinstance(record.get("image_path"), str)]
        duplicates = len(image_paths) - len(set(image_paths))
        missing_files = 0
        empty_files = 0
        unsafe_paths = 0
        for image_path_value in image_paths:
            image_path, path_error = _safe_relative_path(root, image_path_value)
            if image_path is None:
                unsafe_paths += 1
                continue
            if not image_path.is_file():
                missing_files += 1
                continue
            try:
                if image_path.stat().st_size <= 0:
                    empty_files += 1
            except OSError:
                missing_files += 1
        summary[stream_name] = {
            "frame_count": len(records),
            "missing_files": missing_files,
            "empty_files": empty_files,
            "unsafe_paths": unsafe_paths,
            "duplicate_image_paths": duplicates,
            "first_image_path": image_paths[0] if image_paths else None,
            "last_image_path": image_paths[-1] if image_paths else None,
        }
    return summary


def _combined_units(record: dict[str, Any] | None, entry: dict[str, Any] | None) -> dict[str, Any]:
    units: dict[str, Any] = {}
    if isinstance(entry, dict) and isinstance(entry.get("units"), dict):
        units.update(entry["units"])
    if isinstance(record, dict) and isinstance(record.get("units"), dict):
        units.update(record["units"])
    return units


def _normalized_unit(units: dict[str, Any], key: str) -> str | None:
    value = units.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace(" ", "_")


def _tcp_position_units_guess(records: list[dict[str, Any]], entry: dict[str, Any] | None) -> str:
    units = _combined_units(records[0] if records else None, entry)
    unit = _normalized_unit(units, "tcp_position")
    if unit in {"mm", "millimeter", "millimeters"}:
        return "mm"
    if unit in {"m", "meter", "meters"}:
        return "m"
    if records and _finite_vector(records[0].get("actual_tcp_position"), 6):
        values = [abs(float(value)) for value in records[0]["actual_tcp_position"][:3]]
        if max(values, default=0.0) > 10.0:
            return "mm"
        if max(values, default=0.0) <= 2.0:
            return "m"
    return "unknown"


def _is_synthetic_episode(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
) -> bool:
    collection_method = metadata.get("collection_method") if isinstance(metadata, dict) else None
    recorder_version = metadata.get("recorder_version") if isinstance(metadata, dict) else None
    return any(
        [
            isinstance(collection_method, str) and "synthetic" in collection_method.lower(),
            isinstance(recorder_version, str) and "synthetic" in recorder_version.lower(),
            isinstance(recorder_report, dict) and recorder_report.get("synthetic") is True,
            isinstance(streams_index, dict) and streams_index.get("synthetic") is True,
        ]
    )


def _orientation_guard(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
) -> tuple[str, bool]:
    if _is_synthetic_episode(metadata, recorder_report, streams_index):
        return "synthetic episode: converter uses guarded synthetic rotation-vector-degrees policy", True
    metadata = metadata if isinstance(metadata, dict) else {}
    recorder_report = recorder_report if isinstance(recorder_report, dict) else {}
    convention = metadata.get("tcp_orientation_convention") or recorder_report.get("tcp_orientation_convention")
    if convention == ROTATION_VECTOR_DEGREES:
        return "tcp_orientation_convention=rotation_vector_degrees", True
    if convention == ROTATION_VECTOR_RADIANS:
        return "tcp_orientation_convention=rotation_vector_radians", True
    return "non-synthetic episode requires explicit rotation-vector TCP orientation convention before conversion", False


def _robot_state_summary(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
    stream_entry: dict[str, Any] | None,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    record_count = len(records)
    tcp_valid = sum(1 for record in records if _finite_vector(record.get("actual_tcp_position"), 6))
    joint_position_valid = sum(1 for record in records if _finite_vector(record.get("actual_joint_position"), 6))
    joint_velocity_valid = sum(1 for record in records if _finite_vector(record.get("actual_joint_velocity"), 6))
    guard_message, guard_ready = _orientation_guard(metadata, recorder_report, streams_index)
    return {
        "record_count": record_count,
        "tcp_position_units_guess": _tcp_position_units_guess(records, stream_entry),
        "orientation_convention_guard": guard_message,
        "orientation_convention_ready": guard_ready,
        "has_actual_tcp_position": record_count > 0 and tcp_valid == record_count,
        "has_actual_joint_position": record_count > 0 and joint_position_valid == record_count,
        "has_actual_joint_velocity": record_count > 0 and joint_velocity_valid == record_count,
        "valid_actual_tcp_position_records": tcp_valid,
        "valid_actual_joint_position_records": joint_position_valid,
        "valid_actual_joint_velocity_records": joint_velocity_valid,
    }


def _joint_summary(stream_entry: dict[str, Any] | None, records: list[dict[str, Any]]) -> dict[str, Any]:
    record_count = len(records)
    first_names = records[0].get("joint_names") if records else None
    units = _combined_units(records[0] if records else None, stream_entry)
    return {
        "record_count": record_count,
        "joint_names": first_names if isinstance(first_names, list) else None,
        "joint_name_count": len(first_names) if isinstance(first_names, list) else None,
        "joint_names_unique": len(set(first_names)) == len(first_names) if isinstance(first_names, list) else None,
        "has_position": record_count > 0 and all(_finite_vector(record.get("position"), 6) for record in records),
        "has_velocity": record_count > 0 and all(_finite_vector(record.get("velocity"), 6) for record in records),
        "units": units,
    }


def _wrench_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    external_count = sum(1 for record in records if _finite_vector(record.get("external_tcp_force"), 6))
    raw_count = sum(1 for record in records if _finite_vector(record.get("raw_force_torque"), 6))
    valid_count = sum(
        1
        for record in records
        if _finite_vector(record.get("external_tcp_force"), 6) or _finite_vector(record.get("raw_force_torque"), 6)
    )
    preferred_source = "none"
    if external_count:
        preferred_source = "external_tcp_force"
    elif raw_count:
        preferred_source = "raw_force_torque"
    return {
        "external_tcp_force_records": external_count,
        "raw_force_torque_records": raw_count,
        "valid_wrench_records": valid_count,
        "preferred_source": preferred_source,
    }


def _gripper_summary(stream_summary: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    position_count = sum(1 for record in records if _is_finite_number(record.get("gripper_position")))
    width_count = sum(1 for record in records if _is_finite_number(record.get("gripper_width_m")))
    field_used = "none"
    if position_count:
        field_used = "gripper_position"
    elif width_count:
        field_used = "gripper_width_m"
    return {
        "present": bool(stream_summary.get("present")),
        "record_count": len(records),
        "usable_records": max(position_count, width_count),
        "field_used": field_used,
    }


def _command_context_summary(stream_summary: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    action_like_records = sum(
        1 for record in records if any(key in record for key in ["action_label", "measured_action", "action"])
    )
    command_kinds = sorted({record["command_kind"] for record in records if isinstance(record.get("command_kind"), str)})
    return {
        "present": bool(stream_summary.get("present")),
        "record_count": len(records),
        "used_as_action_label": False,
        "policy": "diagnostic only",
        "action_like_field_records": action_like_records,
        "command_kinds": command_kinds,
    }


def _event_summary(root: Path) -> tuple[dict[str, Any], list[str]]:
    records, errors = _read_jsonl_objects(root / "events.jsonl", "events")
    event_names = [record.get("event") for record in records if isinstance(record.get("event"), str)]
    has_end_event = any(
        event_name == "success"
        or "fail" in event_name.lower()
        or "stop" in event_name.lower()
        or "end" in event_name.lower()
        for event_name in event_names
    )
    return (
        {
            "event_count": len(records),
            "first_event": event_names[0] if event_names else None,
            "last_event": event_names[-1] if event_names else None,
            "has_start_event": "episode_start" in event_names,
            "has_end_event": has_end_event,
        },
        errors,
    )


def _conversion_blockers(
    validation_ok: bool,
    required_streams: dict[str, dict[str, Any]],
    timeline: dict[str, Any],
    camera_summary: dict[str, dict[str, Any]],
    robot_state_summary: dict[str, Any],
    joint_summary: dict[str, Any],
    wrench_summary: dict[str, Any],
    conversion_readiness_errors: list[str],
) -> list[str]:
    blockers: list[str] = []
    if not validation_ok:
        blockers.append("raw-real validator failed")
    for stream_name, summary in required_streams.items():
        if not summary["present"] or not summary["record_count"]:
            blockers.append(f"required stream {stream_name} is missing or empty")
    if int(timeline.get("frame_count") or 0) < 2:
        blockers.append("robot_state_rt must contain at least 2 records for conversion")
    if not timeline.get("required_streams_aligned"):
        blockers.append("required converter streams do not align with robot_state_rt record_index values")
    for stream_name, summary in camera_summary.items():
        if summary["frame_count"] <= 0:
            blockers.append(f"{stream_name} has no camera frames")
        if summary["missing_files"]:
            blockers.append(f"{stream_name} has {summary['missing_files']} missing image files")
        if summary["empty_files"]:
            blockers.append(f"{stream_name} has {summary['empty_files']} empty image files")
        if summary["unsafe_paths"]:
            blockers.append(f"{stream_name} has {summary['unsafe_paths']} unsafe image paths")
    if not robot_state_summary["has_actual_tcp_position"]:
        blockers.append("robot_state_rt does not have valid actual_tcp_position for every record")
    if not robot_state_summary["orientation_convention_ready"]:
        blockers.append("TCP orientation convention is not explicitly supported for non-synthetic conversion")
    if not joint_summary["has_position"] or not joint_summary["has_velocity"]:
        blockers.append("joint_states does not have valid position and velocity vectors for every record")
    if wrench_summary["valid_wrench_records"] != int(timeline.get("frame_count") or 0):
        blockers.append("robot_state_rt does not have a valid wrench vector for every primary record")
    blockers.extend(conversion_readiness_errors)
    return _unique_strings(blockers)


def _recommendations(
    ready_for_conversion: bool,
    conversion_blockers: list[str],
    validation_warnings: list[str],
    camera_summary: dict[str, dict[str, Any]],
    gripper_summary: dict[str, Any],
    robot_state_summary: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if ready_for_conversion:
        recommendations.append("Run raw-real to processed converter.")
    else:
        recommendations.append("Fix blocking raw-real inspection findings before conversion.")
    if conversion_blockers:
        recommendations.append("Resolve conversion_blockers listed in this report.")
    for stream_name, summary in camera_summary.items():
        if summary["missing_files"] or summary["empty_files"] or summary["unsafe_paths"]:
            recommendations.append(f"Fix {stream_name} frame file paths and image files.")
    if any("timestamp" in warning.lower() for warning in validation_warnings):
        recommendations.append("Check camera and robot timestamp synchronization.")
    if any("source_stamp" in blocker for blocker in conversion_blockers):
        recommendations.append("Check camera and robot source_stamp synchronization before conversion.")
    if not robot_state_summary["orientation_convention_ready"]:
        recommendations.append("Verify TCP orientation convention before real conversion.")
    if not gripper_summary["present"]:
        recommendations.append("Add gripper_state if gripper actions matter.")
    return _unique_strings(recommendations)


def _validation_dict(result: Any) -> dict[str, Any]:
    return {
        "ok": bool(result.ok),
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }


def write_inspection_report(report: dict[str, Any], output: str | Path, pretty: bool = True) -> Path:
    """Write a JSON raw-real inspection report."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    output_path.write_text(json.dumps(report, indent=indent, sort_keys=pretty) + "\n", encoding="utf-8")
    return output_path


def inspect_raw_real_episode(
    episode_dir: str | Path,
    output_json: str | Path | None = None,
    require_convertible: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serializable raw-real inspection/preflight report."""

    root = Path(episode_dir)
    validation = validate_raw_real_episode(root)
    metadata, metadata_errors = _read_json_object(root / "metadata.json", "metadata")
    recorder_report, recorder_errors = _read_json_object(root / "recorder_report.json", "recorder_report")
    streams_index, streams_index_errors = _read_json_object(root / "streams" / "index.json", "streams/index")
    streams = streams_index.get("streams") if isinstance(streams_index, dict) else None
    if streams is not None and not isinstance(streams, dict):
        streams_index_errors.append(f"{root / 'streams' / 'index.json'}: streams must be a JSON object")
        streams = None

    stream_names = list(REQUIRED_STREAM_NAMES) + list(OPTIONAL_STREAM_NAMES)
    if isinstance(streams, dict):
        for stream_name in sorted(streams):
            if stream_name not in stream_names:
                stream_names.append(stream_name)

    streams_summary: dict[str, dict[str, Any]] = {}
    records_by_stream: dict[str, list[dict[str, Any]]] = {}
    stream_read_errors: list[str] = []
    for stream_name in stream_names:
        summary, records, errors = _summarize_stream(root, stream_name, _stream_entry(streams, stream_name))
        streams_summary[stream_name] = summary
        records_by_stream[stream_name] = records
        stream_read_errors.extend(errors)

    required_streams = _required_streams_summary(streams_summary)
    optional_streams = _optional_streams_summary(streams_summary)
    timeline = _timeline_summary(records_by_stream)
    timestamps = _timestamp_summary(metadata, records_by_stream)
    camera_summary = _camera_summary(root, records_by_stream)
    robot_state_summary = _robot_state_summary(
        metadata,
        recorder_report,
        streams_index,
        _stream_entry(streams, "robot_state_rt"),
        records_by_stream.get("robot_state_rt", []),
    )
    joint_summary = _joint_summary(_stream_entry(streams, "joint_states"), records_by_stream.get("joint_states", []))
    wrench_summary = _wrench_summary(records_by_stream.get("robot_state_rt", []))
    gripper_summary = _gripper_summary(
        streams_summary.get("gripper_state", {}), records_by_stream.get("gripper_state", [])
    )
    command_context_summary = _command_context_summary(
        streams_summary.get("command_context", {}), records_by_stream.get("command_context", [])
    )
    event_summary, event_read_errors = _event_summary(root)
    conversion_readiness_errors = raw_real_conversion_readiness_errors(
        metadata,
        recorder_report,
        streams_index,
        streams,
        records_by_stream,
    )

    conversion_blockers = _conversion_blockers(
        validation.ok,
        required_streams,
        timeline,
        camera_summary,
        robot_state_summary,
        joint_summary,
        wrench_summary,
        conversion_readiness_errors,
    )
    ready_for_conversion = validation.ok and not conversion_blockers
    inspection_errors = metadata_errors + recorder_errors + streams_index_errors + stream_read_errors + event_read_errors
    errors = _unique_strings(list(validation.errors) + inspection_errors + conversion_blockers)
    warnings = _unique_strings(list(validation.warnings))

    if not validation.ok or (require_convertible and not ready_for_conversion):
        status = "failed"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    recommendations = _recommendations(
        ready_for_conversion,
        conversion_blockers,
        validation.warnings,
        camera_summary,
        gripper_summary,
        robot_state_summary,
    )

    report: dict[str, Any] = {
        "status": status,
        "episode_dir": str(root),
        "schema_version": metadata.get("schema_version") if isinstance(metadata, dict) else None,
        "episode_id": metadata.get("episode_id") if isinstance(metadata, dict) else None,
        "task_instruction": metadata.get("task_instruction") if isinstance(metadata, dict) else None,
        "robot_type": metadata.get("robot_type") if isinstance(metadata, dict) else None,
        "collection_method": metadata.get("collection_method") if isinstance(metadata, dict) else None,
        "geometry_type": metadata.get("geometry_type") if isinstance(metadata, dict) else None,
        "orientation_type": metadata.get("orientation_type") if isinstance(metadata, dict) else None,
        "success": metadata.get("success") if isinstance(metadata, dict) else None,
        "failure_reason": metadata.get("failure_reason") if isinstance(metadata, dict) else None,
        "fps": metadata.get("fps") if isinstance(metadata, dict) else None,
        "ready_for_conversion": ready_for_conversion,
        "require_convertible": require_convertible,
        "validation": _validation_dict(validation),
        "streams": streams_summary,
        "required_streams": required_streams,
        "optional_streams": optional_streams,
        "timeline": timeline,
        "timestamps": timestamps,
        "camera_summary": camera_summary,
        "robot_state_summary": robot_state_summary,
        "joint_summary": joint_summary,
        "wrench_summary": wrench_summary,
        "gripper_summary": gripper_summary,
        "command_context_summary": command_context_summary,
        "event_summary": event_summary,
        "conversion_readiness_errors": conversion_readiness_errors,
        "conversion_blockers": conversion_blockers,
        "warnings": warnings,
        "errors": errors,
        "recommendations": recommendations,
    }

    if output_json is not None:
        write_inspection_report(report, output_json, pretty=True)
    return report


def _status(value: bool) -> str:
    return "yes" if value else "no"


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def print_inspection_summary(report: dict[str, Any]) -> None:
    """Print a compact human-readable raw-real inspection summary."""

    timeline = report["timeline"]
    timestamps = report["timestamps"]
    print("Raw-Real Episode Inspection")
    print(f"episode_dir: {report['episode_dir']}")
    print(f"status: {report['status']}")
    print(f"validation_ok: {_status(report['validation']['ok'])}")
    print(f"ready_for_conversion: {_status(report['ready_for_conversion'])}")
    print(f"schema_version: {report['schema_version']}")
    print(f"episode_id: {report['episode_id']}")
    print(f"task_instruction: {report['task_instruction']}")
    print(f"robot_type: {report['robot_type']}")
    print(f"fps: {_format_number(report['fps'])}")
    print(f"primary_timeline: {timeline['primary_stream']}")
    print(f"frame_count: {timeline['frame_count']}")
    print(f"required_streams_aligned: {_status(timeline['required_streams_aligned'])}")
    print(f"expected_dt: {_format_number(timestamps['expected_dt'])}")

    print("Streams")
    for stream_name in REQUIRED_STREAM_NAMES + OPTIONAL_STREAM_NAMES:
        stream = report["streams"].get(stream_name, {})
        print(
            f"{stream_name}: present={_status(bool(stream.get('present')))} "
            f"records={stream.get('record_count')} declared={stream.get('declared_count')}"
        )

    print("Camera Files")
    for stream_name, summary in report["camera_summary"].items():
        print(
            f"{stream_name}: frames={summary['frame_count']} missing={summary['missing_files']} "
            f"empty={summary['empty_files']} duplicates={summary['duplicate_image_paths']}"
        )

    if report["conversion_blockers"]:
        print("Conversion Blockers")
        for blocker in report["conversion_blockers"]:
            print(f"BLOCKER: {blocker}")
    if report["warnings"]:
        print("Warnings")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
    if report["errors"]:
        print("Errors")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    print("Recommendations")
    for recommendation in report["recommendations"]:
        print(f"- {recommendation}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and preflight a raw_real_v0 episode.")
    parser.add_argument("episode_dir", help="Path to raw-real episode directory")
    parser.add_argument("--json-out", help="Optional JSON inspection report output path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON report written with --json-out")
    convertible_group = parser.add_mutually_exclusive_group()
    convertible_group.add_argument("--require-convertible", dest="require_convertible", action="store_true")
    convertible_group.add_argument("--no-require-convertible", dest="require_convertible", action="store_false")
    parser.set_defaults(require_convertible=True)
    args = parser.parse_args(argv)

    try:
        report = inspect_raw_real_episode(args.episode_dir, require_convertible=args.require_convertible)
        if args.json_out:
            write_inspection_report(report, args.json_out, pretty=args.pretty)
    except (OSError, ValueError) as exc:
        print(f"FAILED: raw-real inspection failed: {args.episode_dir}")
        print(str(exc))
        return 1

    print_inspection_summary(report)
    if args.require_convertible:
        return 0 if report["ready_for_conversion"] else 1
    return 0 if report["validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
