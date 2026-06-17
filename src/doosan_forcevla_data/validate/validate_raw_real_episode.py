"""Validate a future raw real Doosan episode directory.

The validator is intentionally offline and dependency-light.  It reads JSON,
JSONL, and file paths only; it does not import ROS packages and it does not
communicate with a live robot.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.raw_real_schema import (
    OPTIONAL_STREAM_NAMES,
    RAW_REAL_SCHEMA_VERSION,
    REQUIRED_METADATA_KEYS,
    REQUIRED_STREAM_NAMES,
    REQUIRED_TOP_LEVEL_FILES,
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


JSONL_STREAM_NAMES = {
    "joint_states",
    "robot_state_rt",
    "tf",
    "tf_static",
    "command_context",
    "gripper_state",
}

CAMERA_STREAM_NAMES = {"external_camera", "wrist_camera"}
CONVERTER_REQUIRED_ALIGNMENT_STREAMS = ["joint_states", "external_camera", "wrist_camera"]
CONVERTER_ALIGNED_OPTIONAL_STREAMS = ["gripper_state"]
ROTATION_VECTOR_DEGREES = "rotation_vector_degrees"
ROTATION_VECTOR_RADIANS = "rotation_vector_radians"
SOURCE_STAMP_SECONDS_UNIT = "seconds"

STRICT_LAB_PROVENANCE_KEYS = [
    "exact_doosan_namespace",
    "external_camera_topic",
    "wrist_camera_topic",
    "read_data_rt_service",
    "tcp_frame",
    "flange_frame",
    "tool_frame",
    "force_torque_source",
    "gripper_state_source",
]
UNKNOWN_PROVENANCE_MARKERS = {
    "unknown",
    "unverified",
    "unset",
    "todo",
    "tbd",
    "none",
    "null",
    "n/a",
    "na",
}

SUPPORTED_TCP_POSITION_UNITS = {"mm", "millimeter", "millimeters", "m", "meter", "meters"}
SUPPORTED_TCP_ORIENTATION_UNITS = {"deg", "degree", "degrees", "rad", "radian", "radians"}
SUPPORTED_JOINT_POSITION_UNITS = {"deg", "degree", "degrees", "rad", "radian", "radians"}
SUPPORTED_JOINT_VELOCITY_UNITS = {
    "deg/s",
    "deg_per_s",
    "degree/s",
    "degrees/s",
    "degrees_per_second",
    "rad/s",
    "rad_per_s",
    "radian/s",
    "radians/s",
    "radians_per_second",
}
DEGREE_UNITS = {"deg", "degree", "degrees"}
RADIAN_UNITS = {"rad", "radian", "radians"}


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _is_path_under_root(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _is_numeric_list(value: Any, expected_len: int) -> bool:
    return isinstance(value, list) and len(value) == expected_len and all(_is_finite_number(item) for item in value)


def _normalized_unit(units: dict[str, Any], key: str) -> str | None:
    value = units.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace(" ", "_")


def _combined_units(record: dict[str, Any] | None, stream_entry: dict[str, Any] | None) -> dict[str, Any]:
    units: dict[str, Any] = {}
    if isinstance(stream_entry, dict) and isinstance(stream_entry.get("units"), dict):
        units.update(stream_entry["units"])
    if isinstance(record, dict) and isinstance(record.get("units"), dict):
        units.update(record["units"])
    return units


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


def _tcp_orientation_convention(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
) -> Any:
    if isinstance(metadata, dict) and metadata.get("tcp_orientation_convention") is not None:
        return metadata.get("tcp_orientation_convention")
    if isinstance(recorder_report, dict):
        return recorder_report.get("tcp_orientation_convention")
    return None


def _supported_unit_error(
    units: dict[str, Any],
    key: str,
    supported: set[str],
    context: str,
) -> str | None:
    unit = _normalized_unit(units, key)
    if unit in supported:
        return None
    return f"{context}: unsupported or missing {key} unit: {unit!r}"


def _orientation_unit_matches_convention(unit: str | None, convention: Any, context: str) -> str | None:
    if convention == ROTATION_VECTOR_DEGREES and unit not in DEGREE_UNITS:
        return f"{context}: tcp_orientation unit {unit!r} does not match tcp_orientation_convention='rotation_vector_degrees'"
    if convention == ROTATION_VECTOR_RADIANS and unit not in RADIAN_UNITS:
        return f"{context}: tcp_orientation unit {unit!r} does not match tcp_orientation_convention='rotation_vector_radians'"
    return None



def _is_unknown_provenance_value(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return (
        normalized in UNKNOWN_PROVENANCE_MARKERS
        or normalized.startswith("unknown")
        or normalized.startswith("todo")
        or normalized.startswith("tbd")
    )


def _strict_lab_provenance_required(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
) -> bool:
    if isinstance(metadata, dict) and (
        metadata.get("lab_provenance_required") is True
        or metadata.get("strict_lab_provenance") is True
    ):
        return True
    if isinstance(recorder_report, dict) and (
        recorder_report.get("lab_provenance_required") is True
        or recorder_report.get("strict_lab_provenance") is True
    ):
        return True
    return False


def _strict_lab_provenance_errors(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
    streams: dict[str, Any] | None,
    records_by_stream: dict[str, list[dict[str, Any]]],
) -> list[str]:
    if not _strict_lab_provenance_required(metadata, recorder_report):
        return [
            "non-synthetic raw-real conversion requires strict lab provenance; "
            "set lab_provenance_required=true or strict_lab_provenance=true and provide verified lab provenance"
        ]

    errors: list[str] = []
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    source_workspace = metadata_dict.get("source_workspace")
    if not isinstance(source_workspace, dict):
        errors.append("strict lab provenance: metadata.source_workspace must be a JSON object")
    else:
        if source_workspace.get("verified") is not True:
            errors.append("strict lab provenance: metadata.source_workspace.verified must be true")
        for key in ["path", "git_commit", "git_remote", "git_branch"]:
            if _is_unknown_provenance_value(source_workspace.get(key)):
                errors.append(f"strict lab provenance: metadata.source_workspace.{key} must be known")

    live_graph = metadata_dict.get("live_graph_verification")
    if not isinstance(live_graph, dict):
        errors.append("strict lab provenance: metadata.live_graph_verification must be a JSON object")
    else:
        if live_graph.get("time_sync_verified") is not True:
            errors.append("strict lab provenance: metadata.live_graph_verification.time_sync_verified must be true")
        for key in STRICT_LAB_PROVENANCE_KEYS:
            if _is_unknown_provenance_value(live_graph.get(key)):
                errors.append(f"strict lab provenance: metadata.live_graph_verification.{key} must be known")

    if isinstance(streams, dict):
        for stream_name, entry in streams.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("verified") is not True:
                errors.append(f"strict lab provenance: stream {stream_name} verified must be true")
            if _is_unknown_provenance_value(entry.get("source_name")):
                errors.append(f"strict lab provenance: stream {stream_name} source_name must be known")

    for stream_name in sorted(CAMERA_STREAM_NAMES):
        for idx, record in enumerate(records_by_stream.get(stream_name, [])):
            if _is_unknown_provenance_value(record.get("frame_id")):
                errors.append(f"strict lab provenance: {stream_name} camera record {idx} frame_id must be known")
                break

    return errors


def _read_json_object(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: could not read {label} JSON object: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: {label} must be a JSON object")
        return None
    return data


def _read_jsonl_objects(path: Path, errors: list[str], label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: could not read {label} JSONL: {exc}")
        return records

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"{path}: line {line_number} is empty")
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path}: line {line_number} must be a JSON object")
            continue
        records.append(data)
    return records


def _check_metadata(metadata: dict[str, Any], path: Path, errors: list[str]) -> None:
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        errors.append(f"{path}: metadata missing required keys: {', '.join(missing)}")

    if metadata.get("schema_version") != RAW_REAL_SCHEMA_VERSION:
        errors.append(f"{path}: schema_version must be {RAW_REAL_SCHEMA_VERSION!r}")

    for key in [
        "episode_id",
        "task_instruction",
        "geometry_type",
        "orientation_type",
        "collection_method",
        "action_label_primary",
        "robot_type",
        "recorder_version",
    ]:
        if key in metadata and not _is_non_empty_string(metadata[key]):
            errors.append(f"{path}: {key} must be a non-empty string")

    if "success" in metadata and not isinstance(metadata["success"], bool):
        errors.append(f"{path}: success must be a boolean")

    failure_reason = metadata.get("failure_reason")
    if failure_reason is not None and not _is_non_empty_string(failure_reason):
        errors.append(f"{path}: failure_reason must be null or a non-empty string")
    if metadata.get("success") is False and not _is_non_empty_string(failure_reason):
        errors.append(f"{path}: failure_reason must be a non-empty string when success is false")

    fps = metadata.get("fps")
    if not _is_finite_number(fps) or float(fps) <= 0.0:
        errors.append(f"{path}: fps must be a positive finite number")

    source_workspace = metadata.get("source_workspace")
    if "source_workspace" in metadata:
        if not isinstance(source_workspace, dict):
            errors.append(f"{path}: source_workspace must be a JSON object")
        else:
            if not _is_non_empty_string(source_workspace.get("path")):
                errors.append(f"{path}: source_workspace.path must be a non-empty string")
            if "verified" in source_workspace and not isinstance(source_workspace["verified"], bool):
                errors.append(f"{path}: source_workspace.verified must be a boolean when present")


def _source_stamp_seconds(value: Any) -> float | None:
    if _is_finite_number(value):
        return float(value)
    if not isinstance(value, dict):
        return None
    sec = value.get("sec")
    nanosec = value.get("nanosec")
    if not _is_finite_number(sec) or not _is_finite_number(nanosec):
        return None
    if not 0.0 <= float(nanosec) < 1_000_000_000.0:
        return None
    return float(sec) + float(nanosec) * 1e-9


def _required_streams_use_numeric_source_stamp(records_by_stream: dict[str, list[dict[str, Any]]]) -> bool:
    for stream_name in REQUIRED_STREAM_NAMES:
        for record in records_by_stream.get(stream_name, []):
            if _is_finite_number(record.get("source_stamp")):
                return True
    return False


def _numeric_source_stamp_timebase_errors(
    streams_index: dict[str, Any] | None,
    records_by_stream: dict[str, list[dict[str, Any]]],
) -> list[str]:
    if not _required_streams_use_numeric_source_stamp(records_by_stream):
        return []

    timebase = streams_index.get("timebase") if isinstance(streams_index, dict) else None
    source_stamp_unit = timebase.get("source_stamp_unit") if isinstance(timebase, dict) else None
    if source_stamp_unit == SOURCE_STAMP_SECONDS_UNIT:
        return []
    return [
        "source_stamp unit/timebase: streams/index.json timebase.source_stamp_unit must be "
        f"'seconds' for non-synthetic numeric source_stamp values; got {source_stamp_unit!r}"
    ]


def _validate_common_record_fields(
    records: list[dict[str, Any]],
    path: Path,
    stream_name: str,
    errors: list[str],
) -> None:
    previous_source: float | None = None
    previous_receipt: float | None = None
    previous_monotonic: float | None = None

    for expected_index, record in enumerate(records):
        context = f"{path}: {stream_name} record {expected_index}"
        record_index = record.get("record_index")
        if record_index != expected_index or isinstance(record_index, bool):
            errors.append(f"{context}: record_index must be sequential from 0")

        if "source_stamp" not in record:
            errors.append(f"{context}: source_stamp is missing")
        else:
            source = _source_stamp_seconds(record["source_stamp"])
            if source is None:
                errors.append(f"{context}: source_stamp must be numeric or an object with finite sec/nanosec")
            else:
                if previous_source is not None and source < previous_source:
                    errors.append(f"{context}: source_stamp must be monotonic nondecreasing")
                previous_source = source

        receipt_stamp = record.get("receipt_stamp")
        if not _is_finite_number(receipt_stamp):
            errors.append(f"{context}: receipt_stamp must be a finite number")
        else:
            receipt = float(receipt_stamp)
            if previous_receipt is not None and receipt < previous_receipt:
                errors.append(f"{context}: receipt_stamp must be monotonic nondecreasing")
            previous_receipt = receipt

        monotonic_stamp = record.get("monotonic_stamp")
        if not _is_finite_number(monotonic_stamp):
            errors.append(f"{context}: monotonic_stamp must be a finite number")
        else:
            monotonic = float(monotonic_stamp)
            if previous_monotonic is not None and monotonic < previous_monotonic:
                errors.append(f"{context}: monotonic_stamp must be monotonic nondecreasing")
            previous_monotonic = monotonic


def _check_numeric_list(
    record: dict[str, Any],
    key: str,
    expected_len: int,
    context: str,
    errors: list[str],
) -> None:
    values = record.get(key)
    if not isinstance(values, list):
        errors.append(f"{context}: {key} must be a list of length {expected_len}")
        return
    if len(values) != expected_len:
        errors.append(f"{context}: {key} length must be {expected_len}, got {len(values)}")
        return
    for idx, value in enumerate(values):
        if not _is_finite_number(value):
            errors.append(f"{context}: {key}[{idx}] must be a finite number")


def _stream_path(root: Path, stream_name: str, entry: dict[str, Any], errors: list[str]) -> Path | None:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"streams/index.json: stream {stream_name} missing non-empty path")
        return None

    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        errors.append(f"streams/index.json: stream {stream_name} path must be relative to episode root")
        return None
    path = root / relative_path
    if not _is_path_under_root(root, path):
        errors.append(f"streams/index.json: stream {stream_name} path must stay inside episode root")
        return None
    if not path.exists():
        errors.append(f"{path}: stream {stream_name} path does not exist")
        return None
    return path


def _check_stream_record_count(
    stream_entry: dict[str, Any],
    stream_name: str,
    records: list[dict[str, Any]],
    path: Path,
    required: bool,
    errors: list[str],
) -> None:
    record_count = stream_entry.get("record_count")
    if _is_non_negative_int(record_count) and record_count != len(records):
        errors.append(
            f"{path}: stream {stream_name} record_count {record_count} does not match actual record count {len(records)}"
        )
    if required and not records:
        errors.append(f"{path}: required stream {stream_name} must contain at least one record")


def _validate_required_stream_entry(
    root: Path,
    stream_name: str,
    entry: Any,
    errors: list[str],
) -> Path | None:
    if not isinstance(entry, dict):
        errors.append(f"streams/index.json: stream {stream_name} entry must be a JSON object")
        return None

    source_name = entry.get("source_name")
    if not isinstance(source_name, str) or not source_name:
        errors.append(f"streams/index.json: stream {stream_name} missing non-empty source_name")

    source_type = entry.get("source_type")
    if not isinstance(source_type, str) or not source_type:
        errors.append(f"streams/index.json: stream {stream_name} missing non-empty source_type")

    if entry.get("required") is not True:
        errors.append(f"streams/index.json: stream {stream_name} required must be true")

    if not _is_non_negative_int(entry.get("record_count")):
        errors.append(f"streams/index.json: stream {stream_name} record_count must be a non-negative integer")

    return _stream_path(root, stream_name, entry, errors)


def _validate_optional_stream_entry(
    root: Path,
    stream_name: str,
    entry: Any,
    errors: list[str],
) -> Path | None:
    if not isinstance(entry, dict):
        errors.append(f"streams/index.json: optional stream {stream_name} entry must be a JSON object")
        return None
    if "required" in entry and entry.get("required") is not False:
        errors.append(f"streams/index.json: optional stream {stream_name} required must be false when present")
    if "record_count" in entry and not _is_non_negative_int(entry.get("record_count")):
        errors.append(
            f"streams/index.json: optional stream {stream_name} record_count must be a non-negative integer"
        )
    if "path" not in entry:
        return None
    return _stream_path(root, stream_name, entry, errors)


def _add_stream_warnings(streams: dict[str, Any], warnings: list[str]) -> None:
    for stream_name in OPTIONAL_STREAM_NAMES:
        if stream_name not in streams:
            warnings.append(f"optional stream {stream_name} is absent")

    for stream_name, entry in streams.items():
        if not isinstance(entry, dict):
            continue
        source_name = entry.get("source_name")
        if isinstance(source_name, str) and "unknown" in source_name.lower():
            warnings.append(f"stream {stream_name} source_name is unverified or unknown: {source_name}")
        if entry.get("verified") is False:
            warnings.append(f"stream {stream_name} has verified: false")


def _validate_joint_states(
    records: list[dict[str, Any]],
    stream_entry: dict[str, Any],
    path: Path,
    errors: list[str],
) -> None:
    for idx, record in enumerate(records):
        context = f"{path}: joint_states record {idx}"
        joint_names = record.get("joint_names")
        if not isinstance(joint_names, list) or len(joint_names) != 6:
            errors.append(f"{context}: joint_names must be a list of length 6")
        else:
            for joint_idx, joint_name in enumerate(joint_names):
                if not _is_non_empty_string(joint_name):
                    errors.append(f"{context}: joint_names[{joint_idx}] must be a non-empty string")
            if all(isinstance(joint_name, str) for joint_name in joint_names) and len(set(joint_names)) != len(
                joint_names
            ):
                errors.append(f"{context}: joint_names must be unique")
        _check_numeric_list(record, "position", 6, context, errors)
        _check_numeric_list(record, "velocity", 6, context, errors)
        if not isinstance(record.get("units"), dict) and not isinstance(stream_entry.get("units"), dict):
            errors.append(f"{context}: units object must exist on the record or stream entry")


def _validate_robot_state_rt(records: list[dict[str, Any]], path: Path, errors: list[str]) -> None:
    for idx, record in enumerate(records):
        context = f"{path}: robot_state_rt record {idx}"
        _check_numeric_list(record, "actual_tcp_position", 6, context, errors)

        has_external_tcp_force = "external_tcp_force" in record
        has_raw_force_torque = "raw_force_torque" in record
        if not has_external_tcp_force and not has_raw_force_torque:
            errors.append(f"{context}: external_tcp_force or raw_force_torque must exist")
        if has_external_tcp_force:
            _check_numeric_list(record, "external_tcp_force", 6, context, errors)
        if has_raw_force_torque:
            _check_numeric_list(record, "raw_force_torque", 6, context, errors)

        for key in ["robot_mode", "robot_state", "control_mode"]:
            if not _is_non_empty_string(record.get(key)):
                errors.append(f"{context}: {key} must be a non-empty string")


def _validate_tf_records(records: list[dict[str, Any]], path: Path, stream_name: str, errors: list[str]) -> None:
    for idx, record in enumerate(records):
        transforms = record.get("transforms")
        if not isinstance(transforms, list):
            errors.append(f"{path}: {stream_name} record {idx}: transforms must be a list")
            continue
        for transform_idx, transform in enumerate(transforms):
            context = f"{path}: {stream_name} record {idx} transform {transform_idx}"
            if not isinstance(transform, dict):
                errors.append(f"{context}: transform must be a JSON object")
                continue
            for key in ["parent_frame", "child_frame"]:
                if not _is_non_empty_string(transform.get(key)):
                    errors.append(f"{context}: {key} must be a non-empty string")
            _check_numeric_list(transform, "translation", 3, context, errors)
            rotation = transform.get("rotation_xyzw", transform.get("rotation"))
            if not isinstance(rotation, list):
                errors.append(f"{context}: rotation_xyzw must be a list of length 4")
            elif len(rotation) != 4:
                errors.append(f"{context}: rotation_xyzw length must be 4, got {len(rotation)}")
            else:
                for rotation_idx, value in enumerate(rotation):
                    if not _is_finite_number(value):
                        errors.append(f"{context}: rotation_xyzw[{rotation_idx}] must be a finite number")


def _validate_gripper_state(records: list[dict[str, Any]], path: Path, errors: list[str]) -> None:
    for idx, record in enumerate(records):
        context = f"{path}: gripper_state record {idx}"
        has_position = "gripper_position" in record
        has_width = "gripper_width_m" in record
        if not has_position and not has_width:
            errors.append(f"{context}: gripper_position or gripper_width_m must exist")
        if has_position and not _is_finite_number(record.get("gripper_position")):
            errors.append(f"{context}: gripper_position must be a finite number")
        if has_width:
            width = record.get("gripper_width_m")
            if not _is_finite_number(width):
                errors.append(f"{context}: gripper_width_m must be a finite number")
            elif float(width) < 0.0:
                errors.append(f"{context}: gripper_width_m must be non-negative")


def _validate_command_context(
    records: list[dict[str, Any]],
    path: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    for idx, record in enumerate(records):
        context = f"{path}: command_context record {idx}"
        if "command_kind" in record and not _is_non_empty_string(record.get("command_kind")):
            errors.append(f"{context}: command_kind must be a non-empty string when present")
        if "commanded_twist" in record:
            _check_numeric_list(record, "commanded_twist", 6, context, errors)
        action_like_keys = [key for key in ["action_label", "measured_action", "action"] if key in record]
        if action_like_keys:
            warnings.append(
                f"{context}: action-like fields are diagnostic only and are not action labels: {', '.join(action_like_keys)}"
            )


def _validate_jsonl_stream(
    path: Path,
    stream_name: str,
    stream_entry: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    required: bool,
) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"{path}: stream {stream_name} must be a JSONL file")
        return []

    records = _read_jsonl_objects(path, errors, stream_name)
    _check_stream_record_count(stream_entry, stream_name, records, path, required, errors)
    _validate_common_record_fields(records, path, stream_name, errors)

    if stream_name == "joint_states":
        _validate_joint_states(records, stream_entry, path, errors)
    elif stream_name == "robot_state_rt":
        _validate_robot_state_rt(records, path, errors)
    elif stream_name in {"tf", "tf_static"}:
        _validate_tf_records(records, path, stream_name, errors)
    elif stream_name == "gripper_state":
        _validate_gripper_state(records, path, errors)
    elif stream_name == "command_context":
        _validate_command_context(records, path, errors, warnings)
    return records


def _validate_camera_index(
    root: Path,
    stream_path: Path,
    stream_name: str,
    stream_entry: dict[str, Any],
    errors: list[str],
) -> list[dict[str, Any]]:
    if not stream_path.is_dir():
        errors.append(f"{stream_path}: camera stream {stream_name} must be a directory")
        return []

    index_path = stream_path / "index.jsonl"
    if not index_path.is_file():
        errors.append(f"{index_path}: camera stream {stream_name} index.jsonl is missing")
        return []

    records = _read_jsonl_objects(index_path, errors, f"{stream_name} index")
    _check_stream_record_count(stream_entry, stream_name, records, index_path, required=True, errors=errors)
    _validate_common_record_fields(records, index_path, stream_name, errors)

    for idx, record in enumerate(records):
        context = f"{index_path}: {stream_name} camera record {idx}"
        image_path_value = record.get("image_path")
        if not isinstance(image_path_value, str) or not image_path_value:
            errors.append(f"{context}: image_path must be a non-empty string")
        else:
            image_relative = Path(image_path_value)
            if image_relative.is_absolute() or ".." in image_relative.parts:
                errors.append(f"{context}: image_path must be relative to episode root")
            else:
                image_path = root / image_relative
                if not _is_path_under_root(root, image_path):
                    errors.append(f"{context}: image_path must stay inside episode root")
                elif not image_path.is_file():
                    errors.append(f"{context}: image_path does not exist: {image_path}")
                else:
                    try:
                        image_size = image_path.stat().st_size
                    except OSError as exc:
                        errors.append(f"{context}: could not stat image_path {image_path}: {exc}")
                    else:
                        if image_size <= 0:
                            errors.append(f"{context}: image_path must reference a non-empty file: {image_path}")

        for key in ["width", "height", "channels"]:
            if not _is_positive_int(record.get(key)):
                errors.append(f"{context}: {key} must be a positive integer")

        for key in ["encoding", "frame_id"]:
            if not isinstance(record.get(key), str) or not record.get(key):
                errors.append(f"{context}: {key} must be a non-empty string")

    return records


def _validate_events(
    records: list[dict[str, Any]],
    path: Path,
    metadata: dict[str, Any] | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not records:
        errors.append(f"{path}: events must contain at least one record")
        return

    previous_timestamp: float | None = None
    event_names: list[str] = []
    for idx, record in enumerate(records):
        context = f"{path}: event record {idx}"
        timestamp = record.get("timestamp")
        if not _is_finite_number(timestamp):
            errors.append(f"{context}: timestamp must be a finite number")
        else:
            timestamp_float = float(timestamp)
            if previous_timestamp is not None and timestamp_float < previous_timestamp:
                errors.append(f"{context}: timestamp must be monotonic nondecreasing")
            previous_timestamp = timestamp_float

        event_name = record.get("event")
        if not _is_non_empty_string(event_name):
            errors.append(f"{context}: event must be a non-empty string")
        else:
            event_names.append(event_name)

    if event_names and event_names[0] != "episode_start":
        warnings.append(f"{path}: first event is not episode_start")
    if metadata is None or not isinstance(metadata.get("success"), bool):
        return
    if metadata["success"] and "success" not in event_names:
        warnings.append(f"{path}: metadata success is true but events do not contain success")
    if metadata["success"] is False and not any("fail" in event_name.lower() for event_name in event_names):
        warnings.append(f"{path}: metadata success is false but events do not contain a failure event")


def _record_index_set(records: list[dict[str, Any]]) -> set[int]:
    indexes: set[int] = set()
    for record in records:
        record_index = record.get("record_index")
        if _is_non_negative_int(record_index):
            indexes.add(record_index)
    return indexes


def _alignment_details(primary_indexes: set[int], candidate_indexes: set[int]) -> str:
    details: list[str] = []
    missing = sorted(primary_indexes - candidate_indexes)
    extra = sorted(candidate_indexes - primary_indexes)
    if missing:
        details.append(f"missing robot_state_rt record_index values {missing[:10]}")
    if extra:
        details.append(f"extra record_index values {extra[:10]}")
    return ", ".join(details) if details else "index sets differ"


def _check_record_index_alignment(
    records_by_stream: dict[str, list[dict[str, Any]]],
    errors: list[str],
    warnings: list[str],
) -> None:
    primary_records = records_by_stream.get("robot_state_rt", [])
    primary_indexes = _record_index_set(primary_records)
    if not primary_indexes:
        return

    for stream_name in CONVERTER_REQUIRED_ALIGNMENT_STREAMS:
        if stream_name not in records_by_stream:
            continue
        candidate_indexes = _record_index_set(records_by_stream[stream_name])
        if candidate_indexes != primary_indexes:
            errors.append(
                f"{stream_name}: record_index alignment with robot_state_rt failed: "
                f"{_alignment_details(primary_indexes, candidate_indexes)}"
            )

    for stream_name in CONVERTER_ALIGNED_OPTIONAL_STREAMS:
        if stream_name not in records_by_stream:
            continue
        candidate_indexes = _record_index_set(records_by_stream[stream_name])
        if candidate_indexes != primary_indexes:
            errors.append(
                f"{stream_name}: record_index alignment with robot_state_rt failed: "
                f"{_alignment_details(primary_indexes, candidate_indexes)}"
            )

    for stream_name in ["tf", "command_context"]:
        if stream_name not in records_by_stream:
            continue
        candidate_indexes = _record_index_set(records_by_stream[stream_name])
        if candidate_indexes and candidate_indexes != primary_indexes:
            warnings.append(
                f"{stream_name}: record_index differs from robot_state_rt: "
                f"{_alignment_details(primary_indexes, candidate_indexes)}"
            )


def _records_by_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        record_index = record.get("record_index")
        if _is_non_negative_int(record_index) and record_index not in indexed:
            indexed[record_index] = record
    return indexed


def timestamp_tolerance_seconds(metadata: dict[str, Any] | None) -> float:
    if metadata is not None:
        fps = metadata.get("fps")
        if _is_finite_number(fps) and float(fps) > 0.0:
            return max(0.1, 2.0 / float(fps))
    return 0.1


def _source_stamp_alignment_errors(
    records_by_stream: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any] | None,
) -> list[str]:
    robot_by_index = _records_by_index(records_by_stream.get("robot_state_rt", []))
    if not robot_by_index:
        return []

    errors: list[str] = []
    tolerance = timestamp_tolerance_seconds(metadata)
    for stream_name in ["external_camera", "wrist_camera"]:
        stream_by_index = _records_by_index(records_by_stream.get(stream_name, []))
        common_indexes = sorted(set(robot_by_index) & set(stream_by_index))
        for record_index in common_indexes:
            robot_stamp = _source_stamp_seconds(robot_by_index[record_index].get("source_stamp"))
            stream_stamp = _source_stamp_seconds(stream_by_index[record_index].get("source_stamp"))
            if robot_stamp is None or stream_stamp is None:
                continue
            if abs(stream_stamp - robot_stamp) > tolerance:
                errors.append(
                    f"{stream_name}: source_stamp differs from robot_state_rt by more than "
                    f"{tolerance:.3f}s at record_index {record_index}; raw_real_v0 conversion requires "
                    "aligned episode-level record_index values"
                )
                break
    return errors


def raw_real_conversion_readiness_errors(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
    streams: dict[str, Any] | None,
    records_by_stream: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Return validator errors for raw_real_v0 data the converter would reject."""

    errors = _source_stamp_alignment_errors(records_by_stream, metadata)
    if _is_synthetic_episode(metadata, recorder_report, streams_index):
        return errors
    if not isinstance(streams, dict):
        return errors

    errors.extend(_numeric_source_stamp_timebase_errors(streams_index, records_by_stream))

    convention = _tcp_orientation_convention(metadata, recorder_report)
    if convention not in {ROTATION_VECTOR_DEGREES, ROTATION_VECTOR_RADIANS}:
        errors.append(
            "metadata/recorder_report: tcp_orientation_convention must be one of "
            "'rotation_vector_degrees' or 'rotation_vector_radians' for non-synthetic conversion; "
            "tcp_orientation_convention_verified alone is not sufficient"
        )

    errors.extend(_strict_lab_provenance_errors(metadata, recorder_report, streams, records_by_stream))

    robot_entry = streams.get("robot_state_rt") if isinstance(streams.get("robot_state_rt"), dict) else {}
    joint_entry = streams.get("joint_states") if isinstance(streams.get("joint_states"), dict) else {}
    robot_records = records_by_stream.get("robot_state_rt", [])
    joint_by_index = _records_by_index(records_by_stream.get("joint_states", []))

    for idx, robot_record in enumerate(robot_records):
        context = f"robot_state_rt record {idx}"
        robot_units = _combined_units(robot_record, robot_entry)

        tcp_position_error = _supported_unit_error(
            robot_units,
            "tcp_position",
            SUPPORTED_TCP_POSITION_UNITS,
            f"{context} actual_tcp_position",
        )
        if tcp_position_error is not None:
            errors.append(tcp_position_error)

        tcp_orientation_error = _supported_unit_error(
            robot_units,
            "tcp_orientation",
            SUPPORTED_TCP_ORIENTATION_UNITS,
            f"{context} actual_tcp_position[3:6]",
        )
        if tcp_orientation_error is not None:
            errors.append(tcp_orientation_error)
        else:
            mismatch_error = _orientation_unit_matches_convention(
                _normalized_unit(robot_units, "tcp_orientation"),
                convention,
                f"{context} actual_tcp_position[3:6]",
            )
            if mismatch_error is not None:
                errors.append(mismatch_error)

        record_index = robot_record.get("record_index")
        joint_record = joint_by_index.get(record_index) if _is_non_negative_int(record_index) else None
        joint_units = _combined_units(joint_record, joint_entry)

        if _is_numeric_list(robot_record.get("actual_joint_position"), 6):
            joint_position_error = _supported_unit_error(
                robot_units,
                "joint_position",
                SUPPORTED_JOINT_POSITION_UNITS,
                f"{context} actual_joint_position",
            )
            if joint_position_error is not None:
                errors.append(joint_position_error)
        elif not isinstance(joint_record, dict) or not _is_numeric_list(joint_record.get("position"), 6):
            errors.append(
                f"{context} actual_joint_position is missing/invalid and joint_states record_index "
                f"{record_index!r} position is not valid for fallback"
            )
        else:
            joint_position_error = _supported_unit_error(
                joint_units,
                "position",
                SUPPORTED_JOINT_POSITION_UNITS,
                f"joint_states record_index {record_index} position",
            )
            if joint_position_error is not None:
                errors.append(joint_position_error)

        if _is_numeric_list(robot_record.get("actual_joint_velocity"), 6):
            joint_velocity_error = _supported_unit_error(
                robot_units,
                "joint_velocity",
                SUPPORTED_JOINT_VELOCITY_UNITS,
                f"{context} actual_joint_velocity",
            )
            if joint_velocity_error is not None:
                errors.append(joint_velocity_error)
        elif not isinstance(joint_record, dict) or not _is_numeric_list(joint_record.get("velocity"), 6):
            errors.append(
                f"{context} actual_joint_velocity is missing/invalid and joint_states record_index "
                f"{record_index!r} velocity is not valid for fallback"
            )
        else:
            joint_velocity_error = _supported_unit_error(
                joint_units,
                "velocity",
                SUPPORTED_JOINT_VELOCITY_UNITS,
                f"joint_states record_index {record_index} velocity",
            )
            if joint_velocity_error is not None:
                errors.append(joint_velocity_error)

    return errors


def _warn_timestamp_mismatches(
    records_by_stream: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any] | None,
    warnings: list[str],
) -> None:
    robot_by_index = _records_by_index(records_by_stream.get("robot_state_rt", []))
    if not robot_by_index:
        return
    tolerance = timestamp_tolerance_seconds(metadata)

    for stream_name in [
        "joint_states",
        "external_camera",
        "wrist_camera",
        "gripper_state",
        "command_context",
    ]:
        if stream_name not in records_by_stream:
            continue
        stream_by_index = _records_by_index(records_by_stream[stream_name])
        common_indexes = sorted(set(robot_by_index) & set(stream_by_index))
        for stamp_key in ["receipt_stamp", "monotonic_stamp"]:
            for record_index in common_indexes:
                robot_stamp = robot_by_index[record_index].get(stamp_key)
                stream_stamp = stream_by_index[record_index].get(stamp_key)
                if not _is_finite_number(robot_stamp) or not _is_finite_number(stream_stamp):
                    continue
                if abs(float(stream_stamp) - float(robot_stamp)) > tolerance:
                    warnings.append(
                        f"{stream_name}: {stamp_key} differs from robot_state_rt by more than "
                        f"{tolerance:.3f}s at record_index {record_index}"
                    )
                    break


def validate_raw_real_episode(root_dir: str | Path) -> ValidationResult:
    """Validate a raw real episode directory."""

    root = Path(root_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if not root.exists():
        return ValidationResult(False, [f"{root}: episode directory does not exist"], warnings)
    if not root.is_dir():
        return ValidationResult(False, [f"{root}: episode path is not a directory"], warnings)

    for relative_path in REQUIRED_TOP_LEVEL_FILES:
        path = root / relative_path
        if not path.is_file():
            errors.append(f"{path}: required file is missing")
        elif not _is_path_under_root(root, path):
            errors.append(f"{path}: required file must stay inside episode root")
    if errors:
        return ValidationResult(False, errors, warnings)

    metadata_path = root / "metadata.json"
    metadata = _read_json_object(metadata_path, errors, "metadata")
    if metadata is not None:
        _check_metadata(metadata, metadata_path, errors)

    _read_json_object(root / "calibration_refs.json", errors, "calibration_refs")
    recorder_report = _read_json_object(root / "recorder_report.json", errors, "recorder_report")
    events = _read_jsonl_objects(root / "events.jsonl", errors, "events")
    _validate_events(events, root / "events.jsonl", metadata, errors, warnings)

    streams_index_path = root / "streams" / "index.json"
    streams_index = _read_json_object(streams_index_path, errors, "streams/index")
    if streams_index is None:
        return ValidationResult(False, errors, warnings)

    if streams_index.get("schema_version") != RAW_REAL_SCHEMA_VERSION:
        errors.append(f"{streams_index_path}: schema_version must be {RAW_REAL_SCHEMA_VERSION!r}")

    streams = streams_index.get("streams")
    if not isinstance(streams, dict):
        errors.append(f"{streams_index_path}: streams must be a JSON object")
        return ValidationResult(False, errors, warnings)

    _add_stream_warnings(streams, warnings)

    required_stream_paths: dict[str, Path] = {}
    for stream_name in REQUIRED_STREAM_NAMES:
        if stream_name not in streams:
            errors.append(f"{streams_index_path}: required stream is missing: {stream_name}")
            continue
        path = _validate_required_stream_entry(root, stream_name, streams[stream_name], errors)
        if path is not None and isinstance(streams[stream_name], dict):
            required_stream_paths[stream_name] = path

    optional_stream_paths: dict[str, Path] = {}
    for stream_name in OPTIONAL_STREAM_NAMES:
        if stream_name not in streams:
            continue
        path = _validate_optional_stream_entry(root, stream_name, streams[stream_name], errors)
        if path is not None and isinstance(streams[stream_name], dict):
            optional_stream_paths[stream_name] = path

    records_by_stream: dict[str, list[dict[str, Any]]] = {}

    for stream_name, path in required_stream_paths.items():
        stream_entry = streams[stream_name]
        if not isinstance(stream_entry, dict):
            continue
        if stream_name in JSONL_STREAM_NAMES:
            records_by_stream[stream_name] = _validate_jsonl_stream(
                path,
                stream_name,
                stream_entry,
                errors,
                warnings,
                required=True,
            )
        elif stream_name in CAMERA_STREAM_NAMES:
            records_by_stream[stream_name] = _validate_camera_index(
                root,
                path,
                stream_name,
                stream_entry,
                errors,
            )

    for stream_name, path in optional_stream_paths.items():
        stream_entry = streams[stream_name]
        if isinstance(stream_entry, dict) and stream_name in JSONL_STREAM_NAMES:
            records_by_stream[stream_name] = _validate_jsonl_stream(
                path,
                stream_name,
                stream_entry,
                errors,
                warnings,
                required=False,
            )

    _check_record_index_alignment(records_by_stream, errors, warnings)
    errors.extend(
        raw_real_conversion_readiness_errors(
            metadata,
            recorder_report,
            streams_index,
            streams,
            records_by_stream,
        )
    )
    _warn_timestamp_mismatches(records_by_stream, metadata, warnings)

    return ValidationResult(not errors, errors, warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a raw real episode directory.")
    parser.add_argument("episode_dir", help="Path to raw real episode directory")
    args = parser.parse_args(argv)

    result = validate_raw_real_episode(args.episode_dir)
    if result.ok:
        print(f"OK: raw real episode is valid: {args.episode_dir}")
        return 0

    print(f"FAILED: raw real episode is invalid: {args.episode_dir}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
