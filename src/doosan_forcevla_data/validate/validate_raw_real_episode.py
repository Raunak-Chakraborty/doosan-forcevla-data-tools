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


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


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

    if "success" in metadata and not isinstance(metadata["success"], bool):
        errors.append(f"{path}: success must be a boolean")

    fps = metadata.get("fps")
    if not _is_finite_number(fps) or float(fps) <= 0.0:
        errors.append(f"{path}: fps must be a positive finite number")


def _valid_source_stamp(value: Any) -> bool:
    if _is_finite_number(value):
        return True
    if not isinstance(value, dict):
        return False
    sec = value.get("sec")
    nanosec = value.get("nanosec")
    if not _is_finite_number(sec) or not _is_finite_number(nanosec):
        return False
    return 0.0 <= float(nanosec) < 1_000_000_000.0


def _validate_common_record_fields(
    records: list[dict[str, Any]],
    path: Path,
    stream_name: str,
    errors: list[str],
) -> None:
    previous_receipt: float | None = None
    previous_monotonic: float | None = None

    for expected_index, record in enumerate(records):
        context = f"{path}: {stream_name} record {expected_index}"
        record_index = record.get("record_index")
        if record_index != expected_index or isinstance(record_index, bool):
            errors.append(f"{context}: record_index must be sequential from 0")

        if "source_stamp" not in record:
            errors.append(f"{context}: source_stamp is missing")
        elif not _valid_source_stamp(record["source_stamp"]):
            errors.append(f"{context}: source_stamp must be numeric or an object with finite sec/nanosec")

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
    if not path.exists():
        errors.append(f"{path}: stream {stream_name} path does not exist")
        return None
    return path


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
        _check_numeric_list(record, "position", 6, context, errors)
        _check_numeric_list(record, "velocity", 6, context, errors)
        if not isinstance(record.get("units"), dict) and not isinstance(stream_entry.get("units"), dict):
            errors.append(f"{context}: units object must exist on the record or stream entry")


def _validate_robot_state_rt(records: list[dict[str, Any]], path: Path, errors: list[str]) -> None:
    for idx, record in enumerate(records):
        context = f"{path}: robot_state_rt record {idx}"
        _check_numeric_list(record, "actual_tcp_position", 6, context, errors)
        _check_numeric_list(record, "actual_joint_position", 6, context, errors)
        _check_numeric_list(record, "actual_joint_velocity", 6, context, errors)

        has_external_tcp_force = "external_tcp_force" in record
        has_raw_force_torque = "raw_force_torque" in record
        if not has_external_tcp_force and not has_raw_force_torque:
            errors.append(f"{context}: external_tcp_force or raw_force_torque must exist")
        if has_external_tcp_force:
            _check_numeric_list(record, "external_tcp_force", 6, context, errors)
        if has_raw_force_torque:
            _check_numeric_list(record, "raw_force_torque", 6, context, errors)

        for key in ["robot_mode", "robot_state", "control_mode"]:
            if key not in record:
                errors.append(f"{context}: {key} must exist")


def _validate_tf_records(records: list[dict[str, Any]], path: Path, stream_name: str, errors: list[str]) -> None:
    for idx, record in enumerate(records):
        transforms = record.get("transforms")
        if not isinstance(transforms, list):
            errors.append(f"{path}: {stream_name} record {idx}: transforms must be a list")


def _validate_jsonl_stream(
    path: Path,
    stream_name: str,
    stream_entry: dict[str, Any],
    errors: list[str],
) -> None:
    if not path.is_file():
        errors.append(f"{path}: stream {stream_name} must be a JSONL file")
        return

    records = _read_jsonl_objects(path, errors, stream_name)
    _validate_common_record_fields(records, path, stream_name, errors)

    if stream_name == "joint_states":
        _validate_joint_states(records, stream_entry, path, errors)
    elif stream_name == "robot_state_rt":
        _validate_robot_state_rt(records, path, errors)
    elif stream_name in {"tf", "tf_static"}:
        _validate_tf_records(records, path, stream_name, errors)


def _validate_camera_index(root: Path, stream_path: Path, stream_name: str, errors: list[str]) -> None:
    if not stream_path.is_dir():
        errors.append(f"{stream_path}: camera stream {stream_name} must be a directory")
        return

    index_path = stream_path / "index.jsonl"
    if not index_path.is_file():
        errors.append(f"{index_path}: camera stream {stream_name} index.jsonl is missing")
        return

    records = _read_jsonl_objects(index_path, errors, f"{stream_name} index")
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
            elif not (root / image_relative).is_file():
                errors.append(f"{context}: image_path does not exist: {root / image_relative}")

        for key in ["width", "height", "channels"]:
            if not _is_positive_int(record.get(key)):
                errors.append(f"{context}: {key} must be a positive integer")

        for key in ["encoding", "frame_id"]:
            if not isinstance(record.get(key), str) or not record.get(key):
                errors.append(f"{context}: {key} must be a non-empty string")


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
    if errors:
        return ValidationResult(False, errors, warnings)

    metadata_path = root / "metadata.json"
    metadata = _read_json_object(metadata_path, errors, "metadata")
    if metadata is not None:
        _check_metadata(metadata, metadata_path, errors)

    _read_json_object(root / "calibration_refs.json", errors, "calibration_refs")
    _read_json_object(root / "recorder_report.json", errors, "recorder_report")
    _read_jsonl_objects(root / "events.jsonl", errors, "events")

    streams_index_path = root / "streams" / "index.json"
    streams_index = _read_json_object(streams_index_path, errors, "streams/index")
    if streams_index is None:
        return ValidationResult(False, errors, warnings)

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

    for stream_name, path in required_stream_paths.items():
        stream_entry = streams[stream_name]
        if not isinstance(stream_entry, dict):
            continue
        if stream_name in JSONL_STREAM_NAMES:
            _validate_jsonl_stream(path, stream_name, stream_entry, errors)
        elif stream_name in CAMERA_STREAM_NAMES:
            _validate_camera_index(root, path, stream_name, errors)

    for stream_name, path in optional_stream_paths.items():
        stream_entry = streams[stream_name]
        if isinstance(stream_entry, dict) and stream_name in JSONL_STREAM_NAMES:
            _validate_jsonl_stream(path, stream_name, stream_entry, errors)

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
