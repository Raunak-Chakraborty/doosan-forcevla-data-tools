"""Validate a simple v0 processed JSONL episode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_raw_episode import ValidationResult


REQUIRED_METADATA_KEYS = [
    "source_raw_episode",
    "dataset_name",
    "robot_type",
    "fps",
    "quaternion_convention",
    "model_state_dim",
    "action_dim",
    "action_label_primary",
    "frame_count",
    "task_instruction",
    "geometry_type",
    "orientation_type",
    "collection_method",
    "success",
    "failure_reason",
    "notes",
]

REQUIRED_FRAME_KEYS = [
    "frame_index",
    "timestamp",
    "external_rgb_path",
    "tcp_rgb_path",
    "model_state",
    "measured_action",
    "action_is_terminal_padding",
]

STRICT_METADATA_SCHEMA_VERSION = "processed_jsonl_v1"
ROTATION_VECTOR_RADIANS = "rotation_vector_radians"
METER_UNITS = {"m", "meter", "meters", "metre", "metres"}
RADIAN_UNITS = {"rad", "radian", "radians"}


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read_metadata(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: could not read metadata JSON: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: metadata must be a JSON object")
        return None
    return data


def _read_frames(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: could not read frames JSONL: {exc}")
        return frames

    if not lines:
        errors.append(f"{path}: frames.jsonl has no frame lines")
        return frames

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"{path}: line {line_number} is empty")
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(frame, dict):
            errors.append(f"{path}: line {line_number} must be a JSON object")
            continue
        frames.append(frame)
    return frames


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _check_numeric_vector(
    frame_idx: int,
    frame: dict[str, Any],
    key: str,
    expected_len: int,
    errors: list[str],
) -> None:
    value = frame.get(key)
    if not isinstance(value, list):
        errors.append(f"frame {frame_idx}: {key} must be a list")
        return
    if len(value) != expected_len:
        errors.append(f"frame {frame_idx}: {key} length must be {expected_len}, got {len(value)}")
        return
    for value_idx, item in enumerate(value):
        if not _is_finite_number(item):
            errors.append(f"frame {frame_idx}: {key}[{value_idx}] must be a finite number")


def _requires_new_layout_metadata(metadata: dict[str, Any]) -> bool:
    return metadata.get("processed_metadata_schema_version") == STRICT_METADATA_SCHEMA_VERSION


def _normalized_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace(" ", "_")


def _layout_entries_by_index(layout: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(layout, list):
        return {}
    entries: dict[int, dict[str, Any]] = {}
    for entry in layout:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        entries[index] = entry
    return entries


def _check_layout(
    metadata_path: Path,
    metadata: dict[str, Any],
    key: str,
    expected_dim: int,
    errors: list[str],
) -> None:
    layout = metadata.get(key)
    if layout is None:
        if _requires_new_layout_metadata(metadata):
            errors.append(f"{metadata_path}: {key} is required for {STRICT_METADATA_SCHEMA_VERSION}")
        return
    if not isinstance(layout, list):
        errors.append(f"{metadata_path}: {key} must be a list")
        return
    if len(layout) != expected_dim:
        errors.append(f"{metadata_path}: {key} must contain {expected_dim} entries, got {len(layout)}")

    indexes: list[int] = []
    for entry_idx, entry in enumerate(layout):
        if not isinstance(entry, dict):
            errors.append(f"{metadata_path}: {key}[{entry_idx}] must be a JSON object")
            continue
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            errors.append(f"{metadata_path}: {key}[{entry_idx}].index must be an integer")
        else:
            indexes.append(index)
        for required_key in ["name", "source", "unit"]:
            if not _is_non_empty_string(entry.get(required_key)):
                errors.append(f"{metadata_path}: {key}[{entry_idx}].{required_key} must be a non-empty string")

    if sorted(indexes) != list(range(expected_dim)):
        errors.append(f"{metadata_path}: {key} indexes must be unique and contiguous 0..{expected_dim - 1}")


def _check_output_unit_metadata(metadata_path: Path, metadata: dict[str, Any], errors: list[str]) -> None:
    if not _requires_new_layout_metadata(metadata):
        return
    if metadata.get("tcp_orientation_output_convention") != ROTATION_VECTOR_RADIANS:
        errors.append(f"{metadata_path}: tcp_orientation_output_convention must be {ROTATION_VECTOR_RADIANS!r}")
    if _normalized_text(metadata.get("tcp_orientation_output_unit")) not in RADIAN_UNITS:
        errors.append(f"{metadata_path}: tcp_orientation_output_unit must be radians")
    if _normalized_text(metadata.get("tcp_position_output_unit")) not in METER_UNITS:
        errors.append(f"{metadata_path}: tcp_position_output_unit must be metres")

    model_layout = _layout_entries_by_index(metadata.get("model_state_layout"))
    for index in range(3):
        if index in model_layout and _normalized_text(model_layout[index].get("unit")) not in METER_UNITS:
            errors.append(f"{metadata_path}: model_state_layout[{index}].unit must be metres")
    for index in range(3, 6):
        if index in model_layout and _normalized_text(model_layout[index].get("unit")) not in RADIAN_UNITS:
            errors.append(f"{metadata_path}: model_state_layout[{index}].unit must be radians")

    action_layout = _layout_entries_by_index(metadata.get("measured_action_layout"))
    for index in range(3):
        if index in action_layout and _normalized_text(action_layout[index].get("unit")) not in METER_UNITS:
            errors.append(f"{metadata_path}: measured_action_layout[{index}].unit must be metres")
    for index in range(3, 6):
        if index in action_layout and _normalized_text(action_layout[index].get("unit")) not in RADIAN_UNITS:
            errors.append(f"{metadata_path}: measured_action_layout[{index}].unit must be radians")


def _check_gripper_placeholder_metadata(metadata_path: Path, metadata: dict[str, Any], errors: list[str]) -> None:
    if metadata.get("gripper_state_is_placeholder") is not True:
        return
    for key in ["gripper_state_source", "gripper_state_provenance"]:
        value = metadata.get(key)
        if isinstance(value, str) and "measured" in value.lower():
            errors.append(f"{metadata_path}: placeholder gripper metadata must not label {key} as measured")
    selected = metadata.get("selected_streams")
    if isinstance(selected, dict):
        gripper_value = selected.get("gripper_state")
        if isinstance(gripper_value, str) and "measured" in gripper_value.lower():
            errors.append(f"{metadata_path}: placeholder selected_streams.gripper_state must not be labelled measured")


def _check_terminal_action_policy(
    metadata_path: Path,
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    padding_indices: list[int],
    errors: list[str],
) -> None:
    policy = metadata.get("terminal_action_policy")
    if policy is None:
        if _requires_new_layout_metadata(metadata):
            errors.append(f"{metadata_path}: terminal_action_policy is required for {STRICT_METADATA_SCHEMA_VERSION}")
        return
    if not isinstance(policy, dict):
        errors.append(f"{metadata_path}: terminal_action_policy must be a JSON object")
        return

    if policy.get("final_observation_retained") is not True:
        errors.append(f"{metadata_path}: terminal_action_policy.final_observation_retained must be true")
    if policy.get("final_action_padded") is not True:
        errors.append(f"{metadata_path}: terminal_action_policy.final_action_padded must be true")
    if policy.get("exporters_must_exclude_terminal_padding_rows") is not True:
        errors.append(f"{metadata_path}: terminal_action_policy.exporters_must_exclude_terminal_padding_rows must be true")
    if policy.get("padding_count") != len(padding_indices):
        errors.append(
            f"{metadata_path}: terminal_action_policy.padding_count {policy.get('padding_count')!r} "
            f"does not match row padding count {len(padding_indices)}"
        )
    if policy.get("terminal_padding_frame_indices") != padding_indices:
        errors.append(
            f"{metadata_path}: terminal_action_policy.terminal_padding_frame_indices must be {padding_indices!r}"
        )

    padding_value = policy.get("padding_value")
    if not (
        isinstance(padding_value, list)
        and len(padding_value) == ACTION_DIM
        and all(_is_finite_number(value) and abs(float(value)) <= 1e-12 for value in padding_value)
    ):
        errors.append(f"{metadata_path}: terminal_action_policy.padding_value must be {ACTION_DIM} finite zeros")
    elif frames and frames[-1].get("measured_action") != padding_value:
        errors.append("final frame: measured_action must equal terminal_action_policy.padding_value")


def _resolve_image_path(processed_root: Path, metadata: dict[str, Any], value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = processed_root / path
    if candidate.exists():
        return candidate
    source_raw = metadata.get("source_raw_episode")
    if isinstance(source_raw, str) and source_raw:
        return Path(source_raw) / path
    return candidate


def validate_processed_episode(processed_episode_dir: str | Path) -> ValidationResult:
    """Validate a processed episode directory and return clear error messages."""

    root = Path(processed_episode_dir)
    errors: list[str] = []

    if not root.exists():
        return ValidationResult(False, [f"{root}: processed episode directory does not exist"])
    if not root.is_dir():
        return ValidationResult(False, [f"{root}: processed episode path is not a directory"])

    metadata_path = root / "metadata_processed.json"
    frames_path = root / "frames.jsonl"
    if not metadata_path.is_file():
        errors.append(f"{metadata_path}: required file is missing")
    if not frames_path.is_file():
        errors.append(f"{frames_path}: required file is missing")
    if errors:
        return ValidationResult(False, errors)

    metadata = _read_metadata(metadata_path, errors)
    frames = _read_frames(frames_path, errors)
    if metadata is None:
        return ValidationResult(False, errors)

    missing_metadata = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing_metadata:
        errors.append(f"{metadata_path}: missing required keys: {', '.join(missing_metadata)}")

    if metadata.get("model_state_dim") != MODEL_STATE_DIM:
        errors.append(f"{metadata_path}: model_state_dim must be {MODEL_STATE_DIM}")
    if metadata.get("action_dim") != ACTION_DIM:
        errors.append(f"{metadata_path}: action_dim must be {ACTION_DIM}")
    declared_model_state_dim = metadata.get("model_state_dim") if metadata.get("model_state_dim") == MODEL_STATE_DIM else MODEL_STATE_DIM
    declared_action_dim = metadata.get("action_dim") if metadata.get("action_dim") == ACTION_DIM else ACTION_DIM
    _check_layout(metadata_path, metadata, "model_state_layout", MODEL_STATE_DIM, errors)
    _check_layout(metadata_path, metadata, "measured_action_layout", ACTION_DIM, errors)
    _check_output_unit_metadata(metadata_path, metadata, errors)
    _check_gripper_placeholder_metadata(metadata_path, metadata, errors)

    frame_count = metadata.get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool):
        errors.append(f"{metadata_path}: frame_count must be an integer")
    elif frame_count != len(frames):
        errors.append(
            f"{metadata_path}: frame_count {frame_count} does not match frames.jsonl lines {len(frames)}"
        )

    previous_timestamp: float | None = None
    padding_indices: list[int] = []
    for idx, frame in enumerate(frames):
        missing_frame_keys = [key for key in REQUIRED_FRAME_KEYS if key not in frame]
        if missing_frame_keys:
            errors.append(f"frame {idx}: missing required keys: {', '.join(missing_frame_keys)}")
            continue

        if frame.get("frame_index") != idx:
            errors.append(f"frame {idx}: frame_index must equal {idx}")

        timestamp = frame.get("timestamp")
        if not _is_finite_number(timestamp):
            errors.append(f"frame {idx}: timestamp must be a finite number")
        else:
            timestamp_float = float(timestamp)
            if previous_timestamp is not None and timestamp_float <= previous_timestamp:
                errors.append(f"frame {idx}: timestamps must be strictly increasing")
            previous_timestamp = timestamp_float

        _check_numeric_vector(idx, frame, "model_state", declared_model_state_dim, errors)
        _check_numeric_vector(idx, frame, "measured_action", declared_action_dim, errors)

        padding = frame.get("action_is_terminal_padding")
        if not isinstance(padding, bool):
            errors.append(f"frame {idx}: action_is_terminal_padding must be a boolean")
        elif padding:
            padding_indices.append(idx)

        for image_key in ["external_rgb_path", "tcp_rgb_path"]:
            image_path = _resolve_image_path(root, metadata, frame.get(image_key))
            if image_path is None:
                errors.append(f"frame {idx}: {image_key} must be a non-empty string")
            elif not image_path.is_file():
                errors.append(f"frame {idx}: {image_key} does not exist: {image_path}")

    if len(padding_indices) != 1:
        errors.append(
            f"frames.jsonl: expected exactly one terminal padding frame, got {len(padding_indices)}"
        )
    elif padding_indices[0] != len(frames) - 1:
        errors.append(
            f"frames.jsonl: terminal padding frame must be final frame, got frame {padding_indices[0]}"
        )

    if frames:
        final_action = frames[-1].get("measured_action")
        if not (
            isinstance(final_action, list)
            and len(final_action) == declared_action_dim
            and all(_is_finite_number(value) and abs(float(value)) <= 1e-12 for value in final_action)
        ):
            errors.append("final frame: measured_action must be all zeros")

    _check_terminal_action_policy(metadata_path, metadata, frames, padding_indices, errors)

    return ValidationResult(not errors, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a v0 processed JSONL episode directory.")
    parser.add_argument("processed_episode_dir", help="Path to processed episode directory")
    args = parser.parse_args(argv)

    result = validate_processed_episode(args.processed_episode_dir)
    if result.ok:
        print(f"OK: processed episode is valid: {args.processed_episode_dir}")
        return 0

    print(f"INVALID: processed episode failed validation: {args.processed_episode_dir}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
