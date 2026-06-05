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

        _check_numeric_vector(idx, frame, "model_state", MODEL_STATE_DIM, errors)
        _check_numeric_vector(idx, frame, "measured_action", ACTION_DIM, errors)

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
            and len(final_action) == ACTION_DIM
            and all(_is_finite_number(value) and abs(float(value)) <= 1e-12 for value in final_action)
        ):
            errors.append("final frame: measured_action must be all zeros")

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
