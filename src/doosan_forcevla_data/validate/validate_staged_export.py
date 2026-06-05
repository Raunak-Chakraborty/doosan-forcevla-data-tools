"""Validate staged LeRobot / ForceVLA dry-run JSONL export records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.plan_lerobot_export import VALID_PROFILES
from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_raw_episode import ValidationResult


REQUIRED_METADATA_KEYS = [
    "source_processed_episode",
    "source_export_plan",
    "profile",
    "dataset_name",
    "robot_type",
    "fps",
    "task_instruction",
    "geometry_type",
    "orientation_type",
    "exported_frame_count",
    "observation_state_dim",
    "action_dim",
    "terminal_padding_excluded",
    "notes",
]

REQUIRED_FRAME_KEYS = [
    "frame_index",
    "timestamp",
    "observation.image",
    "observation.wrist_image",
    "observation.state",
    "action",
    "task",
]


def _read_json_object(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: could not read valid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: expected a JSON object")
        return None
    return data


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: could not read JSONL: {exc}")
        return frames

    if not lines:
        errors.append(f"{path}: frames.jsonl has no records")
        return frames

    for line_number, line in enumerate(lines, start=1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path}: line {line_number} must be a JSON object")
            continue
        frames.append(data)
    return frames


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _check_vector(
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
    for idx, item in enumerate(value):
        if not _is_finite_number(item):
            errors.append(f"frame {frame_idx}: {key}[{idx}] must be a finite number")


def validate_staged_export(staged_export_dir: str | Path) -> ValidationResult:
    """Validate a staged dry-run export directory."""

    root = Path(staged_export_dir)
    errors: list[str] = []

    if not root.exists():
        return ValidationResult(False, [f"{root}: staged export directory does not exist"])
    if not root.is_dir():
        return ValidationResult(False, [f"{root}: staged export path is not a directory"])

    metadata_path = root / "metadata_staged.json"
    frames_path = root / "frames.jsonl"
    if not metadata_path.is_file():
        errors.append(f"{metadata_path}: required file is missing")
    if not frames_path.is_file():
        errors.append(f"{frames_path}: required file is missing")
    if errors:
        return ValidationResult(False, errors)

    metadata = _read_json_object(metadata_path, errors)
    frames = _read_jsonl(frames_path, errors)
    if metadata is None:
        return ValidationResult(False, errors)

    missing_metadata = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing_metadata:
        errors.append(f"{metadata_path}: missing required keys: {', '.join(missing_metadata)}")

    profile = metadata.get("profile")
    if profile not in VALID_PROFILES:
        errors.append(f"{metadata_path}: profile must be one of {', '.join(sorted(VALID_PROFILES))}")

    expected_state_dim = 13 if profile == "forcevla_13d" else MODEL_STATE_DIM
    if profile in VALID_PROFILES and metadata.get("observation_state_dim") != expected_state_dim:
        errors.append(f"{metadata_path}: observation_state_dim must be {expected_state_dim}")
    if metadata.get("action_dim") != ACTION_DIM:
        errors.append(f"{metadata_path}: action_dim must be {ACTION_DIM}")
    if metadata.get("terminal_padding_excluded") is not True:
        errors.append(f"{metadata_path}: terminal_padding_excluded must be true")

    exported_frame_count = metadata.get("exported_frame_count")
    if isinstance(exported_frame_count, bool) or not isinstance(exported_frame_count, int):
        errors.append(f"{metadata_path}: exported_frame_count must be an integer")
    elif exported_frame_count != len(frames):
        errors.append(
            f"{metadata_path}: exported_frame_count {exported_frame_count} does not match frame records {len(frames)}"
        )

    for idx, frame in enumerate(frames):
        missing_frame = [key for key in REQUIRED_FRAME_KEYS if key not in frame]
        if missing_frame:
            errors.append(f"frame {idx}: missing required keys: {', '.join(missing_frame)}")
            continue

        if "action_is_terminal_padding" in frame or "terminal_padding" in frame:
            errors.append(f"frame {idx}: staged records must not contain terminal padding fields")

        _check_vector(idx, frame, "observation.state", expected_state_dim, errors)
        _check_vector(idx, frame, "action", ACTION_DIM, errors)

        for image_key in ["observation.image", "observation.wrist_image"]:
            image_value = frame.get(image_key)
            if not isinstance(image_value, str) or not image_value:
                errors.append(f"frame {idx}: {image_key} must be a non-empty path string")
            elif not Path(image_value).is_file():
                errors.append(f"frame {idx}: {image_key} path does not exist: {image_value}")

        task = frame.get("task")
        if not isinstance(task, str) or not task.strip():
            errors.append(f"frame {idx}: task must be a non-empty string")

    return ValidationResult(not errors, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a staged LeRobot / ForceVLA dry-run export.")
    parser.add_argument("staged_export_dir", help="Path to staged export directory")
    args = parser.parse_args(argv)

    result = validate_staged_export(args.staged_export_dir)
    if result.ok:
        print(f"OK: staged export is valid: {args.staged_export_dir}")
        return 0

    print(f"INVALID: staged export failed validation: {args.staged_export_dir}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
