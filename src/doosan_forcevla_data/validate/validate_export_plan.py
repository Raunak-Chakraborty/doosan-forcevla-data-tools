"""Validate a dry-run LeRobot / ForceVLA export manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.plan_lerobot_export import VALID_PROFILES
from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_raw_episode import ValidationResult


REQUIRED_KEYS = [
    "source_processed_episode",
    "profile",
    "dataset_name",
    "robot_type",
    "fps",
    "task_instruction",
    "geometry_type",
    "orientation_type",
    "input_frame_count",
    "exported_frame_count",
    "excluded_terminal_padding_frame_count",
    "terminal_padding_excluded",
    "lerobot_like_keys",
    "observation_state_dim",
    "action_dim",
    "image_streams",
    "image_availability",
    "first_exported_record_preview",
    "notes",
]

REQUIRED_LEROBOT_KEYS = [
    "observation.image",
    "observation.wrist_image",
    "observation.state",
    "action",
    "task",
]


def _read_manifest(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: could not read valid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: export plan must be a JSON object")
        return None
    return data


def _nested_existing_count(manifest: dict[str, Any], key: str) -> int | None:
    image_availability = manifest.get("image_availability")
    if not isinstance(image_availability, dict):
        return None
    entry = image_availability.get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get("existing_count")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def validate_export_plan(path: str | Path) -> ValidationResult:
    """Validate a dry-run export manifest."""

    manifest_path = Path(path)
    errors: list[str] = []

    if not manifest_path.is_file():
        return ValidationResult(False, [f"{manifest_path}: export plan file does not exist"])

    manifest = _read_manifest(manifest_path, errors)
    if manifest is None:
        return ValidationResult(False, errors)

    missing = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing:
        errors.append(f"{manifest_path}: missing required keys: {', '.join(missing)}")

    profile = manifest.get("profile")
    if profile not in VALID_PROFILES:
        errors.append(f"{manifest_path}: profile must be one of {', '.join(sorted(VALID_PROFILES))}")

    exported_frame_count = manifest.get("exported_frame_count")
    input_frame_count = manifest.get("input_frame_count")
    excluded_count = manifest.get("excluded_terminal_padding_frame_count")

    if isinstance(exported_frame_count, bool) or not isinstance(exported_frame_count, int):
        errors.append(f"{manifest_path}: exported_frame_count must be an integer")
    elif exported_frame_count <= 0:
        errors.append(f"{manifest_path}: exported_frame_count must be > 0")

    if isinstance(input_frame_count, bool) or not isinstance(input_frame_count, int):
        errors.append(f"{manifest_path}: input_frame_count must be an integer")
    if isinstance(excluded_count, bool) or not isinstance(excluded_count, int):
        errors.append(f"{manifest_path}: excluded_terminal_padding_frame_count must be an integer")
    elif excluded_count != 1:
        errors.append(f"{manifest_path}: excluded_terminal_padding_frame_count must be 1")

    if manifest.get("terminal_padding_excluded") is not True:
        errors.append(f"{manifest_path}: terminal_padding_excluded must be true")
    elif isinstance(input_frame_count, int) and isinstance(exported_frame_count, int):
        if input_frame_count <= exported_frame_count:
            errors.append(
                f"{manifest_path}: input_frame_count must be greater than exported_frame_count when terminal padding is excluded"
            )

    lerobot_like_keys = manifest.get("lerobot_like_keys")
    if not isinstance(lerobot_like_keys, dict):
        errors.append(f"{manifest_path}: lerobot_like_keys must be an object")
    else:
        missing_lerobot_keys = [key for key in REQUIRED_LEROBOT_KEYS if key not in lerobot_like_keys]
        if missing_lerobot_keys:
            errors.append(
                f"{manifest_path}: lerobot_like_keys missing: {', '.join(missing_lerobot_keys)}"
            )

    expected_state_dim = 13 if profile == "forcevla_13d" else MODEL_STATE_DIM
    if profile in VALID_PROFILES and manifest.get("observation_state_dim") != expected_state_dim:
        errors.append(f"{manifest_path}: observation_state_dim must be {expected_state_dim} for {profile}")
    if manifest.get("action_dim") != ACTION_DIM:
        errors.append(f"{manifest_path}: action_dim must be {ACTION_DIM}")

    if isinstance(exported_frame_count, int):
        image_count = _nested_existing_count(manifest, "observation.image")
        wrist_count = _nested_existing_count(manifest, "observation.wrist_image")
        if image_count != exported_frame_count:
            errors.append(
                f"{manifest_path}: observation.image existing_count must equal exported_frame_count"
            )
        if wrist_count != exported_frame_count:
            errors.append(
                f"{manifest_path}: observation.wrist_image existing_count must equal exported_frame_count"
            )

    preview = manifest.get("first_exported_record_preview")
    if not isinstance(preview, dict):
        errors.append(f"{manifest_path}: first_exported_record_preview must be an object")
    else:
        if profile in VALID_PROFILES and preview.get("observation_state_length") != expected_state_dim:
            errors.append(
                f"{manifest_path}: first_exported_record_preview observation_state_length must be {expected_state_dim}"
            )
        if preview.get("action_length") != ACTION_DIM:
            errors.append(
                f"{manifest_path}: first_exported_record_preview action_length must be {ACTION_DIM}"
            )

    return ValidationResult(not errors, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a dry-run LeRobot / ForceVLA export plan.")
    parser.add_argument("export_plan", help="Path to dry-run export plan JSON")
    args = parser.parse_args(argv)

    result = validate_export_plan(args.export_plan)
    if result.ok:
        print(f"OK: export plan is valid: {args.export_plan}")
        return 0

    print(f"INVALID: export plan failed validation: {args.export_plan}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
