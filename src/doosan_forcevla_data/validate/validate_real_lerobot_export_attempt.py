"""Validate a dependency-optional real LeRobot export attempt report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_raw_episode import ValidationResult


VALID_PROFILES = {"forcevla_13d", "doosan_full_25d"}
REQUIRED_KEYS = [
    "source_skeleton",
    "output_dir",
    "mode",
    "profile",
    "total_frames",
    "state_dim",
    "action_dim",
    "dependencies",
    "parquet_ready",
    "video_ready",
    "lerobot_api_available",
    "parquet_written",
    "videos_written",
    "metadata_written",
    "skipped_reasons",
    "next_recommended_action",
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


def _is_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _expected_state_dim(profile: Any) -> int | None:
    if profile == "forcevla_13d":
        return 13
    if profile == "doosan_full_25d":
        return MODEL_STATE_DIM
    return None


def validate_real_lerobot_export_attempt(export_attempt_dir: str | Path) -> ValidationResult:
    """Validate a local real-export attempt directory."""

    root = Path(export_attempt_dir)
    errors: list[str] = []

    if not root.exists():
        return ValidationResult(False, [f"{root}: export attempt directory does not exist"])
    if not root.is_dir():
        return ValidationResult(False, [f"{root}: export attempt path is not a directory"])

    report_path = root / "export_attempt_report.json"
    if not report_path.is_file():
        return ValidationResult(False, [f"{report_path}: required report file is missing"])

    report = _read_json_object(report_path, errors)
    if report is None:
        return ValidationResult(False, errors)

    missing = [key for key in REQUIRED_KEYS if key not in report]
    if missing:
        errors.append(f"{report_path}: missing required keys: {', '.join(missing)}")

    profile = report.get("profile")
    if profile not in VALID_PROFILES:
        errors.append(f"{report_path}: profile must be one of {', '.join(sorted(VALID_PROFILES))}")
    expected_state_dim = _expected_state_dim(profile)
    if expected_state_dim is not None and report.get("state_dim") != expected_state_dim:
        errors.append(f"{report_path}: state_dim must be {expected_state_dim} for {profile}")
    if report.get("action_dim") != ACTION_DIM:
        errors.append(f"{report_path}: action_dim must be {ACTION_DIM}")

    total_frames = report.get("total_frames")
    if not _is_int(total_frames) or total_frames <= 0:
        errors.append(f"{report_path}: total_frames must be a positive integer")

    for key in [
        "metadata_written",
        "parquet_written",
        "videos_written",
        "parquet_ready",
        "video_ready",
        "lerobot_api_available",
    ]:
        if not isinstance(report.get(key), bool):
            errors.append(f"{report_path}: {key} must be a boolean")

    skipped_reasons = report.get("skipped_reasons")
    if not isinstance(skipped_reasons, list):
        errors.append(f"{report_path}: skipped_reasons must be a list")
    elif not all(isinstance(reason, str) for reason in skipped_reasons):
        errors.append(f"{report_path}: skipped_reasons entries must be strings")

    if report.get("parquet_written") is True:
        parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
        if not parquet_path.is_file():
            errors.append(f"{parquet_path}: parquet_written is true but parquet file is missing")

    if report.get("videos_written") is True:
        for relative_path in [
            "videos/observation.image/episode_000000.mp4",
            "videos/observation.wrist_image/episode_000000.mp4",
        ]:
            video_path = root / relative_path
            if not video_path.is_file():
                errors.append(f"{video_path}: videos_written is true but video file is missing")

    return ValidationResult(not errors, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a local real LeRobot export attempt.")
    parser.add_argument("export_attempt_dir", help="Path to local real-export attempt directory")
    args = parser.parse_args(argv)

    result = validate_real_lerobot_export_attempt(args.export_attempt_dir)
    if result.ok:
        print(f"OK: real LeRobot export attempt is valid: {args.export_attempt_dir}")
        return 0

    print(f"INVALID: real LeRobot export attempt failed validation: {args.export_attempt_dir}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
