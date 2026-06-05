"""Validate the structure of a v0 raw episode."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from doosan_forcevla_data.schema.raw_schema import RAW_EPISODE_PATHS, REQUIRED_METADATA_KEYS


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def _read_csv_rows(path: Path, errors: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:
        errors.append(f"{path}: could not read CSV: {exc}")
        return [], []

    if not fieldnames:
        errors.append(f"{path}: CSV is missing a header")
    if not rows:
        errors.append(f"{path}: CSV has no data rows")
    return fieldnames, rows


def _check_required_fields(
    path: Path, fieldnames: list[str], required_fields: Iterable[str], errors: list[str]
) -> None:
    missing = [field for field in required_fields if field not in fieldnames]
    if missing:
        errors.append(f"{path}: missing required fields: {', '.join(missing)}")


def _parse_finite_float(path: Path, row_idx: int, field: str, value: str, errors: list[str]) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{path}: row {row_idx} field {field} is not a float: {value!r}")
        return
    if not math.isfinite(number):
        errors.append(f"{path}: row {row_idx} field {field} is not finite")


def _check_numeric_fields(
    path: Path, rows: list[dict[str, str]], fields: Iterable[str], errors: list[str]
) -> None:
    for row_idx, row in enumerate(rows, start=2):
        for field in fields:
            if field in row:
                _parse_finite_float(path, row_idx, field, row[field], errors)


def _check_increasing_timestamps(path: Path, rows: list[dict[str, str]], errors: list[str]) -> None:
    previous: float | None = None
    for row_idx, row in enumerate(rows, start=2):
        if "timestamp" not in row:
            errors.append(f"{path}: missing timestamp field")
            return
        try:
            timestamp = float(row["timestamp"])
        except (TypeError, ValueError):
            errors.append(f"{path}: row {row_idx} timestamp is not a float: {row.get('timestamp')!r}")
            return
        if not math.isfinite(timestamp):
            errors.append(f"{path}: row {row_idx} timestamp is not finite")
            return
        if previous is not None and timestamp <= previous:
            errors.append(f"{path}: timestamps must be strictly increasing at row {row_idx}")
            return
        previous = timestamp


def _check_metadata(path: Path, errors: list[str]) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: could not read metadata JSON: {exc}")
        return

    if not isinstance(data, dict):
        errors.append(f"{path}: metadata must be a JSON object")
        return

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in data]
    if missing:
        errors.append(f"{path}: metadata missing required keys: {', '.join(missing)}")
    if "fps" in data:
        try:
            fps = float(data["fps"])
        except (TypeError, ValueError):
            errors.append(f"{path}: fps must be numeric")
        else:
            if not math.isfinite(fps) or fps <= 0.0:
                errors.append(f"{path}: fps must be positive and finite")
    if "success" in data and not isinstance(data["success"], bool):
        errors.append(f"{path}: success must be a boolean")


def validate_raw_episode(episode_dir: str | Path) -> ValidationResult:
    """Validate a raw episode directory and return clear error messages."""

    root = Path(episode_dir)
    errors: list[str] = []

    if not root.exists():
        return ValidationResult(False, [f"{root}: episode directory does not exist"])
    if not root.is_dir():
        return ValidationResult(False, [f"{root}: episode path is not a directory"])

    required_files = [
        RAW_EPISODE_PATHS.metadata,
        RAW_EPISODE_PATHS.tcp_pose,
        RAW_EPISODE_PATHS.joint_states,
        RAW_EPISODE_PATHS.wrench,
        RAW_EPISODE_PATHS.events,
    ]
    for relative_path in required_files:
        path = root / relative_path
        if not path.is_file():
            errors.append(f"{path}: required file is missing")

    for relative_path in [RAW_EPISODE_PATHS.external_rgb, RAW_EPISODE_PATHS.tcp_rgb]:
        path = root / relative_path
        if not path.is_dir():
            errors.append(f"{path}: required image folder is missing")

    if errors:
        return ValidationResult(False, errors)

    _check_metadata(root / RAW_EPISODE_PATHS.metadata, errors)

    csv_specs = {
        RAW_EPISODE_PATHS.tcp_pose: ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
        RAW_EPISODE_PATHS.joint_states: ["timestamp"]
        + [f"joint_pos_{idx}" for idx in range(6)]
        + [f"joint_vel_{idx}" for idx in range(6)],
        RAW_EPISODE_PATHS.wrench: ["timestamp", "fx", "fy", "fz", "tx", "ty", "tz"],
        RAW_EPISODE_PATHS.events: ["timestamp", "event"],
    }

    commanded_twist_path = root / RAW_EPISODE_PATHS.commanded_twist
    if commanded_twist_path.is_file():
        csv_specs[RAW_EPISODE_PATHS.commanded_twist] = [
            "timestamp",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
            "gripper_velocity",
        ]

    for relative_path, required_fields in csv_specs.items():
        path = root / relative_path
        fieldnames, rows = _read_csv_rows(path, errors)
        _check_required_fields(path, fieldnames, required_fields, errors)
        _check_increasing_timestamps(path, rows, errors)
        numeric_fields = [field for field in required_fields if field != "event"]
        _check_numeric_fields(path, rows, numeric_fields, errors)

    return ValidationResult(not errors, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a v0 raw episode directory.")
    parser.add_argument("episode_dir", help="Path to raw episode directory")
    args = parser.parse_args(argv)

    result = validate_raw_episode(args.episode_dir)
    if result.ok:
        print(f"OK: raw episode is valid: {args.episode_dir}")
        return 0

    print(f"INVALID: raw episode failed validation: {args.episode_dir}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
