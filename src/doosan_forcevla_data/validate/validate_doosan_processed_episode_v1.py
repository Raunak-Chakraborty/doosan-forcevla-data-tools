"""Validate the Patch-8 production two-camera processed episode."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import OBSERVATION_STATE_DIM
from doosan_forcevla_data.convert.doosan_measured_action_v1 import ACTION_DIM
from doosan_forcevla_data.convert.doosan_processed_episode_v1 import (
    CAMERA_SPECS,
    EXTERNAL_CAMERA_KEY,
    FPS,
    PROCESSED_SCHEMA_ID,
    SYNTHETIC_MODEL_SLOT,
    TCP_CAMERA_KEY,
    VIDEO_RELATIVE_PATHS,
    _probe_video,
)


@dataclass(frozen=True)
class ProcessedValidationResult:
    ok: bool
    errors: tuple[str, ...]
    frame_count: int


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}: empty line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: line {line_number} is not an object")
        rows.append(value)
    return rows


def _finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
    )


def validate_doosan_processed_episode_v1(path: str | Path) -> ProcessedValidationResult:
    root = Path(path)
    errors: list[str] = []
    metadata_path = root / "metadata_processed.json"
    frames_path = root / "frames.jsonl"

    try:
        metadata = _read_json(metadata_path)
        rows = _read_jsonl(frames_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ProcessedValidationResult(False, (str(exc),), 0)

    if metadata.get("schema_version") != PROCESSED_SCHEMA_ID:
        errors.append("metadata schema_version mismatch")
    if metadata.get("fps") != FPS:
        errors.append(f"metadata fps must be {FPS}")
    if metadata.get("state_dim") != OBSERVATION_STATE_DIM:
        errors.append(f"state_dim must be {OBSERVATION_STATE_DIM}")
    if metadata.get("action_dim") != ACTION_DIM:
        errors.append(f"action_dim must be {ACTION_DIM}")
    if metadata.get("frame_count") != len(rows):
        errors.append("frame_count does not equal frames.jsonl row count")
    if metadata.get("measured_action_count") != len(rows):
        errors.append("measured_action_count does not equal row count")
    synchronized_state_count = metadata.get("synchronized_state_count")
    if (
        isinstance(synchronized_state_count, bool)
        or not isinstance(synchronized_state_count, int)
        or synchronized_state_count < len(rows) + 1
    ):
        errors.append("synchronized_state_count must be at least action-bearing row count + 1")
    excluded_terminal = metadata.get("excluded_terminal_reference_index")
    if isinstance(excluded_terminal, bool) or not isinstance(excluded_terminal, int) or excluded_terminal < 0:
        errors.append("excluded_terminal_reference_index must be a non-negative integer")

    actionless = metadata.get("excluded_actionless_reference_indices")
    if actionless is not None:
        if not (
            isinstance(actionless, list)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in actionless)
            and actionless == sorted(set(actionless))
        ):
            errors.append("excluded_actionless_reference_indices must be sorted unique non-negative integers")
        elif excluded_terminal not in actionless:
            errors.append("excluded terminal reference must be actionless")

    dropped = metadata.get("dropped_reference_indices")
    dropped_count = metadata.get("dropped_reference_count")
    if dropped is not None or dropped_count is not None:
        if not (
            isinstance(dropped, list)
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in dropped)
            and dropped == sorted(set(dropped))
        ):
            errors.append("dropped_reference_indices must be sorted unique non-negative integers")
        elif dropped_count != len(dropped):
            errors.append("dropped_reference_count does not match dropped_reference_indices")
    if metadata.get("terminal_action_emitted") is not False:
        errors.append("terminal_action_emitted must be false")
    if metadata.get("physical_camera_count") != 2:
        errors.append("physical_camera_count must be exactly 2")
    if set(metadata.get("cameras", {})) != {TCP_CAMERA_KEY, EXTERNAL_CAMERA_KEY}:
        errors.append("metadata cameras must contain exactly tcp_camera and external_camera_2")
    if metadata.get("synthetic_model_slot") != SYNTHETIC_MODEL_SLOT:
        errors.append("synthetic right-wrist model-slot metadata mismatch")
    if not isinstance(metadata.get("task"), str) or not metadata.get("task", "").strip():
        errors.append("task must be a non-empty exact source string")

    previous_reference_timestamp: int | None = None
    previous_reference_index: int | None = None
    row_reference_indices: list[int] = []
    for index, row in enumerate(rows):
        if row.get("frame_index") != index:
            errors.append(f"row {index}: frame_index mismatch")
        reference_index = row.get("reference_index")
        if isinstance(reference_index, bool) or not isinstance(reference_index, int) or reference_index < 0:
            errors.append(f"row {index}: reference_index must be non-negative int")
        else:
            if previous_reference_index is not None and reference_index <= previous_reference_index:
                errors.append(f"row {index}: reference_index must be strictly increasing")
            previous_reference_index = reference_index
            row_reference_indices.append(reference_index)
            if row.get("action_target_reference_index") != reference_index + 1:
                errors.append(f"row {index}: action target must be original reference+1")
        timestamp = row.get("reference_timestamp_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            errors.append(f"row {index}: reference_timestamp_ns must be int")
        elif previous_reference_timestamp is not None and timestamp <= previous_reference_timestamp:
            errors.append(f"row {index}: reference timestamps are not strictly increasing")
        if isinstance(timestamp, int) and not isinstance(timestamp, bool):
            previous_reference_timestamp = timestamp
        expected_lerobot_timestamp = index / FPS
        value = row.get("lerobot_timestamp")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or abs(float(value) - expected_lerobot_timestamp) > 1e-12:
            errors.append(f"row {index}: LeRobot timestamp must equal frame_index/{FPS}")
        if not _finite_vector(row.get("observation_state_25d"), OBSERVATION_STATE_DIM):
            errors.append(f"row {index}: invalid observation_state_25d")
        if not _finite_vector(row.get("action_7d"), ACTION_DIM):
            errors.append(f"row {index}: invalid action_7d")
        cameras = row.get("cameras")
        if not isinstance(cameras, dict) or set(cameras) != {TCP_CAMERA_KEY, EXTERNAL_CAMERA_KEY}:
            errors.append(f"row {index}: camera selection set mismatch")
            continue
        tcp = cameras[TCP_CAMERA_KEY]
        ext = cameras[EXTERNAL_CAMERA_KEY]
        if not isinstance(tcp, dict) or tcp.get("source_index") != row.get("reference_index"):
            errors.append(f"row {index}: TCP source index must equal original reference index")
        for camera, entry in ((TCP_CAMERA_KEY, tcp), (EXTERNAL_CAMERA_KEY, ext)):
            if not isinstance(entry, dict):
                errors.append(f"row {index}: {camera} selection must be object")
                continue
            source_index = entry.get("source_index")
            header_timestamp = entry.get("header_timestamp_ns")
            if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
                errors.append(f"row {index}: {camera} source_index invalid")
            if isinstance(header_timestamp, bool) or not isinstance(header_timestamp, int) or header_timestamp < 0:
                errors.append(f"row {index}: {camera} header timestamp invalid")

    if metadata.get("excluded_terminal_reference_index") in row_reference_indices:
        errors.append("terminal reference is present as a training row")
    if isinstance(actionless, list) and set(actionless) & set(row_reference_indices):
        errors.append("actionless reference is present as a training row")

    for camera in (TCP_CAMERA_KEY, EXTERNAL_CAMERA_KEY):
        spec = CAMERA_SPECS[camera]
        if metadata.get("cameras", {}).get(camera) != spec:
            errors.append(f"camera spec mismatch for {camera}")
        video_path = root / VIDEO_RELATIVE_PATHS[camera]
        if not video_path.is_file():
            errors.append(f"missing native video for {camera}: {video_path}")
            continue
        try:
            _probe_video(video_path, camera=camera, expected_frames=len(rows))
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))

    return ProcessedValidationResult(not errors, tuple(errors), len(rows))


__all__ = ["ProcessedValidationResult", "validate_doosan_processed_episode_v1"]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a Patch-8 production Doosan processed episode."
    )
    parser.add_argument("processed_episode")
    args = parser.parse_args(argv)
    try:
        result = validate_doosan_processed_episode_v1(args.processed_episode)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 1
    if not result.ok:
        print("INVALID:")
        for error in result.errors:
            print(f"- {error}")
        return 1
    print(f"VALID: frame_count={result.frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
