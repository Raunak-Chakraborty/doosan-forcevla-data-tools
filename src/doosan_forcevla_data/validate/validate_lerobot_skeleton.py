"""Validate a local LeRobot-style skeleton export."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.plan_lerobot_export import VALID_PROFILES
from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_raw_episode import ValidationResult


REQUIRED_INFO_KEYS = [
    "codebase_version",
    "robot_type",
    "total_episodes",
    "total_frames",
    "total_tasks",
    "fps",
    "chunks_size",
    "total_chunks",
    "data_path",
    "video_path",
    "splits",
    "features",
    "export_profile",
    "notes",
]

REQUIRED_FRAME_KEYS = [
    "observation.image",
    "observation.wrist_image",
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    "index",
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


def _read_single_jsonl(path: Path, name: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: could not read {name}: {exc}")
        return None
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) != 1:
        errors.append(f"{path}: expected exactly one {name} line, got {len(non_empty)}")
        return None
    try:
        data = json.loads(non_empty[0])
    except json.JSONDecodeError as exc:
        errors.append(f"{path}: invalid JSONL {name} line: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: {name} line must be a JSON object")
        return None
    return data


def _read_frame_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: could not read frame JSONL: {exc}")
        return frames
    if not lines:
        errors.append(f"{path}: frame JSONL has no records")
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


def _is_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _expected_state_dim(profile: Any) -> int | None:
    if profile == "forcevla_13d":
        return 13
    if profile == "doosan_full_25d":
        return MODEL_STATE_DIM
    return None


def _check_vector(frame_idx: int, frame: dict[str, Any], key: str, expected_len: int, errors: list[str]) -> None:
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


def _check_relative_image_path(
    root: Path,
    frame_idx: int,
    frame: dict[str, Any],
    key: str,
    expected_prefix: str,
    errors: list[str],
) -> None:
    value = frame.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"frame {frame_idx}: {key} must be a non-empty relative path string")
        return
    rel_path = Path(value)
    if rel_path.is_absolute():
        errors.append(f"frame {frame_idx}: {key} must be relative, got absolute path: {value}")
        return
    if ".." in rel_path.parts:
        errors.append(f"frame {frame_idx}: {key} must not contain '..': {value}")
        return
    if not value.startswith(expected_prefix):
        errors.append(f"frame {frame_idx}: {key} must be under {expected_prefix}")
        return
    image_path = root / rel_path
    if not image_path.exists():
        errors.append(f"frame {frame_idx}: {key} path does not exist under output: {value}")
    elif not image_path.is_file():
        errors.append(f"frame {frame_idx}: {key} path is not a file: {value}")


def _feature_shape(info: dict[str, Any], key: str) -> Any:
    features = info.get("features")
    if not isinstance(features, dict):
        return None
    feature = features.get(key)
    if not isinstance(feature, dict):
        return None
    return feature.get("shape")


def validate_lerobot_skeleton(skeleton_dir: str | Path) -> ValidationResult:
    """Validate a local LeRobot-style skeleton export directory."""

    root = Path(skeleton_dir)
    errors: list[str] = []

    if not root.exists():
        return ValidationResult(False, [f"{root}: skeleton output directory does not exist"])
    if not root.is_dir():
        return ValidationResult(False, [f"{root}: skeleton output path is not a directory"])

    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    stats_path = root / "meta" / "episodes_stats.jsonl"
    for required_path in [info_path, tasks_path, episodes_path, stats_path]:
        if not required_path.is_file():
            errors.append(f"{required_path}: required file is missing")
    if errors:
        return ValidationResult(False, errors)

    info = _read_json_object(info_path, errors)
    task = _read_single_jsonl(tasks_path, "task", errors)
    episode = _read_single_jsonl(episodes_path, "episode", errors)
    stats = _read_single_jsonl(stats_path, "episode stats", errors)
    if info is None or task is None or episode is None or stats is None:
        return ValidationResult(False, errors)

    missing_info = [key for key in REQUIRED_INFO_KEYS if key not in info]
    if missing_info:
        errors.append(f"{info_path}: missing required keys: {', '.join(missing_info)}")

    profile = info.get("export_profile", episode.get("export_profile"))
    if profile not in VALID_PROFILES:
        errors.append(f"{info_path}: export_profile must be one of {', '.join(sorted(VALID_PROFILES))}")
    state_dim = _expected_state_dim(profile)

    if info.get("codebase_version") != "v2.1":
        errors.append(f"{info_path}: codebase_version must be v2.1")
    if info.get("total_episodes") != 1:
        errors.append(f"{info_path}: total_episodes must be 1")
    if info.get("total_tasks") != 1:
        errors.append(f"{info_path}: total_tasks must be 1")
    if info.get("total_chunks") != 1:
        errors.append(f"{info_path}: total_chunks must be 1")

    data_path_template = info.get("data_path")
    if not isinstance(data_path_template, str):
        errors.append(f"{info_path}: data_path must be a string")
    else:
        if not data_path_template.endswith(".jsonl"):
            errors.append(f"{info_path}: data_path must end with .jsonl for the skeleton writer")
        if data_path_template.endswith(".parquet") or ".parquet" in data_path_template:
            errors.append(f"{info_path}: data_path must not point to parquet yet")

    notes = info.get("notes")
    if not isinstance(notes, dict):
        errors.append(f"{info_path}: notes must be an object")
    else:
        if notes.get("skeleton_only") is not True:
            errors.append(f"{info_path}: notes.skeleton_only must be true")
        if notes.get("parquet_written") is not False:
            errors.append(f"{info_path}: notes.parquet_written must be false")
        if notes.get("videos_encoded") is not False:
            errors.append(f"{info_path}: notes.videos_encoded must be false")
        if notes.get("terminal_padding_excluded") is not True:
            errors.append(f"{info_path}: notes.terminal_padding_excluded must be true")
        if notes.get("image_mode") not in {"symlink", "copy"}:
            errors.append(f"{info_path}: notes.image_mode must be symlink or copy")

    if state_dim is not None and _feature_shape(info, "observation.state") != [state_dim]:
        errors.append(f"{info_path}: observation.state feature shape must be [{state_dim}]")
    if _feature_shape(info, "action") != [ACTION_DIM]:
        errors.append(f"{info_path}: action feature shape must be [{ACTION_DIM}]")

    if not _is_int(task.get("task_index")):
        errors.append(f"{tasks_path}: task_index must be an integer")
    if not isinstance(task.get("task"), str) or not task.get("task", "").strip():
        errors.append(f"{tasks_path}: task must be a non-empty string")

    if not _is_int(episode.get("episode_index")):
        errors.append(f"{episodes_path}: episode_index must be an integer")
        episode_index = 0
    else:
        episode_index = int(episode["episode_index"])
    if not _is_int(episode.get("task_index")):
        errors.append(f"{episodes_path}: task_index must be an integer")
    elif _is_int(task.get("task_index")) and episode["task_index"] != task["task_index"]:
        errors.append(f"{episodes_path}: task_index must match tasks.jsonl")
    if not _is_int(episode.get("length")):
        errors.append(f"{episodes_path}: length must be an integer")
        expected_length = None
    else:
        expected_length = int(episode["length"])
    if profile in VALID_PROFILES and episode.get("export_profile") != profile:
        errors.append(f"{episodes_path}: export_profile must match info.json")

    if stats.get("episode_index") != episode_index:
        errors.append(f"{stats_path}: episode_index must match episodes.jsonl")
    if stats.get("stats") != {}:
        errors.append(f"{stats_path}: stats must be an empty object stub")

    data_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.jsonl"
    if not data_path.is_file():
        errors.append(f"{data_path}: required frame JSONL file is missing")
        return ValidationResult(False, errors)

    frames = _read_frame_jsonl(data_path, errors)
    total_frames = info.get("total_frames")
    if not _is_int(total_frames):
        errors.append(f"{info_path}: total_frames must be an integer")
    elif total_frames != len(frames):
        errors.append(f"{info_path}: total_frames {total_frames} does not match frame records {len(frames)}")
    if expected_length is not None and expected_length != len(frames):
        errors.append(f"{episodes_path}: length {expected_length} does not match frame records {len(frames)}")

    for idx, frame in enumerate(frames):
        missing_frame_keys = [key for key in REQUIRED_FRAME_KEYS if key not in frame]
        if missing_frame_keys:
            errors.append(f"frame {idx}: missing required keys: {', '.join(missing_frame_keys)}")
            continue

        if "action_is_terminal_padding" in frame or "terminal_padding" in frame:
            errors.append(f"frame {idx}: terminal padding fields must not be present")
        if state_dim is not None:
            _check_vector(idx, frame, "observation.state", state_dim, errors)
        _check_vector(idx, frame, "action", ACTION_DIM, errors)

        if not _is_finite_number(frame.get("timestamp")):
            errors.append(f"frame {idx}: timestamp must be finite")
        for int_key in ["frame_index", "episode_index", "task_index", "index"]:
            if not _is_int(frame.get(int_key)):
                errors.append(f"frame {idx}: {int_key} must be an integer")
        if frame.get("episode_index") != episode_index:
            errors.append(f"frame {idx}: episode_index must match episodes.jsonl")
        if _is_int(task.get("task_index")) and frame.get("task_index") != task["task_index"]:
            errors.append(f"frame {idx}: task_index must match tasks.jsonl")
        if not isinstance(frame.get("task"), str) or not frame.get("task", "").strip():
            errors.append(f"frame {idx}: task must be a non-empty string")

        _check_relative_image_path(
            root,
            idx,
            frame,
            "observation.image",
            "image_staging/observation.image/",
            errors,
        )
        _check_relative_image_path(
            root,
            idx,
            frame,
            "observation.wrist_image",
            "image_staging/observation.wrist_image/",
            errors,
        )

    return ValidationResult(not errors, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a local LeRobot-style skeleton export.")
    parser.add_argument("skeleton_dir", help="Path to LeRobot-style skeleton export directory")
    args = parser.parse_args(argv)

    result = validate_lerobot_skeleton(args.skeleton_dir)
    if result.ok:
        print(f"OK: LeRobot skeleton is valid: {args.skeleton_dir}")
        return 0

    print(f"INVALID: LeRobot skeleton failed validation: {args.skeleton_dir}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
