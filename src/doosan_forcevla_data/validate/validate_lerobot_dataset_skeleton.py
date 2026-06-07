"""Validate a multi-episode local LeRobot-style skeleton dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM


REQUIRED_INFO_KEYS = {
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
}

VALID_PROFILES = {"forcevla_13d", "doosan_full_25d"}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        records.append(data)
    return records


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunks_size = info.get("chunks_size", 1000)
    if not _is_int(chunks_size) or chunks_size <= 0:
        chunks_size = 1000
    return episode_index // chunks_size


def _format_data_path(info: dict[str, Any], episode_index: int) -> Path:
    template = info.get("data_path")
    if not isinstance(template, str) or not template:
        raise ValueError("info.json data_path must be a non-empty string")
    return Path(
        template.format(
            episode_chunk=_episode_chunk(info, episode_index),
            episode_index=episode_index,
        )
    )


def _feature_shape(info: dict[str, Any], key: str) -> Any:
    features = info.get("features")
    if not isinstance(features, dict):
        return None
    feature = features.get(key)
    if not isinstance(feature, dict):
        return None
    return feature.get("shape")


def _expected_state_dim(profile: Any) -> int | None:
    if profile == "forcevla_13d":
        return 13
    if profile == "doosan_full_25d":
        return MODEL_STATE_DIM
    return None


def _check_vector(value: Any, dim: int, name: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{name} must be a list")
        return
    if len(value) != dim:
        errors.append(f"{name} length must be {dim}, got {len(value)}")
        return
    for idx, item in enumerate(value):
        if not _is_number(item):
            errors.append(f"{name}[{idx}] must be a number")


def _check_relative_file(root: Path, value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{name} must be a non-empty relative path string")
        return
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        errors.append(f"{name} must be a safe relative path: {value}")
        return
    path = root / rel
    if not path.is_file():
        errors.append(f"{name} path does not exist under dataset root: {value}")


def validate_lerobot_dataset_skeleton(root_dir: str | Path) -> ValidationResult:
    root = Path(root_dir)
    errors: list[str] = []
    warnings: list[str] = []

    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    stats_path = root / "meta" / "episodes_stats.jsonl"

    try:
        info = _read_json_object(info_path)
        tasks = _read_jsonl(tasks_path)
        episodes = _read_jsonl(episodes_path)
        stats = _read_jsonl(stats_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return ValidationResult(False, [f"could not read skeleton metadata: {exc}"], warnings)

    missing = sorted(REQUIRED_INFO_KEYS - set(info))
    for key in missing:
        errors.append(f"{info_path}: missing required key {key}")

    profile = info.get("export_profile")
    expected_state_dim = _expected_state_dim(profile)
    state_shape = _feature_shape(info, "observation.state")
    action_shape = _feature_shape(info, "action")

    if profile not in VALID_PROFILES:
        errors.append("export_profile must be forcevla_13d or doosan_full_25d")
    if expected_state_dim is not None and state_shape != [expected_state_dim]:
        errors.append(f"observation.state shape must be [{expected_state_dim}] for {profile}")
    if action_shape != [ACTION_DIM]:
        errors.append(f"action shape must be [{ACTION_DIM}]")
    if _feature_shape(info, "prompt") != [1]:
        errors.append("prompt feature must exist with shape [1]")
    if _feature_shape(info, "task") != [1]:
        errors.append("task feature must exist with shape [1]")

    total_episodes = info.get("total_episodes")
    total_tasks = info.get("total_tasks")
    total_frames = info.get("total_frames")
    total_chunks = info.get("total_chunks")

    if not _is_int(total_episodes) or total_episodes <= 0:
        errors.append("total_episodes must be a positive integer")
    elif total_episodes != len(episodes):
        errors.append(f"total_episodes {total_episodes} does not match episodes.jsonl records {len(episodes)}")

    if not _is_int(total_tasks) or total_tasks <= 0:
        errors.append("total_tasks must be a positive integer")
    elif total_tasks != len(tasks):
        errors.append(f"total_tasks {total_tasks} does not match tasks.jsonl records {len(tasks)}")

    if not _is_int(total_frames) or total_frames <= 0:
        errors.append("total_frames must be a positive integer")

    if not _is_int(total_chunks) or total_chunks <= 0:
        errors.append("total_chunks must be a positive integer")

    if info.get("codebase_version") != "v2.1":
        errors.append("codebase_version must be v2.1")

    notes = info.get("notes")
    if not isinstance(notes, dict):
        errors.append("notes must be an object")
    else:
        if notes.get("skeleton_only") is not True:
            errors.append("notes.skeleton_only must be true")
        if notes.get("multi_episode_skeleton") is not True:
            errors.append("notes.multi_episode_skeleton must be true")
        if notes.get("parquet_written") is not False:
            errors.append("notes.parquet_written must be false")
        if notes.get("videos_encoded") is not False:
            errors.append("notes.videos_encoded must be false")

    task_by_index: dict[int, str] = {}
    for idx, task in enumerate(tasks):
        task_index = task.get("task_index")
        task_text = task.get("task")
        if not _is_int(task_index):
            errors.append(f"{tasks_path}: task record {idx} task_index must be integer")
            continue
        if task_index in task_by_index:
            errors.append(f"{tasks_path}: duplicate task_index {task_index}")
        if not isinstance(task_text, str) or not task_text.strip():
            errors.append(f"{tasks_path}: task record {idx} task must be non-empty string")
        else:
            task_by_index[int(task_index)] = task_text

    stats_by_episode: dict[int, dict[str, Any]] = {}
    for idx, stat in enumerate(stats):
        episode_index = stat.get("episode_index")
        if not _is_int(episode_index):
            errors.append(f"{stats_path}: stats record {idx} episode_index must be integer")
            continue
        stats_by_episode[int(episode_index)] = stat

    expected_episode_indices = list(range(len(episodes)))
    actual_episode_indices: list[int] = []
    total_frame_count_seen = 0
    global_indices: list[int] = []

    for episode_pos, episode in enumerate(episodes):
        episode_index = episode.get("episode_index")
        task_index = episode.get("task_index")
        length = episode.get("length")

        if not _is_int(episode_index):
            errors.append(f"{episodes_path}: episode record {episode_pos} episode_index must be integer")
            continue
        episode_index = int(episode_index)
        actual_episode_indices.append(episode_index)

        if episode_index != episode_pos:
            errors.append(f"{episodes_path}: episode_index must be sequential from 0; got {episode_index} at record {episode_pos}")

        if not _is_int(task_index) or int(task_index) not in task_by_index:
            errors.append(f"{episodes_path}: episode {episode_index} task_index must reference tasks.jsonl")
            task_text = None
        else:
            task_text = task_by_index[int(task_index)]

        if not _is_int(length) or int(length) <= 0:
            errors.append(f"{episodes_path}: episode {episode_index} length must be positive integer")
            continue

        if episode_index not in stats_by_episode:
            errors.append(f"{stats_path}: missing stats record for episode {episode_index}")

        try:
            frames_path = root / _format_data_path(info, episode_index)
            frames = _read_jsonl(frames_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"could not read frames for episode {episode_index}: {exc}")
            continue

        if len(frames) != int(length):
            errors.append(f"episode {episode_index}: length {length} does not match frame records {len(frames)}")

        total_frame_count_seen += len(frames)

        for frame_pos, frame in enumerate(frames):
            _check_vector(frame.get("observation.state"), expected_state_dim or 0, f"episode {episode_index} frame {frame_pos} observation.state", errors)
            _check_vector(frame.get("action"), ACTION_DIM, f"episode {episode_index} frame {frame_pos} action", errors)
            _check_relative_file(root, frame.get("observation.image"), f"episode {episode_index} frame {frame_pos} observation.image", errors)
            _check_relative_file(root, frame.get("observation.wrist_image"), f"episode {episode_index} frame {frame_pos} observation.wrist_image", errors)

            for int_key in ["frame_index", "episode_index", "task_index", "index"]:
                if not _is_int(frame.get(int_key)):
                    errors.append(f"episode {episode_index} frame {frame_pos}: {int_key} must be integer")

            if frame.get("episode_index") != episode_index:
                errors.append(f"episode {episode_index} frame {frame_pos}: episode_index must match episodes.jsonl")
            if frame.get("task_index") != task_index:
                errors.append(f"episode {episode_index} frame {frame_pos}: task_index must match episodes.jsonl")
            if task_text is not None:
                if frame.get("task") != task_text:
                    errors.append(f"episode {episode_index} frame {frame_pos}: task must match tasks.jsonl")
                if frame.get("prompt") != task_text:
                    errors.append(f"episode {episode_index} frame {frame_pos}: prompt must match tasks.jsonl")

            if _is_int(frame.get("index")):
                global_indices.append(int(frame["index"]))

    if actual_episode_indices != expected_episode_indices:
        errors.append(f"episode indices must be sequential: expected {expected_episode_indices}, got {actual_episode_indices}")

    if _is_int(total_frames) and total_frame_count_seen != int(total_frames):
        errors.append(f"total_frames {total_frames} does not match observed frame count {total_frame_count_seen}")

    if global_indices and global_indices != list(range(len(global_indices))):
        errors.append("frame index field 'index' must be globally sequential from 0")

    if _is_int(total_chunks) and _is_int(total_episodes):
        expected_chunks = (int(total_episodes) - 1) // int(info.get("chunks_size", 1000)) + 1
        if int(total_chunks) != expected_chunks:
            errors.append(f"total_chunks must be {expected_chunks}, got {total_chunks}")

    return ValidationResult(not errors, errors, warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a multi-episode LeRobot skeleton dataset.")
    parser.add_argument("root")
    args = parser.parse_args(argv)

    result = validate_lerobot_dataset_skeleton(args.root)
    if result.ok:
        print(f"OK: multi-episode LeRobot skeleton is valid: {args.root}")
        return 0

    print(f"FAILED: multi-episode LeRobot skeleton is invalid: {args.root}")
    for error in result.errors:
        print(f"ERROR: {error}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
