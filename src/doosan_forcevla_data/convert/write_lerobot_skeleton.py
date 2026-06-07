"""Write a local LeRobot-style skeleton export without parquet or videos."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.plan_lerobot_export import VALID_PROFILES
from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_staged_export import validate_staged_export


IMAGE_MODES = {"symlink", "copy"}
DEFAULT_PROFILE = "forcevla_13d"
CHUNKS_SIZE = 1000


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        records.append(record)
    return records


def _optional_processed_metadata(staged_metadata: dict[str, Any]) -> dict[str, Any]:
    source_processed = staged_metadata.get("source_processed_episode")
    if not isinstance(source_processed, str) or not source_processed:
        return {}
    metadata_path = Path(source_processed) / "metadata_processed.json"
    if not metadata_path.is_file():
        return {}
    return _read_json_object(metadata_path)


def _state_dim_for_profile(profile: str) -> int:
    if profile == "forcevla_13d":
        return 13
    if profile == "doosan_full_25d":
        return MODEL_STATE_DIM
    raise ValueError(f"unsupported export profile: {profile}")


def _check_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _finite_vector(values: Any, expected_len: int, name: str) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    if len(values) != expected_len:
        raise ValueError(f"{name} length must be {expected_len}, got {len(values)}")
    result: list[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}[{idx}] must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{idx}] must be finite")
        result.append(number)
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _int_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _task_string(frame: dict[str, Any], staged_metadata: dict[str, Any]) -> str:
    task = frame.get("task", staged_metadata.get("task_instruction"))
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    return task


def _remove_existing_staged_target(target: Path) -> None:
    if not target.exists() and not target.is_symlink():
        return
    if target.is_dir() and not target.is_symlink():
        raise ValueError(f"staged image target is a directory and cannot be replaced: {target}")
    target.unlink()


def _stage_image(source_value: Any, target: Path, image_mode: str) -> None:
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("image path must be a non-empty string")
    source = Path(source_value).resolve()
    if not source.is_file():
        raise ValueError(f"source image does not exist: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_staged_target(target)

    try:
        if image_mode == "symlink":
            target.symlink_to(source)
        elif image_mode == "copy":
            shutil.copy2(source, target)
        else:
            raise ValueError(f"unsupported image_mode: {image_mode}")
    except OSError as exc:
        raise ValueError(
            f"could not stage image with mode {image_mode}: {source} -> {target}: {exc}"
        ) from exc


def _staged_image_relative_path(
    output_root: Path,
    image_key: str,
    episode_name: str,
    frame_ordinal: int,
    source_value: Any,
    image_mode: str,
) -> str:
    if not isinstance(source_value, str) or not source_value:
        raise ValueError(f"{image_key} must be a non-empty string")
    suffix = Path(source_value).suffix
    target = output_root / "image_staging" / image_key / episode_name / f"{frame_ordinal:06d}{suffix}"
    _stage_image(source_value, target, image_mode)
    return target.relative_to(output_root).as_posix()


def _camera_feature(storage: str = "video") -> dict[str, Any]:
    """Return a LeRobot v2.1-compatible camera feature entry.

    The real export target stores RGB streams as MP4 videos.  The skeleton still
    keeps staged image files for inspection/encoding, but the final dataset
    metadata should use a real LeRobot camera dtype, not a private
    ``image_reference`` placeholder.
    """

    if storage not in {"image", "video"}:
        raise ValueError(f"unsupported camera storage: {storage}")
    return {
        "dtype": storage,
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
    }


def _features_for_profile(profile: str) -> dict[str, dict[str, Any]]:
    state_dim = _state_dim_for_profile(profile)
    return {
        "observation.image": _camera_feature("video"),
        "observation.wrist_image": _camera_feature("video"),
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action": {"dtype": "float32", "shape": [ACTION_DIM]},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task": {"dtype": "string", "shape": [1]},
        "prompt": {"dtype": "string", "shape": [1]},
    }


def _write_jsonl_line(path: Path, record: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_lerobot_skeleton(
    staged_export_dir: str | Path,
    output_dir: str | Path,
    episode_index: int = 0,
    task_index: int = 0,
    profile: str = DEFAULT_PROFILE,
    image_mode: str = "symlink",
    overwrite: bool = False,
) -> Path:
    """Write a local inspectable LeRobot-style skeleton export."""

    if profile not in VALID_PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(VALID_PROFILES))}")
    if image_mode not in IMAGE_MODES:
        raise ValueError(f"image_mode must be one of: {', '.join(sorted(IMAGE_MODES))}")
    _check_non_negative_int(episode_index, "episode_index")
    _check_non_negative_int(task_index, "task_index")

    staged_root = Path(staged_export_dir)
    output_root = Path(output_dir)
    output_exists = output_root.exists() or output_root.is_symlink()
    if output_exists and (output_root.is_symlink() or not output_root.is_dir()):
        raise ValueError(f"output path exists and is not a directory: {output_root}")
    if output_exists and not overwrite:
        raise ValueError(f"output directory already exists; pass --overwrite to replace it: {output_root}")

    validation = validate_staged_export(staged_root)
    if not validation.ok:
        message = "staged export validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in validation.errors
        )
        raise ValueError(message)

    staged_metadata = _read_json_object(staged_root / "metadata_staged.json")
    staged_profile = staged_metadata.get("profile")
    if staged_profile != profile:
        raise ValueError(
            f"profile argument must match staged metadata profile: {profile!r} != {staged_profile!r}"
        )

    processed_metadata = _optional_processed_metadata(staged_metadata)
    frames = _read_jsonl(staged_root / "frames.jsonl")
    expected_state_dim = _state_dim_for_profile(profile)
    exported_frame_count = len(frames)
    episode_name = f"episode_{episode_index:06d}"

    if output_exists and overwrite:
        shutil.rmtree(output_root)

    meta_dir = output_root / "meta"
    data_dir = output_root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    task = _task_string(frames[0], staged_metadata) if frames else str(staged_metadata.get("task_instruction", ""))
    if not task.strip():
        raise ValueError("task must be available from staged frames or metadata")

    info = {
        "codebase_version": "v2.1",
        "robot_type": staged_metadata.get("robot_type"),
        "total_episodes": 1,
        "total_frames": exported_frame_count,
        "total_tasks": 1,
        "fps": staged_metadata.get("fps"),
        "chunks_size": CHUNKS_SIZE,
        "total_chunks": 1,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.jsonl",
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
        "splits": {"train": "0:1"},
        "features": _features_for_profile(profile),
        "export_profile": profile,
        "notes": {
            "skeleton_only": True,
            "parquet_written": False,
            "videos_encoded": False,
            "image_mode": image_mode,
            "terminal_padding_excluded": True,
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    _write_jsonl_line(meta_dir / "tasks.jsonl", {"task_index": task_index, "task": task})

    episode_record = {
        "episode_index": episode_index,
        "task_index": task_index,
        "length": exported_frame_count,
        "success": processed_metadata.get("success"),
        "geometry_type": staged_metadata.get("geometry_type"),
        "orientation_type": staged_metadata.get("orientation_type"),
        "collection_method": processed_metadata.get("collection_method"),
        "export_profile": profile,
        "source_staged_export": str(staged_root.resolve()),
    }
    _write_jsonl_line(meta_dir / "episodes.jsonl", episode_record)
    _write_jsonl_line(meta_dir / "episodes_stats.jsonl", {"episode_index": episode_index, "stats": {}})

    data_path = data_dir / f"{episode_name}.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for ordinal, frame in enumerate(frames):
            state = _finite_vector(frame.get("observation.state"), expected_state_dim, "observation.state")
            action = _finite_vector(frame.get("action"), ACTION_DIM, "action")
            frame_index = _int_value(frame.get("frame_index"), "frame_index")
            timestamp = _finite_float(frame.get("timestamp"), "timestamp")
            frame_task = _task_string(frame, staged_metadata)

            image_rel = _staged_image_relative_path(
                output_root,
                "observation.image",
                episode_name,
                ordinal,
                frame.get("observation.image"),
                image_mode,
            )
            wrist_image_rel = _staged_image_relative_path(
                output_root,
                "observation.wrist_image",
                episode_name,
                ordinal,
                frame.get("observation.wrist_image"),
                image_mode,
            )

            output_frame = {
                "observation.image": image_rel,
                "observation.wrist_image": wrist_image_rel,
                "observation.state": state,
                "action": action,
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": episode_index,
                "task_index": task_index,
                "index": frame_index,
                "task": frame_task,
                "prompt": frame_task,
            }
            handle.write(json.dumps(output_frame, separators=(",", ":")) + "\n")

    return output_root


def _read_info(output_root: Path) -> dict[str, Any]:
    return _read_json_object(output_root / "meta" / "info.json")


def _print_summary(output_root: Path) -> None:
    info = _read_info(output_root)
    print("Local LeRobot Skeleton Export")
    print(f"output path: {output_root}")
    print(f"profile: {info['export_profile']}")
    print(f"total frames: {info['total_frames']}")
    print(f"image mode: {info['notes']['image_mode']}")
    print(f"data path template: {info['data_path']}")
    print("notes: skeleton only; no parquet written; no videos encoded")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a local LeRobot-style skeleton export.")
    parser.add_argument("--staged", required=True, help="Staged export directory")
    parser.add_argument("--output", required=True, help="LeRobot-style skeleton output directory")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--profile", choices=sorted(VALID_PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--image-mode", choices=sorted(IMAGE_MODES), default="symlink")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing skeleton output directory")
    args = parser.parse_args(argv)

    try:
        output_root = write_lerobot_skeleton(
            args.staged,
            args.output,
            episode_index=args.episode_index,
            task_index=args.task_index,
            profile=args.profile,
            image_mode=args.image_mode,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"FAILED: could not write LeRobot skeleton: {args.output}")
        print(str(exc))
        return 1

    _print_summary(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
