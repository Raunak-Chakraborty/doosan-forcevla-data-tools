"""Write a multi-episode local LeRobot-style skeleton dataset.

This module keeps the existing single-episode writer untouched and adds a
dataset-level writer that combines multiple staged episode exports into one
LeRobot-style dataset root.

It writes JSONL skeleton data only:
  - meta/info.json
  - meta/tasks.jsonl
  - meta/episodes.jsonl
  - meta/episodes_stats.jsonl
  - data/chunk-XXX/episode_YYYYYY.jsonl

It does not write parquet files, encode videos, upload to Hugging Face, or use
ROS.  Real parquet/video export can be adapted after this skeleton layer is
validated.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.write_lerobot_skeleton import (
    ACTION_DIM,
    CHUNKS_SIZE,
    DEFAULT_PROFILE,
    IMAGE_MODES,
    VALID_PROFILES,
    _features_for_profile,
    _finite_float,
    _finite_vector,
    _int_value,
    _optional_processed_metadata,
    _read_json_object,
    _read_jsonl,
    _staged_image_relative_path,
    _state_dim_for_profile,
    _task_string,
)
from doosan_forcevla_data.validate.validate_staged_export import validate_staged_export


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _check_staged_roots(staged_export_dirs: list[str | Path]) -> list[Path]:
    if not staged_export_dirs:
        raise ValueError("at least one staged export directory is required")

    roots: list[Path] = []
    seen: set[Path] = set()
    for value in staged_export_dirs:
        root = Path(value)
        resolved = root.resolve()
        if resolved in seen:
            raise ValueError(f"duplicate staged export directory: {root}")
        seen.add(resolved)
        roots.append(root)
    return roots


def _episode_chunk(episode_index: int) -> int:
    return episode_index // CHUNKS_SIZE


def _total_chunks(total_episodes: int) -> int:
    if total_episodes <= 0:
        return 0
    return _episode_chunk(total_episodes - 1) + 1


def write_lerobot_dataset_skeleton(
    staged_export_dirs: list[str | Path],
    output_dir: str | Path,
    profile: str = DEFAULT_PROFILE,
    image_mode: str = "symlink",
    overwrite: bool = False,
) -> Path:
    """Combine multiple staged episode exports into one local skeleton dataset."""

    if profile not in VALID_PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(VALID_PROFILES))}")
    if image_mode not in IMAGE_MODES:
        raise ValueError(f"image_mode must be one of: {', '.join(sorted(IMAGE_MODES))}")

    staged_roots = _check_staged_roots(staged_export_dirs)
    output_root = Path(output_dir)

    output_exists = output_root.exists() or output_root.is_symlink()
    if output_exists and (output_root.is_symlink() or not output_root.is_dir()):
        raise ValueError(f"output path exists and is not a directory: {output_root}")
    if output_exists and not overwrite:
        raise ValueError(f"output directory already exists; pass --overwrite to replace it: {output_root}")
    if output_exists and overwrite:
        shutil.rmtree(output_root)

    expected_state_dim = _state_dim_for_profile(profile)

    episode_payloads: list[dict[str, Any]] = []
    task_to_index: dict[str, int] = {}
    task_records: list[dict[str, Any]] = []
    total_frames = 0

    for episode_index, staged_root in enumerate(staged_roots):
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
                f"profile argument must match staged metadata profile for {staged_root}: "
                f"{profile!r} != {staged_profile!r}"
            )

        processed_metadata = _optional_processed_metadata(staged_metadata)
        frames = _read_jsonl(staged_root / "frames.jsonl")
        if not frames:
            raise ValueError(f"staged export contains no frames: {staged_root}")

        task = _task_string(frames[0], staged_metadata)
        if not task.strip():
            raise ValueError(f"task must be available from staged frames or metadata: {staged_root}")

        if task not in task_to_index:
            task_to_index[task] = len(task_records)
            task_records.append({"task_index": task_to_index[task], "task": task})

        task_index = task_to_index[task]

        episode_payloads.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "task": task,
                "staged_root": staged_root,
                "staged_metadata": staged_metadata,
                "processed_metadata": processed_metadata,
                "frames": frames,
            }
        )
        total_frames += len(frames)

    meta_dir = output_root / "meta"
    data_root = output_root / "data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    fps_values = {payload["staged_metadata"].get("fps") for payload in episode_payloads}
    fps = episode_payloads[0]["staged_metadata"].get("fps")
    fps_consistent = len(fps_values) == 1

    robot_types = {payload["staged_metadata"].get("robot_type") for payload in episode_payloads}
    robot_type = episode_payloads[0]["staged_metadata"].get("robot_type")
    robot_type_consistent = len(robot_types) == 1

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": len(episode_payloads),
        "total_frames": total_frames,
        "total_tasks": len(task_records),
        "fps": fps,
        "chunks_size": CHUNKS_SIZE,
        "total_chunks": _total_chunks(len(episode_payloads)),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.jsonl",
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
        "splits": {"train": f"0:{len(episode_payloads)}"},
        "features": _features_for_profile(profile),
        "export_profile": profile,
        "notes": {
            "skeleton_only": True,
            "parquet_written": False,
            "videos_encoded": False,
            "image_mode": image_mode,
            "terminal_padding_excluded": True,
            "multi_episode_skeleton": True,
            "fps_consistent": fps_consistent,
            "robot_type_consistent": robot_type_consistent,
        },
    }

    if not fps_consistent:
        raise ValueError("all staged exports must have the same fps for one LeRobot dataset")
    if not robot_type_consistent:
        raise ValueError("all staged exports must have the same robot_type for one LeRobot dataset")

    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(meta_dir / "tasks.jsonl", task_records)

    episode_records: list[dict[str, Any]] = []
    episode_stats_records: list[dict[str, Any]] = []

    global_index = 0
    for payload in episode_payloads:
        episode_index = int(payload["episode_index"])
        task_index = int(payload["task_index"])
        staged_root = payload["staged_root"]
        staged_metadata = payload["staged_metadata"]
        processed_metadata = payload["processed_metadata"]
        frames = payload["frames"]

        episode_name = f"episode_{episode_index:06d}"
        chunk_dir = output_root / "data" / f"chunk-{_episode_chunk(episode_index):03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        data_path = chunk_dir / f"{episode_name}.jsonl"

        episode_record = {
            "episode_index": episode_index,
            "task_index": task_index,
            "length": len(frames),
            "success": processed_metadata.get("success"),
            "geometry_type": staged_metadata.get("geometry_type"),
            "orientation_type": staged_metadata.get("orientation_type"),
            "collection_method": processed_metadata.get("collection_method"),
            "export_profile": profile,
            "source_staged_export": str(staged_root.resolve()),
        }
        episode_records.append(episode_record)
        episode_stats_records.append({"episode_index": episode_index, "stats": {}})

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
                    "index": global_index,
                    "task": frame_task,
                    "prompt": frame_task,
                }
                handle.write(json.dumps(output_frame, separators=(",", ":")) + "\n")
                global_index += 1

    _write_jsonl(meta_dir / "episodes.jsonl", episode_records)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats_records)

    return output_root


def _print_summary(output_root: Path) -> None:
    info = _read_json_object(output_root / "meta" / "info.json")
    print("Multi-Episode Local LeRobot Skeleton Export")
    print(f"output path: {output_root}")
    print(f"profile: {info['export_profile']}")
    print(f"total episodes: {info['total_episodes']}")
    print(f"total frames: {info['total_frames']}")
    print(f"total tasks: {info['total_tasks']}")
    print(f"image mode: {info['notes']['image_mode']}")
    print(f"data path template: {info['data_path']}")
    print("notes: skeleton only; no parquet written; no videos encoded")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a multi-episode local LeRobot-style skeleton export.")
    parser.add_argument("--staged", action="append", required=True, help="Staged export directory; repeat for each episode")
    parser.add_argument("--output", required=True, help="LeRobot-style skeleton output directory")
    parser.add_argument("--profile", choices=sorted(VALID_PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--image-mode", choices=sorted(IMAGE_MODES), default="symlink")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing skeleton output directory")
    args = parser.parse_args(argv)

    try:
        output_root = write_lerobot_dataset_skeleton(
            args.staged,
            args.output,
            profile=args.profile,
            image_mode=args.image_mode,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"FAILED: could not write multi-episode LeRobot skeleton: {args.output}")
        print(str(exc))
        return 1

    _print_summary(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
