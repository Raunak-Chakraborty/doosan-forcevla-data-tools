"""Write a dependency-optional real export from a multi-episode skeleton dataset.

Input:
  multi-episode LeRobot-style skeleton dataset
  - meta/info.json
  - meta/tasks.jsonl
  - meta/episodes.jsonl
  - meta/episodes_stats.jsonl
  - data/chunk-XXX/episode_YYYYYY.jsonl

Output, when dependencies are available:
  - parquet files per episode
  - MP4 videos per camera stream and episode
  - metadata copied/adapted for real export
  - export_attempt_report.json

This does not upload to Hugging Face and does not use ROS.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from doosan_forcevla_data.inspect.check_export_dependencies import check_export_dependencies
from doosan_forcevla_data.validate.validate_lerobot_dataset_skeleton import validate_lerobot_dataset_skeleton
from doosan_forcevla_data.convert.write_real_lerobot_export import (
    REPORT_NAME,
    _encode_video_with_fallback,
    _frame_image_paths,
    _remove_file_if_present,
    _summarize_video_backends,
    _verify_video,
)


MODES = {"dry-run", "write-if-available"}
IMAGE_KEYS = ["observation.image", "observation.wrist_image"]


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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunks_size = info.get("chunks_size", 1000)
    if not isinstance(chunks_size, int) or isinstance(chunks_size, bool) or chunks_size <= 0:
        chunks_size = 1000
    return episode_index // chunks_size


def _format_data_path(info: dict[str, Any], episode_index: int) -> Path:
    template = info.get("data_path")
    if not isinstance(template, str) or not template:
        raise ValueError("meta/info.json must contain a non-empty data_path template")
    return Path(
        template.format(
            episode_chunk=_episode_chunk(info, episode_index),
            episode_index=episode_index,
        )
    )


def _load_dataset_skeleton(
    skeleton_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    info = _read_json_object(skeleton_root / "meta" / "info.json")
    tasks = _read_jsonl(skeleton_root / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(skeleton_root / "meta" / "episodes.jsonl")
    stats = _read_jsonl(skeleton_root / "meta" / "episodes_stats.jsonl")

    frames_by_episode: dict[int, list[dict[str, Any]]] = {}
    for episode in episodes:
        episode_index = episode.get("episode_index")
        if not isinstance(episode_index, int) or isinstance(episode_index, bool):
            raise ValueError("episodes.jsonl episode_index must be an integer")
        frames_path = skeleton_root / _format_data_path(info, int(episode_index))
        frames_by_episode[int(episode_index)] = _read_jsonl(frames_path)

    return info, tasks, episodes, stats, frames_by_episode


def _feature_dim(info: dict[str, Any], key: str) -> int | None:
    features = info.get("features")
    if not isinstance(features, dict):
        return None
    feature = features.get(key)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if isinstance(shape, list) and len(shape) == 1 and isinstance(shape[0], int):
        return int(shape[0])
    return None


def _dependency_summary(dependencies: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "available": bool(entry.get("available")),
            "version": entry.get("version"),
            "detail": str(entry.get("detail", "")),
        }
        for key, entry in dependencies.items()
    }


def _bool_dependency(dependencies: dict[str, dict[str, object]], key: str) -> bool:
    entry = dependencies.get(key)
    return bool(isinstance(entry, dict) and entry.get("available") is True)


def _parquet_path(output_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = _episode_chunk(info, episode_index)
    return output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _write_parquet_episode(
    output_root: Path,
    info: dict[str, Any],
    episode_index: int,
    frames: list[dict[str, Any]],
) -> Path:
    pa = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")

    parquet_path = _parquet_path(output_root, info, episode_index)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_file_if_present(parquet_path)

    table = pa.table(
        {
            "observation.state": pa.array(
                [[float(value) for value in frame["observation.state"]] for frame in frames],
                type=pa.list_(pa.float32()),
            ),
            "action": pa.array(
                [[float(value) for value in frame["action"]] for frame in frames],
                type=pa.list_(pa.float32()),
            ),
            "timestamp": pa.array([float(frame["timestamp"]) for frame in frames], type=pa.float64()),
            "frame_index": pa.array([int(frame["frame_index"]) for frame in frames], type=pa.int64()),
            "episode_index": pa.array([int(frame["episode_index"]) for frame in frames], type=pa.int64()),
            "task_index": pa.array([int(frame["task_index"]) for frame in frames], type=pa.int64()),
            "index": pa.array([int(frame["index"]) for frame in frames], type=pa.int64()),
            "task": pa.array([str(frame["task"]) for frame in frames], type=pa.string()),
            "prompt": pa.array([str(frame["prompt"]) for frame in frames], type=pa.string()),
        }
    )
    pq.write_table(table, parquet_path)
    return parquet_path


def _write_videos_episode(
    skeleton_root: Path,
    output_root: Path,
    episode_index: int,
    frames: list[dict[str, Any]],
    fps: float,
    dependencies: dict[str, dict[str, Any]],
) -> tuple[list[Path], dict[str, str], list[str]]:
    if not any(_bool_dependency(dependencies, key) for key in ["imageio_ffmpeg", "imageio", "cv2"]):
        raise ValueError("video encoding requires imageio_ffmpeg, imageio, or cv2")

    written: list[Path] = []
    video_backends: dict[str, str] = {}
    backend_errors: list[str] = []
    for key in IMAGE_KEYS:
        image_paths = _frame_image_paths(skeleton_root, frames, key)
        output_path = output_root / "videos" / key / f"episode_{episode_index:06d}.mp4"
        backend_name, errors = _encode_video_with_fallback(image_paths, output_path, fps, dependencies)
        _verify_video(output_path)
        written.append(output_path)
        video_backends[key] = backend_name
        backend_errors.extend(f"{key}: {error}" for error in errors)
    return written, video_backends, backend_errors


def _adapt_info(
    info: dict[str, Any],
    source_skeleton: Path,
    parquet_written: bool,
    videos_written: bool,
) -> dict[str, Any]:
    adapted = json.loads(json.dumps(info))
    notes = adapted.setdefault("notes", {})
    if not isinstance(notes, dict):
        notes = {}
        adapted["notes"] = notes

    notes["skeleton_only"] = False
    notes["real_export_scaffold"] = True
    notes["multi_episode_real_export"] = True
    notes["source_skeleton"] = str(source_skeleton.resolve())
    notes["parquet_written"] = parquet_written
    notes["videos_encoded"] = videos_written

    if parquet_written:
        adapted["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    else:
        adapted["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.jsonl.placeholder"

    if videos_written:
        features = adapted.get("features")
        if isinstance(features, dict):
            for key in IMAGE_KEYS:
                feature = features.get(key)
                if isinstance(feature, dict):
                    feature["dtype"] = "video"

    return adapted


def _write_metadata(
    output_root: Path,
    info: dict[str, Any],
    tasks: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    stats: list[dict[str, Any]],
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(meta_dir / "tasks.jsonl", tasks)
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", stats)


def _base_report(
    skeleton_root: Path,
    output_root: Path,
    mode: str,
    info: dict[str, Any],
    dependencies: dict[str, dict[str, object]],
    parquet_ready: bool,
    video_ready: bool,
) -> dict[str, Any]:
    return {
        "source_skeleton": str(skeleton_root.resolve()),
        "output_dir": str(output_root.resolve()),
        "mode": mode,
        "profile": info.get("export_profile"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
        "state_dim": _feature_dim(info, "observation.state"),
        "action_dim": _feature_dim(info, "action"),
        "dependencies": _dependency_summary(dependencies),
        "parquet_ready": parquet_ready,
        "video_ready": video_ready,
        "lerobot_api_available": _bool_dependency(dependencies, "lerobot"),
        "parquet_written": False,
        "videos_written": False,
        "video_backend": None,
        "video_backends": {},
        "video_backend_errors": [],
        "metadata_written": False,
        "per_episode": [],
        "skipped_reasons": [],
        "next_recommended_action": "Run this command on the lab ForceVLA environment before treating readiness as final.",
    }


def write_report(report: dict[str, Any], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def write_real_lerobot_dataset_export(
    skeleton_dir: str | Path,
    output_dir: str | Path,
    mode: str = "dry-run",
) -> Path:
    """Attempt a dependency-optional real export for a multi-episode skeleton dataset."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(MODES))}")

    skeleton_root = Path(skeleton_dir)
    output_root = Path(output_dir)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output_root}")

    validation = validate_lerobot_dataset_skeleton(skeleton_root)
    if not validation.ok:
        message = "multi-episode skeleton validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in validation.errors
        )
        raise ValueError(message)

    info, tasks, episodes, stats, frames_by_episode = _load_dataset_skeleton(skeleton_root)
    dependencies = check_export_dependencies()
    parquet_ready = _bool_dependency(dependencies, "pyarrow")
    video_ready = _bool_dependency(dependencies, "imageio_ffmpeg") or (
        _bool_dependency(dependencies, "ffmpeg")
        and (
            _bool_dependency(dependencies, "imageio")
            or _bool_dependency(dependencies, "cv2")
            or _bool_dependency(dependencies, "PIL")
        )
    )

    report = _base_report(
        skeleton_root=skeleton_root,
        output_root=output_root,
        mode=mode,
        info=info,
        dependencies=dependencies,
        parquet_ready=parquet_ready,
        video_ready=video_ready,
    )

    if mode == "dry-run":
        report["skipped_reasons"].append("dry-run mode: parquet, videos, and metadata copy were skipped")
        report["next_recommended_action"] = (
            "Run with --mode write-if-available locally or on the lab ForceVLA environment to attempt outputs."
        )
        return write_report(report, output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    fps = float(info.get("fps", 30.0))

    parquet_successes = 0
    video_successes = 0
    expected_episodes = len(episodes)

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        frames = frames_by_episode[episode_index]
        episode_report = {
            "episode_index": episode_index,
            "length": len(frames),
            "parquet_written": False,
            "videos_written": False,
            "parquet_path": str(_parquet_path(output_root, info, episode_index)),
            "video_paths": [
                str(output_root / "videos" / key / f"episode_{episode_index:06d}.mp4")
                for key in IMAGE_KEYS
            ],
            "video_backend": None,
            "video_backends": {},
            "video_backend_errors": [],
            "skipped_reasons": [],
        }

        if not parquet_ready:
            _remove_file_if_present(_parquet_path(output_root, info, episode_index))
            episode_report["skipped_reasons"].append("pyarrow is not available; parquet writing skipped")
        else:
            try:
                _write_parquet_episode(output_root, info, episode_index, frames)
                episode_report["parquet_written"] = True
                parquet_successes += 1
            except Exception as exc:
                _remove_file_if_present(_parquet_path(output_root, info, episode_index))
                reason = f"parquet writing failed: {exc}"
                episode_report["skipped_reasons"].append(reason)
                report["skipped_reasons"].append(f"episode {episode_index}: {reason}")

        video_paths = [
            output_root / "videos" / key / f"episode_{episode_index:06d}.mp4"
            for key in IMAGE_KEYS
        ]
        if not video_ready:
            for video_path in video_paths:
                _remove_file_if_present(video_path)
            episode_report["skipped_reasons"].append(
                "video dependencies unavailable; requires imageio_ffmpeg or ffmpeg with imageio, cv2, or PIL readiness"
            )
        elif not any(_bool_dependency(dependencies, key) for key in ["imageio_ffmpeg", "imageio", "cv2"]):
            for video_path in video_paths:
                _remove_file_if_present(video_path)
            episode_report["skipped_reasons"].append(
                "video encoding skipped: imageio_ffmpeg, imageio, and cv2 are unavailable"
            )
        else:
            try:
                _, video_backends, backend_errors = _write_videos_episode(
                    skeleton_root=skeleton_root,
                    output_root=output_root,
                    episode_index=episode_index,
                    frames=frames,
                    fps=fps,
                    dependencies=dependencies,
                )
                episode_report["video_backends"] = video_backends
                episode_report["video_backend"] = _summarize_video_backends(video_backends)
                episode_report["video_backend_errors"] = backend_errors
                for key, backend_name in video_backends.items():
                    report["video_backends"][f"episode_{episode_index:06d}:{key}"] = backend_name
                report["video_backend_errors"].extend(
                    f"episode {episode_index}: {error}" for error in backend_errors
                )
                episode_report["videos_written"] = True
                video_successes += 1
            except Exception as exc:
                for video_path in video_paths:
                    _remove_file_if_present(video_path)
                reason = f"video encoding failed: {exc}"
                episode_report["skipped_reasons"].append(reason)
                report["skipped_reasons"].append(f"episode {episode_index}: {reason}")

        report["per_episode"].append(episode_report)

    report["parquet_written"] = parquet_successes == expected_episodes
    report["videos_written"] = video_successes == expected_episodes
    report["video_backend"] = _summarize_video_backends(report["video_backends"])

    if not report["parquet_written"] and parquet_ready:
        report["skipped_reasons"].append("one or more episode parquet files were not written")
    if not report["videos_written"] and video_ready:
        report["skipped_reasons"].append("one or more episode video sets were not written")

    if not parquet_ready:
        report["skipped_reasons"].append("pyarrow is not available; parquet writing skipped for all episodes")
    if not video_ready:
        report["skipped_reasons"].append(
            "video dependencies unavailable; requires imageio_ffmpeg or ffmpeg with imageio, cv2, or PIL readiness"
        )

    adapted_info = _adapt_info(
        info=info,
        source_skeleton=skeleton_root,
        parquet_written=report["parquet_written"],
        videos_written=report["videos_written"],
    )
    _write_metadata(output_root, adapted_info, tasks, episodes, stats)
    report["metadata_written"] = True

    if report["parquet_written"] and report["videos_written"]:
        report["next_recommended_action"] = "Validate this multi-episode output on the lab ForceVLA environment."
    elif not report["parquet_written"]:
        report["next_recommended_action"] = "Run on the lab ForceVLA environment with pyarrow available before real parquet export."
    else:
        report["next_recommended_action"] = "Parquet was written; resolve video skipped reasons before final video export."

    return write_report(report, output_root)


def _print_summary(report_path: Path) -> None:
    report = _read_json_object(report_path)
    print("Local Multi-Episode Real LeRobot Export Attempt")
    print(f"report: {report_path}")
    print(f"mode: {report['mode']}")
    print(f"profile: {report['profile']}")
    print(f"total_episodes: {report['total_episodes']}")
    print(f"total_frames: {report['total_frames']}")
    print(f"parquet_ready: {report['parquet_ready']}")
    print(f"video_ready: {report['video_ready']}")
    print(f"parquet_written: {report['parquet_written']}")
    print(f"videos_written: {report['videos_written']}")
    print(f"metadata_written: {report['metadata_written']}")
    if report["skipped_reasons"]:
        print("skipped reasons:")
        for reason in report["skipped_reasons"]:
            print(f"- {reason}")
    print(f"next_recommended_action: {report['next_recommended_action']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attempt a local real export for a multi-episode LeRobot skeleton.")
    parser.add_argument("--skeleton", required=True, help="Multi-episode skeleton dataset directory")
    parser.add_argument("--output", required=True, help="Output real LeRobot dataset directory")
    parser.add_argument("--mode", choices=sorted(MODES), default="dry-run")
    args = parser.parse_args(argv)

    try:
        report_path = write_real_lerobot_dataset_export(args.skeleton, args.output, mode=args.mode)
    except ValueError as exc:
        print(f"FAILED: could not run multi-episode real export: {args.output}")
        print(str(exc))
        return 1

    _print_summary(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
