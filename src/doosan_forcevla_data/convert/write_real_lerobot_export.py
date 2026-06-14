"""Dependency-optional local real LeRobot export scaffold."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from doosan_forcevla_data.inspect.preflight_real_export import preflight_real_export
from doosan_forcevla_data.schema.processed_schema import ACTION_DIM
from doosan_forcevla_data.validate.validate_lerobot_skeleton import validate_lerobot_skeleton


MODES = {"dry-run", "write-if-available"}
REPORT_NAME = "export_attempt_report.json"


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


def _read_single_jsonl(path: Path, name: str) -> dict[str, Any]:
    records = _read_jsonl(path)
    if len(records) != 1:
        raise ValueError(f"{path}: expected exactly one {name} record, got {len(records)}")
    return records[0]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _dependency_available(preflight: dict[str, Any], key: str) -> bool:
    entry = preflight.get("dependency_summary", {}).get(key)
    return bool(isinstance(entry, dict) and entry.get("available") is True)


def _remove_file_if_present(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        raise ValueError(f"expected file path but found directory: {path}")
    if path.exists() or path.is_symlink():
        path.unlink()


def _load_skeleton(skeleton_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    info = _read_json_object(skeleton_root / "meta" / "info.json")
    task = _read_single_jsonl(skeleton_root / "meta" / "tasks.jsonl", "task")
    episode = _read_single_jsonl(skeleton_root / "meta" / "episodes.jsonl", "episode")
    stats = _read_single_jsonl(skeleton_root / "meta" / "episodes_stats.jsonl", "episode stats")
    episode_index = episode.get("episode_index")
    if not isinstance(episode_index, int) or isinstance(episode_index, bool):
        episode_index = 0
    frames = _read_jsonl(skeleton_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.jsonl")
    return info, task, episode, stats, frames


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
            for key in ["observation.image", "observation.wrist_image"]:
                feature = features.get(key)
                if isinstance(feature, dict):
                    feature["dtype"] = "video"
    return adapted


def _write_metadata(
    output_root: Path,
    info: dict[str, Any],
    task: dict[str, Any],
    episode: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(meta_dir / "tasks.jsonl", [task])
    _write_jsonl(meta_dir / "episodes.jsonl", [episode])
    _write_jsonl(meta_dir / "episodes_stats.jsonl", [stats])


def _write_parquet(output_root: Path, episode_index: int, frames: list[dict[str, Any]]) -> Path:
    pa = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")

    parquet_path = output_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
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


def _frame_image_paths(skeleton_root: Path, frames: list[dict[str, Any]], key: str) -> list[Path]:
    paths: list[Path] = []
    for frame in frames:
        value = frame.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty path string")
        path = skeleton_root / value
        if not path.is_file():
            raise ValueError(f"{key} path does not exist: {path}")
        paths.append(path)
    return paths


def _format_fps(fps: float) -> str:
    value = float(fps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    return format(value, ".12g")


def _dependency_flag(dependencies: dict[str, dict[str, Any]], key: str) -> bool:
    return bool(dependencies.get(key, {}).get("available"))


def _encode_video_imageio_ffmpeg(image_paths: list[Path], output_path: Path, fps: float) -> None:
    if not image_paths:
        raise ValueError("video encoding requires at least one image")

    imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_exe:
        raise ValueError("imageio_ffmpeg.get_ffmpeg_exe() returned an empty path")

    suffix = image_paths[0].suffix.lower()
    if not suffix:
        raise ValueError("direct imageio_ffmpeg encoding requires image paths with file suffixes")
    for image_path in image_paths:
        if image_path.suffix.lower() != suffix:
            raise ValueError("direct imageio_ffmpeg encoding requires a single image file suffix per video")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_file_if_present(output_path)
    fps_arg = _format_fps(fps)

    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}_ffmpeg_", dir=output_path.parent) as temp_dir:
        temp_root = Path(temp_dir)
        for index, image_path in enumerate(image_paths):
            staged_path = temp_root / f"frame_{index:06d}{suffix}"
            try:
                staged_path.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, staged_path)

        command = [
            str(ffmpeg_exe),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-framerate",
            fps_arg,
            "-start_number",
            "0",
            "-i",
            str(temp_root / f"frame_%06d{suffix}"),
            "-frames:v",
            str(len(image_paths)),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            fps_arg,
            "-threads",
            "1",
            "-map_metadata",
            "-1",
            "-f",
            "mp4",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            suffix_detail = f": {detail}" if detail else ""
            raise ValueError(
                f"imageio_ffmpeg ffmpeg command failed with exit code {exc.returncode}{suffix_detail}"
            ) from exc


def _encode_video_imageio(image_paths: list[Path], output_path: Path, fps: float) -> None:
    try:
        imageio = importlib.import_module("imageio.v2")
    except ModuleNotFoundError:
        imageio = importlib.import_module("imageio")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_file_if_present(output_path)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for image_path in image_paths:
            writer.append_data(imageio.imread(image_path))


def _encode_video_cv2(image_paths: list[Path], output_path: Path, fps: float) -> None:
    cv2 = importlib.import_module("cv2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_file_if_present(output_path)
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"cv2 could not read image: {image_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height))
    )
    if not writer.isOpened():
        raise ValueError(f"cv2 could not open VideoWriter for {output_path}")
    try:
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cv2 could not read image: {image_path}")
            writer.write(image)
    finally:
        writer.release()


def _verify_video(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"video was not written: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"video file is empty: {path}")


def _encode_video_with_fallback(
    image_paths: list[Path],
    output_path: Path,
    fps: float,
    dependencies: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    backend_errors: list[str] = []
    backends = [
        ("imageio_ffmpeg", _dependency_flag(dependencies, "imageio_ffmpeg"), _encode_video_imageio_ffmpeg),
        ("imageio", _dependency_flag(dependencies, "imageio"), _encode_video_imageio),
        ("cv2", _dependency_flag(dependencies, "cv2"), _encode_video_cv2),
    ]

    for backend_name, available, encoder in backends:
        if not available:
            continue
        try:
            encoder(image_paths, output_path, fps)
        except Exception as exc:
            _remove_file_if_present(output_path)
            backend_errors.append(f"{backend_name} failed: {exc}")
            continue
        return backend_name, backend_errors

    if backend_errors:
        raise ValueError("; ".join(backend_errors))
    raise ValueError("video encoding requires imageio_ffmpeg, imageio, or cv2")


def _summarize_video_backends(video_backends: dict[str, str]) -> str | None:
    unique_backends = sorted(set(video_backends.values()))
    if not unique_backends:
        return None
    if len(unique_backends) == 1:
        return unique_backends[0]
    return "mixed"


def _write_videos(
    skeleton_root: Path,
    output_root: Path,
    episode_index: int,
    frames: list[dict[str, Any]],
    fps: float,
    dependencies: dict[str, dict[str, Any]],
) -> tuple[list[Path], dict[str, str], list[str]]:
    if not any(_dependency_flag(dependencies, key) for key in ["imageio_ffmpeg", "imageio", "cv2"]):
        raise ValueError("video encoding requires imageio_ffmpeg, imageio, or cv2")

    written: list[Path] = []
    video_backends: dict[str, str] = {}
    backend_errors: list[str] = []
    for key in ["observation.image", "observation.wrist_image"]:
        image_paths = _frame_image_paths(skeleton_root, frames, key)
        output_path = output_root / "videos" / key / f"episode_{episode_index:06d}.mp4"
        backend_name, errors = _encode_video_with_fallback(image_paths, output_path, fps, dependencies)
        _verify_video(output_path)
        written.append(output_path)
        video_backends[key] = backend_name
        backend_errors.extend(f"{key}: {error}" for error in errors)
    return written, video_backends, backend_errors


def _base_report(
    source_skeleton: Path,
    output_root: Path,
    mode: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_skeleton": str(source_skeleton.resolve()),
        "output_dir": str(output_root.resolve()),
        "mode": mode,
        "profile": preflight.get("profile"),
        "total_frames": preflight.get("total_frames"),
        "state_dim": preflight.get("state_dim"),
        "action_dim": preflight.get("action_dim"),
        "dependencies": preflight.get("dependency_summary", {}),
        "parquet_ready": bool(preflight.get("parquet_ready")),
        "video_ready": bool(preflight.get("video_ready")),
        "lerobot_api_available": bool(preflight.get("lerobot_api_available")),
        "parquet_written": False,
        "videos_written": False,
        "video_backend": None,
        "video_backends": {},
        "video_backend_errors": [],
        "metadata_written": False,
        "skipped_reasons": [],
        "next_recommended_action": "Run this command on the lab ForceVLA environment before treating readiness as final.",
    }


def write_report(report: dict[str, Any], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def write_real_lerobot_export(
    skeleton_dir: str | Path,
    output_dir: str | Path,
    mode: str = "dry-run",
) -> Path:
    """Attempt a dependency-optional local real export and return the report path."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(MODES))}")
    skeleton_root = Path(skeleton_dir)
    output_root = Path(output_dir)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output_root}")

    validation = validate_lerobot_skeleton(skeleton_root)
    if not validation.ok:
        message = "skeleton validation failed:\n" + "\n".join(f"ERROR: {error}" for error in validation.errors)
        raise ValueError(message)

    preflight = preflight_real_export(skeleton_root)
    info, task, episode, stats, frames = _load_skeleton(skeleton_root)
    episode_index = int(episode.get("episode_index", 0))
    fps = float(info.get("fps", 30.0))
    report = _base_report(skeleton_root, output_root, mode, preflight)

    if mode == "dry-run":
        report["skipped_reasons"].append("dry-run mode: parquet, videos, and metadata copy were skipped")
        report["next_recommended_action"] = (
            "Run with --mode write-if-available locally or on the lab ForceVLA environment to attempt outputs."
        )
        return write_report(report, output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    parquet_path = output_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not report["parquet_ready"]:
        _remove_file_if_present(parquet_path)
        report["skipped_reasons"].append("pyarrow is not available; parquet writing skipped")
    else:
        try:
            _write_parquet(output_root, episode_index, frames)
            report["parquet_written"] = True
        except Exception as exc:
            report["skipped_reasons"].append(f"parquet writing failed: {exc}")
            _remove_file_if_present(parquet_path)

    video_paths = [
        output_root / "videos" / "observation.image" / f"episode_{episode_index:06d}.mp4",
        output_root / "videos" / "observation.wrist_image" / f"episode_{episode_index:06d}.mp4",
    ]
    if not report["video_ready"]:
        for video_path in video_paths:
            _remove_file_if_present(video_path)
        report["skipped_reasons"].append(
            "video dependencies unavailable; requires imageio_ffmpeg or ffmpeg with imageio, cv2, or PIL readiness"
        )
    elif not any(_dependency_available(preflight, key) for key in ["imageio_ffmpeg", "imageio", "cv2"]):
        for video_path in video_paths:
            _remove_file_if_present(video_path)
        report["skipped_reasons"].append("video encoding skipped: imageio_ffmpeg, imageio, and cv2 are unavailable")
    else:
        try:
            _, video_backends, backend_errors = _write_videos(
                skeleton_root, output_root, episode_index, frames, fps, preflight["dependency_summary"]
            )
            report["video_backends"] = video_backends
            report["video_backend"] = _summarize_video_backends(video_backends)
            report["video_backend_errors"] = backend_errors
            report["videos_written"] = True
        except Exception as exc:
            for video_path in video_paths:
                _remove_file_if_present(video_path)
            report["skipped_reasons"].append(f"video encoding failed: {exc}")

    adapted_info = _adapt_info(
        info=info,
        source_skeleton=skeleton_root,
        parquet_written=report["parquet_written"],
        videos_written=report["videos_written"],
    )
    _write_metadata(output_root, adapted_info, task, episode, stats)
    report["metadata_written"] = True

    if report["parquet_written"] and report["videos_written"]:
        report["next_recommended_action"] = "Validate this output on the lab ForceVLA environment."
    elif not report["parquet_written"]:
        report["next_recommended_action"] = "Run on the lab ForceVLA environment with pyarrow available before real parquet export."
    else:
        report["next_recommended_action"] = "Parquet was written; resolve video skipped reasons before final video export."
    return write_report(report, output_root)


def _print_summary(report_path: Path) -> None:
    report = _read_json_object(report_path)
    print("Local Real LeRobot Export Attempt")
    print(f"report: {report_path}")
    print(f"mode: {report['mode']}")
    print(f"profile: {report['profile']}")
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
    print("notes: local only; no Hugging Face upload; no ROS; no any4lerobot")
    print(f"next_recommended_action: {report['next_recommended_action']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attempt a dependency-optional local real LeRobot export.")
    parser.add_argument("--skeleton", required=True, help="Source local LeRobot-style skeleton export")
    parser.add_argument("--output", required=True, help="Output real-export attempt directory")
    parser.add_argument("--mode", choices=sorted(MODES), default="dry-run")
    args = parser.parse_args(argv)

    try:
        report_path = write_real_lerobot_export(args.skeleton, args.output, mode=args.mode)
    except (OSError, ValueError) as exc:
        print(f"FAILED: could not write real export attempt: {args.output}")
        print(str(exc))
        return 1

    _print_summary(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
