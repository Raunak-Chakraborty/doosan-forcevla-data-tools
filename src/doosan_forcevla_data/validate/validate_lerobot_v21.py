"""Strict validation for the direct ForceVLA-compatible LeRobot v2.1 export."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.processed_to_lerobot_v21 import (
    ACTION_DIM,
    CHUNKS_SIZE,
    DATA_PATH_TEMPLATE,
    EXPORTER_VERSION,
    PARQUET_COLUMNS,
    PYARROW_EXTRA_HELP,
    STATE_COLUMNS,
    VIDEO_COLUMNS,
    VIDEO_PATH_TEMPLATE,
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def _import_pyarrow(errors: list[str]) -> tuple[Any | None, Any | None]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        errors.append(f"pyarrow is required to validate LeRobot v2.1 Parquet: {exc}. {PYARROW_EXTRA_HELP}")
        return None, None
    return pa, pq


def _read_json_object(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: could not read JSON object: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: expected a JSON object")
        return None
    return data


def _read_jsonl_objects(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: could not read JSONL: {exc}")
        return records
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"{path}: line {line_number} is empty")
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path}: line {line_number} must be a JSON object")
            continue
        records.append(data)
    return records


def _episode_chunk(episode_index: int) -> int:
    return episode_index // CHUNKS_SIZE


def _format_data_path(episode_index: int) -> Path:
    return Path(DATA_PATH_TEMPLATE.format(episode_chunk=_episode_chunk(episode_index), episode_index=episode_index))


def _format_video_path(video_key: str, episode_index: int) -> Path:
    return Path(VIDEO_PATH_TEMPLATE.format(video_key=video_key, episode_index=episode_index))


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_is_finite_number(item) for item in value)


def _ratio_to_float(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            denominator = float(den)
            if denominator == 0.0:
                return None
            return float(num) / denominator
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _ffprobe_video(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if shutil.which("ffprobe") is None:
        errors.append("ffprobe command is required to validate videos")
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"{path}: ffprobe failed: {exc}")
        return None
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"{path}: ffprobe returned invalid JSON: {exc}")
        return None
    streams = data.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        errors.append(f"{path}: ffprobe found no video stream")
        return None
    stream = streams[0]
    frame_text = stream.get("nb_read_frames") or stream.get("nb_frames")
    frames = int(frame_text) if isinstance(frame_text, str) and frame_text.isdigit() else None
    return {
        "frames": frames,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": _ratio_to_float(stream.get("avg_frame_rate")) or _ratio_to_float(stream.get("r_frame_rate")),
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
    }


def _check_info(info_path: Path, info: dict[str, Any], errors: list[str]) -> None:
    expected = {
        "codebase_version": "v2.1",
        "chunks_size": CHUNKS_SIZE,
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            errors.append(f"{info_path}: {key} must be {value!r}")
    for key in ["robot_type", "total_episodes", "total_frames", "total_tasks", "total_chunks", "fps", "total_videos"]:
        if key not in info:
            errors.append(f"{info_path}: missing {key}")
    if not _is_finite_number(info.get("fps")) or float(info.get("fps", 0.0)) <= 0.0:
        errors.append(f"{info_path}: fps must be positive and finite")
    if info.get("total_videos") != info.get("total_episodes") * len(VIDEO_COLUMNS):
        errors.append(f"{info_path}: total_videos must equal total_episodes * {len(VIDEO_COLUMNS)}")
    features = info.get("features")
    if not isinstance(features, dict):
        errors.append(f"{info_path}: features must be a JSON object")
        return
    for key, length in STATE_COLUMNS.items():
        feature = features.get(key)
        if not isinstance(feature, dict):
            errors.append(f"{info_path}: features.{key} is missing")
            continue
        if feature.get("dtype") != "float32" or feature.get("shape") != [length]:
            errors.append(f"{info_path}: features.{key} must be float32 shape [{length}]")
    for key in VIDEO_COLUMNS:
        feature = features.get(key)
        if not isinstance(feature, dict):
            errors.append(f"{info_path}: features.{key} is missing")
            continue
        shape = feature.get("shape")
        if feature.get("dtype") != "video" or not (
            isinstance(shape, list)
            and len(shape) == 3
            and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in shape)
        ):
            errors.append(f"{info_path}: features.{key} must be video with positive [height,width,channels] shape")


def _check_parquet_rows(root: Path, pq: Any, info: dict[str, Any], episode: dict[str, Any], errors: list[str]) -> int:
    episode_index = episode.get("episode_index")
    length = episode.get("length")
    task_index = episode.get("task_index")
    if not isinstance(episode_index, int) or isinstance(episode_index, bool):
        errors.append("episodes.jsonl: episode_index must be an integer")
        return 0
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        errors.append(f"episode {episode_index}: length must be a positive integer")
        return 0
    if not isinstance(task_index, int) or isinstance(task_index, bool):
        errors.append(f"episode {episode_index}: task_index must be an integer")
        return 0
    parquet_path = root / _format_data_path(episode_index)
    if not parquet_path.is_file():
        errors.append(f"{parquet_path}: parquet file is missing")
        return 0
    try:
        table = pq.read_table(parquet_path)
    except Exception as exc:
        errors.append(f"{parquet_path}: could not read parquet: {exc}")
        return 0
    if table.num_rows != length:
        errors.append(f"{parquet_path}: row count {table.num_rows} does not match episode length {length}")
    missing = [key for key in PARQUET_COLUMNS if key not in table.column_names]
    if missing:
        errors.append(f"{parquet_path}: missing columns: {', '.join(missing)}")
        return table.num_rows
    image_columns = [key for key in table.column_names if "images" in key]
    if image_columns:
        errors.append(f"{parquet_path}: direct ForceVLA export must not include image struct columns: {image_columns!r}")
    rows = table.to_pylist()
    fps = float(info.get("fps", 0.0)) if _is_finite_number(info.get("fps")) else 0.0
    for idx, row in enumerate(rows):
        if row.get("frame_index") != idx:
            errors.append(f"{parquet_path}: row {idx} frame_index must equal {idx}")
        if row.get("episode_index") != episode_index:
            errors.append(f"{parquet_path}: row {idx} episode_index must equal {episode_index}")
        if row.get("task_index") != task_index:
            errors.append(f"{parquet_path}: row {idx} task_index must equal {task_index}")
        timestamp = row.get("timestamp")
        if not _is_finite_number(timestamp):
            errors.append(f"{parquet_path}: row {idx} timestamp must be finite")
        elif fps > 0.0 and abs(float(timestamp) - idx / fps) > 1e-9:
            errors.append(f"{parquet_path}: row {idx} timestamp must equal frame_index / fps")
        for key, length_expected in STATE_COLUMNS.items():
            if not _finite_vector(row.get(key), length_expected):
                errors.append(f"{parquet_path}: row {idx} {key} must be finite vector length {length_expected}")
        quat = row.get("observation.state.ee_quat")
        if _finite_vector(quat, 4):
            norm = math.sqrt(sum(float(value) * float(value) for value in quat))
            if abs(norm - 1.0) > 1e-4:
                errors.append(f"{parquet_path}: row {idx} observation.state.ee_quat norm must be 1, got {norm}")
    if rows:
        final_action = rows[-1].get("action")
        if not _finite_vector(final_action, ACTION_DIM) or any(abs(float(value)) > 1e-12 for value in final_action):
            errors.append(f"{parquet_path}: final action must be the retained zero terminal action")
    return table.num_rows


def _check_video_files(root: Path, info: dict[str, Any], episode: dict[str, Any], errors: list[str]) -> None:
    episode_index = episode.get("episode_index")
    length = episode.get("length")
    if not isinstance(episode_index, int) or not isinstance(length, int):
        return
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    fps = float(info.get("fps", 0.0)) if _is_finite_number(info.get("fps")) else None
    for video_key in VIDEO_COLUMNS:
        video_path = root / _format_video_path(video_key, episode_index)
        if not video_path.is_file():
            errors.append(f"{video_path}: video file is missing")
            continue
        report = _ffprobe_video(video_path, errors)
        if report is None:
            continue
        if report["frames"] != length:
            errors.append(f"{video_path}: decoded frames {report['frames']!r} must equal episode length {length}")
        feature = features.get(video_key)
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if isinstance(shape, list) and len(shape) >= 2:
            if report["height"] != shape[0] or report["width"] != shape[1]:
                errors.append(f"{video_path}: dimensions {(report['height'], report['width'])!r} do not match feature shape {shape[:2]!r}")
        if fps is not None and report["fps"] is not None and abs(float(report["fps"]) - fps) > 1e-6:
            errors.append(f"{video_path}: fps {report['fps']!r} must equal info fps {fps}")


def _check_provenance(path: Path, provenance: dict[str, Any] | None, errors: list[str]) -> None:
    if provenance is None:
        return
    if provenance.get("exporter_version") != EXPORTER_VERSION:
        errors.append(f"{path}: exporter_version must be {EXPORTER_VERSION!r}")
    terminal = provenance.get("terminal_action_policy")
    if not isinstance(terminal, dict) or terminal.get("final_zero_terminal_action_retained") is not True:
        errors.append(f"{path}: provenance must state final_zero_terminal_action_retained=true")
    forcevla = provenance.get("target_loader_reference")
    if not isinstance(forcevla, dict) or forcevla.get("forcevla_v5_config") != "forcevla_sfp_all_trimmed_v5":
        errors.append(f"{path}: provenance must record forcevla_sfp_all_trimmed_v5 reference")


def validate_lerobot_v21(dataset_dir: str | Path) -> ValidationResult:
    root = Path(dataset_dir)
    errors: list[str] = []
    if not root.exists():
        return ValidationResult(False, [f"{root}: dataset directory does not exist"])
    if not root.is_dir():
        return ValidationResult(False, [f"{root}: dataset path is not a directory"])

    required = [
        root / "meta" / "info.json",
        root / "meta" / "tasks.jsonl",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "episodes_stats.jsonl",
        root / "meta" / "export_provenance.json",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"{path}: required file is missing")
    if errors:
        return ValidationResult(False, errors)

    info_path = root / "meta" / "info.json"
    info = _read_json_object(info_path, errors)
    tasks = _read_jsonl_objects(root / "meta" / "tasks.jsonl", errors)
    episodes = _read_jsonl_objects(root / "meta" / "episodes.jsonl", errors)
    episode_stats = _read_jsonl_objects(root / "meta" / "episodes_stats.jsonl", errors)
    provenance = _read_json_object(root / "meta" / "export_provenance.json", errors)
    if info is None:
        return ValidationResult(False, errors)

    _check_info(info_path, info, errors)
    _check_provenance(root / "meta" / "export_provenance.json", provenance, errors)
    if info.get("total_episodes") != len(episodes):
        errors.append(f"{info_path}: total_episodes must match episodes.jsonl length")
    if info.get("total_tasks") != len(tasks):
        errors.append(f"{info_path}: total_tasks must match tasks.jsonl length")
    if len(episode_stats) != len(episodes):
        errors.append("episodes_stats.jsonl length must match episodes.jsonl length")
    task_indexes = {record.get("task_index") for record in tasks if isinstance(record.get("task_index"), int)}
    for record in tasks:
        if not isinstance(record.get("task"), str) or not record["task"].strip():
            errors.append("tasks.jsonl: each task must have a non-empty task string")
    for episode in episodes:
        if episode.get("task_index") not in task_indexes:
            errors.append(f"episode {episode.get('episode_index')}: task_index is not present in tasks.jsonl")

    pa, pq = _import_pyarrow(errors)
    total_rows = 0
    if pq is not None:
        for episode in episodes:
            total_rows += _check_parquet_rows(root, pq, info, episode, errors)
    if info.get("total_frames") != total_rows:
        errors.append(f"{info_path}: total_frames {info.get('total_frames')!r} must equal parquet rows {total_rows}")
    for episode in episodes:
        _check_video_files(root, info, episode, errors)
    for record in episode_stats:
        stats = record.get("stats")
        if not isinstance(stats, dict) or not stats:
            errors.append(f"episode {record.get('episode_index')}: episodes_stats stats must be non-empty")
        elif "action" not in stats or "observation.state.wrench" not in stats:
            errors.append(f"episode {record.get('episode_index')}: episodes_stats must include action and wrench stats")
    return ValidationResult(not errors, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a direct LeRobot v2.1 ForceVLA-compatible dataset.")
    parser.add_argument("dataset_dir", help="LeRobot v2.1 dataset directory")
    args = parser.parse_args(argv)
    result = validate_lerobot_v21(args.dataset_dir)
    if result.ok:
        print(f"OK: LeRobot v2.1 dataset is valid: {args.dataset_dir}")
        return 0
    print(f"INVALID: LeRobot v2.1 dataset failed validation: {args.dataset_dir}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
