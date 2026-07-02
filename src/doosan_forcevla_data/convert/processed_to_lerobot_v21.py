"""Direct processed JSONL to ForceVLA-compatible LeRobot v2.1 export.

This writer intentionally avoids LeRobot, NumPy, Pillow, OpenCV, and training
framework imports. PyArrow is imported lazily only when writing Parquet, and
videos are encoded with the system ffmpeg command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM


EXPORTER_VERSION = "processed_to_lerobot_v21_direct_v1"
CHUNKS_SIZE = 1000
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/{video_key}/episode_{episode_index:06d}.mp4"
PYARROW_EXTRA_HELP = "Install the optional exporter dependency with `pip install -e .[lerobot-v21]`."

STATE_COLUMNS = {
    "observation.state.ee_pos": 3,
    "observation.state.ee_quat": 4,
    "observation.state.gripper_pos": 2,
    "observation.state.wrench": 6,
    "observation.state.joint_pos": 6,
    "observation.state.joint_vel": 6,
    "action": ACTION_DIM,
}

VIDEO_COLUMNS = {
    "observation.images.center": {
        "frame_key": "external_rgb_path",
        "processed_camera": "external_camera_1",
        "description": "external_camera_1 mapped to ForceVLA center image slot",
    },
    "observation.images.left": {
        "frame_key": "tcp_rgb_path",
        "processed_camera": "tcp_camera",
        "description": "tcp_camera mapped to ForceVLA left image slot",
    },
    "observation.images.right": {
        "frame_key": "external_camera_2_rgb_path",
        "processed_camera": "external_camera_2",
        "description": "external_camera_2 mapped to ForceVLA right image slot",
    },
}

PARQUET_COLUMNS = [
    "frame_index",
    "episode_index",
    "timestamp",
    "task_index",
    "observation.state.joint_pos",
    "observation.state.ee_pos",
    "observation.state.ee_quat",
    "observation.state.gripper_pos",
    "observation.state.wrench",
    "action",
    "observation.state.joint_vel",
]


@dataclass(frozen=True)
class ProcessedEpisode:
    root: Path
    metadata: dict[str, Any]
    frames: list[dict[str, Any]]
    task: str


@dataclass(frozen=True)
class ExportedEpisode:
    episode_index: int
    task_index: int
    source_root: Path
    frame_count: int
    parquet_path: Path
    video_paths: dict[str, Path]
    video_reports: dict[str, dict[str, Any]]
    stats: dict[str, Any]


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(f"PyArrow is required to write LeRobot v2.1 Parquet. {PYARROW_EXTRA_HELP}") from exc
    return pa, pq


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: could not read JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{path}: could not read JSONL: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{path}: line {line_number} is empty")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        records.append(record)
    return records


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _finite_float(value: Any, context: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _finite_vector(value: Any, length: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} must be a list of length {length}")
    return [_finite_float(item, f"{context}[{idx}]") for idx, item in enumerate(value)]


def _positive_fps(value: Any, context: str) -> float:
    fps = _finite_float(value, context)
    if fps <= 0.0:
        raise ValueError(f"{context} must be positive")
    return fps


def _format_fps(fps: float) -> str:
    return format(_positive_fps(fps, "fps"), ".12g")


def _episode_chunk(episode_index: int) -> int:
    return episode_index // CHUNKS_SIZE


def _total_chunks(total_episodes: int) -> int:
    if total_episodes <= 0:
        return 0
    return _episode_chunk(total_episodes - 1) + 1


def _contains_path(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _safe_output_state(output_root: Path, overwrite: bool) -> None:
    if not output_root.exists() and not output_root.is_symlink():
        return
    if output_root.is_symlink() or not output_root.is_dir():
        if overwrite:
            return
        raise FileExistsError(f"output path exists and is not a directory: {output_root}")
    entries = list(output_root.iterdir())
    if entries and not overwrite:
        raise FileExistsError(f"output directory already exists and is non-empty: {output_root}")


def _finalize_staging(staging_root: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.exists() or output_root.is_symlink():
        if output_root.is_symlink() or not output_root.is_dir():
            if not overwrite:
                raise FileExistsError(f"output path exists and is not a directory: {output_root}")
            output_root.unlink()
        else:
            entries = list(output_root.iterdir())
            if entries and not overwrite:
                raise FileExistsError(f"output directory already exists and is non-empty: {output_root}")
            shutil.rmtree(output_root)
    staging_root.replace(output_root)


def _resolve_image_path(processed_root: Path, metadata: dict[str, Any], value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("image path value must be a non-empty string")
    path = Path(value)
    if path.is_absolute():
        return path
    processed_candidate = processed_root / path
    if processed_candidate.is_file():
        return processed_candidate
    source_raw = metadata.get("source_raw_episode")
    if isinstance(source_raw, str) and source_raw:
        return Path(source_raw) / path
    return processed_candidate


def _load_processed_episode(processed_episode_dir: str | Path, task_override: str | None = None) -> ProcessedEpisode:
    root = Path(processed_episode_dir)
    if not root.is_dir():
        raise ValueError(f"processed episode directory does not exist: {root}")
    metadata = _read_json_object(root / "metadata_processed.json")
    frames = _read_jsonl_objects(root / "frames.jsonl")
    if not frames:
        raise ValueError(f"{root}: frames.jsonl has no frames")
    if metadata.get("model_state_dim") != MODEL_STATE_DIM:
        raise ValueError(f"{root}: model_state_dim must be {MODEL_STATE_DIM}")
    if metadata.get("action_dim") != ACTION_DIM:
        raise ValueError(f"{root}: action_dim must be {ACTION_DIM}")
    declared_count = metadata.get("frame_count")
    if declared_count != len(frames):
        raise ValueError(f"{root}: frame_count {declared_count!r} does not match {len(frames)} frame rows")
    _positive_fps(metadata.get("fps"), f"{root}: metadata_processed.json fps")

    padding_indices: list[int] = []
    for idx, frame in enumerate(frames):
        if frame.get("frame_index") != idx:
            raise ValueError(f"{root}: frame {idx} frame_index must equal row index")
        _finite_float(frame.get("timestamp"), f"{root}: frame {idx} timestamp")
        _finite_vector(frame.get("model_state"), MODEL_STATE_DIM, f"{root}: frame {idx} model_state")
        _finite_vector(frame.get("measured_action"), ACTION_DIM, f"{root}: frame {idx} measured_action")
        if frame.get("action_is_terminal_padding") is True:
            padding_indices.append(idx)
        for spec in VIDEO_COLUMNS.values():
            image_path = _resolve_image_path(root, metadata, frame.get(spec["frame_key"]))
            if not image_path.is_file():
                raise ValueError(f"{root}: frame {idx} {spec['frame_key']} does not exist: {image_path}")
    if padding_indices != [len(frames) - 1]:
        raise ValueError(f"{root}: expected exactly one final terminal padding frame, got {padding_indices!r}")
    final_action = _finite_vector(frames[-1].get("measured_action"), ACTION_DIM, f"{root}: final measured_action")
    if any(abs(value) > 1e-12 for value in final_action):
        raise ValueError(f"{root}: final terminal action must be all zeros")

    task = task_override or metadata.get("task_instruction")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"{root}: task must be supplied by --task or metadata task_instruction")
    return ProcessedEpisode(root=root, metadata=metadata, frames=frames, task=task.strip())


def _rotvec_to_quat_wxyz(rotvec: Sequence[float]) -> list[float]:
    rx, ry, rz = (float(rotvec[0]), float(rotvec[1]), float(rotvec[2]))
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return [1.0, 0.5 * rx, 0.5 * ry, 0.5 * rz]
    half = 0.5 * theta
    scale = math.sin(half) / theta
    quat = [math.cos(half), rx * scale, ry * scale, rz * scale]
    norm = math.sqrt(sum(value * value for value in quat))
    if norm == 0.0:
        return [1.0, 0.0, 0.0, 0.0]
    quat = [value / norm for value in quat]
    if quat[0] < 0.0:
        quat = [-value for value in quat]
    return quat


def _split_state_and_action(frame: dict[str, Any]) -> dict[str, list[float]]:
    state = _finite_vector(frame.get("model_state"), MODEL_STATE_DIM, "model_state")
    action = _finite_vector(frame.get("measured_action"), ACTION_DIM, "measured_action")
    gripper = float(state[6])
    return {
        "observation.state.ee_pos": state[0:3],
        "observation.state.ee_quat": _rotvec_to_quat_wxyz(state[3:6]),
        "observation.state.gripper_pos": [gripper, gripper],
        "observation.state.wrench": state[7:13],
        "observation.state.joint_pos": state[13:19],
        "observation.state.joint_vel": state[19:25],
        "action": action,
    }


def _stats_for_vectors(values: list[list[float]]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": [], "std": [], "min": [], "max": []}
    dim = len(values[0])
    count = len(values)
    means = [sum(row[idx] for row in values) / count for idx in range(dim)]
    variances = [sum((row[idx] - means[idx]) ** 2 for row in values) / count for idx in range(dim)]
    return {
        "count": count,
        "mean": means,
        "std": [math.sqrt(value) for value in variances],
        "min": [min(row[idx] for row in values) for idx in range(dim)],
        "max": [max(row[idx] for row in values) for idx in range(dim)],
    }


def _episode_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key in STATE_COLUMNS:
        stats[key] = _stats_for_vectors([row[key] for row in rows])
    stats["timestamp"] = _stats_for_vectors([[float(row["timestamp"])] for row in rows])
    return stats


def _build_rows(episode: ProcessedEpisode, episode_index: int, task_index: int, fps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, frame in enumerate(episode.frames):
        split = _split_state_and_action(frame)
        rows.append(
            {
                "frame_index": row_index,
                "episode_index": episode_index,
                "timestamp": row_index / fps,
                "task_index": task_index,
                **split,
            }
        )
    return rows


def _write_parquet_episode(output_root: Path, episode_index: int, rows: list[dict[str, Any]]) -> Path:
    pa, pq = _import_pyarrow()
    parquet_path = output_root / "data" / f"chunk-{_episode_chunk(episode_index):03d}" / f"episode_{episode_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "frame_index": pa.array([int(row["frame_index"]) for row in rows], type=pa.int64()),
        "episode_index": pa.array([int(row["episode_index"]) for row in rows], type=pa.int64()),
        "timestamp": pa.array([float(row["timestamp"]) for row in rows], type=pa.float64()),
        "task_index": pa.array([int(row["task_index"]) for row in rows], type=pa.int64()),
    }
    for key in [
        "observation.state.joint_pos",
        "observation.state.ee_pos",
        "observation.state.ee_quat",
        "observation.state.gripper_pos",
        "observation.state.wrench",
        "action",
        "observation.state.joint_vel",
    ]:
        arrays[key] = pa.array([[float(value) for value in row[key]] for row in rows], type=pa.list_(pa.float32()))
    table = pa.table({key: arrays[key] for key in PARQUET_COLUMNS})
    pq.write_table(table, parquet_path)
    return parquet_path


def _image_paths_for_video(episode: ProcessedEpisode, video_key: str) -> list[Path]:
    spec = VIDEO_COLUMNS[video_key]
    frame_key = spec["frame_key"]
    paths = [_resolve_image_path(episode.root, episode.metadata, frame.get(frame_key)) for frame in episode.frames]
    suffixes = {path.suffix.lower() for path in paths}
    if len(suffixes) != 1:
        raise ValueError(f"{episode.root}: {video_key} image suffixes must be uniform, got {sorted(suffixes)!r}")
    if not next(iter(suffixes)):
        raise ValueError(f"{episode.root}: {video_key} image paths must have a file suffix")
    return paths


def _run_subprocess(command: list[str], context: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{context}: required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{context}: command failed with exit code {exc.returncode}{suffix}") from exc


def _encode_video(image_paths: list[Path], output_path: Path, fps: float) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("video encoding requires at least one image")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps_arg = _format_fps(fps)
    suffix = image_paths[0].suffix.lower()
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}_frames_", dir=output_path.parent) as temp_dir:
        temp_root = Path(temp_dir)
        for idx, source_path in enumerate(image_paths):
            staged_path = temp_root / f"frame_{idx:06d}{suffix}"
            try:
                staged_path.symlink_to(source_path.resolve())
            except OSError:
                shutil.copy2(source_path, staged_path)
        command = [
            "ffmpeg",
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
            "mpeg4",
            "-q:v",
            "2",
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
        _run_subprocess(command, f"encoding {output_path}")
    return _probe_video(output_path, expected_frames=len(image_paths), expected_fps=fps)


def _parse_ratio(value: Any) -> float | None:
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


def _probe_video(path: Path, expected_frames: int | None = None, expected_fps: float | None = None) -> dict[str, Any]:
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
    completed = _run_subprocess(command, f"probing {path}")
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"probing {path}: ffprobe did not return JSON") from exc
    streams = data.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(f"probing {path}: no video stream found")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise RuntimeError(f"probing {path}: invalid stream JSON")
    frame_text = stream.get("nb_read_frames") or stream.get("nb_frames")
    decoded_frames = int(frame_text) if isinstance(frame_text, str) and frame_text.isdigit() else None
    fps = _parse_ratio(stream.get("avg_frame_rate")) or _parse_ratio(stream.get("r_frame_rate"))
    report = {
        "path": str(path),
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": int(stream.get("width")) if isinstance(stream.get("width"), int) else stream.get("width"),
        "height": int(stream.get("height")) if isinstance(stream.get("height"), int) else stream.get("height"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "fps": fps,
        "decoded_frames": decoded_frames,
        "duration": stream.get("duration"),
    }
    if expected_frames is not None and decoded_frames != expected_frames:
        raise RuntimeError(f"{path}: decoded frame count {decoded_frames!r} does not equal expected {expected_frames}")
    if expected_fps is not None and fps is not None and abs(float(fps) - float(expected_fps)) > 1e-6:
        raise RuntimeError(f"{path}: fps {fps!r} does not equal expected {expected_fps}")
    return report


def _features(video_shape: list[int]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state.joint_pos": {"dtype": "float32", "shape": [6]},
        "observation.state.ee_pos": {"dtype": "float32", "shape": [3]},
        "observation.state.ee_quat": {"dtype": "float32", "shape": [4]},
        "observation.state.gripper_pos": {"dtype": "float32", "shape": [2]},
        "observation.state.wrench": {"dtype": "float32", "shape": [6]},
        "action": {"dtype": "float32", "shape": [ACTION_DIM]},
        "observation.state.joint_vel": {"dtype": "float32", "shape": [6]},
    }
    for video_key in VIDEO_COLUMNS:
        features[video_key] = {
            "dtype": "video",
            "shape": video_shape,
            "names": ["height", "width", "channels"],
        }
    return features


def _video_shape_from_metadata(episodes: list[ProcessedEpisode]) -> list[int]:
    heights: set[int] = set()
    widths: set[int] = set()
    channels: set[int] = set()
    for episode in episodes:
        streams = episode.metadata.get("processed_camera_streams")
        if not isinstance(streams, dict):
            continue
        for spec in VIDEO_COLUMNS.values():
            entry = streams.get(spec["processed_camera"])
            if not isinstance(entry, dict):
                continue
            width = entry.get("width")
            height = entry.get("height")
            channel_count = entry.get("channels")
            if isinstance(width, int) and not isinstance(width, bool):
                widths.add(width)
            if isinstance(height, int) and not isinstance(height, bool):
                heights.add(height)
            if isinstance(channel_count, int) and not isinstance(channel_count, bool):
                channels.add(channel_count)
    if len(widths) == len(heights) == len(channels) == 1:
        return [next(iter(heights)), next(iter(widths)), next(iter(channels))]
    return [480, 640, 3]


def _write_metadata(
    output_root: Path,
    episodes: list[ProcessedEpisode],
    exported_episodes: list[ExportedEpisode],
    task_records: list[dict[str, Any]],
    fps: float,
    robot_type: str,
    video_shape: list[int],
) -> None:
    total_frames = sum(item.frame_count for item in exported_episodes)
    total_episodes = len(exported_episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_records),
        "total_chunks": _total_chunks(total_episodes),
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "splits": {"train": f"0:{total_episodes}"},
        "total_videos": total_episodes * len(VIDEO_COLUMNS),
        "features": _features(video_shape),
    }
    _write_json(output_root / "meta" / "info.json", info)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", task_records)
    _write_jsonl(
        output_root / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": item.episode_index,
                "task_index": item.task_index,
                "length": item.frame_count,
            }
            for item in exported_episodes
        ],
    )
    _write_jsonl(
        output_root / "meta" / "episodes_stats.jsonl",
        [{"episode_index": item.episode_index, "stats": item.stats} for item in exported_episodes],
    )
    _write_json(
        output_root / "meta" / "export_provenance.json",
        _provenance(episodes, exported_episodes, fps, robot_type),
    )


def _provenance(
    episodes: list[ProcessedEpisode],
    exported_episodes: list[ExportedEpisode],
    fps: float,
    robot_type: str,
) -> dict[str, Any]:
    return {
        "exporter_version": EXPORTER_VERSION,
        "target_codebase_version": "LeRobot v2.1",
        "target_loader_reference": {
            "forcevla_path": "/home/horus/robotics_thesis/forcevla/forcevla_fork_tshiamor",
            "forcevla_commit": "9b93324e42b3b7708aeeb80df8de37f91163df13",
            "forcevla_v5_config": "forcevla_sfp_all_trimmed_v5",
            "forcevla_v5_repo_id": "tshiamor/aic_gt_sfp_all_trimmed_v5",
        },
        "robot_type": robot_type,
        "fps": fps,
        "camera_mapping": {
            key: {
                "processed_frame_key": spec["frame_key"],
                "processed_camera": spec["processed_camera"],
                "description": spec["description"],
            }
            for key, spec in VIDEO_COLUMNS.items()
        },
        "state_mapping": {
            "observation.state.ee_pos": "processed model_state[0:3] TCP xyz meters",
            "observation.state.ee_quat": "processed model_state[3:6] rotation-vector radians converted to WXYZ quaternion for executable ForceVLA SfpStateTransform",
            "observation.state.gripper_pos": "processed model_state[6] duplicated as [g, g]; synthetic placeholder if source gripper is placeholder",
            "observation.state.wrench": "processed model_state[7:13] copied unchanged; source frame/compensation semantics remain metadata-provenance only",
            "observation.state.joint_pos": "processed model_state[13:19] joint positions radians",
            "observation.state.joint_vel": "processed model_state[19:25] joint velocities radians_per_second",
            "action": "processed measured_action copied row-for-row, including final zero terminal action",
        },
        "timestamp_policy": "LeRobot timestamps are regularized to frame_index / fps for action-horizon loading; processed source timestamps stay in source frames.jsonl.",
        "terminal_action_policy": {
            "final_observation_retained": True,
            "final_zero_terminal_action_retained": True,
            "source_processed_metadata_exporters_must_exclude_terminal_padding_rows_overridden_for_direct_v21_contract": True,
        },
        "guide_conflicts": [
            {
                "topic": "quaternion_order",
                "guide_claim": "GUIDE.md table says observation.state.ee_quat is xyzw",
                "executable_source": "SfpStateTransform reads w=q[0], xyz=q[1:4]; exporter writes WXYZ",
            }
        ],
        "training_config_not_copied_to_dataset": [
            "UR5e robot_type",
            "SFP task text",
            "20 Hz SFP fps",
            "v5 loss_weights=(1,1,1,5,5,5,1)",
            "v4/v5 stabilization trimming policy",
        ],
        "source_episodes": [
            {
                "episode_index": item.episode_index,
                "source_processed_episode": str(episodes[idx].root.resolve()),
                "source_frame_count": len(episodes[idx].frames),
                "exported_frame_count": item.frame_count,
                "task_index": item.task_index,
                "task": episodes[idx].task,
                "source_task_instruction": episodes[idx].metadata.get("task_instruction"),
                "source_fps": episodes[idx].metadata.get("fps"),
                "source_robot_type": episodes[idx].metadata.get("robot_type"),
                "source_quaternion_convention_label": episodes[idx].metadata.get("quaternion_convention"),
                "source_gripper_placeholder": episodes[idx].metadata.get("gripper_state_is_placeholder"),
                "source_wrench_metadata": episodes[idx].metadata.get("wrench_source_metadata"),
                "video_reports": {
                    video_key: {
                        **report,
                        "path": str(_relative_video_path(video_key, item.episode_index)),
                    }
                    for video_key, report in item.video_reports.items()
                },
            }
            for idx, item in enumerate(exported_episodes)
        ],
    }


def _relative_video_path(video_key: str, episode_index: int) -> Path:
    return Path(VIDEO_PATH_TEMPLATE.format(video_key=video_key, episode_index=episode_index))


def export_processed_to_lerobot_v21(
    processed_episode_dirs: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    *,
    task: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Export one or more processed episodes as a local LeRobot v2.1 dataset."""

    if isinstance(processed_episode_dirs, (str, Path)):
        processed_values: list[str | Path] = [processed_episode_dirs]
    else:
        processed_values = list(processed_episode_dirs)
    if not processed_values:
        raise ValueError("at least one processed episode directory is required")

    output_root = Path(output_dir)
    for value in processed_values:
        processed_root = Path(value)
        if _contains_path(processed_root, output_root):
            raise ValueError(f"output directory cannot be inside a processed input episode: {output_root}")
    _safe_output_state(output_root, overwrite=overwrite)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging_root.exists() or staging_root.is_symlink():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    try:
        episodes = [_load_processed_episode(value, task_override=task) for value in processed_values]
        fps_values = {_positive_fps(episode.metadata.get("fps"), f"{episode.root}: fps") for episode in episodes}
        if len(fps_values) != 1:
            raise ValueError(f"all processed episodes must have the same fps, got {sorted(fps_values)!r}")
        fps = next(iter(fps_values))
        robot_types = {episode.metadata.get("robot_type") for episode in episodes}
        if len(robot_types) != 1 or not isinstance(next(iter(robot_types)), str):
            raise ValueError(f"all processed episodes must have the same string robot_type, got {sorted(map(str, robot_types))!r}")
        robot_type = str(next(iter(robot_types)))

        task_to_index: dict[str, int] = {}
        task_records: list[dict[str, Any]] = []
        exported: list[ExportedEpisode] = []
        for episode_index, episode in enumerate(episodes):
            if episode.task not in task_to_index:
                task_to_index[episode.task] = len(task_records)
                task_records.append({"task_index": task_to_index[episode.task], "task": episode.task})
            task_index = task_to_index[episode.task]
            rows = _build_rows(episode, episode_index, task_index, fps)
            parquet_path = _write_parquet_episode(staging_root, episode_index, rows)
            video_paths: dict[str, Path] = {}
            video_reports: dict[str, dict[str, Any]] = {}
            for video_key in VIDEO_COLUMNS:
                output_path = staging_root / "videos" / video_key / f"episode_{episode_index:06d}.mp4"
                report = _encode_video(_image_paths_for_video(episode, video_key), output_path, fps)
                video_paths[video_key] = output_path
                video_reports[video_key] = report
            exported.append(
                ExportedEpisode(
                    episode_index=episode_index,
                    task_index=task_index,
                    source_root=episode.root,
                    frame_count=len(rows),
                    parquet_path=parquet_path,
                    video_paths=video_paths,
                    video_reports=video_reports,
                    stats=_episode_stats(rows),
                )
            )
        _write_metadata(staging_root, episodes, exported, task_records, fps, robot_type, _video_shape_from_metadata(episodes))
        _finalize_staging(staging_root, output_root, overwrite=overwrite)
    except Exception:
        if staging_root.exists() or staging_root.is_symlink():
            shutil.rmtree(staging_root)
        raise

    return output_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export processed JSONL episodes directly to LeRobot v2.1 Parquet/videos.")
    parser.add_argument(
        "--processed",
        required=True,
        action="append",
        help="Processed episode directory. Repeat for multi-episode output.",
    )
    parser.add_argument("--output", required=True, help="Output LeRobot v2.1 dataset directory")
    parser.add_argument("--task", help="Task text override for all input episodes")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args(argv)
    try:
        output = export_processed_to_lerobot_v21(args.processed, args.output, task=args.task, overwrite=args.overwrite)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"OK: wrote LeRobot v2.1 dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
