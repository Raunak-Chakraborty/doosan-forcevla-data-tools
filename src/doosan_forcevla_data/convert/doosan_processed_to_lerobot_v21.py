"""Patch-8 export from the production processed episode to pinned LeRobot v2.1.

The exporter is intentionally independent of the legacy three-camera processed
writer.  It targets the exact LeRobot checkout pinned by the frozen ForceVLA
thesis repository and preserves the Patch-8 two-camera contract:

* ``observation.images.external_camera_2`` is the physical D435I video;
* ``observation.images.tcp_camera`` is the physical D405 video;
* no third physical image feature is exported;
* ``observation.state`` is the exact 25D Patch-5/6 state;
* ``action`` is the exact 7D Patch-7 measured action;
* the terminal synchronized reference without a measured action is absent;
* the episode task string is copied exactly from acquisition metadata.

PyArrow is imported lazily so the ROS/Jazzy environment can still import the
package.  The actual LeRobot export is expected to run in the pinned ForceVLA
Python environment, where PyArrow is available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    OBSERVATION_STATE_DIM,
    OBSERVATION_STATE_FIELDS,
)
from doosan_forcevla_data.convert.doosan_measured_action_v1 import (
    ACTION_DIM,
    ACTION_FIELDS,
)
from doosan_forcevla_data.convert.doosan_processed_episode_v1 import (
    CAMERA_SPECS,
    EXTERNAL_CAMERA_KEY,
    FPS,
    PROCESSED_SCHEMA_ID,
    ROBOT_TYPE,
    TCP_CAMERA_KEY,
    VIDEO_RELATIVE_PATHS,
    _probe_video,
)
from doosan_forcevla_data.validate.validate_doosan_processed_episode_v1 import (
    validate_doosan_processed_episode_v1,
)


EXPORT_SCHEMA_ID = "doosan_forcevla_lerobot_v21_export_v1"
LEROBOT_CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

TCP_VIDEO_KEY = "observation.images.tcp_camera"
EXTERNAL_VIDEO_KEY = "observation.images.external_camera_2"
VIDEO_FEATURE_TO_CAMERA = {
    TCP_VIDEO_KEY: TCP_CAMERA_KEY,
    EXTERNAL_VIDEO_KEY: EXTERNAL_CAMERA_KEY,
}

EXPECTED_FORCEVLA_COMMIT = "9b61abef116f207d587d10aaf30170b73757c3e0"
EXPECTED_LEROBOT_COMMIT = "e7aea92dd833f83d163820dcf2e58250307697a4"
EXPECTED_DLIMP_COMMIT = "5edaa4691567873d495633f2708982b42edf1972"


class LeRobotExportError(ValueError):
    """Raised when a production Patch-8 LeRobot export is ambiguous."""


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "PyArrow is required for Patch-8 LeRobot export; run the exporter "
            "inside the frozen ForceVLA environment."
        ) from exc
    return pa, pq


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeRobotExportError(f"{path}: could not read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LeRobotExportError(f"{path}: expected JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LeRobotExportError(f"{path}: could not read JSONL: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise LeRobotExportError(f"{path}: empty line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LeRobotExportError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise LeRobotExportError(f"{path}: line {line_number} is not an object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _finite_vector(value: Any, expected: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise LeRobotExportError(f"{context}: expected list length {expected}")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise LeRobotExportError(f"{context}[{index}]: expected finite number")
        converted = float(item)
        if not math.isfinite(converted):
            raise LeRobotExportError(f"{context}[{index}]: value is not finite")
        result.append(converted)
    return result


def _feature_stats(vectors: Sequence[Sequence[float]]) -> dict[str, list[float] | list[int]]:
    if not vectors:
        raise LeRobotExportError("cannot compute statistics for an empty feature")
    width = len(vectors[0])
    if width <= 0 or any(len(row) != width for row in vectors):
        raise LeRobotExportError("statistics input has inconsistent vector widths")

    count = len(vectors)
    minimum: list[float] = []
    maximum: list[float] = []
    mean: list[float] = []
    std: list[float] = []

    for column in range(width):
        values = [float(row[column]) for row in vectors]
        if not all(math.isfinite(value) for value in values):
            raise LeRobotExportError("statistics input contains non-finite values")
        mu = sum(values) / count
        variance = sum((value - mu) ** 2 for value in values) / count
        minimum.append(min(values))
        maximum.append(max(values))
        mean.append(mu)
        std.append(math.sqrt(max(variance, 0.0)))

    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "count": [count],
    }


def _scalar_stats(values: Sequence[int | float]) -> dict[str, list[float] | list[int]]:
    return _feature_stats([[float(value)] for value in values])


def _features() -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float64",
            "shape": [OBSERVATION_STATE_DIM],
            "names": list(OBSERVATION_STATE_FIELDS),
        },
        "action": {
            "dtype": "float64",
            "shape": [ACTION_DIM],
            "names": list(ACTION_FIELDS),
        },
    }
    for video_key, camera in VIDEO_FEATURE_TO_CAMERA.items():
        spec = CAMERA_SPECS[camera]
        features[video_key] = {
            "dtype": "video",
            "shape": [int(spec["height"]), int(spec["width"]), 3],
            "names": ["height", "width", "channels"],
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def _dataset_rows(processed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(processed_rows):
        if row.get("frame_index") != index or row.get("reference_index") != index:
            raise LeRobotExportError(f"processed row {index}: non-contiguous source/reference index")
        timestamp = row.get("lerobot_timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise LeRobotExportError(f"processed row {index}: invalid LeRobot timestamp")
        expected_timestamp = index / FPS
        if abs(float(timestamp) - expected_timestamp) > 1e-12:
            raise LeRobotExportError(f"processed row {index}: timestamp is not frame_index/{FPS}")
        result.append(
            {
                "observation.state": _finite_vector(
                    row.get("observation_state_25d"), OBSERVATION_STATE_DIM, f"row {index} state"
                ),
                "action": _finite_vector(row.get("action_7d"), ACTION_DIM, f"row {index} action"),
                "timestamp": float(expected_timestamp),
                "frame_index": index,
                "episode_index": 0,
                "index": index,
                "task_index": 0,
            }
        )
    return result


def _write_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    pa, pq = _import_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "observation.state": pa.array(
            [row["observation.state"] for row in rows],
            type=pa.list_(pa.float64()),
        ),
        "action": pa.array([row["action"] for row in rows], type=pa.list_(pa.float64())),
        "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float32()),
        "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
        "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
        "index": pa.array([row["index"] for row in rows], type=pa.int64()),
        "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
    }
    table = pa.table(columns)
    pq.write_table(table, path, compression="zstd")


def _copy_video(processed_root: Path, staging: Path, *, video_key: str, camera: str, frame_count: int) -> dict[str, Any]:
    source = processed_root / VIDEO_RELATIVE_PATHS[camera]
    if not source.is_file():
        raise LeRobotExportError(f"missing processed native video: {source}")
    relative = Path(
        VIDEO_PATH_TEMPLATE.format(
            episode_chunk=0,
            video_key=video_key,
            episode_index=0,
        )
    )
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    report = _probe_video(destination, camera=camera, expected_frames=frame_count).to_dict()
    report["dataset_relative_path"] = str(relative)
    report["physical_camera"] = True
    return report


def export_doosan_processed_to_lerobot_v21(
    processed_episode_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export one production Patch-8 processed episode as one LeRobot v2.1 episode."""

    processed_root = Path(processed_episode_dir).resolve()
    output = Path(output_dir).resolve()
    if not processed_root.is_dir():
        raise LeRobotExportError(f"processed episode does not exist: {processed_root}")
    if output == processed_root or processed_root in output.parents:
        raise LeRobotExportError("LeRobot output cannot be the processed episode or live inside it")

    validation = validate_doosan_processed_episode_v1(processed_root)
    if not validation.ok:
        raise LeRobotExportError(
            "processed episode validation failed:\n" + "\n".join(validation.errors)
        )

    metadata = _read_json(processed_root / "metadata_processed.json")
    if metadata.get("schema_version") != PROCESSED_SCHEMA_ID:
        raise LeRobotExportError("processed schema is not the production Patch-8 schema")
    if metadata.get("robot_type") != ROBOT_TYPE:
        raise LeRobotExportError(f"robot_type must be {ROBOT_TYPE!r}")
    task = metadata.get("task")
    if not isinstance(task, str) or not task.strip():
        raise LeRobotExportError("processed task must be a non-empty exact source string")
    task = task.strip()

    processed_rows = _read_jsonl(processed_root / "frames.jsonl")
    rows = _dataset_rows(processed_rows)
    frame_count = len(rows)
    if frame_count <= 0:
        raise LeRobotExportError("cannot export an empty processed episode")
    if metadata.get("frame_count") != frame_count:
        raise LeRobotExportError("processed frame_count changed during export")
    if metadata.get("excluded_terminal_reference_index") != frame_count:
        raise LeRobotExportError("terminal reference policy does not match N action-bearing rows")

    if output.exists() or output.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        else:
            staging.unlink()
    staging.mkdir(parents=True)

    try:
        parquet_relative = Path(
            DATA_PATH_TEMPLATE.format(episode_chunk=0, episode_index=0)
        )
        _write_parquet(staging / parquet_relative, rows)

        video_reports = {
            video_key: _copy_video(
                processed_root,
                staging,
                video_key=video_key,
                camera=camera,
                frame_count=frame_count,
            )
            for video_key, camera in VIDEO_FEATURE_TO_CAMERA.items()
        }

        info = {
            "codebase_version": LEROBOT_CODEBASE_VERSION,
            "robot_type": ROBOT_TYPE,
            "total_episodes": 1,
            "total_frames": frame_count,
            "total_tasks": 1,
            "total_videos": 2,
            "total_chunks": 1,
            "chunks_size": CHUNKS_SIZE,
            "fps": FPS,
            "splits": {"train": "0:1"},
            "data_path": DATA_PATH_TEMPLATE,
            "video_path": VIDEO_PATH_TEMPLATE,
            "features": _features(),
        }
        _write_json(staging / "meta" / "info.json", info)
        _write_jsonl(staging / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
        _write_jsonl(
            staging / "meta" / "episodes.jsonl",
            [{"episode_index": 0, "tasks": [task], "length": frame_count}],
        )

        states = [row["observation.state"] for row in rows]
        actions = [row["action"] for row in rows]
        timestamps = [row["timestamp"] for row in rows]
        frame_indices = [row["frame_index"] for row in rows]
        episode_indices = [row["episode_index"] for row in rows]
        indices = [row["index"] for row in rows]
        task_indices = [row["task_index"] for row in rows]

        episode_stats = {
            "observation.state": _feature_stats(states),
            "action": _feature_stats(actions),
            "timestamp": _scalar_stats(timestamps),
            "frame_index": _scalar_stats(frame_indices),
            "episode_index": _scalar_stats(episode_indices),
            "index": _scalar_stats(indices),
            "task_index": _scalar_stats(task_indices),
        }
        _write_jsonl(
            staging / "meta" / "episodes_stats.jsonl",
            [{"episode_index": 0, "stats": episode_stats}],
        )

        provenance = {
            "schema_version": EXPORT_SCHEMA_ID,
            "source_processed_schema": PROCESSED_SCHEMA_ID,
            "source_processed_episode": str(processed_root),
            "source_raw_episode": metadata.get("source_raw_episode"),
            "source_episode_index": metadata.get("source_episode_index"),
            "task": task,
            "target_lerobot_codebase_version": LEROBOT_CODEBASE_VERSION,
            "target_forcevla_commit": EXPECTED_FORCEVLA_COMMIT,
            "target_lerobot_commit": EXPECTED_LEROBOT_COMMIT,
            "target_dlimp_commit": EXPECTED_DLIMP_COMMIT,
            "row_policy": metadata.get("row_policy"),
            "frame_count": frame_count,
            "state_dim": OBSERVATION_STATE_DIM,
            "action_dim": ACTION_DIM,
            "terminal_policy": {
                "synchronized_state_count": metadata.get("synchronized_state_count"),
                "excluded_terminal_reference_index": metadata.get("excluded_terminal_reference_index"),
                "terminal_action_emitted": False,
                "synthetic_terminal_zero_action": False,
            },
            "physical_camera_count": 2,
            "camera_mapping": {
                TCP_VIDEO_KEY: {
                    "source_camera": TCP_CAMERA_KEY,
                    "native_shape_hwc": [480, 640, 3],
                    "forcevla_slot": "left_wrist_0_rgb",
                },
                EXTERNAL_VIDEO_KEY: {
                    "source_camera": EXTERNAL_CAMERA_KEY,
                    "native_shape_hwc": [480, 848, 3],
                    "forcevla_slot": "base_0_rgb",
                },
            },
            "synthetic_right_wrist": {
                "dataset_feature_present": False,
                "constructed_in_forcevla_adapter": True,
                "forcevla_slot": "right_wrist_0_rgb",
                "image_mask": False,
            },
            "video_reports": video_reports,
            "timestamp_policy": metadata.get("lerobot_timestamp_policy"),
            "original_ros_timestamps_retained_in": "source processed frames.jsonl",
        }
        _write_json(staging / "meta" / "export_provenance.json", provenance)

        if output.exists() or output.is_symlink():
            if output.is_dir() and not output.is_symlink():
                shutil.rmtree(output)
            else:
                output.unlink()
        staging.replace(output)
    except Exception:
        if staging.exists() or staging.is_symlink():
            shutil.rmtree(staging)
        raise

    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export one production Patch-8 processed episode to the pinned LeRobot v2.1 layout."
    )
    parser.add_argument("processed_episode")
    parser.add_argument("output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_doosan_processed_to_lerobot_v21(
            args.processed_episode,
            args.output,
            overwrite=args.overwrite,
        )
    except (LeRobotExportError, RuntimeError, OSError, FileExistsError) as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"OK: wrote Patch-8 LeRobot v2.1 dataset: {result}")
    return 0


__all__ = [
    "EXPORT_SCHEMA_ID",
    "EXTERNAL_VIDEO_KEY",
    "TCP_VIDEO_KEY",
    "VIDEO_FEATURE_TO_CAMERA",
    "export_doosan_processed_to_lerobot_v21",
]


if __name__ == "__main__":
    raise SystemExit(main())
