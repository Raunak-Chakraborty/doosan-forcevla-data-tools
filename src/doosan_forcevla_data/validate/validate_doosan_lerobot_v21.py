"""Validate the production Patch-8 LeRobot v2.1 export without importing LeRobot."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    OBSERVATION_STATE_DIM,
    OBSERVATION_STATE_FIELDS,
)
from doosan_forcevla_data.convert.doosan_measured_action_v1 import ACTION_DIM, ACTION_FIELDS
from doosan_forcevla_data.convert.doosan_processed_episode_v1 import (
    CAMERA_SPECS,
    EXTERNAL_CAMERA_KEY,
    FPS,
    ROBOT_TYPE,
    TCP_CAMERA_KEY,
    _probe_video,
)
from doosan_forcevla_data.convert.doosan_processed_to_lerobot_v21 import (
    EXTERNAL_VIDEO_KEY,
    EXPORT_SCHEMA_ID,
    LEROBOT_CODEBASE_VERSION,
    TCP_VIDEO_KEY,
    VIDEO_FEATURE_TO_CAMERA,
)


@dataclass(frozen=True)
class LeRobotValidationResult:
    ok: bool
    errors: tuple[str, ...]
    frame_count: int


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}: empty line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: line {line_number} not object")
        rows.append(value)
    return rows


def _import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyArrow is required to validate the Patch-8 LeRobot Parquet") from exc
    return pq


def validate_doosan_lerobot_v21(path: str | Path) -> LeRobotValidationResult:
    root = Path(path)
    errors: list[str] = []
    try:
        info = _read_json(root / "meta" / "info.json")
        provenance = _read_json(root / "meta" / "export_provenance.json")
        tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
        episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
        episodes_stats = _read_jsonl(root / "meta" / "episodes_stats.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return LeRobotValidationResult(False, (str(exc),), 0)

    if info.get("codebase_version") != LEROBOT_CODEBASE_VERSION:
        errors.append("codebase_version must be v2.1")
    if info.get("robot_type") != ROBOT_TYPE:
        errors.append(f"robot_type must be {ROBOT_TYPE}")
    if info.get("fps") != FPS:
        errors.append(f"fps must be {FPS}")
    if info.get("total_episodes") != 1 or info.get("total_tasks") != 1:
        errors.append("Patch-8 export must contain exactly one episode and one task")
    if info.get("total_videos") != 2:
        errors.append("Patch-8 export must contain exactly two physical videos")
    if info.get("total_chunks") != 1 or info.get("chunks_size") != 1000:
        errors.append("LeRobot chunk metadata mismatch")
    if info.get("splits") != {"train": "0:1"}:
        errors.append("LeRobot split metadata mismatch")

    frame_count = info.get("total_frames")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        errors.append("total_frames must be positive int")
        frame_count = 0

    features = info.get("features")
    required_features = {
        "observation.state",
        "action",
        TCP_VIDEO_KEY,
        EXTERNAL_VIDEO_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    if not isinstance(features, dict) or set(features) != required_features:
        errors.append("LeRobot feature set must contain exactly state/action/two videos/default fields")
    else:
        if features["observation.state"] != {
            "dtype": "float64",
            "shape": [OBSERVATION_STATE_DIM],
            "names": list(OBSERVATION_STATE_FIELDS),
        }:
            errors.append("observation.state feature contract mismatch")
        if features["action"] != {
            "dtype": "float64",
            "shape": [ACTION_DIM],
            "names": list(ACTION_FIELDS),
        }:
            errors.append("action feature contract mismatch")
        for video_key, camera in VIDEO_FEATURE_TO_CAMERA.items():
            spec = CAMERA_SPECS[camera]
            expected = {
                "dtype": "video",
                "shape": [int(spec["height"]), int(spec["width"]), 3],
                "names": ["height", "width", "channels"],
            }
            if features[video_key] != expected:
                errors.append(f"video feature contract mismatch for {video_key}")

    if len(tasks) != 1:
        errors.append("tasks.jsonl must contain exactly one task")
        task = None
    else:
        task = tasks[0].get("task")
        if tasks[0].get("task_index") != 0 or set(tasks[0]) != {"task_index", "task"}:
            errors.append("tasks.jsonl structure mismatch")
        if not isinstance(task, str) or not task.strip():
            errors.append("tasks.jsonl must contain one non-empty exact source task")
            task = None

    if episodes != [{"episode_index": 0, "length": frame_count, "tasks": [task]}]:
        errors.append("episodes.jsonl structure/content mismatch")
    if len(episodes_stats) != 1 or episodes_stats[0].get("episode_index") != 0:
        errors.append("episodes_stats.jsonl must contain exactly episode 0")

    if provenance.get("schema_version") != EXPORT_SCHEMA_ID:
        errors.append("export provenance schema mismatch")
    if provenance.get("frame_count") != frame_count:
        errors.append("export provenance frame_count mismatch")
    if provenance.get("physical_camera_count") != 2:
        errors.append("export provenance physical_camera_count must be 2")
    synthetic = provenance.get("synthetic_right_wrist")
    if synthetic != {
        "constructed_in_forcevla_adapter": True,
        "dataset_feature_present": False,
        "forcevla_slot": "right_wrist_0_rgb",
        "image_mask": False,
    }:
        errors.append("synthetic right-wrist provenance mismatch")
    terminal = provenance.get("terminal_policy")
    if not isinstance(terminal, dict) or terminal.get("terminal_action_emitted") is not False or terminal.get("synthetic_terminal_zero_action") is not False:
        errors.append("terminal-action provenance mismatch")

    parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    if not parquet_path.is_file():
        errors.append(f"missing parquet file: {parquet_path}")
    else:
        try:
            pq = _import_pyarrow_parquet()
            table = pq.read_table(parquet_path)
            if table.num_rows != frame_count:
                errors.append("Parquet row count does not equal total_frames")
            expected_columns = {
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            }
            if set(table.column_names) != expected_columns:
                errors.append("Parquet column set mismatch")
            if table.num_rows:
                rows = table.to_pylist()
                for index, row in enumerate(rows):
                    state = row.get("observation.state")
                    action = row.get("action")
                    if not isinstance(state, list) or len(state) != OBSERVATION_STATE_DIM or not all(math.isfinite(float(x)) for x in state):
                        errors.append(f"Parquet row {index}: invalid 25D state")
                        break
                    if not isinstance(action, list) or len(action) != ACTION_DIM or not all(math.isfinite(float(x)) for x in action):
                        errors.append(f"Parquet row {index}: invalid 7D action")
                        break
                    if row.get("frame_index") != index or row.get("index") != index:
                        errors.append(f"Parquet row {index}: frame/global index mismatch")
                        break
                    if row.get("episode_index") != 0 or row.get("task_index") != 0:
                        errors.append(f"Parquet row {index}: episode/task index mismatch")
                        break
                    timestamp = float(row.get("timestamp"))
                    if abs(timestamp - index / FPS) > 2e-6:
                        errors.append(f"Parquet row {index}: timestamp mismatch")
                        break
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            errors.append(str(exc))

    for video_key, camera in VIDEO_FEATURE_TO_CAMERA.items():
        video_path = root / "videos" / "chunk-000" / video_key / "episode_000000.mp4"
        if not video_path.is_file():
            errors.append(f"missing video: {video_path}")
            continue
        try:
            _probe_video(video_path, camera=camera, expected_frames=frame_count)
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))

    return LeRobotValidationResult(not errors, tuple(errors), int(frame_count))


__all__ = ["LeRobotValidationResult", "validate_doosan_lerobot_v21"]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a production Patch-8 Doosan LeRobot v2.1 export."
    )
    parser.add_argument("dataset_root")
    args = parser.parse_args(argv)
    try:
        result = validate_doosan_lerobot_v21(args.dataset_root)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
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
