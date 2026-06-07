"""Smoke-test ForceVLA-style observation construction from a local LeRobot export.

This intentionally avoids ``LeRobotDataset.__getitem__`` video decoding because
the current lab ForceVLA environment can load LeRobot metadata/dataset objects,
but ``torchvision.io.VideoReader`` is unavailable.  Instead, this smoke test
reads:

* parquet rows with pyarrow
* MP4 frames with PyAV

and verifies that the exported dataset can be converted into the observation
pieces needed by the ForceVLA/OpenPI transform path:

* observation.image
* observation.wrist_image
* observation.state
* action
* prompt

This is a validation/debug tool, not a training dataloader.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CAMERA_KEYS = ["observation.image", "observation.wrist_image"]


def _require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required for this smoke test because it reads LeRobot parquet files. "
            "Run this inside the validated lab ForceVLA environment."
        ) from exc
    return pq


def _require_av():
    try:
        import av
    except Exception as exc:
        raise RuntimeError(
            "PyAV is required for this smoke test because it decodes LeRobot MP4 videos. "
            "Run this inside the validated lab ForceVLA environment."
        ) from exc
    return av


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


def _finite_vector(values: Any, expected_len: int, name: str) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    if len(values) != expected_len:
        raise ValueError(f"{name} length must be {expected_len}, got {len(values)}")

    result: list[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}[{idx}] must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{idx}] must be finite")
        result.append(number)
    return result


def _feature_shape(info: dict[str, Any], key: str) -> list[int] | None:
    features = info.get("features")
    if not isinstance(features, dict):
        return None
    feature = features.get(key)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if isinstance(shape, tuple):
        shape = list(shape)
    if isinstance(shape, list) and all(isinstance(v, int) for v in shape):
        return shape
    return None


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunks_size = info.get("chunks_size", 1000)
    if not isinstance(chunks_size, int) or chunks_size <= 0:
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


def _format_video_path(info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    template = info.get("video_path")
    if not isinstance(template, str) or not template:
        raise ValueError("meta/info.json must contain a non-empty video_path template")
    return Path(
        template.format(
            episode_chunk=_episode_chunk(info, episode_index),
            episode_index=episode_index,
            video_key=video_key,
        )
    )


def _read_parquet_columns(parquet_path: Path) -> dict[str, Any]:
    pq = _require_pyarrow_parquet()
    table = pq.read_table(parquet_path)
    return {
        "table": table,
        "columns": table.to_pydict(),
    }


def _decode_video_and_select_frame(video_path: Path, row_index: int) -> tuple[dict[str, Any], Any]:
    if row_index < 0:
        raise ValueError("row_index must be non-negative")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    selected_frame = None
    selected_shape = None
    selected_dtype = None
    decoded_frames = 0

    av = _require_av()
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream_summary = {
            "stream_frames_metadata": int(stream.frames) if stream.frames is not None else None,
            "average_rate": str(stream.average_rate) if stream.average_rate is not None else None,
            "width": int(stream.width),
            "height": int(stream.height),
        }

        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if decoded_frames == row_index:
                selected_frame = rgb
                selected_shape = list(rgb.shape)
                selected_dtype = str(rgb.dtype)
            decoded_frames += 1
    finally:
        container.close()

    if selected_frame is None:
        raise IndexError(f"{video_path}: row_index {row_index} is outside decoded frame count {decoded_frames}")

    summary = {
        "path": str(video_path),
        "exists": True,
        "size_bytes": video_path.stat().st_size,
        "decoded_frames": decoded_frames,
        "selected_frame_shape": selected_shape,
        "selected_frame_dtype": selected_dtype,
        **stream_summary,
    }
    return summary, selected_frame


def build_smoke_observation(
    dataset_root: str | Path,
    episode_index: int = 0,
    row_index: int = 0,
    expected_state_dim: int = 13,
    expected_action_dim: int = 7,
    strict_video_shape: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one ForceVLA-style observation sample and a JSON-safe report."""

    root = Path(dataset_root)
    errors: list[str] = []
    warnings: list[str] = []

    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"

    info = _read_json_object(info_path)
    tasks = _read_jsonl(tasks_path)
    episodes = _read_jsonl(episodes_path)

    parquet_path = root / _format_data_path(info, episode_index)
    parquet_data = _read_parquet_columns(parquet_path)
    table = parquet_data["table"]
    columns = parquet_data["columns"]

    if table.num_rows <= 0:
        raise ValueError(f"{parquet_path}: parquet table has no rows")
    if row_index < 0 or row_index >= table.num_rows:
        raise IndexError(f"row_index {row_index} outside parquet row count {table.num_rows}")

    required_columns = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
        "index",
        "task",
        "prompt",
    ]
    missing_columns = [key for key in required_columns if key not in columns]
    if missing_columns:
        raise ValueError(f"{parquet_path}: missing parquet columns: {missing_columns}")

    state = _finite_vector(columns["observation.state"][row_index], expected_state_dim, "observation.state")
    action = _finite_vector(columns["action"][row_index], expected_action_dim, "action")
    prompt = columns["prompt"][row_index]
    task = columns["task"][row_index]

    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("prompt must be a non-empty string")
    if not isinstance(task, str) or not task.strip():
        errors.append("task must be a non-empty string")
    if isinstance(prompt, str) and isinstance(task, str) and prompt != task:
        warnings.append("prompt does not equal task")

    video_reports: dict[str, dict[str, Any]] = {}
    video_frames: dict[str, Any] = {}
    for video_key in CAMERA_KEYS:
        video_path = root / _format_video_path(info, episode_index, video_key)
        video_report, frame = _decode_video_and_select_frame(video_path, row_index)
        video_reports[video_key] = video_report
        video_frames[video_key] = frame

        metadata_shape = _feature_shape(info, video_key)
        if metadata_shape is not None:
            video_report["metadata_shape"] = metadata_shape
            selected_shape = video_report.get("selected_frame_shape")
            if selected_shape != metadata_shape:
                message = (
                    f"{video_key}: decoded frame shape {selected_shape} does not match "
                    f"metadata shape {metadata_shape}"
                )
                if strict_video_shape:
                    errors.append(message)
                else:
                    warnings.append(message)

        if video_report["decoded_frames"] != table.num_rows:
            errors.append(
                f"{video_key}: decoded video frame count {video_report['decoded_frames']} "
                f"does not match parquet rows {table.num_rows}"
            )

    observation = {
        "observation.image": video_frames["observation.image"],
        "observation.wrist_image": video_frames["observation.wrist_image"],
        "observation.state": state,
        "action": action,
        "prompt": prompt,
    }

    report = {
        "ok": not errors,
        "dataset_root": str(root.resolve()),
        "episode_index": episode_index,
        "row_index": row_index,
        "parquet_path": str(parquet_path),
        "parquet_rows": table.num_rows,
        "parquet_columns": table.column_names,
        "metadata": {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "data_path": info.get("data_path"),
            "video_path": info.get("video_path"),
        },
        "tasks_count": len(tasks),
        "episodes_count": len(episodes),
        "sample": {
            "timestamp": columns["timestamp"][row_index],
            "frame_index": columns["frame_index"][row_index],
            "episode_index": columns["episode_index"][row_index],
            "task_index": columns["task_index"][row_index],
            "index": columns["index"][row_index],
            "task": task,
            "prompt": prompt,
            "state_dim": len(state),
            "action_dim": len(action),
            "state_first_values": state[: min(6, len(state))],
            "action": action,
        },
        "videos": video_reports,
        "forcevla_observation_summary": {
            "keys": list(observation.keys()),
            "observation.image.shape": list(observation["observation.image"].shape),
            "observation.wrist_image.shape": list(observation["observation.wrist_image"].shape),
            "observation.state.length": len(observation["observation.state"]),
            "action.length": len(observation["action"]),
            "prompt": observation["prompt"],
        },
        "warnings": warnings,
        "errors": errors,
    }

    return observation, report


def smoke_forcevla_observation_builder(
    dataset_root: str | Path,
    episode_index: int = 0,
    row_index: int = 0,
    expected_state_dim: int = 13,
    expected_action_dim: int = 7,
    strict_video_shape: bool = False,
) -> dict[str, Any]:
    """Return only the JSON-safe report for CLI/tests."""

    _, report = build_smoke_observation(
        dataset_root=dataset_root,
        episode_index=episode_index,
        row_index=row_index,
        expected_state_dim=expected_state_dim,
        expected_action_dim=expected_action_dim,
        strict_video_shape=strict_video_shape,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test ForceVLA-style observation construction from local parquet/videos."
    )
    parser.add_argument("dataset_root", help="Local LeRobot-style dataset export root.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--expected-state-dim", type=int, default=13)
    parser.add_argument("--expected-action-dim", type=int, default=7)
    parser.add_argument("--strict-video-shape", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON report output path.")
    args = parser.parse_args(argv)

    report = smoke_forcevla_observation_builder(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        row_index=args.row_index,
        expected_state_dim=args.expected_state_dim,
        expected_action_dim=args.expected_action_dim,
        strict_video_shape=args.strict_video_shape,
    )

    text = json.dumps(report, indent=2)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote smoke report: {output_path}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
