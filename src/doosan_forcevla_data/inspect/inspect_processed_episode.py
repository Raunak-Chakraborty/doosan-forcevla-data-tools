"""Inspect a validated v0 processed episode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode


WRENCH_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
NONZERO_EPS = 1e-12


def _read_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")
    return data


def _read_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        frame = json.loads(line)
        if not isinstance(frame, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        frames.append(frame)
    return frames


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _resolve_image_path(processed_root: Path, metadata: dict[str, Any], value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = processed_root / path
    if candidate.exists():
        return candidate
    source_raw = metadata.get("source_raw_episode")
    if isinstance(source_raw, str) and source_raw:
        return Path(source_raw) / path
    return candidate


def summarize_processed_episode(processed_episode_dir: str | Path) -> dict[str, Any]:
    """Return a statistics summary for a validated processed episode."""

    root = Path(processed_episode_dir)
    validation = validate_processed_episode(root)
    if not validation.ok:
        message = "processed episode validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in validation.errors
        )
        raise ValueError(message)

    metadata = _read_metadata(root / "metadata_processed.json")
    frames = _read_frames(root / "frames.jsonl")
    frame_count = len(frames)
    fps = float(metadata.get("fps", 0.0))
    nominal_dt_seconds = 1.0 / fps if fps > 0.0 else 0.0
    timestamp_start_seconds = float(frames[0]["timestamp"]) if frames else 0.0
    timestamp_end_seconds = float(frames[-1]["timestamp"]) if frames else 0.0
    timestamp_span_seconds = timestamp_end_seconds - timestamp_start_seconds if frames else 0.0
    nominal_frame_coverage_seconds = frame_count / fps if fps > 0.0 else 0.0

    non_terminal_frames = [frame for frame in frames if not frame["action_is_terminal_padding"]]
    translation_norms = [_norm(frame["measured_action"][:3]) for frame in non_terminal_frames]
    rotation_norms = [_norm(frame["measured_action"][3:6]) for frame in non_terminal_frames]

    nonzero_translation_count = sum(1 for value in translation_norms if value > NONZERO_EPS)
    nonzero_rotation_count = sum(1 for value in rotation_norms if value > NONZERO_EPS)

    max_abs_wrench_values = [0.0] * len(WRENCH_NAMES)
    force_norms: list[float] = []
    torque_norms: list[float] = []
    for frame in frames:
        wrench = [float(value) for value in frame["model_state"][7:13]]
        for idx, value in enumerate(wrench):
            max_abs_wrench_values[idx] = max(max_abs_wrench_values[idx], abs(value))
        force_norms.append(_norm(wrench[:3]))
        torque_norms.append(_norm(wrench[3:6]))

    external_count = 0
    tcp_count = 0
    for frame in frames:
        external_path = _resolve_image_path(root, metadata, frame.get("external_rgb_path"))
        tcp_path = _resolve_image_path(root, metadata, frame.get("tcp_rgb_path"))
        if external_path is not None and external_path.is_file():
            external_count += 1
        if tcp_path is not None and tcp_path.is_file():
            tcp_count += 1

    summary = {
        "episode_path": str(root),
        "frame_count": frame_count,
        "fps": fps,
        "nominal_dt_seconds": nominal_dt_seconds,
        "timestamp_start_seconds": timestamp_start_seconds,
        "timestamp_end_seconds": timestamp_end_seconds,
        "timestamp_span_seconds": timestamp_span_seconds,
        "nominal_frame_coverage_seconds": nominal_frame_coverage_seconds,
        "model_state_dim": metadata.get("model_state_dim"),
        "action_dim": metadata.get("action_dim"),
        "task_instruction": metadata.get("task_instruction"),
        "geometry_type": metadata.get("geometry_type"),
        "orientation_type": metadata.get("orientation_type"),
        "collection_method": metadata.get("collection_method"),
        "success": metadata.get("success"),
        "mean_translation_step_norm": _mean(translation_norms),
        "max_translation_step_norm": max(translation_norms, default=0.0),
        "mean_rotation_step_norm": _mean(rotation_norms),
        "max_rotation_step_norm": max(rotation_norms, default=0.0),
        "nonzero_translation_action_count": nonzero_translation_count,
        "nonzero_rotation_action_count": nonzero_rotation_count,
        "max_abs_wrench": dict(zip(WRENCH_NAMES, max_abs_wrench_values)),
        "max_force_norm": max(force_norms, default=0.0),
        "max_torque_norm": max(torque_norms, default=0.0),
        "external_rgb_files_existing": external_count,
        "tcp_rgb_files_existing": tcp_count,
    }
    return summary


def _format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact human-readable processed episode summary."""

    print("Processed Episode Summary")
    print(f"episode path: {summary['episode_path']}")
    print(f"frame count: {summary['frame_count']}")
    print(f"fps: {_format_float(summary['fps'])}")
    print(f"timestamp start seconds: {_format_float(summary['timestamp_start_seconds'])}")
    print(f"timestamp end seconds: {_format_float(summary['timestamp_end_seconds'])}")
    print(f"timestamp span seconds: {_format_float(summary['timestamp_span_seconds'])}")
    print(f"nominal dt seconds: {_format_float(summary['nominal_dt_seconds'])}")
    print(f"nominal frame coverage seconds: {_format_float(summary['nominal_frame_coverage_seconds'])}")
    print(f"model_state_dim: {summary['model_state_dim']}")
    print(f"action_dim: {summary['action_dim']}")
    print(f"task_instruction: {summary['task_instruction']}")
    print(f"geometry_type: {summary['geometry_type']}")
    print(f"orientation_type: {summary['orientation_type']}")
    print(f"collection_method: {summary['collection_method']}")
    print(f"success: {summary['success']}")

    print("Action Statistics (non-terminal frames)")
    print(f"mean translation step norm: {_format_float(summary['mean_translation_step_norm'])}")
    print(f"max translation step norm: {_format_float(summary['max_translation_step_norm'])}")
    print(f"mean rotation step norm: {_format_float(summary['mean_rotation_step_norm'])}")
    print(f"max rotation step norm: {_format_float(summary['max_rotation_step_norm'])}")
    print(f"nonzero translation actions: {summary['nonzero_translation_action_count']}")
    print(f"nonzero rotation actions: {summary['nonzero_rotation_action_count']}")

    print("Wrench Statistics")
    max_abs_wrench = summary["max_abs_wrench"]
    for name in WRENCH_NAMES:
        print(f"max abs {name}: {_format_float(max_abs_wrench[name])}")
    print(f"max force norm: {_format_float(summary['max_force_norm'])}")
    print(f"max torque norm: {_format_float(summary['max_torque_norm'])}")

    print("Image Stream Status")
    print(f"external_rgb_path files existing: {summary['external_rgb_files_existing']} / {summary['frame_count']}")
    print(f"tcp_rgb_path files existing: {summary['tcp_rgb_files_existing']} / {summary['frame_count']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a v0 processed JSONL episode.")
    parser.add_argument("processed_episode_dir", help="Path to processed episode directory")
    args = parser.parse_args(argv)

    try:
        summary = summarize_processed_episode(args.processed_episode_dir)
    except ValueError as exc:
        print(f"INVALID: processed episode failed inspection: {args.processed_episode_dir}")
        print(str(exc))
        return 1

    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
