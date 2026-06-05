"""Create a dry-run LeRobot / ForceVLA export manifest from v0 processed JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode


VALID_PROFILES = {"forcevla_13d", "doosan_full_25d"}


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _read_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        frame = json.loads(line)
        if not isinstance(frame, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        frames.append(frame)
    return frames


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


def _state_dim_for_profile(profile: str) -> int:
    if profile == "forcevla_13d":
        return 13
    if profile == "doosan_full_25d":
        return MODEL_STATE_DIM
    raise ValueError(f"unsupported export profile: {profile}")


def _state_for_profile(model_state: list[float], profile: str) -> list[float]:
    if profile == "forcevla_13d":
        return model_state[:13]
    if profile == "doosan_full_25d":
        return model_state
    raise ValueError(f"unsupported export profile: {profile}")


def build_lerobot_export_plan(processed_episode_dir: str | Path, profile: str) -> dict[str, Any]:
    """Build a dry-run export manifest without writing training data."""

    if profile not in VALID_PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(VALID_PROFILES))}")

    root = Path(processed_episode_dir)
    validation = validate_processed_episode(root)
    if not validation.ok:
        message = "processed episode validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in validation.errors
        )
        raise ValueError(message)

    metadata = _read_json_object(root / "metadata_processed.json")
    frames = _read_frames(root / "frames.jsonl")
    exported_frames = [frame for frame in frames if not frame["action_is_terminal_padding"]]
    excluded_count = len(frames) - len(exported_frames)
    observation_state_dim = _state_dim_for_profile(profile)

    image_count = 0
    wrist_image_count = 0
    for frame in exported_frames:
        image_path = _resolve_image_path(root, metadata, frame.get("external_rgb_path"))
        wrist_image_path = _resolve_image_path(root, metadata, frame.get("tcp_rgb_path"))
        if image_path is not None and image_path.is_file():
            image_count += 1
        if wrist_image_path is not None and wrist_image_path.is_file():
            wrist_image_count += 1

    first_preview: dict[str, Any] | None = None
    if exported_frames:
        first = exported_frames[0]
        first_state = _state_for_profile(first["model_state"], profile)
        first_preview = {
            "frame_index": first["frame_index"],
            "timestamp": first["timestamp"],
            "observation_state_length": len(first_state),
            "action_length": len(first["measured_action"]),
            "task": metadata.get("task_instruction"),
        }

    return {
        "source_processed_episode": str(root.resolve()),
        "profile": profile,
        "dataset_name": metadata.get("dataset_name"),
        "robot_type": metadata.get("robot_type"),
        "fps": metadata.get("fps"),
        "task_instruction": metadata.get("task_instruction"),
        "geometry_type": metadata.get("geometry_type"),
        "orientation_type": metadata.get("orientation_type"),
        "input_frame_count": len(frames),
        "exported_frame_count": len(exported_frames),
        "excluded_terminal_padding_frame_count": excluded_count,
        "terminal_padding_excluded": True,
        "lerobot_like_keys": {
            "observation.image": "external_rgb_path",
            "observation.wrist_image": "tcp_rgb_path",
            "observation.state": "model_state[:13]" if profile == "forcevla_13d" else "model_state",
            "action": "measured_action",
            "task": "task_instruction",
        },
        "observation_state_dim": observation_state_dim,
        "action_dim": ACTION_DIM,
        "image_streams": {
            "observation.image": {"source_key": "external_rgb_path"},
            "observation.wrist_image": {"source_key": "tcp_rgb_path"},
        },
        "image_availability": {
            "observation.image": {"existing_count": image_count},
            "observation.wrist_image": {"existing_count": wrist_image_count},
        },
        "first_exported_record_preview": first_preview,
        "notes": [
            "dry run only",
            "no parquet written",
            "no videos encoded",
            "no Hugging Face upload",
        ],
    }


def write_lerobot_export_plan(
    processed_episode_dir: str | Path,
    profile: str,
    output: str | Path,
) -> Path:
    """Write a dry-run export manifest and return its path."""

    manifest = build_lerobot_export_plan(processed_episode_dir, profile)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


def _print_summary(manifest: dict[str, Any], output_path: Path) -> None:
    print("Dry-Run LeRobot Export Plan")
    print(f"output: {output_path}")
    print(f"source processed episode: {manifest['source_processed_episode']}")
    print(f"profile: {manifest['profile']}")
    print(f"dataset_name: {manifest['dataset_name']}")
    print(f"robot_type: {manifest['robot_type']}")
    print(f"fps: {manifest['fps']}")
    print(f"input frames: {manifest['input_frame_count']}")
    print(f"exported frames: {manifest['exported_frame_count']}")
    print(f"excluded terminal padding frames: {manifest['excluded_terminal_padding_frame_count']}")
    print(f"observation.state dim: {manifest['observation_state_dim']}")
    print(f"action dim: {manifest['action_dim']}")
    image_count = manifest["image_availability"]["observation.image"]["existing_count"]
    wrist_count = manifest["image_availability"]["observation.wrist_image"]["existing_count"]
    exported_count = manifest["exported_frame_count"]
    print(f"observation.image files existing: {image_count} / {exported_count}")
    print(f"observation.wrist_image files existing: {wrist_count} / {exported_count}")
    print("notes: dry run only; no parquet written; no videos encoded; no Hugging Face upload")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a dry-run LeRobot / ForceVLA export manifest.")
    parser.add_argument("--processed", required=True, help="Processed episode directory")
    parser.add_argument("--profile", required=True, choices=sorted(VALID_PROFILES))
    parser.add_argument("--output", required=True, help="Output JSON manifest path")
    args = parser.parse_args(argv)

    try:
        output_path = write_lerobot_export_plan(args.processed, args.profile, args.output)
        manifest = _read_json_object(output_path)
    except ValueError as exc:
        print(f"FAILED: could not plan export for processed episode: {args.processed}")
        print(str(exc))
        return 1

    _print_summary(manifest, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
