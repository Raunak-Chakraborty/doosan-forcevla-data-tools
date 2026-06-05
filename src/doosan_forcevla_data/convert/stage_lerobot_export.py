"""Stage a dry-run LeRobot / ForceVLA export as inspectable JSONL records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.plan_lerobot_export import VALID_PROFILES
from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_export_plan import validate_export_plan
from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode


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


def _resolve_image_path(processed_root: Path, metadata: dict[str, Any], value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty image path string")

    path = Path(value)
    if path.is_absolute():
        if path.is_file():
            return path.resolve()
        raise ValueError(f"{key} path does not exist: {path}")

    processed_candidate = processed_root / path
    if processed_candidate.is_file():
        return processed_candidate.resolve()

    source_raw = metadata.get("source_raw_episode")
    if isinstance(source_raw, str) and source_raw:
        raw_candidate = Path(source_raw) / path
        if raw_candidate.is_file():
            return raw_candidate.resolve()

    raise ValueError(
        f"{key} path could not be resolved: {value!r}; tried relative to processed episode"
        " and source_raw_episode"
    )


def _state_for_profile(model_state: list[float], profile: str) -> list[float]:
    if profile == "forcevla_13d":
        return model_state[:13]
    if profile == "doosan_full_25d":
        return model_state
    raise ValueError(f"unsupported export profile: {profile}")


def _state_dim_for_profile(profile: str) -> int:
    if profile == "forcevla_13d":
        return 13
    if profile == "doosan_full_25d":
        return MODEL_STATE_DIM
    raise ValueError(f"unsupported export profile: {profile}")


def _check_finite_vector(values: list[Any], expected_len: int, name: str) -> list[float]:
    if len(values) != expected_len:
        raise ValueError(f"{name} length must be {expected_len}, got {len(values)}")
    floats: list[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}[{idx}] must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{idx}] must be finite")
        floats.append(number)
    return floats


def stage_lerobot_export(
    processed_episode_dir: str | Path,
    export_plan: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create an inspectable staged JSONL export from a dry-run export plan."""

    processed_root = Path(processed_episode_dir)
    export_plan_path = Path(export_plan)
    output_root = Path(output_dir)

    processed_validation = validate_processed_episode(processed_root)
    if not processed_validation.ok:
        message = "processed episode validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in processed_validation.errors
        )
        raise ValueError(message)

    plan_validation = validate_export_plan(export_plan_path)
    if not plan_validation.ok:
        message = "export plan validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in plan_validation.errors
        )
        raise ValueError(message)

    metadata = _read_json_object(processed_root / "metadata_processed.json")
    frames = _read_frames(processed_root / "frames.jsonl")
    plan = _read_json_object(export_plan_path)
    profile = str(plan["profile"])
    if profile not in VALID_PROFILES:
        raise ValueError(f"unsupported export profile: {profile}")

    source_from_plan = Path(str(plan["source_processed_episode"])).resolve()
    if source_from_plan != processed_root.resolve():
        raise ValueError(
            "export plan source_processed_episode does not match --processed: "
            f"{source_from_plan} != {processed_root.resolve()}"
        )

    expected_state_dim = _state_dim_for_profile(profile)
    exported_frames = [frame for frame in frames if not frame["action_is_terminal_padding"]]
    if len(exported_frames) != plan["exported_frame_count"]:
        raise ValueError("exported frame count does not match export plan")

    output_root.mkdir(parents=True, exist_ok=True)

    staged_metadata = {
        "source_processed_episode": str(processed_root.resolve()),
        "source_export_plan": str(export_plan_path.resolve()),
        "profile": profile,
        "dataset_name": metadata.get("dataset_name"),
        "robot_type": metadata.get("robot_type"),
        "fps": metadata.get("fps"),
        "task_instruction": metadata.get("task_instruction"),
        "geometry_type": metadata.get("geometry_type"),
        "orientation_type": metadata.get("orientation_type"),
        "exported_frame_count": len(exported_frames),
        "observation_state_dim": expected_state_dim,
        "action_dim": ACTION_DIM,
        "terminal_padding_excluded": True,
        "notes": [
            "staging dry run only",
            "no parquet written",
            "no videos encoded",
            "images are referenced, not copied",
            "no Hugging Face upload",
        ],
    }
    (output_root / "metadata_staged.json").write_text(
        json.dumps(staged_metadata, indent=2) + "\n", encoding="utf-8"
    )

    image_refs_checked = 0
    with (output_root / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for frame in exported_frames:
            state = _check_finite_vector(
                _state_for_profile(frame["model_state"], profile), expected_state_dim, "observation.state"
            )
            action = _check_finite_vector(frame["measured_action"], ACTION_DIM, "action")
            image_path = _resolve_image_path(
                processed_root, metadata, frame.get("external_rgb_path"), "observation.image"
            )
            wrist_image_path = _resolve_image_path(
                processed_root, metadata, frame.get("tcp_rgb_path"), "observation.wrist_image"
            )
            image_refs_checked += 2

            staged_frame = {
                "frame_index": frame["frame_index"],
                "timestamp": frame["timestamp"],
                "observation.image": str(image_path),
                "observation.wrist_image": str(wrist_image_path),
                "observation.state": state,
                "action": action,
                "task": metadata.get("task_instruction"),
            }
            handle.write(json.dumps(staged_frame, separators=(",", ":")) + "\n")

    return output_root


def _read_staged_metadata(output_root: Path) -> dict[str, Any]:
    return _read_json_object(output_root / "metadata_staged.json")


def _print_summary(output_root: Path) -> None:
    metadata = _read_staged_metadata(output_root)
    exported_frames = int(metadata["exported_frame_count"])
    print("Staged LeRobot / ForceVLA Export Dry Run")
    print(f"profile: {metadata['profile']}")
    print(f"output path: {output_root}")
    print(f"exported frames: {exported_frames}")
    print(f"observation.state dim: {metadata['observation_state_dim']}")
    print(f"action dim: {metadata['action_dim']}")
    print(f"image references checked: {exported_frames * 2}")
    print("notes: staging only; no parquet written; no videos encoded; images referenced, not copied")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage a dry-run LeRobot / ForceVLA export as JSONL.")
    parser.add_argument("--processed", required=True, help="Processed episode directory")
    parser.add_argument("--export-plan", required=True, help="Dry-run export plan JSON path")
    parser.add_argument("--output", required=True, help="Staged export output directory")
    args = parser.parse_args(argv)

    try:
        output_root = stage_lerobot_export(args.processed, args.export_plan, args.output)
    except ValueError as exc:
        print(f"FAILED: could not stage export: {args.output}")
        print(str(exc))
        return 1

    _print_summary(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
