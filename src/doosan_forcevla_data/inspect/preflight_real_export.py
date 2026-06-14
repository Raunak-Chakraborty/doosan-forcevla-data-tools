"""Preflight a local LeRobot-style skeleton before real export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doosan_forcevla_data.inspect.check_export_dependencies import (
    check_export_dependencies,
    implemented_video_backend_ready,
)
from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_lerobot_skeleton import validate_lerobot_skeleton


VALID_PROFILES = {"forcevla_13d", "doosan_full_25d"}


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _read_single_jsonl(path: Path, name: str) -> dict[str, Any]:
    records = _read_jsonl(path)
    if len(records) != 1:
        raise ValueError(f"{path}: expected exactly one {name} record, got {len(records)}")
    return records[0]


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


def _feature_shape(info: dict[str, Any], key: str) -> Any:
    features = info.get("features")
    if not isinstance(features, dict):
        return None
    feature = features.get(key)
    if not isinstance(feature, dict):
        return None
    return feature.get("shape")


def _expected_state_dim(profile: Any) -> int | None:
    if profile == "forcevla_13d":
        return 13
    if profile == "doosan_full_25d":
        return MODEL_STATE_DIM
    return None


def _bool_dependency(dependencies: dict[str, dict[str, object]], key: str) -> bool:
    entry = dependencies.get(key)
    return bool(isinstance(entry, dict) and entry.get("available") is True)


def _dependency_summary(dependencies: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "available": bool(entry.get("available")),
            "version": entry.get("version"),
            "detail": str(entry.get("detail", "")),
        }
        for key, entry in dependencies.items()
    }


def _empty_report(root: Path) -> dict[str, Any]:
    return {
        "skeleton_dir": str(root),
        "profile": None,
        "total_frames": 0,
        "state_dim": None,
        "action_dim": None,
        "skeleton_valid": False,
        "schema_valid": False,
        "prompt_task_compatible": False,
        "image_staging_complete": False,
        "image_counts": {"observation.image": 0, "observation.wrist_image": 0},
        "image_file_type_counts": {"symlink": 0, "regular_file": 0},
        "dependency_summary": {},
        "parquet_ready": False,
        "video_ready": False,
        "lerobot_api_available": False,
        "real_export_ready": False,
        "video_export_ready": False,
        "metadata": {},
        "warnings": [],
        "errors": [],
        "next_recommended_action": "Fix skeleton readability or validation errors, then rerun preflight.",
    }


def _check_relative_image(root: Path, frame: dict[str, Any], key: str, errors: list[str]) -> Path | None:
    value = frame.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{key} must be a non-empty relative path string")
        return None
    rel_path = Path(value)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        errors.append(f"{key} must be a safe relative path: {value}")
        return None
    path = root / rel_path
    if not path.is_file():
        errors.append(f"{key} path does not exist under skeleton root: {value}")
        return None
    return path


def preflight_real_export(skeleton_dir: str | Path) -> dict[str, Any]:
    """Inspect a skeleton export and report readiness for future real export."""

    root = Path(skeleton_dir)
    report = _empty_report(root)
    warnings: list[str] = report["warnings"]
    errors: list[str] = report["errors"]

    dependencies = check_export_dependencies()
    report["dependency_summary"] = _dependency_summary(dependencies)
    parquet_ready = _bool_dependency(dependencies, "pyarrow")
    video_ready = implemented_video_backend_ready(dependencies)
    report["parquet_ready"] = parquet_ready
    report["video_ready"] = video_ready
    report["lerobot_api_available"] = _bool_dependency(dependencies, "lerobot")

    warnings.append("Missing laptop dependencies are not final ForceVLA compatibility blockers.")
    warnings.append("Run this same preflight on the lab workstation inside the validated ForceVLA environment.")
    warnings.append("This command does not write parquet or videos.")

    validation = validate_lerobot_skeleton(root)
    report["skeleton_valid"] = validation.ok
    if not validation.ok:
        errors.extend(validation.errors)

    try:
        info = _read_json_object(root / "meta" / "info.json")
        task = _read_single_jsonl(root / "meta" / "tasks.jsonl", "task")
        episode = _read_single_jsonl(root / "meta" / "episodes.jsonl", "episode")
        stats = _read_single_jsonl(root / "meta" / "episodes_stats.jsonl", "episode stats")
        episode_index = episode.get("episode_index")
        if not isinstance(episode_index, int) or isinstance(episode_index, bool):
            episode_index = 0
        frames_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.jsonl"
        frames = _read_jsonl(frames_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"could not read skeleton export: {exc}")
        report["real_export_ready"] = False
        report["video_export_ready"] = False
        return report

    profile = info.get("export_profile", episode.get("export_profile"))
    expected_state_dim = _expected_state_dim(profile)
    state_shape = _feature_shape(info, "observation.state")
    action_shape = _feature_shape(info, "action")
    state_dim = state_shape[0] if isinstance(state_shape, list) and len(state_shape) == 1 else None
    action_dim = action_shape[0] if isinstance(action_shape, list) and len(action_shape) == 1 else None

    report["profile"] = profile
    report["total_frames"] = len(frames)
    report["state_dim"] = state_dim
    report["action_dim"] = action_dim
    report["metadata"] = {
        "codebase_version": info.get("codebase_version"),
        "image_mode": info.get("notes", {}).get("image_mode") if isinstance(info.get("notes"), dict) else None,
        "data_path": info.get("data_path"),
        "video_path": info.get("video_path"),
        "task_index": task.get("task_index"),
        "episode_index": episode.get("episode_index"),
        "episodes_stats_stub": stats.get("stats") == {},
    }

    schema_errors: list[str] = []
    if profile not in VALID_PROFILES:
        schema_errors.append("export_profile must be forcevla_13d or doosan_full_25d")
    if expected_state_dim is not None and state_dim != expected_state_dim:
        schema_errors.append(f"observation.state dim must be {expected_state_dim} for {profile}")
    if action_dim != ACTION_DIM:
        schema_errors.append(f"action dim must be {ACTION_DIM}")
    if _feature_shape(info, "prompt") != [1]:
        schema_errors.append("prompt feature must exist with shape [1]")
    if _feature_shape(info, "task") != [1]:
        schema_errors.append("task feature must exist with shape [1]")

    total_frames = info.get("total_frames")
    if total_frames != len(frames):
        schema_errors.append("info.json total_frames must match JSONL row count")
    if episode.get("length") != len(frames):
        schema_errors.append("episodes.jsonl length must match JSONL row count")

    notes = info.get("notes")
    if info.get("codebase_version") != "v2.1":
        schema_errors.append("info.json codebase_version must be v2.1")
    if not isinstance(notes, dict):
        schema_errors.append("info.json notes must be an object")
    else:
        if notes.get("skeleton_only") is not True:
            schema_errors.append("notes.skeleton_only must be true")
        if notes.get("parquet_written") is not False:
            schema_errors.append("notes.parquet_written must be false")
        if notes.get("videos_encoded") is not False:
            schema_errors.append("notes.videos_encoded must be false")
    data_path = info.get("data_path")
    if not isinstance(data_path, str) or not data_path.endswith(".jsonl"):
        schema_errors.append("data_path must currently end with .jsonl")
    if not isinstance(info.get("video_path"), str) or not info.get("video_path"):
        schema_errors.append("video_path template must exist")

    prompt_task_compatible = True
    image_counts = {"observation.image": 0, "observation.wrist_image": 0}
    file_type_counts = {"symlink": 0, "regular_file": 0}
    image_errors: list[str] = []

    for idx, frame in enumerate(frames):
        if "action_is_terminal_padding" in frame or "terminal_padding" in frame:
            schema_errors.append(f"frame {idx}: terminal padding fields must be absent")
        task_value = frame.get("task")
        prompt_value = frame.get("prompt")
        if not isinstance(task_value, str) or not task_value.strip():
            schema_errors.append(f"frame {idx}: task must be a non-empty string")
            prompt_task_compatible = False
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            schema_errors.append(f"frame {idx}: prompt must be a non-empty string")
            prompt_task_compatible = False
        elif isinstance(task_value, str) and prompt_value != task_value:
            schema_errors.append(f"frame {idx}: prompt must equal task")
            prompt_task_compatible = False

        for image_key in ["observation.image", "observation.wrist_image"]:
            image_path = _check_relative_image(root, frame, image_key, image_errors)
            if image_path is None:
                continue
            image_counts[image_key] += 1
            if image_path.is_symlink():
                file_type_counts["symlink"] += 1
            elif image_path.is_file():
                file_type_counts["regular_file"] += 1

    image_staging_complete = not image_errors and all(count == len(frames) for count in image_counts.values())
    schema_valid = not schema_errors

    report["schema_valid"] = schema_valid
    report["prompt_task_compatible"] = prompt_task_compatible
    report["image_staging_complete"] = image_staging_complete
    report["image_counts"] = image_counts
    report["image_file_type_counts"] = file_type_counts
    errors.extend(schema_errors)
    errors.extend(image_errors)

    structural_ready = bool(report["skeleton_valid"] and schema_valid and image_staging_complete)
    report["real_export_ready"] = structural_ready and parquet_ready
    report["video_export_ready"] = structural_ready and video_ready

    if not structural_ready:
        report["next_recommended_action"] = "Fix skeleton/schema/image staging issues, then rerun preflight."
    elif not parquet_ready:
        report["next_recommended_action"] = (
            "Run this preflight on the lab ForceVLA environment and confirm pyarrow before parquet writing."
        )
    elif not video_ready:
        report["next_recommended_action"] = (
            "Parquet prerequisites appear available; confirm video dependencies before MP4 writing."
        )
    else:
        report["next_recommended_action"] = (
            "Implement real parquet/video export next, after lab ForceVLA environment validation."
        )

    return report


def _status(value: bool) -> str:
    return "yes" if value else "no"


def print_preflight_report(report: dict[str, Any]) -> None:
    """Print a readable real-export preflight report."""

    print("Real Export Preflight")
    print(f"skeleton_dir: {report['skeleton_dir']}")
    print(f"profile: {report['profile']}")
    print(f"total_frames: {report['total_frames']}")
    print(f"state_dim: {report['state_dim']}")
    print(f"action_dim: {report['action_dim']}")
    print("")
    print("Structural Checks")
    print(f"skeleton_valid: {_status(report['skeleton_valid'])}")
    print(f"schema_valid: {_status(report['schema_valid'])}")
    print(f"prompt_task_compatible: {_status(report['prompt_task_compatible'])}")
    print(f"image_staging_complete: {_status(report['image_staging_complete'])}")
    print(f"image_mode: {report['metadata'].get('image_mode')}")
    print(f"observation.image files: {report['image_counts']['observation.image']}")
    print(f"observation.wrist_image files: {report['image_counts']['observation.wrist_image']}")
    print(f"symlinked image files: {report['image_file_type_counts']['symlink']}")
    print(f"regular image files: {report['image_file_type_counts']['regular_file']}")
    print("")
    print("Dependency Readiness")
    print(f"parquet_ready: {_status(report['parquet_ready'])}")
    print(f"video_ready: {_status(report['video_ready'])}")
    print(f"lerobot_api_available: {_status(report['lerobot_api_available'])}")
    print(f"real_export_ready: {_status(report['real_export_ready'])}")
    print(f"video_export_ready: {_status(report['video_export_ready'])}")
    print("")
    print("Notes")
    print("This command does not write parquet or videos.")
    print("Missing laptop dependencies are not final ForceVLA compatibility blockers.")
    print("Run this same preflight on the lab workstation inside the validated ForceVLA environment.")

    if report["warnings"]:
        print("")
        print("Warnings")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
    if report["errors"]:
        print("")
        print("Errors")
        for error in report["errors"]:
            print(f"ERROR: {error}")

    print("")
    print(f"next_recommended_action: {report['next_recommended_action']}")


def write_preflight_report(report: dict[str, Any], output: str | Path) -> Path:
    """Write a JSON preflight report."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight a LeRobot skeleton before real export.")
    parser.add_argument("skeleton_dir", help="Path to local LeRobot-style skeleton export")
    parser.add_argument("--output", help="Optional JSON report output path")
    args = parser.parse_args(argv)

    report = preflight_real_export(args.skeleton_dir)
    if args.output:
        write_preflight_report(report, args.output)
    print_preflight_report(report)
    return 0 if report["skeleton_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
