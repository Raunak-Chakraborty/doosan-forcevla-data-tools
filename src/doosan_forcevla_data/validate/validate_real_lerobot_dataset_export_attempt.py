"""Validate a multi-episode real LeRobot export attempt report and outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_NAME = "export_attempt_report.json"
REQUIRED_REPORT_KEYS = {
    "source_skeleton",
    "output_dir",
    "mode",
    "profile",
    "total_episodes",
    "total_frames",
    "total_tasks",
    "state_dim",
    "action_dim",
    "dependencies",
    "parquet_ready",
    "video_ready",
    "lerobot_api_available",
    "parquet_written",
    "videos_written",
    "metadata_written",
    "per_episode",
    "skipped_reasons",
    "next_recommended_action",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def validate_real_lerobot_dataset_export_attempt(root_dir: str | Path) -> ValidationResult:
    root = Path(root_dir)
    errors: list[str] = []
    warnings: list[str] = []
    report_path = root / REPORT_NAME

    try:
        report = _read_json_object(report_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return ValidationResult(False, [f"could not read {report_path}: {exc}"], warnings)

    missing = sorted(REQUIRED_REPORT_KEYS - set(report))
    for key in missing:
        errors.append(f"{report_path}: missing required key {key}")

    mode = report.get("mode")
    if mode not in {"dry-run", "write-if-available"}:
        errors.append(f"{report_path}: mode must be dry-run or write-if-available")

    total_episodes = report.get("total_episodes")
    total_frames = report.get("total_frames")
    total_tasks = report.get("total_tasks")
    if not _is_int(total_episodes) or total_episodes <= 0:
        errors.append(f"{report_path}: total_episodes must be a positive integer")
        total_episodes = 0
    if not _is_int(total_frames) or total_frames <= 0:
        errors.append(f"{report_path}: total_frames must be a positive integer")
    if not _is_int(total_tasks) or total_tasks <= 0:
        errors.append(f"{report_path}: total_tasks must be a positive integer")

    for key in ["parquet_ready", "video_ready", "lerobot_api_available", "parquet_written", "videos_written", "metadata_written"]:
        if not _is_bool(report.get(key)):
            errors.append(f"{report_path}: {key} must be boolean")

    per_episode = report.get("per_episode")
    if not isinstance(per_episode, list):
        errors.append(f"{report_path}: per_episode must be a list")
        per_episode = []

    if mode == "dry-run":
        if report.get("parquet_written") is not False:
            errors.append(f"{report_path}: dry-run must not write parquet")
        if report.get("videos_written") is not False:
            errors.append(f"{report_path}: dry-run must not write videos")
        if report.get("metadata_written") is not False:
            errors.append(f"{report_path}: dry-run must not write metadata")
        return ValidationResult(not errors, errors, warnings)

    if report.get("metadata_written"):
        for rel in ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl", "meta/episodes_stats.jsonl"]:
            if not (root / rel).is_file():
                errors.append(f"metadata_written is true but missing {rel}")

        try:
            info = _read_json_object(root / "meta" / "info.json")
            tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
            episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
            stats = _read_jsonl(root / "meta" / "episodes_stats.jsonl")
            if _is_int(total_episodes) and len(episodes) != total_episodes:
                errors.append("meta/episodes.jsonl length must match report total_episodes")
            if _is_int(total_tasks) and len(tasks) != total_tasks:
                errors.append("meta/tasks.jsonl length must match report total_tasks")
            if len(stats) != len(episodes):
                errors.append("meta/episodes_stats.jsonl length must match episodes")
            notes = info.get("notes")
            if not isinstance(notes, dict):
                errors.append("meta/info.json notes must be an object")
            else:
                if notes.get("skeleton_only") is not False:
                    errors.append("meta/info.json notes.skeleton_only must be false for real export")
                if notes.get("multi_episode_real_export") is not True:
                    errors.append("meta/info.json notes.multi_episode_real_export must be true")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"could not validate metadata: {exc}")

    if len(per_episode) != total_episodes:
        errors.append("per_episode length must match total_episodes")

    for item in per_episode:
        if not isinstance(item, dict):
            errors.append("per_episode entries must be objects")
            continue
        episode_index = item.get("episode_index")
        if not _is_int(episode_index):
            errors.append("per_episode episode_index must be integer")
            continue

        if item.get("parquet_written"):
            parquet_path = root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
            if not parquet_path.is_file():
                errors.append(f"episode {episode_index}: parquet_written true but missing {parquet_path}")

        if item.get("videos_written"):
            for key in ["observation.image", "observation.wrist_image"]:
                video_path = root / "videos" / key / f"episode_{episode_index:06d}.mp4"
                if not video_path.is_file():
                    errors.append(f"episode {episode_index}: videos_written true but missing {video_path}")

    return ValidationResult(not errors, errors, warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a multi-episode real LeRobot export attempt.")
    parser.add_argument("root")
    args = parser.parse_args(argv)

    result = validate_real_lerobot_dataset_export_attempt(args.root)
    if result.ok:
        print(f"OK: multi-episode real LeRobot export attempt is valid: {args.root}")
        return 0

    print(f"FAILED: multi-episode real LeRobot export attempt is invalid: {args.root}")
    for error in result.errors:
        print(f"ERROR: {error}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
