"""Inspect a direct LeRobot v2.1 ForceVLA-compatible dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doosan_forcevla_data.validate.validate_lerobot_v21 import validate_lerobot_v21


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _import_pyarrow() -> tuple[Any, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


def summarize_lerobot_v21(dataset_dir: str | Path, *, skip_validation: bool = False) -> dict[str, Any]:
    root = Path(dataset_dir)
    if not skip_validation:
        result = validate_lerobot_v21(root)
        if not result.ok:
            raise ValueError("LeRobot v2.1 validation failed:\n" + "\n".join(f"ERROR: {error}" for error in result.errors))
    info = _read_json(root / "meta" / "info.json")
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    provenance = _read_json(root / "meta" / "export_provenance.json")
    _, pq = _import_pyarrow()
    parquet_summaries = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        chunk = episode_index // int(info.get("chunks_size", 1000))
        path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(path)
        parquet_summaries.append(
            {
                "episode_index": episode_index,
                "rows": table.num_rows,
                "columns": table.column_names,
                "schema": str(table.schema),
            }
        )
    return {
        "dataset_path": str(root),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
        "total_videos": info.get("total_videos"),
        "feature_keys": list(info.get("features", {}).keys()) if isinstance(info.get("features"), dict) else [],
        "tasks": tasks,
        "episodes": episodes,
        "parquet": parquet_summaries,
        "camera_mapping": provenance.get("camera_mapping"),
        "state_mapping": provenance.get("state_mapping"),
        "terminal_action_policy": provenance.get("terminal_action_policy"),
        "guide_conflicts": provenance.get("guide_conflicts"),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("LeRobot v2.1 Dataset Summary")
    for key in ["dataset_path", "codebase_version", "robot_type", "fps", "total_episodes", "total_frames", "total_tasks", "total_videos"]:
        print(f"{key}: {summary.get(key)}")
    print("Feature Keys")
    for key in summary["feature_keys"]:
        print(f"- {key}")
    print("Episodes")
    for episode in summary["episodes"]:
        print(f"- episode_index={episode.get('episode_index')} task_index={episode.get('task_index')} length={episode.get('length')}")
    print("Parquet")
    for item in summary["parquet"]:
        print(f"- episode_index={item['episode_index']} rows={item['rows']} columns={len(item['columns'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a direct LeRobot v2.1 ForceVLA-compatible dataset.")
    parser.add_argument("dataset_dir", help="LeRobot v2.1 dataset directory")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    parser.add_argument("--skip-validation", action="store_true", help="Read metadata without running strict validation first")
    args = parser.parse_args(argv)
    try:
        summary = summarize_lerobot_v21(args.dataset_dir, skip_validation=args.skip_validation)
    except (ValueError, OSError, ImportError) as exc:
        print(f"FAILED: {exc}")
        return 1
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
