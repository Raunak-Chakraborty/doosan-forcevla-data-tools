"""Inspect a production Patch-8 LeRobot v2.1 export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doosan_forcevla_data.validate.validate_doosan_lerobot_v21 import (
    validate_doosan_lerobot_v21,
)


def inspect_doosan_lerobot_v21(path: str | Path) -> dict[str, object]:
    root = Path(path)
    result = validate_doosan_lerobot_v21(root)
    if not result.ok:
        raise ValueError("LeRobot export validation failed:\n" + "\n".join(result.errors))
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (root / "meta" / "export_provenance.json").read_text(encoding="utf-8")
    )
    tasks = [
        json.loads(line)
        for line in (root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "codebase_version": info["codebase_version"],
        "robot_type": info["robot_type"],
        "fps": info["fps"],
        "frame_count": info["total_frames"],
        "total_videos": info["total_videos"],
        "video_keys": sorted(
            key for key, feature in info["features"].items() if feature["dtype"] == "video"
        ),
        "state_dim": info["features"]["observation.state"]["shape"][0],
        "action_dim": info["features"]["action"]["shape"][0],
        "task": tasks[0]["task"],
        "terminal_policy": provenance["terminal_policy"],
        "synthetic_right_wrist": provenance["synthetic_right_wrist"],
        "validation": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Patch-8 LeRobot v2.1 export")
    parser.add_argument("dataset_root")
    args = parser.parse_args(argv)
    try:
        summary = inspect_doosan_lerobot_v21(args.dataset_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
