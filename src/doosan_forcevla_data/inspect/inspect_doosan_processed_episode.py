"""Inspect the Patch-8 production processed episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doosan_forcevla_data.validate.validate_doosan_processed_episode_v1 import (
    validate_doosan_processed_episode_v1,
)


def inspect_doosan_processed_episode(path: str | Path) -> dict[str, object]:
    root = Path(path)
    validation = validate_doosan_processed_episode_v1(root)
    if not validation.ok:
        raise ValueError("processed episode validation failed:\n" + "\n".join(validation.errors))
    metadata = json.loads((root / "metadata_processed.json").read_text(encoding="utf-8"))
    return {
        "schema_version": metadata["schema_version"],
        "frame_count": metadata["frame_count"],
        "synchronized_state_count": metadata["synchronized_state_count"],
        "excluded_terminal_reference_index": metadata["excluded_terminal_reference_index"],
        "state_dim": metadata["state_dim"],
        "action_dim": metadata["action_dim"],
        "task": metadata["task"],
        "physical_camera_count": metadata["physical_camera_count"],
        "cameras": metadata["cameras"],
        "synthetic_model_slot": metadata["synthetic_model_slot"],
        "video_reports": metadata["video_reports"],
        "validation": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Patch-8 production processed episode")
    parser.add_argument("processed_episode")
    args = parser.parse_args(argv)
    try:
        summary = inspect_doosan_processed_episode(args.processed_episode)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
