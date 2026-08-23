"""Inspect and fully deserialize a ROS 2 MCAP episode."""

from __future__ import annotations

import argparse
import json
import sys

from doosan_forcevla_data.ingest.ros2_mcap import (
    McapIngestError,
    scan_episode,
)


def main(
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MCAP metadata/sidecars "
            "and deserialize every ROS 2 "
            "message without synchronization "
            "or typed field conversion."
        )
    )

    parser.add_argument(
        "episode_dir",
        help=(
            "Path to a rosbag2 MCAP "
            "episode"
        ),
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "emit compact JSON instead "
            "of indented JSON"
        ),
    )

    args = parser.parse_args(argv)

    try:
        summary = scan_episode(
            args.episode_dir
        )
    except McapIngestError as exc:
        print(
            f"MCAP_INGEST_ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.compact:
        print(
            json.dumps(
                summary.to_dict(),
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                summary.to_dict(),
                indent=2,
                sort_keys=True,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
