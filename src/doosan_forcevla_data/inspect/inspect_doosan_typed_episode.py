"""Inspect the final Doosan raw-v1 episode through Patch-2 typed decoders."""

from __future__ import annotations

import argparse
import json
import sys

from doosan_forcevla_data.ingest.doosan_raw_v1 import (
    scan_typed_episode,
)
from doosan_forcevla_data.ingest.ros2_mcap import (
    McapIngestError,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the final two-camera Doosan raw contract and "
            "decode every message into Patch-2 typed records."
        )
    )

    parser.add_argument(
        "episode_dir",
        help="Path to a rosbag2 MCAP episode",
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact JSON instead of indented JSON",
    )

    args = parser.parse_args(argv)

    try:
        summary = scan_typed_episode(args.episode_dir)
    except McapIngestError as exc:
        print(
            f"DOOSAN_TYPED_DECODE_ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    kwargs = {
        "sort_keys": True,
        "allow_nan": False,
    }

    if not args.compact:
        kwargs["indent"] = 2

    print(
        json.dumps(
            summary.to_dict(),
            **kwargs,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
