"""Inspect Patch-6 binary SCHUNK semantics on one raw Doosan episode."""

from __future__ import annotations

import argparse
import json
import sys

from doosan_forcevla_data.convert.doosan_gripper_semantics_v1 import (
    GripperSemanticsError,
    build_doosan_gripper_episode,
)
from doosan_forcevla_data.sync.doosan_policy_v1 import DoosanPolicyError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Patch-6 held/released SCHUNK semantics using the exact Patch-4 "
            "gripper selections."
        )
    )
    parser.add_argument("episode_dir")
    parser.add_argument("--include-samples", action="store_true")
    args = parser.parse_args(argv)

    try:
        episode = build_doosan_gripper_episode(args.episode_dir)
        payload = episode.summary_dict()
        if args.include_samples:
            payload["samples"] = [sample.to_dict() for sample in episode.samples]
        json.dump(payload, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
        return 0
    except (GripperSemanticsError, DoosanPolicyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
