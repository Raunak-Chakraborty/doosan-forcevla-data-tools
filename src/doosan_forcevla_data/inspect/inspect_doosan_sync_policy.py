"""Inspect the production Doosan synchronization policy on one raw episode."""

from __future__ import annotations

import argparse
import json
import sys

from doosan_forcevla_data.sync.doosan_policy_v1 import (
    DoosanPolicyError,
    build_doosan_sync_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Patch-4 Doosan synchronization plan and emit "
            "timestamp/provenance JSON without decoding image payloads downstream."
        )
    )
    parser.add_argument("episode_dir")
    parser.add_argument(
        "--include-decisions",
        action="store_true",
        help="include per-frame source-index provenance in JSON",
    )
    args = parser.parse_args(argv)

    try:
        result = build_doosan_sync_plan(args.episode_dir)
    except DoosanPolicyError as exc:
        print(f"DOOSAN_SYNC_POLICY_ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            result.to_dict(
                include_decisions=args.include_decisions,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
