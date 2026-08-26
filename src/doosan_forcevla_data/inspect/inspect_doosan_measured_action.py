"""Inspect Patch-7 measured 7D actions on one raw Doosan episode."""

from __future__ import annotations

import argparse
import json
import sys

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    ForceProprioError,
    build_doosan_force_proprio_episode,
)
from doosan_forcevla_data.convert.doosan_gripper_semantics_v1 import (
    GripperSemanticsError,
    build_doosan_gripper_episode,
)
from doosan_forcevla_data.convert.doosan_measured_action_v1 import (
    MeasuredActionError,
    build_doosan_measured_action_episode,
)
from doosan_forcevla_data.sync.doosan_policy_v1 import DoosanPolicyError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Patch-7 measured 7D actions from the exact Patch-5/Patch-6 "
            "synchronized state sequence."
        )
    )
    parser.add_argument("episode_dir")
    parser.add_argument("--include-actions", action="store_true")
    args = parser.parse_args(argv)

    try:
        force_episode = build_doosan_force_proprio_episode(args.episode_dir)
        gripper_episode = build_doosan_gripper_episode(args.episode_dir)
        action_episode = build_doosan_measured_action_episode(
            force_episode,
            gripper_episode,
        )
        payload = action_episode.summary_dict()
        if args.include_actions:
            payload["actions"] = [action.to_dict() for action in action_episode.actions]
        json.dump(payload, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
        return 0
    except (
        ForceProprioError,
        GripperSemanticsError,
        MeasuredActionError,
        DoosanPolicyError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
