"""Inspect Patch-5 Doosan force/proprio semantics on one raw episode."""

from __future__ import annotations

import argparse
import json
import math
import sys

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    DoosanForceProprioEpisode,
    ForceProprioError,
    build_doosan_force_proprio_episode,
)
from doosan_forcevla_data.sync.doosan_policy_v1 import DoosanPolicyError


def _range(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ForceProprioError("cannot summarize an empty or non-finite value set")
    return {"min": min(values), "max": max(values)}


def _component_ranges(episode: DoosanForceProprioEpisode) -> dict[str, object]:
    states = [sample.state for sample in episode.samples]
    return {
        "tcp_position_m": [
            _range([state.tcp_position_m[index] for state in states])
            for index in range(3)
        ],
        "tcp_rotvec_rad": [
            _range([state.tcp_rotvec_rad[index] for state in states])
            for index in range(3)
        ],
        "joint_position_rad": [
            _range([state.joint_position_rad[index] for state in states])
            for index in range(6)
        ],
        "joint_velocity_rad_s": [
            _range([state.joint_velocity_rad_s[index] for state in states])
            for index in range(6)
        ],
        "wrench_base_n_nm": [
            _range([state.wrench_base_n_nm[index] for state in states])
            for index in range(6)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Patch-5 force/proprio semantic state from the exact Patch-4 "
            "RobotStateRt selections."
        )
    )
    parser.add_argument("episode_dir")
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="include all synchronized physical semantic samples in JSON",
    )
    args = parser.parse_args(argv)

    try:
        episode = build_doosan_force_proprio_episode(args.episode_dir)
        result = episode.summary_dict()
        result["component_ranges"] = _component_ranges(episode)
        if args.include_samples:
            result["samples"] = [sample.to_dict() for sample in episode.samples]
    except (DoosanPolicyError, ForceProprioError) as exc:
        print(f"DOOSAN_FORCE_PROPRIO_ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
