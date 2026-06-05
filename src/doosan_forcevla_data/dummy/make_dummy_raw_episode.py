"""Generate a tiny v0 raw episode using only the Python standard library."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from doosan_forcevla_data.schema.raw_schema import RawEpisodeMetadata


DEFAULT_FRAME_COUNT = 20
DEFAULT_FPS = 30


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_ppm(path: Path, frame_idx: int, stream_offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    red = (20 + frame_idx * 7 + stream_offset) % 256
    green = (80 + frame_idx * 3 + stream_offset) % 256
    blue = (140 + frame_idx * 5 + stream_offset) % 256
    pixels = [
        (red, green, blue),
        (red, green // 2, blue),
        (red // 2, green, blue),
        (red, green, blue // 2),
    ]
    body = "\n".join(f"{r} {g} {b}" for r, g, b in pixels)
    path.write_text(f"P3\n2 2\n255\n{body}\n", encoding="ascii")


def make_dummy_raw_episode(
    output: str | Path,
    frame_count: int = DEFAULT_FRAME_COUNT,
    fps: int = DEFAULT_FPS,
) -> Path:
    """Create a tiny deterministic raw episode and return its directory."""

    if frame_count <= 1:
        raise ValueError("frame_count must be greater than 1")
    if fps <= 0:
        raise ValueError("fps must be positive")

    episode_dir = Path(output)
    episode_dir.mkdir(parents=True, exist_ok=True)

    metadata = RawEpisodeMetadata(
        episode_id=episode_dir.name,
        task_instruction="Insert the peg into the hole.",
        geometry_type="round_peg_round_hole",
        orientation_type="vertical_insertion",
        collection_method="dummy",
        action_label_primary="measured_tcp_delta",
        success=True,
        failure_reason=None,
        fps=fps,
        optional_action_streams=["commanded_twist"],
    )
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata.to_json_dict(), indent=2) + "\n", encoding="utf-8"
    )

    tcp_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    wrench_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []

    for idx in range(frame_count):
        timestamp = idx / fps
        fraction = idx / (frame_count - 1)
        angle = 0.08 * fraction
        qz = math.sin(angle / 2.0)
        qw = math.cos(angle / 2.0)

        x = 0.45 + 0.03 * fraction
        y = 0.01 * math.sin(math.pi * fraction)
        z = 0.22 - 0.04 * fraction

        tcp_rows.append(
            {
                "timestamp": f"{timestamp:.9f}",
                "x": f"{x:.9f}",
                "y": f"{y:.9f}",
                "z": f"{z:.9f}",
                "qx": "0.000000000",
                "qy": "0.000000000",
                "qz": f"{qz:.9f}",
                "qw": f"{qw:.9f}",
                "gripper_pos": "0.000000000",
            }
        )

        joint_row: dict[str, object] = {"timestamp": f"{timestamp:.9f}"}
        for joint_idx in range(6):
            joint_row[f"joint_pos_{joint_idx}"] = f"{0.1 * joint_idx + 0.02 * fraction:.9f}"
            joint_row[f"joint_vel_{joint_idx}"] = f"{0.02 / (frame_count / fps):.9f}"
        joint_rows.append(joint_row)

        contact_fraction = max(0.0, (fraction - 0.70) / 0.30)
        wrench_rows.append(
            {
                "timestamp": f"{timestamp:.9f}",
                "fx": f"{0.2 * math.sin(2.0 * math.pi * fraction):.9f}",
                "fy": "0.000000000",
                "fz": f"{12.0 * contact_fraction:.9f}",
                "tx": "0.000000000",
                "ty": f"{0.03 * contact_fraction:.9f}",
                "tz": "0.000000000",
            }
        )

        action_rows.append(
            {
                "timestamp": f"{timestamp:.9f}",
                "vx": "0.030000000",
                "vy": "0.000000000",
                "vz": "-0.040000000",
                "wx": "0.000000000",
                "wy": "0.000000000",
                "wz": "0.080000000",
                "gripper_velocity": "0.000000000",
            }
        )

        _write_ppm(episode_dir / "images" / "external_rgb" / f"{idx:06d}.ppm", idx, 0)
        _write_ppm(episode_dir / "images" / "tcp_rgb" / f"{idx:06d}.ppm", idx, 40)

    _write_csv(
        episode_dir / "robot" / "tcp_pose.csv",
        ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper_pos"],
        tcp_rows,
    )
    _write_csv(
        episode_dir / "robot" / "joint_states.csv",
        ["timestamp"]
        + [f"joint_pos_{idx}" for idx in range(6)]
        + [f"joint_vel_{idx}" for idx in range(6)],
        joint_rows,
    )
    _write_csv(
        episode_dir / "force" / "wrench.csv",
        ["timestamp", "fx", "fy", "fz", "tx", "ty", "tz"],
        wrench_rows,
    )
    _write_csv(
        episode_dir / "actions" / "commanded_twist.csv",
        ["timestamp", "vx", "vy", "vz", "wx", "wy", "wz", "gripper_velocity"],
        action_rows,
    )
    _write_csv(
        episode_dir / "events.csv",
        ["timestamp", "event"],
        [
            {"timestamp": "0.000000000", "event": "episode_start"},
            {"timestamp": f"{(frame_count - 6) / fps:.9f}", "event": "contact_begin"},
            {"timestamp": f"{(frame_count - 1) / fps:.9f}", "event": "success"},
        ],
    )

    return episode_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a tiny dummy v0 raw episode.")
    parser.add_argument("--output", required=True, help="Episode output directory")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args(argv)

    episode_dir = make_dummy_raw_episode(args.output, frame_count=args.frames, fps=args.fps)
    print(f"Wrote dummy raw episode: {episode_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
