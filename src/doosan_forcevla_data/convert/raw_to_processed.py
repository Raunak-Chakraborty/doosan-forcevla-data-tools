"""Convert a validated v0 raw episode into a simple processed JSONL episode."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from doosan_forcevla_data.convert.compute_actions import (
    compute_measured_tcp_delta_action,
    quat_to_rotvec_xyzw,
)
from doosan_forcevla_data.schema.processed_schema import (
    ACTION_DIM,
    MODEL_STATE_DIM,
    QUATERNION_CONVENTION,
)
from doosan_forcevla_data.validate.validate_raw_episode import validate_raw_episode


DATASET_NAME = "doosan_peg_in_hole_v0"
ROBOT_TYPE = "doosan_m1013"


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: could not read JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"{path}: could not read CSV: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: CSV has no data rows")
    return rows


def _finite_float(row: dict[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except KeyError as exc:
        raise ValueError(f"{source}: missing field {field}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: field {field} is not a float: {row.get(field)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}: field {field} is not finite")
    return value


def _tcp_pos(row: dict[str, str], source: str) -> list[float]:
    return [_finite_float(row, field, source) for field in ["x", "y", "z"]]


def _tcp_quat(row: dict[str, str], source: str) -> list[float]:
    return [_finite_float(row, field, source) for field in ["qx", "qy", "qz", "qw"]]


def _gripper_pos(row: dict[str, str]) -> float:
    if "gripper_pos" not in row or row["gripper_pos"] in (None, ""):
        return 0.0
    value = float(row["gripper_pos"])
    if not math.isfinite(value):
        raise ValueError("tcp_pose.csv: gripper_pos is not finite")
    return value


def _image_path(raw_root: Path, stream: str, frame_idx: int) -> str:
    path = raw_root / "images" / stream / f"{frame_idx:06d}.ppm"
    if not path.is_file():
        raise ValueError(f"{path}: expected image file is missing")
    return str(path.resolve())


def _build_model_state(
    tcp_row: dict[str, str],
    joint_row: dict[str, str],
    wrench_row: dict[str, str],
    row_idx: int,
) -> list[float]:
    tcp_source = f"tcp_pose.csv row {row_idx}"
    joint_source = f"joint_states.csv row {row_idx}"
    wrench_source = f"wrench.csv row {row_idx}"

    ee_pos = _tcp_pos(tcp_row, tcp_source)
    ee_axis_angle = quat_to_rotvec_xyzw(_tcp_quat(tcp_row, tcp_source))
    gripper_pos = [_gripper_pos(tcp_row)]
    wrench = [_finite_float(wrench_row, field, wrench_source) for field in ["fx", "fy", "fz", "tx", "ty", "tz"]]
    joint_pos = [_finite_float(joint_row, f"joint_pos_{idx}", joint_source) for idx in range(6)]
    joint_vel = [_finite_float(joint_row, f"joint_vel_{idx}", joint_source) for idx in range(6)]

    model_state = ee_pos + ee_axis_angle + gripper_pos + wrench + joint_pos + joint_vel
    if len(model_state) != MODEL_STATE_DIM:
        raise ValueError(f"model_state length must be {MODEL_STATE_DIM}, got {len(model_state)}")
    if not all(math.isfinite(value) for value in model_state):
        raise ValueError("model_state contains a non-finite value")
    return model_state


def convert_raw_to_processed(raw_episode_dir: str | Path, output_dir: str | Path) -> Path:
    """Convert a validated raw episode into metadata_processed.json and frames.jsonl."""

    raw_root = Path(raw_episode_dir)
    output_root = Path(output_dir)

    validation = validate_raw_episode(raw_root)
    if not validation.ok:
        message = "raw episode validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in validation.errors
        )
        raise ValueError(message)

    metadata = _read_json_object(raw_root / "metadata.json")
    tcp_rows = _read_csv_rows(raw_root / "robot" / "tcp_pose.csv")
    joint_rows = _read_csv_rows(raw_root / "robot" / "joint_states.csv")
    wrench_rows = _read_csv_rows(raw_root / "force" / "wrench.csv")

    frame_count = len(tcp_rows)
    if frame_count != len(joint_rows) or frame_count != len(wrench_rows):
        raise ValueError(
            "raw CSV row counts must match for v0 frame-index alignment: "
            f"tcp_pose={len(tcp_rows)}, joint_states={len(joint_rows)}, wrench={len(wrench_rows)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    processed_metadata = {
        "source_raw_episode": str(raw_root.resolve()),
        "dataset_name": DATASET_NAME,
        "robot_type": ROBOT_TYPE,
        "fps": metadata.get("fps"),
        "quaternion_convention": QUATERNION_CONVENTION,
        "model_state_dim": MODEL_STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_label_primary": "measured_tcp_delta",
        "frame_count": frame_count,
        "task_instruction": metadata.get("task_instruction"),
        "geometry_type": metadata.get("geometry_type"),
        "orientation_type": metadata.get("orientation_type"),
        "collection_method": metadata.get("collection_method"),
        "success": metadata.get("success"),
        "failure_reason": metadata.get("failure_reason"),
        "notes": "Simple v0 JSONL processed manifest; not yet LeRobot parquet export and not yet ForceVLA training-ready.",
    }

    (output_root / "metadata_processed.json").write_text(
        json.dumps(processed_metadata, indent=2) + "\n", encoding="utf-8"
    )

    with (output_root / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for idx, (tcp_row, joint_row, wrench_row) in enumerate(zip(tcp_rows, joint_rows, wrench_rows)):
            row_idx = idx + 2
            timestamp = _finite_float(tcp_row, "timestamp", f"tcp_pose.csv row {row_idx}")
            model_state = _build_model_state(tcp_row, joint_row, wrench_row, row_idx)

            if idx == frame_count - 1:
                measured_action = [0.0] * ACTION_DIM
                action_is_terminal_padding = True
            else:
                next_tcp_row = tcp_rows[idx + 1]
                measured_action = compute_measured_tcp_delta_action(
                    _tcp_pos(tcp_row, f"tcp_pose.csv row {row_idx}"),
                    _tcp_quat(tcp_row, f"tcp_pose.csv row {row_idx}"),
                    _tcp_pos(next_tcp_row, f"tcp_pose.csv row {row_idx + 1}"),
                    _tcp_quat(next_tcp_row, f"tcp_pose.csv row {row_idx + 1}"),
                    gripper_t=_gripper_pos(tcp_row),
                    gripper_t1=_gripper_pos(next_tcp_row),
                )
                action_is_terminal_padding = False

            if len(measured_action) != ACTION_DIM:
                raise ValueError(f"measured_action length must be {ACTION_DIM}, got {len(measured_action)}")

            frame = {
                "frame_index": idx,
                "timestamp": timestamp,
                "external_rgb_path": _image_path(raw_root, "external_rgb", idx),
                "tcp_rgb_path": _image_path(raw_root, "tcp_rgb", idx),
                "model_state": model_state,
                "measured_action": measured_action,
                "action_is_terminal_padding": action_is_terminal_padding,
            }
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")

    return output_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a v0 raw episode to simple processed JSONL.")
    parser.add_argument("--raw", required=True, help="Raw episode directory")
    parser.add_argument("--output", required=True, help="Processed episode output directory")
    args = parser.parse_args(argv)

    try:
        output_dir = convert_raw_to_processed(args.raw, args.output)
    except ValueError as exc:
        print(f"FAILED: could not convert raw episode: {args.raw}")
        print(str(exc))
        return 1

    print(f"OK: wrote processed episode: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
