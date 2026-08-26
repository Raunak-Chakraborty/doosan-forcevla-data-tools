"""Patch-8 production processed episode for the two-camera Doosan dataset.

This module is the ROS/Jazzy side of Patch 8.  It consumes the frozen raw MCAP
contract and the semantic outputs from Patches 5--7, then writes one compact
processed episode consisting of:

* ``metadata_processed.json``
* ``frames.jsonl`` with exactly the action-bearing source observations
* one native-resolution video for each of the two physical cameras

The terminal synchronized observation is deliberately excluded because Patch 7
does not fabricate an action for it.  No third physical camera is generated and
no image is cropped, stretched, or resized during conversion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

from doosan_forcevla_data.convert.doosan_force_proprio_v1 import (
    OBSERVATION_STATE_DIM,
    OBSERVATION_STATE_FIELDS,
    build_doosan_force_proprio_episode,
)
from doosan_forcevla_data.convert.doosan_gripper_semantics_v1 import (
    assemble_forcevla_v2_observation_states,
    build_doosan_gripper_episode,
)
from doosan_forcevla_data.convert.doosan_measured_action_v1 import (
    ACTION_DIM,
    ACTION_FIELDS,
    build_doosan_measured_action_episode,
)
from doosan_forcevla_data.ingest.doosan_raw_v1 import (
    EXTERNAL_IMAGE_TOPIC,
    TCP_IMAGE_TOPIC,
    ImageRecord,
    iter_typed_messages,
)
from doosan_forcevla_data.sync.doosan_policy_v1 import build_doosan_sync_plan


PROCESSED_SCHEMA_ID = "doosan_forcevla_processed_episode_v1"
FORCEVLA_CONTRACT_ID = "doosan_forcevla_dataset_contract_v2"
FPS = 30
ROBOT_TYPE = "doosan_m1013"
REFERENCE_TOPIC = TCP_IMAGE_TOPIC

TCP_CAMERA_KEY = "tcp_camera"
EXTERNAL_CAMERA_KEY = "external_camera_2"

CAMERA_SPECS = {
    TCP_CAMERA_KEY: {
        "topic": TCP_IMAGE_TOPIC,
        "width": 640,
        "height": 480,
        "encoding": "rgb8",
        "forcevla_slot": "left_wrist_0_rgb",
    },
    EXTERNAL_CAMERA_KEY: {
        "topic": EXTERNAL_IMAGE_TOPIC,
        "width": 848,
        "height": 480,
        "encoding": "rgb8",
        "forcevla_slot": "base_0_rgb",
    },
}

SYNTHETIC_MODEL_SLOT = {
    "slot": "right_wrist_0_rgb",
    "construction": "all-zero image inside DoosanForcevlaInputs",
    "image_mask": False,
    "physical_camera": False,
}

VIDEO_RELATIVE_PATHS = {
    TCP_CAMERA_KEY: Path("videos") / "tcp_camera.mp4",
    EXTERNAL_CAMERA_KEY: Path("videos") / "external_camera_2.mp4",
}


class ProcessedEpisodeError(ValueError):
    """Raised when the production Patch-8 processed contract is violated."""


@dataclass(frozen=True)
class VideoReport:
    camera: str
    relative_path: str
    width: int
    height: int
    fps: float
    decoded_frames: int
    codec_name: str | None
    pixel_format: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "camera": self.camera,
            "relative_path": self.relative_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "decoded_frames": self.decoded_frames,
            "codec_name": self.codec_name,
            "pixel_format": self.pixel_format,
        }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessedEpisodeError(f"{path}: could not read JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise ProcessedEpisodeError(f"{path}: expected JSON object")
    return data


def _episode_task(episode_dir: Path) -> str:
    operator = _read_json_object(episode_dir / "episode_operator.json")
    validation = _read_json_object(episode_dir / "episode_validation.json")

    operator_task = operator.get("task")
    validation_task = (
        validation.get("metadata", {})
        .get("custom_data", {})
        .get("task")
    )

    if not isinstance(operator_task, str) or not operator_task.strip():
        raise ProcessedEpisodeError("episode_operator.json task must be a non-empty string")
    if validation_task != operator_task:
        raise ProcessedEpisodeError(
            "episode task provenance mismatch between episode_operator.json and "
            "episode_validation.json"
        )
    return operator_task.strip()


def _episode_index(episode_dir: Path) -> int:
    validation = _read_json_object(episode_dir / "episode_validation.json")
    value = validation.get("metadata", {}).get("custom_data", {}).get("episode_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProcessedEpisodeError("episode_validation metadata episode_index must be non-negative int")
    return value


def _parse_ratio(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0.0:
                return None
            return float(numerator) / denominator_value
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _run_checked(command: list[str], *, context: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{context}: required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{context}: command failed with exit code {exc.returncode}{suffix}") from exc


def _probe_video(path: Path, *, camera: str, expected_frames: int) -> VideoReport:
    spec = CAMERA_SPECS[camera]
    completed = _run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        context=f"probing {path}",
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"probing {path}: ffprobe did not return JSON") from exc
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise RuntimeError(f"probing {path}: expected exactly one video stream")
    stream = streams[0]
    frame_text = stream.get("nb_read_frames") or stream.get("nb_frames")
    decoded_frames = int(frame_text) if isinstance(frame_text, str) and frame_text.isdigit() else None
    fps = _parse_ratio(stream.get("avg_frame_rate")) or _parse_ratio(stream.get("r_frame_rate"))
    width = stream.get("width")
    height = stream.get("height")

    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"{path}: decoded frame count {decoded_frames!r} does not equal {expected_frames}"
        )
    if width != spec["width"] or height != spec["height"]:
        raise RuntimeError(
            f"{path}: video shape {(width, height)!r} does not equal native "
            f"{(spec['width'], spec['height'])!r}"
        )
    if fps is None or abs(float(fps) - FPS) > 1e-6:
        raise RuntimeError(f"{path}: video fps {fps!r} does not equal {FPS}")

    return VideoReport(
        camera=camera,
        relative_path=str(VIDEO_RELATIVE_PATHS[camera]),
        width=int(width),
        height=int(height),
        fps=float(fps),
        decoded_frames=decoded_frames,
        codec_name=stream.get("codec_name") if isinstance(stream.get("codec_name"), str) else None,
        pixel_format=stream.get("pix_fmt") if isinstance(stream.get("pix_fmt"), str) else None,
    )


def _ffmpeg_process(path: Path, *, width: int, height: int) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "mpeg4",
        "-q:v",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-f",
        "mp4",
        str(path),
    ]
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to build the Patch-8 processed episode") from exc


def _rgb_payload(record: ImageRecord, *, camera: str) -> bytes:
    spec = CAMERA_SPECS[camera]
    if record.width != spec["width"] or record.height != spec["height"]:
        raise ProcessedEpisodeError(
            f"{camera}: image shape {(record.width, record.height)!r} differs from frozen native shape"
        )
    if record.encoding.lower() != "rgb8":
        raise ProcessedEpisodeError(f"{camera}: expected rgb8, got {record.encoding!r}")
    expected_step = record.width * 3
    if record.step != expected_step:
        raise ProcessedEpisodeError(
            f"{camera}: expected tightly packed step {expected_step}, got {record.step}"
        )
    raw = bytes(record.data)
    expected_length = record.step * record.height
    if len(raw) != expected_length:
        raise ProcessedEpisodeError(
            f"{camera}: payload length {len(raw)} does not equal step*height {expected_length}"
        )
    return raw


def _single_nearest_source_index(decision: Any, *, source: str, reference_index: int) -> tuple[int, int]:
    selection = decision.selection
    if selection is None:
        raise ProcessedEpisodeError(
            f"reference {reference_index}: required {source} synchronization selection is missing"
        )
    if selection.alpha is not None or len(selection.source_indices) != 1 or len(selection.source_timestamps_ns) != 1:
        raise ProcessedEpisodeError(
            f"reference {reference_index}: {source} must use one non-interpolated nearest sample"
        )
    return int(selection.source_indices[0]), int(selection.source_timestamps_ns[0])


def build_processed_rows(episode_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the 1008 action-bearing semantic rows without materializing pixels."""

    episode = Path(episode_dir).resolve()
    sync_result = build_doosan_sync_plan(episode)
    force_episode = build_doosan_force_proprio_episode(episode)
    gripper_episode = build_doosan_gripper_episode(episode)
    action_episode = build_doosan_measured_action_episode(force_episode, gripper_episode)
    states = assemble_forcevla_v2_observation_states(force_episode, gripper_episode)

    if sync_result.dropped_reference_count != 0:
        raise ProcessedEpisodeError("Patch 8 v1 requires zero dropped Patch-4 references")
    if len(states) != action_episode.state_count:
        raise ProcessedEpisodeError("Patch-5/6 state count differs from Patch-7 state_count")
    if action_episode.action_count != len(states) - 1:
        raise ProcessedEpisodeError("Patch-7 action geometry is inconsistent")

    external_plan = sync_result.plan.source_plan("external_image")
    if len(external_plan.decisions) != len(states):
        raise ProcessedEpisodeError("external-image decision count differs from synchronized state count")

    rows: list[dict[str, Any]] = []
    for frame_index, action in enumerate(action_episode.actions):
        reference_index = action.source_reference_index
        if reference_index != frame_index:
            raise ProcessedEpisodeError("Patch-8 v1 requires contiguous action-bearing references 0..N-2")
        if action.target_reference_index != reference_index + 1:
            raise ProcessedEpisodeError("Patch-8 action target is not the next synchronized reference")

        force_sample = force_episode.samples[reference_index]
        gripper_sample = gripper_episode.samples[reference_index]
        state = tuple(float(value) for value in states[reference_index])
        action_vector = tuple(float(value) for value in action.to_vector())

        if len(state) != OBSERVATION_STATE_DIM or not all(math.isfinite(value) for value in state):
            raise ProcessedEpisodeError(f"reference {reference_index}: invalid 25D state")
        if len(action_vector) != ACTION_DIM or not all(math.isfinite(value) for value in action_vector):
            raise ProcessedEpisodeError(f"reference {reference_index}: invalid 7D action")
        if force_sample.reference_timestamp_ns != action.source_reference_timestamp_ns:
            raise ProcessedEpisodeError("Patch-5 timestamp differs from Patch-7 action source timestamp")
        if gripper_sample.reference_timestamp_ns != action.source_reference_timestamp_ns:
            raise ProcessedEpisodeError("Patch-6 timestamp differs from Patch-7 action source timestamp")

        external_source_index, external_timestamp_ns = _single_nearest_source_index(
            external_plan.decisions[reference_index],
            source="external_camera_2",
            reference_index=reference_index,
        )

        rows.append(
            {
                "frame_index": frame_index,
                "reference_index": reference_index,
                "reference_timestamp_ns": int(action.source_reference_timestamp_ns),
                "lerobot_timestamp": frame_index / FPS,
                "observation_state_25d": list(state),
                "action_7d": list(action_vector),
                "action_target_reference_index": int(action.target_reference_index),
                "action_target_reference_timestamp_ns": int(action.target_reference_timestamp_ns),
                "cameras": {
                    TCP_CAMERA_KEY: {
                        "source_index": reference_index,
                        "header_timestamp_ns": int(action.source_reference_timestamp_ns),
                    },
                    EXTERNAL_CAMERA_KEY: {
                        "source_index": external_source_index,
                        "header_timestamp_ns": external_timestamp_ns,
                    },
                },
            }
        )

    if len(rows) != action_episode.action_count:
        raise ProcessedEpisodeError("processed row count must equal measured action count")
    if rows and rows[-1]["reference_index"] == action_episode.terminal_reference_index:
        raise ProcessedEpisodeError("terminal reference must not become a training row")

    metadata = {
        "schema_version": PROCESSED_SCHEMA_ID,
        "forcevla_contract_id": FORCEVLA_CONTRACT_ID,
        "source_raw_episode": str(episode),
        "source_episode_index": _episode_index(episode),
        "task": _episode_task(episode),
        "robot_type": ROBOT_TYPE,
        "fps": FPS,
        "reference_topic": REFERENCE_TOPIC,
        "synchronized_state_count": action_episode.state_count,
        "frame_count": len(rows),
        "measured_action_count": action_episode.action_count,
        "excluded_terminal_reference_index": action_episode.terminal_reference_index,
        "terminal_action_emitted": False,
        "state_dim": OBSERVATION_STATE_DIM,
        "state_fields": list(OBSERVATION_STATE_FIELDS),
        "action_dim": ACTION_DIM,
        "action_fields": list(ACTION_FIELDS),
        "row_policy": "source observation t paired with measured Patch-7 action t->t+1",
        "lerobot_timestamp_policy": "frame_index / 30; original ROS reference timestamps retained separately",
        "physical_camera_count": 2,
        "cameras": CAMERA_SPECS,
        "synthetic_model_slot": SYNTHETIC_MODEL_SLOT,
        "camera_calibration": {
            TCP_CAMERA_KEY: sync_result.inputs.tcp_calibration.to_dict(),
            EXTERNAL_CAMERA_KEY: sync_result.inputs.external_calibration.to_dict(),
        },
        "video_paths": {key: str(path) for key, path in VIDEO_RELATIVE_PATHS.items()},
        "speedl_primary_action": False,
        "joy_primary_action": False,
    }
    return rows, metadata


def _materialize_native_videos(
    episode_dir: Path,
    rows: Sequence[dict[str, Any]],
    staging_root: Path,
) -> dict[str, VideoReport]:
    # Keep selections in training-row order.  A nearest-neighbour synchronization
    # policy may legitimately select the same physical frame for two adjacent
    # references, so repeated source indices are supported and are emitted as
    # repeated video frames.
    desired_rows: dict[str, list[tuple[int, int, int]]] = {
        TCP_CAMERA_KEY: [],
        EXTERNAL_CAMERA_KEY: [],
    }
    for row in rows:
        reference_index = int(row["reference_index"])
        cameras = row["cameras"]
        for camera in (TCP_CAMERA_KEY, EXTERNAL_CAMERA_KEY):
            source = cameras[camera]
            desired_rows[camera].append(
                (
                    int(source["source_index"]),
                    reference_index,
                    int(source["header_timestamp_ns"]),
                )
            )

    desired_by_source: dict[str, dict[int, list[tuple[int, int]]]] = {
        TCP_CAMERA_KEY: {},
        EXTERNAL_CAMERA_KEY: {},
    }
    for camera, selections in desired_rows.items():
        source_indices = [item[0] for item in selections]
        if any(b < a for a, b in zip(source_indices[:-1], source_indices[1:], strict=True)):
            raise ProcessedEpisodeError(
                f"{camera}: selected source indices must be non-decreasing in training-row order"
            )
        if len(selections) != len(rows):
            raise ProcessedEpisodeError(f"{camera}: expected one selected image per processed row")
        for source_index, reference_index, header_timestamp_ns in selections:
            desired_by_source[camera].setdefault(source_index, []).append(
                (reference_index, header_timestamp_ns)
            )

    processes: dict[str, subprocess.Popen[bytes]] = {}
    written: dict[str, int] = {TCP_CAMERA_KEY: 0, EXTERNAL_CAMERA_KEY: 0}
    topic_source_index: dict[str, int] = {TCP_IMAGE_TOPIC: 0, EXTERNAL_IMAGE_TOPIC: 0}

    try:
        for camera, spec in CAMERA_SPECS.items():
            processes[camera] = _ffmpeg_process(
                staging_root / VIDEO_RELATIVE_PATHS[camera],
                width=int(spec["width"]),
                height=int(spec["height"]),
            )

        topic_to_camera = {
            TCP_IMAGE_TOPIC: TCP_CAMERA_KEY,
            EXTERNAL_IMAGE_TOPIC: EXTERNAL_CAMERA_KEY,
        }

        for topic, record in iter_typed_messages(episode_dir):
            if topic not in topic_to_camera:
                continue
            if not isinstance(record, ImageRecord):
                raise ProcessedEpisodeError(f"{topic}: expected ImageRecord")

            source_index = topic_source_index[topic]
            topic_source_index[topic] += 1
            camera = topic_to_camera[topic]
            selections = desired_by_source[camera].get(source_index)
            if selections is None:
                continue

            payload = _rgb_payload(record, camera=camera)
            for _, expected_header_timestamp_ns in selections:
                if record.stamp.header_timestamp_ns != expected_header_timestamp_ns:
                    raise ProcessedEpisodeError(
                        f"{camera}: selected source index {source_index} header timestamp mismatch"
                    )

                process = processes[camera]
                if process.stdin is None:  # pragma: no cover - defensive
                    raise RuntimeError(f"{camera}: ffmpeg stdin is unavailable")
                try:
                    process.stdin.write(payload)
                except BrokenPipeError as exc:
                    detail = b""
                    if process.stderr is not None:
                        detail = process.stderr.read()
                        process.stderr.close()
                    raise RuntimeError(
                        f"{camera}: ffmpeg terminated while writing frames: "
                        f"{detail.decode('utf-8', errors='replace').strip()}"
                    ) from exc
                written[camera] += 1

        for camera, process in processes.items():
            if process.stdin is not None:
                process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            if process.stderr is not None:
                process.stderr.close()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"{camera}: ffmpeg exited with {return_code}: "
                    f"{stderr.decode('utf-8', errors='replace').strip()}"
                )

    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.wait()

    for camera in written:
        if written[camera] != len(rows):
            raise ProcessedEpisodeError(
                f"{camera}: wrote {written[camera]} frames, expected {len(rows)}"
            )

    return {
        camera: _probe_video(
            staging_root / VIDEO_RELATIVE_PATHS[camera],
            camera=camera,
            expected_frames=len(rows),
        )
        for camera in CAMERA_SPECS
    }


def build_doosan_processed_episode(
    episode_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Build the production two-camera Patch-8 processed episode."""

    episode = Path(episode_dir).resolve()
    output = Path(output_dir).resolve()
    if not episode.is_dir():
        raise ProcessedEpisodeError(f"raw episode directory does not exist: {episode}")
    if output == episode or episode in output.parents:
        raise ProcessedEpisodeError("processed output cannot be the raw episode or live inside it")

    if output.exists() or output.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")

    rows, metadata = build_processed_rows(episode)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        reports = _materialize_native_videos(episode, rows, staging)
        metadata = {
            **metadata,
            "video_reports": {camera: report.to_dict() for camera, report in reports.items()},
        }
        _write_json(staging / "metadata_processed.json", metadata)
        _write_jsonl(staging / "frames.jsonl", rows)

        if output.exists() or output.is_symlink():
            if output.is_dir() and not output.is_symlink():
                shutil.rmtree(output)
            else:
                output.unlink()
        staging.replace(output)
    except Exception:
        if staging.exists() or staging.is_symlink():
            shutil.rmtree(staging)
        raise

    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Patch-8 two-camera production processed episode from one raw MCAP episode."
    )
    parser.add_argument("episode_dir", help="Validated doosan_two_camera_rosbag_raw_v1 episode")
    parser.add_argument("output_dir", help="Output processed episode directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args(argv)

    try:
        output = build_doosan_processed_episode(
            args.episode_dir,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (ProcessedEpisodeError, RuntimeError, OSError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"OK: wrote Patch-8 processed episode: {output}")
    return 0


__all__ = [
    "CAMERA_SPECS",
    "EXTERNAL_CAMERA_KEY",
    "FORCEVLA_CONTRACT_ID",
    "FPS",
    "PROCESSED_SCHEMA_ID",
    "ProcessedEpisodeError",
    "SYNTHETIC_MODEL_SLOT",
    "TCP_CAMERA_KEY",
    "VIDEO_RELATIVE_PATHS",
    "VideoReport",
    "build_doosan_processed_episode",
    "build_processed_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
