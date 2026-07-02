"""Validate a simple v0 processed JSONL episode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.processed_schema import ACTION_DIM, MODEL_STATE_DIM
from doosan_forcevla_data.validate.validate_raw_episode import ValidationResult


REQUIRED_METADATA_KEYS = [
    "source_raw_episode",
    "dataset_name",
    "robot_type",
    "fps",
    "quaternion_convention",
    "model_state_dim",
    "action_dim",
    "action_label_primary",
    "frame_count",
    "task_instruction",
    "geometry_type",
    "orientation_type",
    "collection_method",
    "success",
    "failure_reason",
    "notes",
]

REQUIRED_FRAME_KEYS = [
    "frame_index",
    "timestamp",
    "external_rgb_path",
    "tcp_rgb_path",
    "model_state",
    "measured_action",
    "action_is_terminal_padding",
]

STRICT_METADATA_SCHEMA_VERSION = "processed_jsonl_v1"
PROCESSED_CAMERA_SCHEMA_VERSION = "processed_camera_streams_v1"
ROTATION_VECTOR_RADIANS = "rotation_vector_radians"
METER_UNITS = {"m", "meter", "meters", "metre", "metres"}
RADIAN_UNITS = {"rad", "radian", "radians"}
LEGACY_IMAGE_FRAME_KEYS = ["external_rgb_path", "tcp_rgb_path"]
DEFAULT_THREE_CAMERA_NAMES = ["external_camera_1", "external_camera_2", "tcp_camera"]


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read_metadata(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: could not read metadata JSON: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: metadata must be a JSON object")
        return None
    return data


def _read_frames(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: could not read frames JSONL: {exc}")
        return frames

    if not lines:
        errors.append(f"{path}: frames.jsonl has no frame lines")
        return frames

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"{path}: line {line_number} is empty")
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(frame, dict):
            errors.append(f"{path}: line {line_number} must be a JSON object")
            continue
        frames.append(frame)
    return frames


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _check_numeric_vector(
    frame_idx: int,
    frame: dict[str, Any],
    key: str,
    expected_len: int,
    errors: list[str],
) -> None:
    value = frame.get(key)
    if not isinstance(value, list):
        errors.append(f"frame {frame_idx}: {key} must be a list")
        return
    if len(value) != expected_len:
        errors.append(f"frame {frame_idx}: {key} length must be {expected_len}, got {len(value)}")
        return
    for value_idx, item in enumerate(value):
        if not _is_finite_number(item):
            errors.append(f"frame {frame_idx}: {key}[{value_idx}] must be a finite number")


def _requires_new_layout_metadata(metadata: dict[str, Any]) -> bool:
    return metadata.get("processed_metadata_schema_version") == STRICT_METADATA_SCHEMA_VERSION


def _normalized_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace(" ", "_")


def _layout_entries_by_index(layout: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(layout, list):
        return {}
    entries: dict[int, dict[str, Any]] = {}
    for entry in layout:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        entries[index] = entry
    return entries


def _check_layout(
    metadata_path: Path,
    metadata: dict[str, Any],
    key: str,
    expected_dim: int,
    errors: list[str],
) -> None:
    layout = metadata.get(key)
    if layout is None:
        if _requires_new_layout_metadata(metadata):
            errors.append(f"{metadata_path}: {key} is required for {STRICT_METADATA_SCHEMA_VERSION}")
        return
    if not isinstance(layout, list):
        errors.append(f"{metadata_path}: {key} must be a list")
        return
    if len(layout) != expected_dim:
        errors.append(f"{metadata_path}: {key} must contain {expected_dim} entries, got {len(layout)}")

    indexes: list[int] = []
    for entry_idx, entry in enumerate(layout):
        if not isinstance(entry, dict):
            errors.append(f"{metadata_path}: {key}[{entry_idx}] must be a JSON object")
            continue
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            errors.append(f"{metadata_path}: {key}[{entry_idx}].index must be an integer")
        else:
            indexes.append(index)
        for required_key in ["name", "source", "unit"]:
            if not _is_non_empty_string(entry.get(required_key)):
                errors.append(f"{metadata_path}: {key}[{entry_idx}].{required_key} must be a non-empty string")

    if sorted(indexes) != list(range(expected_dim)):
        errors.append(f"{metadata_path}: {key} indexes must be unique and contiguous 0..{expected_dim - 1}")


def _check_output_unit_metadata(metadata_path: Path, metadata: dict[str, Any], errors: list[str]) -> None:
    if not _requires_new_layout_metadata(metadata):
        return
    if metadata.get("tcp_orientation_output_convention") != ROTATION_VECTOR_RADIANS:
        errors.append(f"{metadata_path}: tcp_orientation_output_convention must be {ROTATION_VECTOR_RADIANS!r}")
    if _normalized_text(metadata.get("tcp_orientation_output_unit")) not in RADIAN_UNITS:
        errors.append(f"{metadata_path}: tcp_orientation_output_unit must be radians")
    if _normalized_text(metadata.get("tcp_position_output_unit")) not in METER_UNITS:
        errors.append(f"{metadata_path}: tcp_position_output_unit must be metres")

    model_layout = _layout_entries_by_index(metadata.get("model_state_layout"))
    for index in range(3):
        if index in model_layout and _normalized_text(model_layout[index].get("unit")) not in METER_UNITS:
            errors.append(f"{metadata_path}: model_state_layout[{index}].unit must be metres")
    for index in range(3, 6):
        if index in model_layout and _normalized_text(model_layout[index].get("unit")) not in RADIAN_UNITS:
            errors.append(f"{metadata_path}: model_state_layout[{index}].unit must be radians")

    action_layout = _layout_entries_by_index(metadata.get("measured_action_layout"))
    for index in range(3):
        if index in action_layout and _normalized_text(action_layout[index].get("unit")) not in METER_UNITS:
            errors.append(f"{metadata_path}: measured_action_layout[{index}].unit must be metres")
    for index in range(3, 6):
        if index in action_layout and _normalized_text(action_layout[index].get("unit")) not in RADIAN_UNITS:
            errors.append(f"{metadata_path}: measured_action_layout[{index}].unit must be radians")


def _check_gripper_placeholder_metadata(metadata_path: Path, metadata: dict[str, Any], errors: list[str]) -> None:
    if metadata.get("gripper_state_is_placeholder") is not True:
        return
    for key in ["gripper_state_source", "gripper_state_provenance"]:
        value = metadata.get(key)
        if isinstance(value, str) and "measured" in value.lower():
            errors.append(f"{metadata_path}: placeholder gripper metadata must not label {key} as measured")
    selected = metadata.get("selected_streams")
    if isinstance(selected, dict):
        gripper_value = selected.get("gripper_state")
        if isinstance(gripper_value, str) and "measured" in gripper_value.lower():
            errors.append(f"{metadata_path}: placeholder selected_streams.gripper_state must not be labelled measured")


def _check_terminal_action_policy(
    metadata_path: Path,
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    padding_indices: list[int],
    errors: list[str],
) -> None:
    policy = metadata.get("terminal_action_policy")
    if policy is None:
        if _requires_new_layout_metadata(metadata):
            errors.append(f"{metadata_path}: terminal_action_policy is required for {STRICT_METADATA_SCHEMA_VERSION}")
        return
    if not isinstance(policy, dict):
        errors.append(f"{metadata_path}: terminal_action_policy must be a JSON object")
        return

    if policy.get("final_observation_retained") is not True:
        errors.append(f"{metadata_path}: terminal_action_policy.final_observation_retained must be true")
    if policy.get("final_action_padded") is not True:
        errors.append(f"{metadata_path}: terminal_action_policy.final_action_padded must be true")
    if policy.get("exporters_must_exclude_terminal_padding_rows") is not True:
        errors.append(f"{metadata_path}: terminal_action_policy.exporters_must_exclude_terminal_padding_rows must be true")
    if policy.get("padding_count") != len(padding_indices):
        errors.append(
            f"{metadata_path}: terminal_action_policy.padding_count {policy.get('padding_count')!r} "
            f"does not match row padding count {len(padding_indices)}"
        )
    if policy.get("terminal_padding_frame_indices") != padding_indices:
        errors.append(
            f"{metadata_path}: terminal_action_policy.terminal_padding_frame_indices must be {padding_indices!r}"
        )

    padding_value = policy.get("padding_value")
    if not (
        isinstance(padding_value, list)
        and len(padding_value) == ACTION_DIM
        and all(_is_finite_number(value) and abs(float(value)) <= 1e-12 for value in padding_value)
    ):
        errors.append(f"{metadata_path}: terminal_action_policy.padding_value must be {ACTION_DIM} finite zeros")
    elif frames and frames[-1].get("measured_action") != padding_value:
        errors.append("final frame: measured_action must equal terminal_action_policy.padding_value")


def _resolve_image_path(processed_root: Path, metadata: dict[str, Any], value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = processed_root / path
    if candidate.exists():
        return candidate
    source_raw = metadata.get("source_raw_episode")
    if isinstance(source_raw, str) and source_raw:
        return Path(source_raw) / path
    return candidate


def _is_path_under(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _decode_image_shape(path: Path) -> tuple[int, int, int] | str:
    try:
        from PIL import Image
    except ImportError:
        return "Pillow/PIL is not available; install Pillow to verify processed camera images"

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            channels = len(image.getbands())
    except Exception as exc:
        return f"Pillow/PIL could not decode image: {exc}"
    return int(width), int(height), int(channels)


def _read_raw_camera_index(index_path: Path, errors: list[str]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{index_path}: could not read raw camera index for source verification: {exc}")
        return records

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{index_path}: line {line_number} is invalid JSON for source verification: {exc}")
            continue
        if not isinstance(record, dict):
            errors.append(f"{index_path}: line {line_number} must be a JSON object for source verification")
            continue
        record_index = record.get("record_index")
        if isinstance(record_index, bool) or not isinstance(record_index, int):
            errors.append(f"{index_path}: line {line_number} record_index must be an integer for source verification")
            continue
        if record_index in records:
            errors.append(f"{index_path}: duplicate record_index {record_index} for source verification")
            continue
        records[record_index] = record
    return records


def _metadata_raw_stream_path(metadata: dict[str, Any], raw_stream: str) -> Path:
    raw_camera_streams = metadata.get("raw_camera_streams")
    if isinstance(raw_camera_streams, dict):
        entry = raw_camera_streams.get(raw_stream)
        if isinstance(entry, dict) and isinstance(entry.get("path"), str) and entry["path"]:
            return Path(entry["path"])
    return Path("streams") / raw_stream


def _source_raw_root(metadata: dict[str, Any]) -> Path | None:
    source_raw = metadata.get("source_raw_episode")
    if not isinstance(source_raw, str) or not source_raw:
        return None
    return Path(source_raw)


def _legacy_camera_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    camera_mapping = metadata.get("camera_mapping") if isinstance(metadata.get("camera_mapping"), dict) else {}
    external_raw_stream = None
    tcp_raw_stream = None
    if isinstance(camera_mapping, dict):
        external = camera_mapping.get("external_rgb_path")
        tcp = camera_mapping.get("tcp_rgb_path")
        if isinstance(external, dict):
            external_raw_stream = external.get("raw_stream")
        if isinstance(tcp, dict):
            tcp_raw_stream = tcp.get("raw_stream")
    return [
        {
            "camera_name": external_raw_stream or "external_camera",
            "raw_stream": external_raw_stream or "external_camera",
            "frame_key": "external_rgb_path",
        },
        {
            "camera_name": tcp_raw_stream or "tcp_camera",
            "raw_stream": tcp_raw_stream or "tcp_camera",
            "frame_key": "tcp_rgb_path",
        },
    ]


def _check_processed_camera_metadata(
    metadata_path: Path,
    metadata: dict[str, Any],
    errors: list[str],
) -> list[dict[str, Any]]:
    if metadata.get("processed_camera_schema_version") is None:
        return _legacy_camera_entries(metadata)

    if metadata.get("processed_camera_schema_version") != PROCESSED_CAMERA_SCHEMA_VERSION:
        errors.append(
            f"{metadata_path}: processed_camera_schema_version must be {PROCESSED_CAMERA_SCHEMA_VERSION!r}"
        )

    processed_camera_streams = metadata.get("processed_camera_streams")
    if not isinstance(processed_camera_streams, dict) or not processed_camera_streams:
        errors.append(f"{metadata_path}: processed_camera_streams must be a non-empty JSON object")
        return []

    names = metadata.get("processed_camera_names")
    if not isinstance(names, list) or not all(_is_non_empty_string(name) for name in names):
        errors.append(f"{metadata_path}: processed_camera_names must be a non-empty list of strings")
        names = list(processed_camera_streams)

    frame_keys = metadata.get("processed_camera_frame_keys")
    if not isinstance(frame_keys, list) or not all(_is_non_empty_string(key) for key in frame_keys):
        errors.append(f"{metadata_path}: processed_camera_frame_keys must be a non-empty list of strings")

    camera_count = metadata.get("processed_camera_count")
    if camera_count != len(names):
        errors.append(f"{metadata_path}: processed_camera_count must equal processed_camera_names length {len(names)}")

    raw_camera_streams = metadata.get("raw_camera_streams")
    if isinstance(raw_camera_streams, dict) and all(name in raw_camera_streams for name in DEFAULT_THREE_CAMERA_NAMES):
        missing = [name for name in DEFAULT_THREE_CAMERA_NAMES if name not in names]
        if missing:
            errors.append(
                f"{metadata_path}: default three-camera raw episode requires processed cameras "
                f"{DEFAULT_THREE_CAMERA_NAMES!r}; missing {missing!r}"
            )

    entries: list[dict[str, Any]] = []
    seen_frame_keys: set[str] = set()
    for name in names:
        entry = processed_camera_streams.get(name)
        if not isinstance(entry, dict):
            errors.append(f"{metadata_path}: processed_camera_streams.{name} must be a JSON object")
            continue
        frame_key = entry.get("frame_key")
        raw_stream = entry.get("raw_stream")
        if not _is_non_empty_string(frame_key):
            errors.append(f"{metadata_path}: processed_camera_streams.{name}.frame_key must be a non-empty string")
            continue
        if not _is_non_empty_string(raw_stream):
            errors.append(f"{metadata_path}: processed_camera_streams.{name}.raw_stream must be a non-empty string")
            continue
        if raw_stream != name:
            errors.append(f"{metadata_path}: processed_camera_streams.{name}.raw_stream must equal {name!r}")
        if name == "external_camera_2" and frame_key != "external_camera_2_rgb_path":
            errors.append(
                f"{metadata_path}: external_camera_2 must use frame_key 'external_camera_2_rgb_path'"
            )
        if frame_key in seen_frame_keys:
            errors.append(f"{metadata_path}: duplicate processed camera frame_key {frame_key!r}")
        seen_frame_keys.add(str(frame_key))

        camera_mapping = metadata.get("camera_mapping")
        if isinstance(camera_mapping, dict):
            mapping = camera_mapping.get(frame_key)
            if not isinstance(mapping, dict) or mapping.get("raw_stream") != raw_stream:
                errors.append(
                    f"{metadata_path}: camera_mapping.{frame_key}.raw_stream must match processed camera raw_stream {raw_stream!r}"
                )
        entries.append(entry)

    return entries


def _check_processed_camera_counts(
    root: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    camera_entries: list[dict[str, Any]],
    errors: list[str],
) -> None:
    for entry in camera_entries:
        camera_name = entry.get("camera_name", entry.get("raw_stream"))
        frame_count = entry.get("frame_count")
        image_count = entry.get("image_count")
        if frame_count is not None and frame_count != len(frames):
            errors.append(f"{metadata_path}: processed_camera_streams.{camera_name}.frame_count must be {len(frames)}")
        if image_count is not None and image_count != len(frames):
            errors.append(f"{metadata_path}: processed_camera_streams.{camera_name}.image_count must be {len(frames)}")

        output_dir = entry.get("output_dir")
        if isinstance(output_dir, str) and output_dir:
            relative = Path(output_dir)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{metadata_path}: processed_camera_streams.{camera_name}.output_dir must be safe and relative")
                continue
            directory = root / relative
            if not directory.is_dir():
                errors.append(f"{directory}: processed camera output directory is missing")
                continue
            files = [path for path in directory.iterdir() if path.is_file()]
            if len(files) != len(frames):
                errors.append(
                    f"{directory}: expected {len(frames)} image files for {camera_name}, got {len(files)}"
                )


def _check_processed_camera_source_mappings(
    root: Path,
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    camera_entries: list[dict[str, Any]],
    errors: list[str],
) -> None:
    source_raw_root = _source_raw_root(metadata)
    if source_raw_root is None or not source_raw_root.is_dir():
        return

    for entry in camera_entries:
        frame_key = entry.get("frame_key")
        raw_stream = entry.get("raw_stream")
        camera_name = entry.get("camera_name", raw_stream)
        if not isinstance(frame_key, str) or not isinstance(raw_stream, str):
            continue

        raw_stream_relative = _metadata_raw_stream_path(metadata, raw_stream)
        if raw_stream_relative.is_absolute() or ".." in raw_stream_relative.parts:
            errors.append(f"processed camera {camera_name}: raw stream path must be safe and relative")
            continue
        raw_stream_dir = source_raw_root / raw_stream_relative
        raw_index = _read_raw_camera_index(raw_stream_dir / "index.jsonl", errors)
        if not raw_index:
            continue

        seen_source_paths: set[str] = set()
        allow_reuse = entry.get("allow_reused_source_images") is True
        for row_idx, frame in enumerate(frames):
            record_index = frame.get("frame_index")
            if isinstance(record_index, bool) or not isinstance(record_index, int):
                continue
            raw_record = raw_index.get(record_index)
            if not isinstance(raw_record, dict):
                errors.append(
                    f"frame {row_idx}: processed camera {camera_name} has no raw {raw_stream} record_index {record_index}"
                )
                continue
            source_value = raw_record.get("image_path")
            if not isinstance(source_value, str) or not source_value:
                errors.append(f"frame {row_idx}: raw {raw_stream} image_path must be a non-empty string")
                continue
            source_relative = Path(source_value)
            if source_relative.is_absolute() or ".." in source_relative.parts:
                errors.append(f"frame {row_idx}: raw {raw_stream} image_path must be safe and relative")
                continue
            if source_value in seen_source_paths and not allow_reuse:
                errors.append(
                    f"frame {row_idx}: raw {raw_stream} image_path {source_value!r} is reused without explicit allowance"
                )
            seen_source_paths.add(source_value)

            source_path = source_raw_root / source_relative
            if not _is_path_under(source_raw_root, source_path) or not _is_path_under(raw_stream_dir, source_path):
                errors.append(f"frame {row_idx}: raw {raw_stream} image_path must stay inside its camera stream directory")
                continue
            if not source_path.is_file():
                errors.append(f"frame {row_idx}: raw {raw_stream} image_path does not exist: {source_path}")
                continue

            processed_path = _resolve_image_path(root, metadata, frame.get(frame_key))
            if processed_path is None or not processed_path.is_file():
                continue
            try:
                same_file = processed_path.resolve() == source_path.resolve()
            except OSError:
                same_file = False
            if same_file:
                continue
            try:
                if processed_path.read_bytes() != source_path.read_bytes():
                    errors.append(
                        f"frame {row_idx}: {frame_key} bytes do not match raw {raw_stream} image_path {source_value!r}"
                    )
                    break
            except OSError as exc:
                errors.append(f"frame {row_idx}: could not compare {frame_key} to raw source bytes: {exc}")
                break


def validate_processed_episode(processed_episode_dir: str | Path) -> ValidationResult:
    """Validate a processed episode directory and return clear error messages."""

    root = Path(processed_episode_dir)
    errors: list[str] = []

    if not root.exists():
        return ValidationResult(False, [f"{root}: processed episode directory does not exist"])
    if not root.is_dir():
        return ValidationResult(False, [f"{root}: processed episode path is not a directory"])

    metadata_path = root / "metadata_processed.json"
    frames_path = root / "frames.jsonl"
    if not metadata_path.is_file():
        errors.append(f"{metadata_path}: required file is missing")
    if not frames_path.is_file():
        errors.append(f"{frames_path}: required file is missing")
    if errors:
        return ValidationResult(False, errors)

    metadata = _read_metadata(metadata_path, errors)
    frames = _read_frames(frames_path, errors)
    if metadata is None:
        return ValidationResult(False, errors)

    missing_metadata = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing_metadata:
        errors.append(f"{metadata_path}: missing required keys: {', '.join(missing_metadata)}")

    if metadata.get("model_state_dim") != MODEL_STATE_DIM:
        errors.append(f"{metadata_path}: model_state_dim must be {MODEL_STATE_DIM}")
    if metadata.get("action_dim") != ACTION_DIM:
        errors.append(f"{metadata_path}: action_dim must be {ACTION_DIM}")
    declared_model_state_dim = metadata.get("model_state_dim") if metadata.get("model_state_dim") == MODEL_STATE_DIM else MODEL_STATE_DIM
    declared_action_dim = metadata.get("action_dim") if metadata.get("action_dim") == ACTION_DIM else ACTION_DIM
    _check_layout(metadata_path, metadata, "model_state_layout", MODEL_STATE_DIM, errors)
    _check_layout(metadata_path, metadata, "measured_action_layout", ACTION_DIM, errors)
    _check_output_unit_metadata(metadata_path, metadata, errors)
    _check_gripper_placeholder_metadata(metadata_path, metadata, errors)
    camera_entries = _check_processed_camera_metadata(metadata_path, metadata, errors)

    frame_count = metadata.get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool):
        errors.append(f"{metadata_path}: frame_count must be an integer")
    elif frame_count != len(frames):
        errors.append(
            f"{metadata_path}: frame_count {frame_count} does not match frames.jsonl lines {len(frames)}"
        )

    previous_timestamp: float | None = None
    padding_indices: list[int] = []
    image_shapes_by_key: dict[str, set[tuple[int, int, int]]] = {
        str(entry["frame_key"]): set()
        for entry in camera_entries
        if isinstance(entry.get("frame_key"), str)
    }
    for idx, frame in enumerate(frames):
        missing_frame_keys = [key for key in REQUIRED_FRAME_KEYS if key not in frame]
        if missing_frame_keys:
            errors.append(f"frame {idx}: missing required keys: {', '.join(missing_frame_keys)}")
            continue

        if frame.get("frame_index") != idx:
            errors.append(f"frame {idx}: frame_index must equal {idx}")

        timestamp = frame.get("timestamp")
        if not _is_finite_number(timestamp):
            errors.append(f"frame {idx}: timestamp must be a finite number")
        else:
            timestamp_float = float(timestamp)
            if previous_timestamp is not None and timestamp_float <= previous_timestamp:
                errors.append(f"frame {idx}: timestamps must be strictly increasing")
            previous_timestamp = timestamp_float

        _check_numeric_vector(idx, frame, "model_state", declared_model_state_dim, errors)
        _check_numeric_vector(idx, frame, "measured_action", declared_action_dim, errors)

        padding = frame.get("action_is_terminal_padding")
        if not isinstance(padding, bool):
            errors.append(f"frame {idx}: action_is_terminal_padding must be a boolean")
        elif padding:
            padding_indices.append(idx)

        image_keys = [str(entry["frame_key"]) for entry in camera_entries if isinstance(entry.get("frame_key"), str)]
        if not image_keys:
            image_keys = list(LEGACY_IMAGE_FRAME_KEYS)
        for image_key in image_keys:
            if image_key not in frame:
                errors.append(f"frame {idx}: missing processed camera image key: {image_key}")
                continue
            image_path = _resolve_image_path(root, metadata, frame.get(image_key))
            if image_path is None:
                errors.append(f"frame {idx}: {image_key} must be a non-empty string")
            elif not image_path.is_file():
                errors.append(f"frame {idx}: {image_key} does not exist: {image_path}")
            else:
                decoded = _decode_image_shape(image_path)
                if isinstance(decoded, str):
                    errors.append(f"frame {idx}: {image_key} is not decodable: {image_path}: {decoded}")
                else:
                    image_shapes_by_key.setdefault(image_key, set()).add(decoded)

    if len(padding_indices) != 1:
        errors.append(
            f"frames.jsonl: expected exactly one terminal padding frame, got {len(padding_indices)}"
        )
    elif padding_indices[0] != len(frames) - 1:
        errors.append(
            f"frames.jsonl: terminal padding frame must be final frame, got frame {padding_indices[0]}"
        )

    if frames:
        final_action = frames[-1].get("measured_action")
        if not (
            isinstance(final_action, list)
            and len(final_action) == declared_action_dim
            and all(_is_finite_number(value) and abs(float(value)) <= 1e-12 for value in final_action)
        ):
            errors.append("final frame: measured_action must be all zeros")

    _check_terminal_action_policy(metadata_path, metadata, frames, padding_indices, errors)
    _check_processed_camera_counts(root, metadata_path, metadata, frames, camera_entries, errors)

    for entry in camera_entries:
        frame_key = entry.get("frame_key")
        camera_name = entry.get("camera_name", entry.get("raw_stream", frame_key))
        if not isinstance(frame_key, str):
            continue
        shapes = image_shapes_by_key.get(frame_key, set())
        if len(shapes) > 1:
            errors.append(f"processed camera {camera_name}: decoded image dimensions are not stable: {sorted(shapes)!r}")
        if len(shapes) == 1:
            decoded_shape = next(iter(shapes))
            declared = (entry.get("width"), entry.get("height"), entry.get("channels"))
            if all(isinstance(value, int) and not isinstance(value, bool) for value in declared):
                expected_shape = (int(declared[0]), int(declared[1]), int(declared[2]))
                if decoded_shape != expected_shape:
                    errors.append(
                        f"processed camera {camera_name}: decoded image dimensions {decoded_shape!r} "
                        f"do not match metadata {(expected_shape)!r}"
                    )

    if metadata.get("processed_camera_schema_version") == PROCESSED_CAMERA_SCHEMA_VERSION:
        _check_processed_camera_source_mappings(root, metadata, frames, camera_entries, errors)

    return ValidationResult(not errors, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a v0 processed JSONL episode directory.")
    parser.add_argument("processed_episode_dir", help="Path to processed episode directory")
    args = parser.parse_args(argv)

    result = validate_processed_episode(args.processed_episode_dir)
    if result.ok:
        print(f"OK: processed episode is valid: {args.processed_episode_dir}")
        return 0

    print(f"INVALID: processed episode failed validation: {args.processed_episode_dir}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
