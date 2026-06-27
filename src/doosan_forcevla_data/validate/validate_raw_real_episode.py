"""Validate a future raw real Doosan episode directory.

The validator is intentionally offline and dependency-light.  It reads JSON,
JSONL, and file paths only; it does not import ROS packages and it does not
communicate with a live robot.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.raw_real_schema import (
    OPTIONAL_STREAM_NAMES,
    RAW_REAL_SCHEMA_VERSION,
    REQUIRED_METADATA_KEYS,
    REQUIRED_STREAM_NAMES,
    REQUIRED_TOP_LEVEL_FILES,
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


JSONL_STREAM_NAMES = {
    "joint_states",
    "robot_state_rt",
    "tf",
    "tf_static",
    "command_context",
    "gripper_state",
}

KNOWN_CAMERA_STREAM_NAMES = {
    "tcp_camera",
    "external_camera_1",
    "external_camera_2",
    "external_camera",
    "wrist_camera",
}
CAMERA_STREAM_KIND_VALUES = {"camera_rgb", "camera_images", "rgb_camera", "camera"}
MODEL_EXTERNAL_IMAGE_KEY = "external_rgb_path"
MODEL_TCP_IMAGE_KEY = "tcp_rgb_path"
MODEL_CAMERA_OUTPUT_KEYS = [MODEL_EXTERNAL_IMAGE_KEY, MODEL_TCP_IMAGE_KEY]
MODEL_CAMERA_MAPPING_KEYS = [
    "model_camera_mapping",
    "model_input_cameras",
    "camera_mapping",
    "selected_camera_streams",
]
EXTERNAL_CAMERA_MAPPING_KEYS = {
    "external_rgb_path",
    "observation.image",
    "observation_image",
    "external_camera",
    "external_camera_1",
    "external",
    "model_external_camera",
}
TCP_CAMERA_MAPPING_KEYS = {
    "tcp_rgb_path",
    "observation.wrist_image",
    "observation_wrist_image",
    "wrist_image",
    "tcp_camera",
    "wrist_camera",
    "tcp",
    "wrist",
    "model_tcp_camera",
}
CONVERTER_REQUIRED_ALIGNMENT_STREAMS = ["joint_states"]
CONVERTER_ALIGNED_OPTIONAL_STREAMS = ["gripper_state"]
ROTATION_VECTOR_DEGREES = "rotation_vector_degrees"
ROTATION_VECTOR_RADIANS = "rotation_vector_radians"

CAMERA_ROBOT_SOURCE_STAMP_TOLERANCE_FPS_FRACTION = 0.5
CAMERA_ROBOT_SOURCE_STAMP_FALLBACK_TOLERANCE_SEC = 0.02
CAMERA_ROBOT_SOURCE_STAMP_OVERRIDE_KEY = "max_camera_robot_source_stamp_offset_sec"
CAMERA_ROBOT_SOURCE_STAMP_MAX_OVERRIDE_SEC = 0.1
CAMERA_ROBOT_SOURCE_STAMP_MAX_OVERRIDE_FRAME_FRACTION = 2.0

SUPPORTED_TCP_ORIENTATION_CONVENTIONS = {ROTATION_VECTOR_DEGREES, ROTATION_VECTOR_RADIANS}
UNSUPPORTED_TCP_ORIENTATION_CONVENTIONS = {
    "doosan_posx_euler_zyz_degrees",
    "doosan_robotstate_actual_tcp_position_euler_zyz_degrees",
    "euler_zyz_degrees",
}
EXPLICIT_SYNTHETIC_COLLECTION_METHODS = {
    "synthetic_raw_real",
    "synthetic_raw_real_fixture",
    "pipeline_smoke",
    "pipeline_smoke_raw_real",
}
EXPLICIT_SYNTHETIC_RECORDER_VERSIONS = {"synthetic_raw_real_generator_v0", "pipeline_smoke_raw_real_generator_v0"}
SOURCE_STAMP_SECONDS_UNIT = "seconds"

WRENCH_SOURCE_FIELDS = ["tcp_wrench", "measured_tcp_wrench", "external_tcp_force", "raw_force_torque"]
WRENCH_MODEL_STATE_ORDER = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
WRENCH_FORCE_UNIT = "N"
WRENCH_TORQUE_UNIT = "Nm"
SUPPORTED_WRENCH_FRAMES = {"base", "base_frame", "flange", "flange_frame", "tcp", "tcp_frame", "tool", "tool_frame"}
SUPPORTED_WRENCH_COMPENSATION = {
    "doosan_internal",
    "estimated_external_tcp_force",
    "raw_flange_sensor",
    "gravity_compensated",
    "not_gravity_compensated",
    "raw",
    "unknown",
}
PLACEHOLDER_GRIPPER_SOURCE_NAMES = {
    "synthetic_constant_pending_gripper_integration",
    "pipeline_smoke_constant_gripper",
    "synthetic_gripper_state",
}

STRICT_LAB_PROVENANCE_KEYS = [
    "exact_doosan_namespace",
    "read_data_rt_service",
    "tcp_frame",
    "flange_frame",
    "tool_frame",
    "force_torque_source",
    "gripper_state_source",
]
UNKNOWN_PROVENANCE_MARKERS = {
    "unknown",
    "unverified",
    "unset",
    "todo",
    "tbd",
    "none",
    "null",
    "n/a",
    "na",
}

SUPPORTED_TCP_POSITION_UNITS = {"mm", "millimeter", "millimeters", "m", "meter", "meters"}
SUPPORTED_TCP_ORIENTATION_UNITS = {"deg", "degree", "degrees", "rad", "radian", "radians"}
SUPPORTED_JOINT_POSITION_UNITS = {"deg", "degree", "degrees", "rad", "radian", "radians"}
SUPPORTED_JOINT_VELOCITY_UNITS = {
    "deg/s",
    "deg_per_s",
    "degree/s",
    "degrees/s",
    "degrees_per_second",
    "rad/s",
    "rad_per_s",
    "radian/s",
    "radians/s",
    "radians_per_second",
}
DEGREE_UNITS = {"deg", "degree", "degrees"}
RADIAN_UNITS = {"rad", "radian", "radians"}

CALIBRATION_REF_ID_KEYS = ("id", "calibration_id", "reference_id", "ref", "reference", "name")
REQUIRED_NON_CAMERA_CALIBRATION_REF_PATHS = [
    ("tcp_tool_calibration", ("tcp_tool_calibration",)),
    ("force_torque_calibration", ("force_torque_calibration",)),
]


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))



def _positive_fps(metadata: dict[str, Any] | None) -> float | None:
    fps = metadata.get("fps") if isinstance(metadata, dict) else None
    if _is_finite_number(fps) and float(fps) > 0.0:
        return float(fps)
    return None

def _is_non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _is_path_under_root(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _is_numeric_list(value: Any, expected_len: int) -> bool:
    return isinstance(value, list) and len(value) == expected_len and all(_is_finite_number(item) for item in value)


def _normalized_unit(units: dict[str, Any], key: str) -> str | None:
    value = units.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace(" ", "_")


def _normalized_label(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _unique_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def is_camera_stream_entry(stream_name: str, entry: Any) -> bool:
    if stream_name in KNOWN_CAMERA_STREAM_NAMES:
        return True
    if not isinstance(entry, dict):
        return False
    for key in ["type", "stream_type", "kind"]:
        if _normalized_label(entry.get(key)) in CAMERA_STREAM_KIND_VALUES:
            return True
    role_values = [
        entry.get("role"),
        entry.get("camera_role"),
        entry.get("source_role"),
        entry.get("model_role"),
    ]
    return any(_camera_target_from_role(role) is not None for role in role_values)


def camera_stream_names(streams: dict[str, Any] | None) -> list[str]:
    if not isinstance(streams, dict):
        return []
    return sorted(name for name, entry in streams.items() if is_camera_stream_entry(name, entry))


def _camera_target_from_mapping_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = _normalized_label(value)
    if normalized in {_normalized_label(key) for key in EXTERNAL_CAMERA_MAPPING_KEYS}:
        return MODEL_EXTERNAL_IMAGE_KEY
    if normalized in {_normalized_label(key) for key in TCP_CAMERA_MAPPING_KEYS}:
        return MODEL_TCP_IMAGE_KEY
    return None


def _camera_target_from_role(value: Any) -> str | None:
    normalized = _normalized_label(value)
    if normalized is None:
        return None
    if normalized in {"tcp_camera", "wrist_camera", "tcp", "wrist"}:
        return MODEL_TCP_IMAGE_KEY
    if normalized in {"external_camera", "external", "external_camera_1", "external_rgb"}:
        return MODEL_EXTERNAL_IMAGE_KEY
    return None


def _camera_target_from_stream_name(stream_name: str) -> str | None:
    if stream_name in {"tcp_camera", "wrist_camera"}:
        return MODEL_TCP_IMAGE_KEY
    if stream_name == "external_camera" or stream_name.startswith("external_camera_"):
        return MODEL_EXTERNAL_IMAGE_KEY
    return None


def _camera_target_for_inference(stream_name: str, entry: dict[str, Any]) -> str | None:
    for key in ["model_input_key", "model_key"]:
        target = _camera_target_from_mapping_key(entry.get(key))
        if target is not None:
            return target
    for key in ["role", "camera_role", "source_role", "model_role"]:
        target = _camera_target_from_role(entry.get(key))
        if target is not None:
            return target
    return _camera_target_from_stream_name(stream_name)


def _mapping_stream_name(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ["stream", "stream_name", "source_stream", "raw_stream", "name"]:
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


def _add_explicit_camera_assignment(
    assignments: dict[str, tuple[str, str]],
    errors: list[str],
    *,
    target: str,
    stream_name: str,
    source: str,
    camera_names: set[str],
) -> None:
    if stream_name not in camera_names:
        errors.append(
            f"camera mapping: {source} selects stream {stream_name!r}, but it is not a declared camera stream"
        )
        return
    previous = assignments.get(target)
    if previous is not None and previous[0] != stream_name:
        errors.append(
            f"camera mapping: conflicting selections for {target}: {previous[0]!r} from {previous[1]}, "
            f"and {stream_name!r} from {source}"
        )
        return
    assignments[target] = (stream_name, source)


def _collect_explicit_camera_assignments(
    metadata: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
    streams: dict[str, Any],
    camera_names: set[str],
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    assignments: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    containers = [("metadata", metadata), ("streams/index.json", streams_index)]

    for container_label, container in containers:
        if not isinstance(container, dict):
            continue
        for mapping_key in MODEL_CAMERA_MAPPING_KEYS:
            mapping = container.get(mapping_key)
            if not isinstance(mapping, dict):
                continue
            for raw_target, raw_value in mapping.items():
                target = _camera_target_from_mapping_key(raw_target)
                stream_name = _mapping_stream_name(raw_value)
                source = f"{container_label}.{mapping_key}.{raw_target}"
                if target is None:
                    continue
                if stream_name is None:
                    errors.append(f"camera mapping: {source} must name a camera stream")
                    continue
                _add_explicit_camera_assignment(
                    assignments,
                    errors,
                    target=target,
                    stream_name=stream_name,
                    source=source,
                    camera_names=camera_names,
                )

    for stream_name, entry in streams.items():
        if stream_name not in camera_names or not isinstance(entry, dict):
            continue
        for key in ["model_input_key", "model_key"]:
            target = _camera_target_from_mapping_key(entry.get(key))
            if target is not None:
                _add_explicit_camera_assignment(
                    assignments,
                    errors,
                    target=target,
                    stream_name=stream_name,
                    source=f"streams/index.json streams.{stream_name}.{key}",
                    camera_names=camera_names,
                )
        if entry.get("used_for_model") is True:
            target = None
            for key in ["role", "camera_role", "source_role", "model_role"]:
                target = _camera_target_from_role(entry.get(key))
                if target is not None:
                    break
            if target is None:
                target = _camera_target_from_stream_name(stream_name)
            if target is None:
                errors.append(
                    f"camera mapping: streams/index.json streams.{stream_name}.used_for_model=true "
                    "requires role/model_input_key identifying external or tcp model image use"
                )
            else:
                _add_explicit_camera_assignment(
                    assignments,
                    errors,
                    target=target,
                    stream_name=stream_name,
                    source=f"streams/index.json streams.{stream_name}.used_for_model",
                    camera_names=camera_names,
                )
    return assignments, errors


def select_model_camera_streams(
    metadata: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
    streams: dict[str, Any] | None,
) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """Select raw camera streams for processed external/tcp image compatibility keys."""

    if not isinstance(streams, dict):
        return {}, ["camera mapping: streams/index.json streams must be a JSON object"], {}

    camera_names_list = camera_stream_names(streams)
    camera_names = set(camera_names_list)
    if not camera_names:
        return {}, ["camera mapping: at least one declared camera_rgb stream is required"], {}

    assignments, errors = _collect_explicit_camera_assignments(metadata, streams_index, streams, camera_names)
    selected: dict[str, str] = {}
    selection_sources: dict[str, str] = {}

    for target in MODEL_CAMERA_OUTPUT_KEYS:
        if target in assignments:
            selected[target], selection_sources[target] = assignments[target]
            continue

        candidates = [
            name
            for name in camera_names_list
            if isinstance(streams.get(name), dict) and _camera_target_for_inference(name, streams[name]) == target
        ]
        if len(candidates) == 1:
            selected[target] = candidates[0]
            selection_sources[target] = "unambiguous legacy/name/role fallback"
            continue
        if not candidates:
            errors.append(
                f"camera mapping: no camera stream selected for {target}; provide metadata.model_camera_mapping "
                "or a camera stream model_input_key"
            )
        else:
            errors.append(
                f"camera mapping: multiple candidate streams for {target}: {candidates!r}; provide explicit "
                "metadata.model_camera_mapping or stream model_input_key instead of guessing"
            )

    return selected, errors, selection_sources


def _combined_units(record: dict[str, Any] | None, stream_entry: dict[str, Any] | None) -> dict[str, Any]:
    units: dict[str, Any] = {}
    if isinstance(stream_entry, dict) and isinstance(stream_entry.get("units"), dict):
        units.update(stream_entry["units"])
    if isinstance(record, dict) and isinstance(record.get("units"), dict):
        units.update(record["units"])
    return units



def is_explicit_synthetic_episode(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
) -> bool:
    collection_method = metadata.get("collection_method") if isinstance(metadata, dict) else None
    recorder_version = metadata.get("recorder_version") if isinstance(metadata, dict) else None
    return any(
        [
            isinstance(metadata, dict) and metadata.get("synthetic") is True,
            collection_method in EXPLICIT_SYNTHETIC_COLLECTION_METHODS,
            recorder_version in EXPLICIT_SYNTHETIC_RECORDER_VERSIONS,
            isinstance(recorder_report, dict) and recorder_report.get("synthetic") is True,
            isinstance(streams_index, dict) and streams_index.get("synthetic") is True,
        ]
    )

def _tcp_orientation_convention(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
) -> Any:
    if isinstance(metadata, dict) and metadata.get("tcp_orientation_convention") is not None:
        return metadata.get("tcp_orientation_convention")
    if isinstance(recorder_report, dict):
        return recorder_report.get("tcp_orientation_convention")
    return None



def _quoted_values(values: set[str]) -> str:
    return ", ".join(f"'{value}'" for value in sorted(values))


def tcp_orientation_convention_readiness_error(convention: Any) -> str | None:
    supported = _quoted_values(SUPPORTED_TCP_ORIENTATION_CONVENTIONS)
    unsupported = _quoted_values(UNSUPPORTED_TCP_ORIENTATION_CONVENTIONS)
    if convention in SUPPORTED_TCP_ORIENTATION_CONVENTIONS:
        return None
    if convention in UNSUPPORTED_TCP_ORIENTATION_CONVENTIONS:
        return (
            "metadata/recorder_report: tcp_orientation_convention "
            f"{convention!r} is recognized but unsupported for conversion; "
            "Doosan native Euler ZYZ must be live-verified and converted before use. "
            f"Supported conversion conventions: {supported}"
        )
    if convention is None:
        return (
            "metadata/recorder_report: tcp_orientation_convention must be one of "
            f"{supported} for non-synthetic conversion; "
            "tcp_orientation_convention_verified alone is not sufficient. "
            f"Recognized but unsupported Doosan/native conventions: {unsupported}"
        )
    return (
        "metadata/recorder_report: unknown tcp_orientation_convention "
        f"{convention!r}; supported conversion conventions: {supported}; "
        f"recognized but unsupported Doosan/native conventions: {unsupported}"
    )

def _supported_unit_error(
    units: dict[str, Any],
    key: str,
    supported: set[str],
    context: str,
) -> str | None:
    unit = _normalized_unit(units, key)
    if unit in supported:
        return None
    return f"{context}: unsupported or missing {key} unit: {unit!r}"


def _orientation_unit_matches_convention(unit: str | None, convention: Any, context: str) -> str | None:
    if convention == ROTATION_VECTOR_DEGREES and unit not in DEGREE_UNITS:
        return f"{context}: tcp_orientation unit {unit!r} does not match tcp_orientation_convention='rotation_vector_degrees'"
    if convention == ROTATION_VECTOR_RADIANS and unit not in RADIAN_UNITS:
        return f"{context}: tcp_orientation unit {unit!r} does not match tcp_orientation_convention='rotation_vector_radians'"
    return None



def _is_unknown_provenance_value(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return (
        normalized in UNKNOWN_PROVENANCE_MARKERS
        or normalized.startswith("unknown")
        or normalized.startswith("todo")
        or normalized.startswith("tbd")
    )


def _strict_lab_provenance_required(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
) -> bool:
    if isinstance(metadata, dict) and (
        metadata.get("lab_provenance_required") is True
        or metadata.get("strict_lab_provenance") is True
    ):
        return True
    if isinstance(recorder_report, dict) and (
        recorder_report.get("lab_provenance_required") is True
        or recorder_report.get("strict_lab_provenance") is True
    ):
        return True
    return False


def _camera_topic_provenance_value(live_graph: dict[str, Any], stream_name: str) -> Any:
    camera_topics = live_graph.get("camera_topics")
    if isinstance(camera_topics, dict) and stream_name in camera_topics:
        return camera_topics.get(stream_name)
    direct_key = f"{stream_name}_topic"
    if direct_key in live_graph:
        return live_graph.get(direct_key)
    legacy_key = None
    if stream_name == "tcp_camera":
        legacy_key = "wrist_camera_topic"
    elif stream_name == "external_camera_1":
        legacy_key = "external_camera_topic"
    elif stream_name == "wrist_camera":
        legacy_key = "wrist_camera_topic"
    elif stream_name == "external_camera":
        legacy_key = "external_camera_topic"
    if legacy_key is not None:
        return live_graph.get(legacy_key)
    return None


def _strict_lab_provenance_errors(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
    streams: dict[str, Any] | None,
    records_by_stream: dict[str, list[dict[str, Any]]],
    *,
    root_dir: str | Path | None = None,
    calibration_refs: dict[str, Any] | None = None,
) -> list[str]:
    if not _strict_lab_provenance_required(metadata, recorder_report):
        return [
            "non-synthetic raw-real conversion requires strict lab provenance; "
            "set lab_provenance_required=true or strict_lab_provenance=true and provide verified lab provenance"
        ]

    errors: list[str] = []
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    source_workspace = metadata_dict.get("source_workspace")
    if not isinstance(source_workspace, dict):
        errors.append("strict lab provenance: metadata.source_workspace must be a JSON object")
    else:
        if source_workspace.get("verified") is not True:
            errors.append("strict lab provenance: metadata.source_workspace.verified must be true")
        for key in ["path", "git_commit", "git_remote", "git_branch"]:
            if _is_unknown_provenance_value(source_workspace.get(key)):
                errors.append(f"strict lab provenance: metadata.source_workspace.{key} must be known")

    live_graph = metadata_dict.get("live_graph_verification")
    if not isinstance(live_graph, dict):
        errors.append("strict lab provenance: metadata.live_graph_verification must be a JSON object")
    else:
        if live_graph.get("time_sync_verified") is not True:
            errors.append("strict lab provenance: metadata.live_graph_verification.time_sync_verified must be true")
        for key in STRICT_LAB_PROVENANCE_KEYS:
            if _is_unknown_provenance_value(live_graph.get(key)):
                errors.append(f"strict lab provenance: metadata.live_graph_verification.{key} must be known")
        for stream_name in camera_stream_names(streams):
            if _is_unknown_provenance_value(_camera_topic_provenance_value(live_graph, stream_name)):
                errors.append(
                    f"strict lab provenance: metadata.live_graph_verification camera topic for {stream_name} "
                    "must be known (camera_topics mapping or <stream_name>_topic)"
                )

    if isinstance(streams, dict):
        for stream_name, entry in streams.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("verified") is not True:
                errors.append(f"strict lab provenance: stream {stream_name} verified must be true")
            if _is_unknown_provenance_value(entry.get("source_name")):
                errors.append(f"strict lab provenance: stream {stream_name} source_name must be known")

    for stream_name in camera_stream_names(streams):
        for idx, record in enumerate(records_by_stream.get(stream_name, [])):
            if _is_unknown_provenance_value(record.get("frame_id")):
                errors.append(f"strict lab provenance: {stream_name} camera record {idx} frame_id must be known")
                break

    return errors




def _declared_image_shape_text(record: dict[str, Any]) -> str:
    return f"{record.get('width')!r}x{record.get('height')!r}x{record.get('channels')!r}"


def _decoded_image_shape_text(shape: tuple[int, int, int]) -> str:
    width, height, channels = shape
    return f"{width}x{height}x{channels}"


def _decode_image_shape(path: Path) -> tuple[int, int, int] | str:
    try:
        from PIL import Image
    except ImportError:
        return "Pillow/PIL is not available; install Pillow to verify non-synthetic camera images"

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            channels = len(image.getbands())
    except Exception as exc:
        return f"Pillow/PIL could not decode image: {exc}"

    return int(width), int(height), int(channels)


def _camera_image_decodability_errors(
    root: Path,
    records_by_stream: dict[str, list[dict[str, Any]]],
    camera_streams: list[str],
) -> list[str]:
    errors: list[str] = []
    for stream_name in camera_streams:
        for idx, record in enumerate(records_by_stream.get(stream_name, [])):
            image_path_value = record.get("image_path")
            if not isinstance(image_path_value, str) or not image_path_value:
                continue
            image_relative = Path(image_path_value)
            if image_relative.is_absolute() or ".." in image_relative.parts:
                continue
            image_path = root / image_relative
            if not _is_path_under_root(root, image_path) or not image_path.is_file():
                continue
            try:
                if image_path.stat().st_size <= 0:
                    continue
            except OSError:
                continue

            declared_shape = _declared_image_shape_text(record)
            decoded = _decode_image_shape(image_path)
            if isinstance(decoded, str):
                errors.append(
                    f"{stream_name} camera record {idx}: image_path {image_path_value} is not decodable "
                    f"for non-synthetic conversion (declared {declared_shape}; resolved {image_path}): {decoded}"
                )
                continue

            if not all(_is_positive_int(record.get(key)) for key in ["width", "height", "channels"]):
                continue
            expected = (int(record["width"]), int(record["height"]), int(record["channels"]))
            if decoded != expected:
                errors.append(
                    f"{stream_name} camera record {idx}: decoded image dimensions do not match declared metadata "
                    f"for image_path {image_path_value}: declared {declared_shape}, "
                    f"decoded {_decoded_image_shape_text(decoded)} (resolved {image_path})"
                )
    return errors


def _nested_value(data: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _is_meaningful_calibration_ref(value: Any) -> bool:
    if isinstance(value, str):
        return not _is_unknown_provenance_value(value)
    if isinstance(value, dict):
        if not value:
            return False
        return any(_is_meaningful_calibration_ref(value.get(key)) for key in CALIBRATION_REF_ID_KEYS if key in value)
    return False


def _calibration_refs_readiness_errors(
    calibration_refs: dict[str, Any] | None,
    camera_streams: list[str],
) -> list[str]:
    if not isinstance(calibration_refs, dict):
        return ["calibration_refs: must be a JSON object for non-synthetic conversion"]

    errors: list[str] = []
    required_paths = list(REQUIRED_NON_CAMERA_CALIBRATION_REF_PATHS)
    for stream_name in camera_streams:
        required_paths.append((f"camera_intrinsics.{stream_name}", ("camera_intrinsics", stream_name)))
        required_paths.append((f"camera_extrinsics.{stream_name}", ("camera_extrinsics", stream_name)))

    for display_path, path in required_paths:
        exists, value = _nested_value(calibration_refs, path)
        if not exists:
            errors.append(f"calibration_refs.{display_path} is required for non-synthetic conversion")
        elif not _is_meaningful_calibration_ref(value):
            errors.append(
                f"calibration_refs.{display_path} must be a non-empty known calibration reference "
                "for non-synthetic conversion"
            )
    return errors

def _selected_wrench_source(record: dict[str, Any]) -> str | None:
    for source in WRENCH_SOURCE_FIELDS:
        if _is_numeric_list(record.get(source), 6):
            return source
    return None


def selected_wrench_sources_for_model_state(records_by_stream: dict[str, list[dict[str, Any]]]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for record in records_by_stream.get("robot_state_rt", []):
        source = _selected_wrench_source(record)
        if source is not None and source not in seen:
            selected.append(source)
            seen.add(source)
    return selected


def _robot_state_wrench_sources_metadata(streams: dict[str, Any] | None) -> Any:
    if not isinstance(streams, dict):
        return None
    robot_entry = streams.get("robot_state_rt")
    if not isinstance(robot_entry, dict):
        return None
    return robot_entry.get("wrench_sources")


def _validated_wrench_metadata(source: str, metadata: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    valid = True
    context = f"wrench metadata: selected source {source}"

    order = metadata.get("order")
    if order != WRENCH_MODEL_STATE_ORDER:
        errors.append(f"{context} order must be {WRENCH_MODEL_STATE_ORDER!r}; got {order!r}")
        valid = False

    force_unit = metadata.get("force_unit")
    if force_unit != WRENCH_FORCE_UNIT:
        errors.append(f"{context} force_unit must be 'N'; got {force_unit!r}")
        valid = False

    torque_unit = metadata.get("torque_unit")
    if torque_unit != WRENCH_TORQUE_UNIT:
        errors.append(f"{context} torque_unit must be 'Nm'; got {torque_unit!r}")
        valid = False

    frame = metadata.get("frame")
    if frame not in SUPPORTED_WRENCH_FRAMES:
        errors.append(f"{context} frame must be one of {sorted(SUPPORTED_WRENCH_FRAMES)!r}; got {frame!r}")
        valid = False

    compensation = metadata.get("compensation")
    if compensation not in SUPPORTED_WRENCH_COMPENSATION:
        errors.append(
            f"{context} compensation must be one of {sorted(SUPPORTED_WRENCH_COMPENSATION)!r}; "
            f"got {compensation!r}"
        )
        valid = False

    approved = metadata.get("approved_for_model_state")
    if approved is not True:
        errors.append(f"{context} approved_for_model_state must be true; got {approved!r}")
        valid = False

    if not valid:
        return None
    validated = {
        "order": list(WRENCH_MODEL_STATE_ORDER),
        "force_unit": force_unit,
        "torque_unit": torque_unit,
        "frame": frame,
        "compensation": compensation,
        "approved_for_model_state": True,
    }
    for optional_key in ["source_name", "source_type", "source_service_or_topic"]:
        value = metadata.get(optional_key)
        if isinstance(value, str) and value.strip():
            validated[optional_key] = value
    return validated


def selected_wrench_metadata_for_model_state(
    streams: dict[str, Any] | None,
    records_by_stream: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    selected_sources = selected_wrench_sources_for_model_state(records_by_stream)
    wrench_sources = _robot_state_wrench_sources_metadata(streams)
    if not isinstance(wrench_sources, dict):
        return {}

    selected_metadata: dict[str, dict[str, Any]] = {}
    for source in selected_sources:
        metadata = wrench_sources.get(source)
        errors: list[str] = []
        if isinstance(metadata, dict):
            validated = _validated_wrench_metadata(source, metadata, errors)
            if validated is not None:
                selected_metadata[source] = validated
    return selected_metadata


def _wrench_metadata_errors(
    streams: dict[str, Any] | None,
    records_by_stream: dict[str, list[dict[str, Any]]],
) -> list[str]:
    selected_sources = selected_wrench_sources_for_model_state(records_by_stream)
    if not selected_sources:
        return []

    errors: list[str] = []
    wrench_sources = _robot_state_wrench_sources_metadata(streams)
    if not isinstance(wrench_sources, dict):
        return [
            "wrench metadata: streams/index.json streams.robot_state_rt.wrench_sources "
            "must be a JSON object for non-synthetic conversion"
        ]

    for source in selected_sources:
        metadata = wrench_sources.get(source)
        if not isinstance(metadata, dict):
            errors.append(
                f"wrench metadata: selected source {source} must have metadata at "
                f"streams/index.json streams.robot_state_rt.wrench_sources.{source}"
            )
            continue
        _validated_wrench_metadata(source, metadata, errors)
    return errors

def _read_json_object(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: could not read {label} JSON object: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: {label} must be a JSON object")
        return None
    return data


def _read_jsonl_objects(path: Path, errors: list[str], label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path}: could not read {label} JSONL: {exc}")
        return records

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"{path}: line {line_number} is empty")
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path}: line {line_number} must be a JSON object")
            continue
        records.append(data)
    return records


def _check_metadata(metadata: dict[str, Any], path: Path, errors: list[str]) -> None:
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        errors.append(f"{path}: metadata missing required keys: {', '.join(missing)}")

    if metadata.get("schema_version") != RAW_REAL_SCHEMA_VERSION:
        errors.append(f"{path}: schema_version must be {RAW_REAL_SCHEMA_VERSION!r}")

    for key in [
        "episode_id",
        "task_instruction",
        "geometry_type",
        "orientation_type",
        "collection_method",
        "action_label_primary",
        "robot_type",
        "recorder_version",
    ]:
        if key in metadata and not _is_non_empty_string(metadata[key]):
            errors.append(f"{path}: {key} must be a non-empty string")

    if "success" in metadata and not isinstance(metadata["success"], bool):
        errors.append(f"{path}: success must be a boolean")

    failure_reason = metadata.get("failure_reason")
    if failure_reason is not None and not _is_non_empty_string(failure_reason):
        errors.append(f"{path}: failure_reason must be null or a non-empty string")
    if metadata.get("success") is False and not _is_non_empty_string(failure_reason):
        errors.append(f"{path}: failure_reason must be a non-empty string when success is false")

    fps = metadata.get("fps")
    if not _is_finite_number(fps) or float(fps) <= 0.0:
        errors.append(f"{path}: fps must be a positive finite number")

    source_workspace = metadata.get("source_workspace")
    if "source_workspace" in metadata:
        if not isinstance(source_workspace, dict):
            errors.append(f"{path}: source_workspace must be a JSON object")
        else:
            if not _is_non_empty_string(source_workspace.get("path")):
                errors.append(f"{path}: source_workspace.path must be a non-empty string")
            if "verified" in source_workspace and not isinstance(source_workspace["verified"], bool):
                errors.append(f"{path}: source_workspace.verified must be a boolean when present")


def _source_stamp_seconds(value: Any) -> float | None:
    if _is_finite_number(value):
        return float(value)
    if not isinstance(value, dict):
        return None
    sec = value.get("sec")
    nanosec = value.get("nanosec")
    if not _is_finite_number(sec) or not _is_finite_number(nanosec):
        return None
    if not 0.0 <= float(nanosec) < 1_000_000_000.0:
        return None
    return float(sec) + float(nanosec) * 1e-9


def _required_streams_use_numeric_source_stamp(
    records_by_stream: dict[str, list[dict[str, Any]]],
    camera_streams: list[str],
) -> bool:
    for stream_name in REQUIRED_STREAM_NAMES + camera_streams:
        for record in records_by_stream.get(stream_name, []):
            if _is_finite_number(record.get("source_stamp")):
                return True
    return False


def _numeric_source_stamp_timebase_errors(
    streams_index: dict[str, Any] | None,
    records_by_stream: dict[str, list[dict[str, Any]]],
    camera_streams: list[str],
) -> list[str]:
    if not _required_streams_use_numeric_source_stamp(records_by_stream, camera_streams):
        return []

    timebase = streams_index.get("timebase") if isinstance(streams_index, dict) else None
    source_stamp_unit = timebase.get("source_stamp_unit") if isinstance(timebase, dict) else None
    if source_stamp_unit == SOURCE_STAMP_SECONDS_UNIT:
        return []
    return [
        "source_stamp unit/timebase: streams/index.json timebase.source_stamp_unit must be "
        f"'seconds' for non-synthetic numeric source_stamp values; got {source_stamp_unit!r}"
    ]


def _validate_common_record_fields(
    records: list[dict[str, Any]],
    path: Path,
    stream_name: str,
    errors: list[str],
) -> None:
    previous_source: float | None = None
    previous_receipt: float | None = None
    previous_monotonic: float | None = None

    for expected_index, record in enumerate(records):
        context = f"{path}: {stream_name} record {expected_index}"
        record_index = record.get("record_index")
        if record_index != expected_index or isinstance(record_index, bool):
            errors.append(f"{context}: record_index must be sequential from 0")

        if "source_stamp" not in record:
            errors.append(f"{context}: source_stamp is missing")
        else:
            source = _source_stamp_seconds(record["source_stamp"])
            if source is None:
                errors.append(f"{context}: source_stamp must be numeric or an object with finite sec/nanosec")
            else:
                if previous_source is not None and source < previous_source:
                    errors.append(f"{context}: source_stamp must be monotonic nondecreasing")
                previous_source = source

        receipt_stamp = record.get("receipt_stamp")
        if not _is_finite_number(receipt_stamp):
            errors.append(f"{context}: receipt_stamp must be a finite number")
        else:
            receipt = float(receipt_stamp)
            if previous_receipt is not None and receipt < previous_receipt:
                errors.append(f"{context}: receipt_stamp must be monotonic nondecreasing")
            previous_receipt = receipt

        monotonic_stamp = record.get("monotonic_stamp")
        if not _is_finite_number(monotonic_stamp):
            errors.append(f"{context}: monotonic_stamp must be a finite number")
        else:
            monotonic = float(monotonic_stamp)
            if previous_monotonic is not None and monotonic < previous_monotonic:
                errors.append(f"{context}: monotonic_stamp must be monotonic nondecreasing")
            previous_monotonic = monotonic


def _check_numeric_list(
    record: dict[str, Any],
    key: str,
    expected_len: int,
    context: str,
    errors: list[str],
) -> None:
    values = record.get(key)
    if not isinstance(values, list):
        errors.append(f"{context}: {key} must be a list of length {expected_len}")
        return
    if len(values) != expected_len:
        errors.append(f"{context}: {key} length must be {expected_len}, got {len(values)}")
        return
    for idx, value in enumerate(values):
        if not _is_finite_number(value):
            errors.append(f"{context}: {key}[{idx}] must be a finite number")


def _stream_path(root: Path, stream_name: str, entry: dict[str, Any], errors: list[str]) -> Path | None:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"streams/index.json: stream {stream_name} missing non-empty path")
        return None

    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        errors.append(f"streams/index.json: stream {stream_name} path must be relative to episode root")
        return None
    path = root / relative_path
    if not _is_path_under_root(root, path):
        errors.append(f"streams/index.json: stream {stream_name} path must stay inside episode root")
        return None
    if not path.exists():
        errors.append(f"{path}: stream {stream_name} path does not exist")
        return None
    return path


def _check_stream_record_count(
    stream_entry: dict[str, Any],
    stream_name: str,
    records: list[dict[str, Any]],
    path: Path,
    required: bool,
    errors: list[str],
) -> None:
    record_count = stream_entry.get("record_count")
    if _is_non_negative_int(record_count) and record_count != len(records):
        errors.append(
            f"{path}: stream {stream_name} record_count {record_count} does not match actual record count {len(records)}"
        )
    if required and not records:
        errors.append(f"{path}: required stream {stream_name} must contain at least one record")


def _validate_required_stream_entry(
    root: Path,
    stream_name: str,
    entry: Any,
    errors: list[str],
) -> Path | None:
    if not isinstance(entry, dict):
        errors.append(f"streams/index.json: stream {stream_name} entry must be a JSON object")
        return None

    source_name = entry.get("source_name")
    if not isinstance(source_name, str) or not source_name:
        errors.append(f"streams/index.json: stream {stream_name} missing non-empty source_name")

    source_type = entry.get("source_type")
    if not isinstance(source_type, str) or not source_type:
        errors.append(f"streams/index.json: stream {stream_name} missing non-empty source_type")

    if entry.get("required") is not True:
        errors.append(f"streams/index.json: stream {stream_name} required must be true")

    if not _is_non_negative_int(entry.get("record_count")):
        errors.append(f"streams/index.json: stream {stream_name} record_count must be a non-negative integer")

    return _stream_path(root, stream_name, entry, errors)


def _validate_optional_stream_entry(
    root: Path,
    stream_name: str,
    entry: Any,
    errors: list[str],
) -> Path | None:
    if not isinstance(entry, dict):
        errors.append(f"streams/index.json: optional stream {stream_name} entry must be a JSON object")
        return None
    if "required" in entry and entry.get("required") is not False:
        errors.append(f"streams/index.json: optional stream {stream_name} required must be false when present")
    if "record_count" in entry and not _is_non_negative_int(entry.get("record_count")):
        errors.append(
            f"streams/index.json: optional stream {stream_name} record_count must be a non-negative integer"
        )
    if "path" not in entry:
        return None
    return _stream_path(root, stream_name, entry, errors)


def _validate_camera_stream_entry(
    root: Path,
    stream_name: str,
    entry: Any,
    errors: list[str],
) -> Path | None:
    if not isinstance(entry, dict):
        errors.append(f"streams/index.json: camera stream {stream_name} entry must be a JSON object")
        return None

    source_name = entry.get("source_name")
    if not isinstance(source_name, str) or not source_name:
        errors.append(f"streams/index.json: camera stream {stream_name} missing non-empty source_name")

    source_type = entry.get("source_type")
    if not isinstance(source_type, str) or not source_type:
        errors.append(f"streams/index.json: camera stream {stream_name} missing non-empty source_type")

    if "required" in entry and not isinstance(entry.get("required"), bool):
        errors.append(f"streams/index.json: camera stream {stream_name} required must be a boolean when present")

    if not _is_non_negative_int(entry.get("record_count")):
        errors.append(f"streams/index.json: camera stream {stream_name} record_count must be a non-negative integer")

    for key in ["type", "stream_type", "kind", "role", "camera_role", "camera_id", "external_camera_id", "model_input_key"]:
        if key in entry and (not isinstance(entry.get(key), str) or not entry.get(key)):
            errors.append(f"streams/index.json: camera stream {stream_name} {key} must be a non-empty string when present")

    return _stream_path(root, stream_name, entry, errors)


def _add_stream_warnings(streams: dict[str, Any], warnings: list[str]) -> None:
    for stream_name in OPTIONAL_STREAM_NAMES:
        if stream_name not in streams:
            warnings.append(f"optional stream {stream_name} is absent")

    for stream_name, entry in streams.items():
        if not isinstance(entry, dict):
            continue
        source_name = entry.get("source_name")
        if isinstance(source_name, str) and "unknown" in source_name.lower():
            warnings.append(f"stream {stream_name} source_name is unverified or unknown: {source_name}")
        if entry.get("verified") is False:
            warnings.append(f"stream {stream_name} has verified: false")


def _validate_joint_states(
    records: list[dict[str, Any]],
    stream_entry: dict[str, Any],
    path: Path,
    errors: list[str],
) -> None:
    for idx, record in enumerate(records):
        context = f"{path}: joint_states record {idx}"
        joint_names = record.get("joint_names")
        if not isinstance(joint_names, list) or len(joint_names) != 6:
            errors.append(f"{context}: joint_names must be a list of length 6")
        else:
            for joint_idx, joint_name in enumerate(joint_names):
                if not _is_non_empty_string(joint_name):
                    errors.append(f"{context}: joint_names[{joint_idx}] must be a non-empty string")
            if all(isinstance(joint_name, str) for joint_name in joint_names) and len(set(joint_names)) != len(
                joint_names
            ):
                errors.append(f"{context}: joint_names must be unique")
        _check_numeric_list(record, "position", 6, context, errors)
        _check_numeric_list(record, "velocity", 6, context, errors)
        if not isinstance(record.get("units"), dict) and not isinstance(stream_entry.get("units"), dict):
            errors.append(f"{context}: units object must exist on the record or stream entry")


def _validate_robot_state_rt(records: list[dict[str, Any]], path: Path, errors: list[str]) -> None:
    for idx, record in enumerate(records):
        context = f"{path}: robot_state_rt record {idx}"
        _check_numeric_list(record, "actual_tcp_position", 6, context, errors)

        present_wrench_fields = [field for field in WRENCH_SOURCE_FIELDS if field in record]
        if not present_wrench_fields:
            errors.append(
                f"{context}: tcp_wrench, measured_tcp_wrench, external_tcp_force, or raw_force_torque must exist"
            )
        for field in present_wrench_fields:
            _check_numeric_list(record, field, 6, context, errors)

        for key in ["robot_mode", "robot_state", "control_mode"]:
            if not _is_non_empty_string(record.get(key)):
                errors.append(f"{context}: {key} must be a non-empty string")


def _validate_tf_records(records: list[dict[str, Any]], path: Path, stream_name: str, errors: list[str]) -> None:
    for idx, record in enumerate(records):
        transforms = record.get("transforms")
        if not isinstance(transforms, list):
            errors.append(f"{path}: {stream_name} record {idx}: transforms must be a list")
            continue
        for transform_idx, transform in enumerate(transforms):
            context = f"{path}: {stream_name} record {idx} transform {transform_idx}"
            if not isinstance(transform, dict):
                errors.append(f"{context}: transform must be a JSON object")
                continue
            for key in ["parent_frame", "child_frame"]:
                if not _is_non_empty_string(transform.get(key)):
                    errors.append(f"{context}: {key} must be a non-empty string")
            _check_numeric_list(transform, "translation", 3, context, errors)
            rotation = transform.get("rotation_xyzw", transform.get("rotation"))
            if not isinstance(rotation, list):
                errors.append(f"{context}: rotation_xyzw must be a list of length 4")
            elif len(rotation) != 4:
                errors.append(f"{context}: rotation_xyzw length must be 4, got {len(rotation)}")
            else:
                for rotation_idx, value in enumerate(rotation):
                    if not _is_finite_number(value):
                        errors.append(f"{context}: rotation_xyzw[{rotation_idx}] must be a finite number")


def _validate_gripper_state(records: list[dict[str, Any]], path: Path, errors: list[str]) -> None:
    for idx, record in enumerate(records):
        context = f"{path}: gripper_state record {idx}"
        has_position = "gripper_position" in record
        has_width = "gripper_width_m" in record
        if not has_position and not has_width:
            errors.append(f"{context}: gripper_position or gripper_width_m must exist")
        if has_position and not _is_finite_number(record.get("gripper_position")):
            errors.append(f"{context}: gripper_position must be a finite number")
        if has_width:
            width = record.get("gripper_width_m")
            if not _is_finite_number(width):
                errors.append(f"{context}: gripper_width_m must be a finite number")
            elif float(width) < 0.0:
                errors.append(f"{context}: gripper_width_m must be non-negative")


def _validate_command_context(
    records: list[dict[str, Any]],
    path: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    for idx, record in enumerate(records):
        context = f"{path}: command_context record {idx}"
        if "command_kind" in record and not _is_non_empty_string(record.get("command_kind")):
            errors.append(f"{context}: command_kind must be a non-empty string when present")
        if "commanded_twist" in record:
            _check_numeric_list(record, "commanded_twist", 6, context, errors)
        action_like_keys = [key for key in ["action_label", "measured_action", "action"] if key in record]
        if action_like_keys:
            warnings.append(
                f"{context}: action-like fields are diagnostic only and are not action labels: {', '.join(action_like_keys)}"
            )


def _validate_jsonl_stream(
    path: Path,
    stream_name: str,
    stream_entry: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    required: bool,
) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"{path}: stream {stream_name} must be a JSONL file")
        return []

    records = _read_jsonl_objects(path, errors, stream_name)
    _check_stream_record_count(stream_entry, stream_name, records, path, required, errors)
    _validate_common_record_fields(records, path, stream_name, errors)

    if stream_name == "joint_states":
        _validate_joint_states(records, stream_entry, path, errors)
    elif stream_name == "robot_state_rt":
        _validate_robot_state_rt(records, path, errors)
    elif stream_name in {"tf", "tf_static"}:
        _validate_tf_records(records, path, stream_name, errors)
    elif stream_name == "gripper_state":
        _validate_gripper_state(records, path, errors)
    elif stream_name == "command_context":
        _validate_command_context(records, path, errors, warnings)
    return records


def _validate_camera_index(
    root: Path,
    stream_path: Path,
    stream_name: str,
    stream_entry: dict[str, Any],
    errors: list[str],
) -> list[dict[str, Any]]:
    if not stream_path.is_dir():
        errors.append(f"{stream_path}: camera stream {stream_name} must be a directory")
        return []

    index_path = stream_path / "index.jsonl"
    if not index_path.is_file():
        errors.append(f"{index_path}: camera stream {stream_name} index.jsonl is missing")
        return []

    records = _read_jsonl_objects(index_path, errors, f"{stream_name} index")
    _check_stream_record_count(stream_entry, stream_name, records, index_path, required=True, errors=errors)
    _validate_common_record_fields(records, index_path, stream_name, errors)

    for idx, record in enumerate(records):
        context = f"{index_path}: {stream_name} camera record {idx}"
        image_path_value = record.get("image_path")
        if not isinstance(image_path_value, str) or not image_path_value:
            errors.append(f"{context}: image_path must be a non-empty string")
        else:
            image_relative = Path(image_path_value)
            if image_relative.is_absolute() or ".." in image_relative.parts:
                errors.append(f"{context}: image_path must be relative to episode root")
            else:
                image_path = root / image_relative
                if not _is_path_under_root(root, image_path):
                    errors.append(f"{context}: image_path must stay inside episode root")
                elif not image_path.is_file():
                    errors.append(f"{context}: image_path does not exist: {image_path}")
                else:
                    try:
                        image_size = image_path.stat().st_size
                    except OSError as exc:
                        errors.append(f"{context}: could not stat image_path {image_path}: {exc}")
                    else:
                        if image_size <= 0:
                            errors.append(f"{context}: image_path must reference a non-empty file: {image_path}")

        for key in ["width", "height", "channels"]:
            if not _is_positive_int(record.get(key)):
                errors.append(f"{context}: {key} must be a positive integer")

        for key in ["encoding", "frame_id"]:
            if not isinstance(record.get(key), str) or not record.get(key):
                errors.append(f"{context}: {key} must be a non-empty string")

        for key in ["camera_role", "camera_id", "external_camera_id"]:
            if key in record and (not isinstance(record.get(key), str) or not record.get(key)):
                errors.append(f"{context}: {key} must be a non-empty string when present")

    return records


def _validate_events(
    records: list[dict[str, Any]],
    path: Path,
    metadata: dict[str, Any] | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    if not records:
        errors.append(f"{path}: events must contain at least one record")
        return

    previous_timestamp: float | None = None
    event_names: list[str] = []
    for idx, record in enumerate(records):
        context = f"{path}: event record {idx}"
        timestamp = record.get("timestamp")
        if not _is_finite_number(timestamp):
            errors.append(f"{context}: timestamp must be a finite number")
        else:
            timestamp_float = float(timestamp)
            if previous_timestamp is not None and timestamp_float < previous_timestamp:
                errors.append(f"{context}: timestamp must be monotonic nondecreasing")
            previous_timestamp = timestamp_float

        event_name = record.get("event")
        if not _is_non_empty_string(event_name):
            errors.append(f"{context}: event must be a non-empty string")
        else:
            event_names.append(event_name)

    if event_names and event_names[0] != "episode_start":
        warnings.append(f"{path}: first event is not episode_start")
    if metadata is None or not isinstance(metadata.get("success"), bool):
        return
    if metadata["success"] and "success" not in event_names:
        warnings.append(f"{path}: metadata success is true but events do not contain success")
    if metadata["success"] is False and not any("fail" in event_name.lower() for event_name in event_names):
        warnings.append(f"{path}: metadata success is false but events do not contain a failure event")


def _record_index_set(records: list[dict[str, Any]]) -> set[int]:
    indexes: set[int] = set()
    for record in records:
        record_index = record.get("record_index")
        if _is_non_negative_int(record_index):
            indexes.add(record_index)
    return indexes


def _alignment_details(primary_indexes: set[int], candidate_indexes: set[int]) -> str:
    details: list[str] = []
    missing = sorted(primary_indexes - candidate_indexes)
    extra = sorted(candidate_indexes - primary_indexes)
    if missing:
        details.append(f"missing robot_state_rt record_index values {missing[:10]}")
    if extra:
        details.append(f"extra record_index values {extra[:10]}")
    return ", ".join(details) if details else "index sets differ"


def _check_record_index_alignment(
    records_by_stream: dict[str, list[dict[str, Any]]],
    camera_streams: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    primary_records = records_by_stream.get("robot_state_rt", [])
    primary_indexes = _record_index_set(primary_records)
    if not primary_indexes:
        return

    for stream_name in CONVERTER_REQUIRED_ALIGNMENT_STREAMS + camera_streams:
        if stream_name not in records_by_stream:
            continue
        candidate_indexes = _record_index_set(records_by_stream[stream_name])
        if candidate_indexes != primary_indexes:
            errors.append(
                f"{stream_name}: record_index alignment with robot_state_rt failed: "
                f"{_alignment_details(primary_indexes, candidate_indexes)}"
            )

    for stream_name in CONVERTER_ALIGNED_OPTIONAL_STREAMS:
        if stream_name not in records_by_stream:
            continue
        candidate_indexes = _record_index_set(records_by_stream[stream_name])
        if candidate_indexes != primary_indexes:
            errors.append(
                f"{stream_name}: record_index alignment with robot_state_rt failed: "
                f"{_alignment_details(primary_indexes, candidate_indexes)}"
            )

    for stream_name in ["tf", "command_context"]:
        if stream_name not in records_by_stream:
            continue
        candidate_indexes = _record_index_set(records_by_stream[stream_name])
        if candidate_indexes and candidate_indexes != primary_indexes:
            warnings.append(
                f"{stream_name}: record_index differs from robot_state_rt: "
                f"{_alignment_details(primary_indexes, candidate_indexes)}"
            )


def _records_by_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        record_index = record.get("record_index")
        if _is_non_negative_int(record_index) and record_index not in indexed:
            indexed[record_index] = record
    return indexed




def _has_finite_gripper_value(record: dict[str, Any]) -> bool:
    return _is_finite_number(record.get("gripper_position")) or _is_finite_number(record.get("gripper_width_m"))

def _default_camera_robot_source_stamp_tolerance(metadata: dict[str, Any] | None) -> tuple[float, str]:
    fps = _positive_fps(metadata)
    if fps is None:
        return (
            CAMERA_ROBOT_SOURCE_STAMP_FALLBACK_TOLERANCE_SEC,
            "default fallback because metadata.fps is missing or invalid",
        )
    tolerance = CAMERA_ROBOT_SOURCE_STAMP_TOLERANCE_FPS_FRACTION / fps
    return tolerance, f"default {CAMERA_ROBOT_SOURCE_STAMP_TOLERANCE_FPS_FRACTION:g}/fps from metadata.fps={fps:g}"


def _max_camera_robot_source_stamp_override(metadata: dict[str, Any] | None) -> tuple[float, str]:
    fps = _positive_fps(metadata)
    if fps is None:
        return CAMERA_ROBOT_SOURCE_STAMP_MAX_OVERRIDE_SEC, "absolute maximum without valid metadata.fps"
    max_by_fps = CAMERA_ROBOT_SOURCE_STAMP_MAX_OVERRIDE_FRAME_FRACTION / fps
    return (
        min(CAMERA_ROBOT_SOURCE_STAMP_MAX_OVERRIDE_SEC, max_by_fps),
        f"min({CAMERA_ROBOT_SOURCE_STAMP_MAX_OVERRIDE_SEC:g}, "
        f"{CAMERA_ROBOT_SOURCE_STAMP_MAX_OVERRIDE_FRAME_FRACTION:g}/fps) for metadata.fps={fps:g}",
    )


def _camera_robot_source_stamp_tolerance_policy(
    metadata: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
) -> tuple[float, str, list[str], list[str]]:
    default_tolerance, default_source = _default_camera_robot_source_stamp_tolerance(metadata)
    if not isinstance(streams_index, dict):
        return default_tolerance, default_source, [], []

    timebase = streams_index.get("timebase")
    if not isinstance(timebase, dict) or CAMERA_ROBOT_SOURCE_STAMP_OVERRIDE_KEY not in timebase:
        return default_tolerance, default_source, [], []

    override = timebase.get(CAMERA_ROBOT_SOURCE_STAMP_OVERRIDE_KEY)
    override_path = f"streams/index.json timebase.{CAMERA_ROBOT_SOURCE_STAMP_OVERRIDE_KEY}"
    if not _is_finite_number(override) or float(override) <= 0.0:
        return (
            default_tolerance,
            default_source,
            [f"{override_path} must be a finite positive number of seconds; got {override!r}"],
            [],
        )

    max_override, max_source = _max_camera_robot_source_stamp_override(metadata)
    override_seconds = float(override)
    if override_seconds > max_override:
        return (
            default_tolerance,
            default_source,
            [
                f"{override_path}={override_seconds:.6f}s exceeds allowed maximum "
                f"{max_override:.6f}s ({max_source})"
            ],
            [],
        )

    warnings: list[str] = []
    if override_seconds > default_tolerance:
        warnings.append(
            f"{override_path}={override_seconds:.6f}s exceeds default camera/robot source_stamp "
            f"tolerance {default_tolerance:.6f}s ({default_source}); use only with documented clock-offset review"
        )
    return override_seconds, f"explicit {override_path}", [], warnings


def timestamp_tolerance_seconds(
    metadata: dict[str, Any] | None,
    streams_index: dict[str, Any] | None = None,
) -> float:
    tolerance, _, _, _ = _camera_robot_source_stamp_tolerance_policy(metadata, streams_index)
    return tolerance


def _source_stamp_alignment_errors(
    records_by_stream: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
    camera_streams: list[str],
) -> list[str]:
    tolerance, tolerance_source, errors, _ = _camera_robot_source_stamp_tolerance_policy(metadata, streams_index)
    robot_by_index = _records_by_index(records_by_stream.get("robot_state_rt", []))
    if not robot_by_index:
        return errors

    for stream_name in camera_streams:
        stream_by_index = _records_by_index(records_by_stream.get(stream_name, []))
        common_indexes = sorted(set(robot_by_index) & set(stream_by_index))
        for record_index in common_indexes:
            robot_stamp = _source_stamp_seconds(robot_by_index[record_index].get("source_stamp"))
            stream_stamp = _source_stamp_seconds(stream_by_index[record_index].get("source_stamp"))
            if robot_stamp is None or stream_stamp is None:
                continue
            offset = abs(stream_stamp - robot_stamp)
            if offset > tolerance:
                errors.append(
                    f"{stream_name}: source_stamp differs from robot_state_rt by {offset:.6f}s "
                    f"at record_index {record_index}; allowed camera/robot source_stamp offset is "
                    f"{tolerance:.6f}s ({tolerance_source}); raw_real_v0 conversion pairs records "
                    "by aligned episode-level record_index values"
                )
                break
    return errors


def _is_placeholder_gripper_value(value: Any) -> bool:
    normalized = _normalized_label(value)
    if normalized is None:
        return False
    return normalized in PLACEHOLDER_GRIPPER_SOURCE_NAMES or (
        ("synthetic" in normalized or "pipeline_smoke" in normalized) and "gripper" in normalized
    )


def _gripper_placeholder_sources(
    streams: dict[str, Any] | None,
    records: list[dict[str, Any]],
) -> list[str]:
    sources: list[str] = []
    entry = streams.get("gripper_state") if isinstance(streams, dict) else None
    if isinstance(entry, dict):
        for key in ["source_name", "source_type"]:
            value = entry.get(key)
            if _is_placeholder_gripper_value(value) and isinstance(value, str):
                sources.append(f"streams.gripper_state.{key}={value}")
        if entry.get("placeholder") is True or entry.get("synthetic_placeholder") is True:
            sources.append("streams.gripper_state.placeholder=true")
    for record_index, record in enumerate(records):
        for key in ["source_name", "source_type"]:
            value = record.get(key)
            if _is_placeholder_gripper_value(value) and isinstance(value, str):
                sources.append(f"gripper_state record {record_index} {key}={value}")
                break
    return _unique_strings(sources)


def _gripper_state_readiness_errors(
    records_by_stream: dict[str, list[dict[str, Any]]],
    streams: dict[str, Any] | None,
) -> list[str]:
    robot_indexes = _record_index_set(records_by_stream.get("robot_state_rt", []))
    if not robot_indexes:
        return []

    gripper_records = records_by_stream.get("gripper_state", [])
    if not gripper_records:
        return [
            "gripper_state is required for non-synthetic conversion; missing gripper state would otherwise "
            "be silently zero-filled as gripper_pos=0.0"
        ]

    errors: list[str] = []
    placeholder_sources = _gripper_placeholder_sources(streams, gripper_records)
    if placeholder_sources:
        errors.append(
            "gripper_state uses synthetic/pipeline-smoke placeholder source metadata that is allowed only for "
            "explicit synthetic or pipeline-smoke episodes, not non-synthetic real training data: "
            + ", ".join(placeholder_sources[:5])
        )
    gripper_by_index = _records_by_index(gripper_records)
    gripper_indexes = set(gripper_by_index)
    if gripper_indexes != robot_indexes:
        errors.append(
            "gripper_state must contain one aligned record for every robot_state_rt record_index for "
            f"non-synthetic conversion; {_alignment_details(robot_indexes, gripper_indexes)}; "
            "silent gripper_pos=0.0 fallback is not allowed"
        )

    for record_index in sorted(robot_indexes & gripper_indexes):
        if not _has_finite_gripper_value(gripper_by_index[record_index]):
            errors.append(
                f"gripper_state record_index {record_index}: non-synthetic conversion requires finite "
                "gripper_position or gripper_width_m; silent gripper_pos=0.0 fallback is not allowed"
            )

    return errors

def raw_real_conversion_readiness_errors(
    metadata: dict[str, Any] | None,
    recorder_report: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
    streams: dict[str, Any] | None,
    records_by_stream: dict[str, list[dict[str, Any]]],
    *,
    root_dir: str | Path | None = None,
    calibration_refs: dict[str, Any] | None = None,
) -> list[str]:
    """Return validator errors for raw_real_v0 data the converter would reject."""

    camera_streams = camera_stream_names(streams)
    errors = _source_stamp_alignment_errors(records_by_stream, metadata, streams_index, camera_streams)
    _, camera_mapping_errors, _ = select_model_camera_streams(metadata, streams_index, streams)
    errors.extend(camera_mapping_errors)
    if is_explicit_synthetic_episode(metadata, recorder_report, streams_index):
        return errors

    if root_dir is None:
        errors.append("raw-real episode root is required to verify non-synthetic camera image decodability")
    else:
        errors.extend(_camera_image_decodability_errors(Path(root_dir), records_by_stream, camera_streams))

    errors.extend(_calibration_refs_readiness_errors(calibration_refs, camera_streams))
    if not isinstance(streams, dict):
        return errors

    errors.extend(_numeric_source_stamp_timebase_errors(streams_index, records_by_stream, camera_streams))
    errors.extend(_gripper_state_readiness_errors(records_by_stream, streams))

    convention = _tcp_orientation_convention(metadata, recorder_report)
    convention_error = tcp_orientation_convention_readiness_error(convention)
    if convention_error is not None:
        errors.append(convention_error)

    errors.extend(_strict_lab_provenance_errors(metadata, recorder_report, streams, records_by_stream))
    errors.extend(_wrench_metadata_errors(streams, records_by_stream))

    robot_entry = streams.get("robot_state_rt") if isinstance(streams.get("robot_state_rt"), dict) else {}
    joint_entry = streams.get("joint_states") if isinstance(streams.get("joint_states"), dict) else {}
    robot_records = records_by_stream.get("robot_state_rt", [])
    joint_by_index = _records_by_index(records_by_stream.get("joint_states", []))

    for idx, robot_record in enumerate(robot_records):
        context = f"robot_state_rt record {idx}"
        robot_units = _combined_units(robot_record, robot_entry)

        tcp_position_error = _supported_unit_error(
            robot_units,
            "tcp_position",
            SUPPORTED_TCP_POSITION_UNITS,
            f"{context} actual_tcp_position",
        )
        if tcp_position_error is not None:
            errors.append(tcp_position_error)

        tcp_orientation_error = _supported_unit_error(
            robot_units,
            "tcp_orientation",
            SUPPORTED_TCP_ORIENTATION_UNITS,
            f"{context} actual_tcp_position[3:6]",
        )
        if tcp_orientation_error is not None:
            errors.append(tcp_orientation_error)
        else:
            mismatch_error = _orientation_unit_matches_convention(
                _normalized_unit(robot_units, "tcp_orientation"),
                convention,
                f"{context} actual_tcp_position[3:6]",
            )
            if mismatch_error is not None:
                errors.append(mismatch_error)

        record_index = robot_record.get("record_index")
        joint_record = joint_by_index.get(record_index) if _is_non_negative_int(record_index) else None
        joint_units = _combined_units(joint_record, joint_entry)

        if _is_numeric_list(robot_record.get("actual_joint_position"), 6):
            joint_position_error = _supported_unit_error(
                robot_units,
                "joint_position",
                SUPPORTED_JOINT_POSITION_UNITS,
                f"{context} actual_joint_position",
            )
            if joint_position_error is not None:
                errors.append(joint_position_error)
        elif not isinstance(joint_record, dict) or not _is_numeric_list(joint_record.get("position"), 6):
            errors.append(
                f"{context} actual_joint_position is missing/invalid and joint_states record_index "
                f"{record_index!r} position is not valid for fallback"
            )
        else:
            joint_position_error = _supported_unit_error(
                joint_units,
                "position",
                SUPPORTED_JOINT_POSITION_UNITS,
                f"joint_states record_index {record_index} position",
            )
            if joint_position_error is not None:
                errors.append(joint_position_error)

        if _is_numeric_list(robot_record.get("actual_joint_velocity"), 6):
            joint_velocity_error = _supported_unit_error(
                robot_units,
                "joint_velocity",
                SUPPORTED_JOINT_VELOCITY_UNITS,
                f"{context} actual_joint_velocity",
            )
            if joint_velocity_error is not None:
                errors.append(joint_velocity_error)
        elif not isinstance(joint_record, dict) or not _is_numeric_list(joint_record.get("velocity"), 6):
            errors.append(
                f"{context} actual_joint_velocity is missing/invalid and joint_states record_index "
                f"{record_index!r} velocity is not valid for fallback"
            )
        else:
            joint_velocity_error = _supported_unit_error(
                joint_units,
                "velocity",
                SUPPORTED_JOINT_VELOCITY_UNITS,
                f"joint_states record_index {record_index} velocity",
            )
            if joint_velocity_error is not None:
                errors.append(joint_velocity_error)

    return errors



def _warn_timestamp_mismatches(
    records_by_stream: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any] | None,
    streams_index: dict[str, Any] | None,
    camera_streams: list[str],
    warnings: list[str],
) -> None:
    robot_by_index = _records_by_index(records_by_stream.get("robot_state_rt", []))
    if not robot_by_index:
        return
    tolerance, tolerance_source, _, tolerance_warnings = _camera_robot_source_stamp_tolerance_policy(metadata, streams_index)
    warnings.extend(tolerance_warnings)

    for stream_name in [
        "joint_states",
        *camera_streams,
        "gripper_state",
    ]:
        stream_by_index = _records_by_index(records_by_stream.get(stream_name, []))
        common_indexes = sorted(set(robot_by_index) & set(stream_by_index))
        for record_index in common_indexes:
            robot_record = robot_by_index[record_index]
            stream_record = stream_by_index[record_index]
            for stamp_key in ["source_stamp", "receipt_stamp"]:
                robot_stamp = robot_record.get(stamp_key)
                stream_stamp = stream_record.get(stamp_key)
                if not _is_finite_number(robot_stamp) or not _is_finite_number(stream_stamp):
                    continue
                offset = abs(float(stream_stamp) - float(robot_stamp))
                if offset > tolerance:
                    warnings.append(
                        f"{stream_name}: {stamp_key} differs from robot_state_rt by {offset:.6f}s "
                        f"at record_index {record_index}; allowed timing offset is {tolerance:.6f}s "
                        f"({tolerance_source})"
                    )
                    break

def validate_raw_real_episode(root_dir: str | Path) -> ValidationResult:
    """Validate a raw real episode directory."""

    root = Path(root_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if not root.exists():
        return ValidationResult(False, [f"{root}: episode directory does not exist"], warnings)
    if not root.is_dir():
        return ValidationResult(False, [f"{root}: episode path is not a directory"], warnings)

    for relative_path in REQUIRED_TOP_LEVEL_FILES:
        path = root / relative_path
        if not path.is_file():
            errors.append(f"{path}: required file is missing")
        elif not _is_path_under_root(root, path):
            errors.append(f"{path}: required file must stay inside episode root")
    if errors:
        return ValidationResult(False, errors, warnings)

    metadata_path = root / "metadata.json"
    metadata = _read_json_object(metadata_path, errors, "metadata")
    if metadata is not None:
        _check_metadata(metadata, metadata_path, errors)

    calibration_refs = _read_json_object(root / "calibration_refs.json", errors, "calibration_refs")
    recorder_report = _read_json_object(root / "recorder_report.json", errors, "recorder_report")
    events = _read_jsonl_objects(root / "events.jsonl", errors, "events")
    _validate_events(events, root / "events.jsonl", metadata, errors, warnings)

    streams_index_path = root / "streams" / "index.json"
    streams_index = _read_json_object(streams_index_path, errors, "streams/index")
    if streams_index is None:
        return ValidationResult(False, errors, warnings)

    if streams_index.get("schema_version") != RAW_REAL_SCHEMA_VERSION:
        errors.append(f"{streams_index_path}: schema_version must be {RAW_REAL_SCHEMA_VERSION!r}")

    streams = streams_index.get("streams")
    if not isinstance(streams, dict):
        errors.append(f"{streams_index_path}: streams must be a JSON object")
        return ValidationResult(False, errors, warnings)

    _add_stream_warnings(streams, warnings)
    declared_camera_streams = camera_stream_names(streams)
    if not declared_camera_streams:
        errors.append(f"{streams_index_path}: at least one camera_rgb stream must be declared in streams")

    required_stream_paths: dict[str, Path] = {}
    for stream_name in REQUIRED_STREAM_NAMES:
        if stream_name not in streams:
            errors.append(f"{streams_index_path}: required stream is missing: {stream_name}")
            continue
        path = _validate_required_stream_entry(root, stream_name, streams[stream_name], errors)
        if path is not None and isinstance(streams[stream_name], dict):
            required_stream_paths[stream_name] = path

    optional_stream_paths: dict[str, Path] = {}
    for stream_name in OPTIONAL_STREAM_NAMES:
        if stream_name not in streams:
            continue
        path = _validate_optional_stream_entry(root, stream_name, streams[stream_name], errors)
        if path is not None and isinstance(streams[stream_name], dict):
            optional_stream_paths[stream_name] = path

    camera_stream_paths: dict[str, Path] = {}
    for stream_name in declared_camera_streams:
        path = _validate_camera_stream_entry(root, stream_name, streams[stream_name], errors)
        if path is not None and isinstance(streams[stream_name], dict):
            camera_stream_paths[stream_name] = path

    records_by_stream: dict[str, list[dict[str, Any]]] = {}

    for stream_name, path in required_stream_paths.items():
        stream_entry = streams[stream_name]
        if not isinstance(stream_entry, dict):
            continue
        if stream_name in JSONL_STREAM_NAMES:
            records_by_stream[stream_name] = _validate_jsonl_stream(
                path,
                stream_name,
                stream_entry,
                errors,
                warnings,
                required=True,
            )
    for stream_name, path in camera_stream_paths.items():
        stream_entry = streams[stream_name]
        if isinstance(stream_entry, dict):
            records_by_stream[stream_name] = _validate_camera_index(
                root,
                path,
                stream_name,
                stream_entry,
                errors,
            )

    for stream_name, path in optional_stream_paths.items():
        stream_entry = streams[stream_name]
        if isinstance(stream_entry, dict) and stream_name in JSONL_STREAM_NAMES:
            records_by_stream[stream_name] = _validate_jsonl_stream(
                path,
                stream_name,
                stream_entry,
                errors,
                warnings,
                required=False,
            )

    _check_record_index_alignment(records_by_stream, declared_camera_streams, errors, warnings)
    errors.extend(
        raw_real_conversion_readiness_errors(
            metadata,
            recorder_report,
            streams_index,
            streams,
            records_by_stream,
            root_dir=root,
            calibration_refs=calibration_refs,
        )
    )
    _warn_timestamp_mismatches(records_by_stream, metadata, streams_index, declared_camera_streams, warnings)

    return ValidationResult(not errors, errors, warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a raw real episode directory.")
    parser.add_argument("episode_dir", help="Path to raw real episode directory")
    args = parser.parse_args(argv)

    result = validate_raw_real_episode(args.episode_dir)
    if result.ok:
        print(f"OK: raw real episode is valid: {args.episode_dir}")
        return 0

    print(f"FAILED: raw real episode is invalid: {args.episode_dir}")
    for error in result.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
