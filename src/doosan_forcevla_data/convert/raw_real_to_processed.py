"""Convert a validated raw_real_v0 episode into a processed JSONL episode.

This converter is intentionally offline and dependency-light. It reads files
created by a passive raw-real recorder or the synthetic raw-real generator; it
does not import ROS packages and does not communicate with a robot.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.compute_actions import compute_measured_tcp_delta_action
from doosan_forcevla_data.schema.processed_schema import (
    ACTION_DIM,
    MODEL_STATE_DIM,
    QUATERNION_CONVENTION,
)
from doosan_forcevla_data.schema.raw_real_units import (
    UnitResolution,
    resolve_tcp_orientation_unit,
    resolve_tcp_position_unit,
)
from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode
from doosan_forcevla_data.validate.validate_raw_real_episode import (
    MODEL_EXTERNAL_IMAGE_KEY,
    MODEL_TCP_IMAGE_KEY,
    camera_stream_names,
    tcp_orientation_convention_readiness_error,
    is_explicit_synthetic_episode,
    raw_real_conversion_readiness_errors,
    select_model_camera_streams,
    selected_wrench_metadata_for_model_state,
    validate_raw_real_episode,
)


DATASET_NAME = "doosan_peg_in_hole_v0"
CONVERTER_VERSION = "raw_real_to_processed_v0"
ROTATION_VECTOR_DEGREES = "rotation_vector_degrees"
ROTATION_VECTOR_RADIANS = "rotation_vector_radians"
LEGACY_SYNTHETIC_ROTATION_VECTOR = "legacy_synthetic_rotation_vector"
PROCESSED_METADATA_SCHEMA_VERSION = "processed_jsonl_v1"
MODEL_STATE_LAYOUT_VERSION = "doosan_model_state_25d_v1"
MEASURED_ACTION_LAYOUT_VERSION = "measured_tcp_delta_7d_v1"
WRENCH_SOURCE_FIELDS = ["tcp_wrench", "measured_tcp_wrench", "external_tcp_force", "raw_force_torque"]


@dataclass(frozen=True)
class OrientationPolicy:
    synthetic: bool
    source_convention: str
    source: str
    legacy_fallback: bool
    note: str


@dataclass(frozen=True)
class TcpConversionInfo:
    position_unit: UnitResolution
    orientation_unit: UnitResolution
    position_conversion: str
    orientation_conversion: str


@dataclass(frozen=True)
class GripperMetadata:
    state_source: str
    state_value_field: str
    state_unit: str
    provenance: str
    is_placeholder: bool
    verified: bool
    valid_for_training: bool


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: could not read JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{path}: could not read JSONL: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{path}: line {line_number} is empty")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        records.append(record)
    return records


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _finite_float(value: Any, context: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _finite_vector(value: Any, expected_len: int, context: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of length {expected_len}")
    if len(value) != expected_len:
        raise ValueError(f"{context} must have length {expected_len}, got {len(value)}")
    return [_finite_float(item, f"{context}[{idx}]") for idx, item in enumerate(value)]


def _maybe_finite_vector(value: Any, expected_len: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected_len:
        return None
    if not all(_is_finite_number(item) for item in value):
        return None
    return [float(item) for item in value]


def _stream_path(raw_root: Path, streams: dict[str, Any], stream_name: str) -> Path:
    entry = streams.get(stream_name)
    if not isinstance(entry, dict):
        raise ValueError(f"streams/index.json: stream {stream_name} entry is missing or invalid")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"streams/index.json: stream {stream_name} missing path")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"streams/index.json: stream {stream_name} path must be safe and relative")
    return raw_root / relative_path


def _stream_entry(streams: dict[str, Any], stream_name: str) -> dict[str, Any]:
    entry = streams.get(stream_name)
    if not isinstance(entry, dict):
        return {}
    return entry


def _load_camera_index(raw_root: Path, streams: dict[str, Any], stream_name: str) -> list[dict[str, Any]]:
    return _read_jsonl_objects(_stream_path(raw_root, streams, stream_name) / "index.jsonl")


def _records_by_index(records: list[dict[str, Any]], stream_name: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row_number, record in enumerate(records):
        record_index = record.get("record_index")
        if isinstance(record_index, bool) or not isinstance(record_index, int):
            raise ValueError(f"{stream_name} record {row_number}: record_index must be an integer")
        if record_index in indexed:
            raise ValueError(f"{stream_name}: duplicate record_index {record_index}")
        indexed[record_index] = record
    return indexed


def _require_aligned_indexes(
    primary_indexes: set[int],
    candidate: dict[int, dict[str, Any]],
    stream_name: str,
) -> None:
    candidate_indexes = set(candidate)
    if candidate_indexes == primary_indexes:
        return
    missing = sorted(primary_indexes - candidate_indexes)
    extra = sorted(candidate_indexes - primary_indexes)
    details: list[str] = []
    if missing:
        details.append(f"missing primary record_index values {missing[:10]}")
    if extra:
        details.append(f"extra record_index values {extra[:10]}")
    raise ValueError(f"{stream_name}: record_index alignment with robot_state_rt failed: {', '.join(details)}")


def _timestamp_seconds(value: Any, context: str) -> float:
    if _is_finite_number(value):
        return float(value)
    if isinstance(value, dict):
        sec = _finite_float(value.get("sec"), f"{context}.sec")
        nanosec = _finite_float(value.get("nanosec"), f"{context}.nanosec")
        if nanosec < 0.0 or nanosec >= 1_000_000_000.0:
            raise ValueError(f"{context}.nanosec must be in [0, 1e9)")
        return sec + nanosec * 1e-9
    raise ValueError(f"{context} must be numeric or an object with sec/nanosec")


def _strictly_increasing(values: list[float]) -> bool:
    return all(values[idx] > values[idx - 1] for idx in range(1, len(values)))


def _relative_timestamps(robot_records: list[dict[str, Any]]) -> list[float]:
    source_times = [
        _timestamp_seconds(record.get("source_stamp"), f"robot_state_rt record {idx} source_stamp")
        for idx, record in enumerate(robot_records)
    ]
    if len(source_times) <= 1 or _strictly_increasing(source_times):
        first = source_times[0]
        return [time - first for time in source_times]

    monotonic_times = [
        _finite_float(record.get("monotonic_stamp"), f"robot_state_rt record {idx} monotonic_stamp")
        for idx, record in enumerate(robot_records)
    ]
    if len(monotonic_times) <= 1 or _strictly_increasing(monotonic_times):
        first = monotonic_times[0]
        return [time - first for time in monotonic_times]

    raise ValueError("robot_state_rt timestamps are not strictly increasing by source_stamp or monotonic_stamp")


def _normalized_unit(units: dict[str, Any], key: str) -> str | None:
    value = units.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace(" ", "_")


def _combined_units(record: dict[str, Any], stream_entry: dict[str, Any]) -> dict[str, Any]:
    units: dict[str, Any] = {}
    entry_units = stream_entry.get("units")
    if isinstance(entry_units, dict):
        units.update(entry_units)
    record_units = record.get("units")
    if isinstance(record_units, dict):
        units.update(record_units)
    return units


def _tcp_position_conversion_name(unit: str) -> str:
    if unit == "mm":
        return "millimeters_to_meters_divide_by_1000"
    if unit == "m":
        return "preserved_meters"
    return f"unsupported_tcp_position_unit_{unit}"


def _tcp_orientation_conversion_name(unit: str) -> str:
    if unit == "deg":
        return "rotation_vector_degrees_to_radians_multiply_by_pi_over_180"
    if unit == "rad":
        return "preserved_rotation_vector_radians"
    return f"unsupported_tcp_orientation_unit_{unit}"


def _convert_tcp_position(values: list[float], unit_resolution: UnitResolution, context: str) -> tuple[list[float], str]:
    if not unit_resolution.ok:
        raise ValueError("\n".join(unit_resolution.errors) or f"{context}: tcp_position unit could not be resolved")
    unit = unit_resolution.unit
    if unit == "mm":
        return [value / 1000.0 for value in values], _tcp_position_conversion_name(unit)
    if unit == "m":
        return values, _tcp_position_conversion_name(unit)
    raise ValueError(f"{context}: unsupported or missing tcp_position unit: {unit!r}")


def _convert_tcp_orientation_rotvec(
    values: list[float],
    unit_resolution: UnitResolution,
    orientation_policy: OrientationPolicy,
    context: str,
) -> tuple[list[float], str]:
    if not unit_resolution.ok:
        raise ValueError("\n".join(unit_resolution.errors) or f"{context}: tcp_orientation unit could not be resolved")
    unit = unit_resolution.unit
    if orientation_policy.source_convention == ROTATION_VECTOR_RADIANS and unit != "rad":
        raise ValueError(
            f"{context}: tcp_orientation unit {unit!r} does not match "
            "tcp_orientation_convention='rotation_vector_radians'"
        )
    if orientation_policy.source_convention == ROTATION_VECTOR_DEGREES and unit != "deg":
        raise ValueError(
            f"{context}: tcp_orientation unit {unit!r} does not match "
            "tcp_orientation_convention='rotation_vector_degrees'"
        )
    if unit == "deg":
        return [math.radians(value) for value in values], _tcp_orientation_conversion_name(unit)
    if unit == "rad":
        return values, _tcp_orientation_conversion_name(unit)
    raise ValueError(f"{context}: unsupported or missing tcp_orientation unit: {unit!r}")


def _convert_joint_position(values: list[float], units: dict[str, Any], synthetic: bool, context: str) -> list[float]:
    unit = _normalized_unit(units, "joint_position") or _normalized_unit(units, "position")
    if unit is None and synthetic:
        unit = "deg"
    if unit in {"deg", "degree", "degrees"}:
        return [math.radians(value) for value in values]
    if unit in {"rad", "radian", "radians"}:
        return values
    raise ValueError(f"{context}: unsupported or missing joint position unit: {unit!r}")


def _convert_joint_velocity(values: list[float], units: dict[str, Any], synthetic: bool, context: str) -> list[float]:
    unit = _normalized_unit(units, "joint_velocity") or _normalized_unit(units, "velocity")
    if unit is None and synthetic:
        unit = "deg/s"
    if unit in {"deg/s", "deg_per_s", "degree/s", "degrees/s", "degrees_per_second"}:
        return [math.radians(value) for value in values]
    if unit in {"rad/s", "rad_per_s", "radian/s", "radians/s", "radians_per_second"}:
        return values
    raise ValueError(f"{context}: unsupported or missing joint velocity unit: {unit!r}")


def _rotvec_to_quat_xyzw(rotvec: list[float]) -> list[float]:
    rx, ry, rz = _finite_vector(rotvec, 3, "rotation vector")
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return [0.5 * rx, 0.5 * ry, 0.5 * rz, 1.0]
    scale = math.sin(angle / 2.0) / angle
    quat = [rx * scale, ry * scale, rz * scale, math.cos(angle / 2.0)]
    norm = math.sqrt(sum(component * component for component in quat))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("rotation vector produced invalid quaternion")
    return [component / norm for component in quat]


def _orientation_policy(
    metadata: dict[str, Any],
    recorder_report: dict[str, Any],
    streams_index: dict[str, Any],
) -> OrientationPolicy:
    synthetic = is_explicit_synthetic_episode(metadata, recorder_report, streams_index)
    if metadata.get("tcp_orientation_convention") is not None:
        convention = metadata.get("tcp_orientation_convention")
        source = "metadata.json.tcp_orientation_convention"
    else:
        convention = recorder_report.get("tcp_orientation_convention")
        source = "recorder_report.json.tcp_orientation_convention"

    if convention == ROTATION_VECTOR_DEGREES:
        return OrientationPolicy(
            synthetic=synthetic,
            source_convention=ROTATION_VECTOR_DEGREES,
            source=source,
            legacy_fallback=False,
            note="tcp_orientation_convention=rotation_vector_degrees",
        )
    if convention == ROTATION_VECTOR_RADIANS:
        return OrientationPolicy(
            synthetic=synthetic,
            source_convention=ROTATION_VECTOR_RADIANS,
            source=source,
            legacy_fallback=False,
            note="tcp_orientation_convention=rotation_vector_radians",
        )

    if convention is None and synthetic:
        return OrientationPolicy(
            synthetic=True,
            source_convention=LEGACY_SYNTHETIC_ROTATION_VECTOR,
            source="legacy synthetic raw_real_v0 compatibility fallback",
            legacy_fallback=True,
            note=(
                "legacy synthetic raw-real compatibility: actual_tcp_position[3:6] is treated as a "
                "rotation vector; orientation units come from explicit unit metadata or the legacy deg fallback"
            ),
        )

    error = tcp_orientation_convention_readiness_error(convention)
    if error is None:
        error = f"tcp_orientation_convention {convention!r} is not implemented by this converter"
    raise ValueError(error)


def _select_joint_vectors(
    robot_record: dict[str, Any],
    robot_entry: dict[str, Any],
    joint_record: dict[str, Any],
    joint_entry: dict[str, Any],
    synthetic: bool,
    frame_index: int,
) -> tuple[list[float], list[float], str]:
    robot_units = _combined_units(robot_record, robot_entry)
    joint_units = _combined_units(joint_record, joint_entry)

    robot_joint_pos = _maybe_finite_vector(robot_record.get("actual_joint_position"), 6)
    if robot_joint_pos is not None:
        joint_pos = _convert_joint_position(
            robot_joint_pos, robot_units, synthetic, f"robot_state_rt record {frame_index} actual_joint_position"
        )
        joint_pos_source = "robot_state_rt.actual_joint_position"
    else:
        joint_pos = _convert_joint_position(
            _finite_vector(joint_record.get("position"), 6, f"joint_states record {frame_index} position"),
            joint_units,
            synthetic,
            f"joint_states record {frame_index} position",
        )
        joint_pos_source = "joint_states.position"

    robot_joint_vel = _maybe_finite_vector(robot_record.get("actual_joint_velocity"), 6)
    if robot_joint_vel is not None:
        joint_vel = _convert_joint_velocity(
            robot_joint_vel, robot_units, synthetic, f"robot_state_rt record {frame_index} actual_joint_velocity"
        )
        joint_vel_source = "robot_state_rt.actual_joint_velocity"
    else:
        joint_vel = _convert_joint_velocity(
            _finite_vector(joint_record.get("velocity"), 6, f"joint_states record {frame_index} velocity"),
            joint_units,
            synthetic,
            f"joint_states record {frame_index} velocity",
        )
        joint_vel_source = "joint_states.velocity"

    return joint_pos, joint_vel, f"{joint_pos_source}; {joint_vel_source}"


def _select_wrench(robot_record: dict[str, Any], frame_index: int) -> tuple[list[float], str]:
    for source in WRENCH_SOURCE_FIELDS:
        wrench = _maybe_finite_vector(robot_record.get(source), 6)
        if wrench is not None:
            return wrench, source
    raise ValueError(
        f"robot_state_rt record {frame_index}: tcp_wrench, measured_tcp_wrench, external_tcp_force, "
        "or raw_force_torque must contain 6 finite values"
    )


def _select_gripper(gripper_record: dict[str, Any] | None, frame_index: int) -> float:
    if gripper_record is None:
        return 0.0
    if _is_finite_number(gripper_record.get("gripper_position")):
        return float(gripper_record["gripper_position"])
    if _is_finite_number(gripper_record.get("gripper_width_m")):
        return float(gripper_record["gripper_width_m"])
    raise ValueError(
        f"gripper_state record {frame_index}: expected finite gripper_position or gripper_width_m"
    )


def _has_gripper_value(record: dict[str, Any]) -> bool:
    return _is_finite_number(record.get("gripper_position")) or _is_finite_number(record.get("gripper_width_m"))


def _require_non_synthetic_gripper_state(
    primary_indexes: set[int],
    gripper_by_index: dict[int, dict[str, Any]],
) -> None:
    if set(gripper_by_index) != primary_indexes:
        missing = sorted(primary_indexes - set(gripper_by_index))
        extra = sorted(set(gripper_by_index) - primary_indexes)
        details: list[str] = []
        if missing:
            details.append(f"missing robot_state_rt record_index values {missing[:10]}")
        if extra:
            details.append(f"extra record_index values {extra[:10]}")
        detail_text = "; ".join(details) if details else "index sets differ"
        raise ValueError(
            "non-synthetic raw-real conversion requires aligned gripper_state records; "
            f"{detail_text}; refusing silent gripper_pos=0.0 fallback"
        )

    for record_index in sorted(primary_indexes):
        if not _has_gripper_value(gripper_by_index[record_index]):
            raise ValueError(
                f"gripper_state record_index {record_index}: non-synthetic raw-real conversion requires finite "
                "gripper_position or gripper_width_m; refusing silent gripper_pos=0.0 fallback"
            )


def _resolve_raw_image(raw_root: Path, image_path_value: Any, frame_index: int, stream_name: str) -> Path:
    if not isinstance(image_path_value, str) or not image_path_value:
        raise ValueError(f"{stream_name} record {frame_index}: image_path must be a non-empty string")
    relative_path = Path(image_path_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{stream_name} record {frame_index}: image_path must be safe and relative")
    image_path = raw_root / relative_path
    if not image_path.is_file():
        raise ValueError(f"{stream_name} record {frame_index}: image file does not exist: {image_path}")
    return image_path


def _frame_image_path(
    *,
    raw_root: Path,
    output_root: Path,
    camera_record: dict[str, Any],
    stream_name: str,
    frame_index: int,
    copy_images: bool,
) -> str:
    raw_image = _resolve_raw_image(raw_root, camera_record.get("image_path"), frame_index, stream_name)
    if not copy_images:
        return str(Path(str(camera_record["image_path"])))

    suffix = raw_image.suffix
    target_relative = Path("images") / stream_name / f"{frame_index:06d}{suffix}"
    target_path = output_root / target_relative
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_image, target_path)
    return target_relative.as_posix()


def _camera_stream_metadata(streams: dict[str, Any], stream_names: list[str]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for stream_name in stream_names:
        entry = streams.get(stream_name)
        if not isinstance(entry, dict):
            continue
        metadata[stream_name] = {
            key: entry[key]
            for key in [
                "path",
                "type",
                "stream_type",
                "kind",
                "role",
                "camera_role",
                "camera_id",
                "external_camera_id",
                "source_name",
                "source_type",
                "model_input_key",
                "used_for_model",
                "record_count",
                "verified",
            ]
            if key in entry
        }
    return metadata


def _processed_camera_mapping(
    selected_camera_streams: dict[str, str],
    selection_sources: dict[str, str],
) -> dict[str, dict[str, str]]:
    return {
        "external_rgb_path": {
            "raw_stream": selected_camera_streams[MODEL_EXTERNAL_IMAGE_KEY],
            "model_input": "observation.image",
            "selection_source": selection_sources.get(MODEL_EXTERNAL_IMAGE_KEY, "unknown"),
        },
        "tcp_rgb_path": {
            "raw_stream": selected_camera_streams[MODEL_TCP_IMAGE_KEY],
            "model_input": "observation.wrist_image",
            "selection_source": selection_sources.get(MODEL_TCP_IMAGE_KEY, "unknown"),
        },
    }


def _contains_path(parent: Path, child: Path) -> bool:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    try:
        child_resolved.relative_to(parent_resolved)
    except ValueError:
        return False
    return True


def _prepare_output(raw_root: Path, output_root: Path, overwrite: bool) -> None:
    if _contains_path(raw_root, output_root):
        raise ValueError(
            f"output directory cannot be inside the raw-real episode directory: {output_root}"
        )
    if output_root.exists() or output_root.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_root}")
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError(f"output path exists and is not a directory: {output_root}")
        if _contains_path(output_root, raw_root):
            raise ValueError(f"refusing to overwrite output path that contains raw-real input: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def _normalized_label(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _is_placeholder_gripper_metadata(value: Any) -> bool:
    label = _normalized_label(value)
    if label is None:
        return False
    return "gripper" in label and ("synthetic" in label or "placeholder" in label or "pipeline_smoke" in label)


def _gripper_metadata(
    streams: dict[str, Any],
    gripper_records: list[dict[str, Any]],
    synthetic: bool,
) -> GripperMetadata:
    entry = streams.get("gripper_state") if isinstance(streams.get("gripper_state"), dict) else None
    if not gripper_records or entry is None:
        return GripperMetadata(
            state_source="synthetic-only fallback; gripper_pos=0.0",
            state_value_field="fallback_zero",
            state_unit="unitless_placeholder_scalar",
            provenance="fallback_default",
            is_placeholder=True,
            verified=False,
            valid_for_training=False,
        )

    uses_position = any(_is_finite_number(record.get("gripper_position")) for record in gripper_records)
    uses_width = any(_is_finite_number(record.get("gripper_width_m")) for record in gripper_records)
    if uses_position:
        value_field = "gripper_position"
        value_unit = "source_gripper_position_scalar"
    elif uses_width:
        value_field = "gripper_width_m"
        value_unit = "m"
    else:
        value_field = "unsupported"
        value_unit = "unknown"

    placeholder = bool(entry.get("placeholder") is True or entry.get("synthetic_placeholder") is True)
    for key in ["source_name", "source_type"]:
        if _is_placeholder_gripper_metadata(entry.get(key)):
            placeholder = True
    for record in gripper_records:
        if record.get("placeholder") is True or record.get("synthetic_placeholder") is True:
            placeholder = True
        for key in ["source_name", "source_type"]:
            if _is_placeholder_gripper_metadata(record.get(key)):
                placeholder = True

    verified = entry.get("verified") is True and not placeholder
    if placeholder:
        provenance = "synthetic_placeholder"
        source = "record_index aligned placeholder gripper_state stream"
    elif synthetic:
        provenance = "synthetic_fixture_gripper"
        source = "record_index aligned synthetic gripper_state stream"
    else:
        provenance = "real_measured"
        source = "record_index aligned measured gripper_state stream"

    return GripperMetadata(
        state_source=source,
        state_value_field=value_field,
        state_unit=value_unit,
        provenance=provenance,
        is_placeholder=placeholder,
        verified=verified,
        valid_for_training=(not synthetic and not placeholder and verified),
    )


def _unique_preserving_order(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value in unique:
            continue
        unique.append(value)
    return unique


def _single_or_list(values: list[Any]) -> Any:
    unique = _unique_preserving_order(values)
    if len(unique) == 1:
        return unique[0]
    return unique


def _unique_unit_metadata(resolutions: list[UnitResolution]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for resolution in resolutions:
        item = resolution.as_metadata()
        if item not in items:
            items.append(item)
    return items


def _tcp_conversion_metadata(
    infos: list[TcpConversionInfo],
    orientation_policy: OrientationPolicy,
) -> dict[str, Any]:
    position_units = [info.position_unit.unit for info in infos]
    orientation_units = [info.orientation_unit.unit for info in infos]
    position_conversions = [info.position_conversion for info in infos]
    orientation_conversions = [info.orientation_conversion for info in infos]
    return {
        "tcp_position_source_unit": _single_or_list(position_units),
        "tcp_position_source_unit_resolution": _unique_unit_metadata([info.position_unit for info in infos]),
        "tcp_position_output_unit": "m",
        "tcp_position_conversion": _single_or_list(position_conversions),
        "tcp_position_values_preserved": all(conversion == "preserved_meters" for conversion in position_conversions),
        "tcp_position_values_converted": any(conversion != "preserved_meters" for conversion in position_conversions),
        "tcp_position_legacy_unit_fallback_used": any(info.position_unit.legacy_fallback for info in infos),
        "tcp_orientation_source_convention": orientation_policy.source_convention,
        "tcp_orientation_source_convention_source": orientation_policy.source,
        "tcp_orientation_source_unit": _single_or_list(orientation_units),
        "tcp_orientation_source_unit_resolution": _unique_unit_metadata([info.orientation_unit for info in infos]),
        "tcp_orientation_output_convention": ROTATION_VECTOR_RADIANS,
        "tcp_orientation_output_unit": "rad",
        "tcp_orientation_conversion": _single_or_list(orientation_conversions),
        "tcp_orientation_values_preserved": all(
            conversion == "preserved_rotation_vector_radians" for conversion in orientation_conversions
        ),
        "tcp_orientation_values_converted": any(
            conversion != "preserved_rotation_vector_radians" for conversion in orientation_conversions
        ),
        "tcp_orientation_legacy_unit_fallback_used": any(info.orientation_unit.legacy_fallback for info in infos),
        "tcp_orientation_convention_legacy_fallback_used": orientation_policy.legacy_fallback,
        "orientation_conversion": orientation_policy.note,
    }


def _conversion_text(values: list[str]) -> str:
    return str(_single_or_list(values))


def _first_text(*values: Any, default: str = "unspecified") -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _wrench_layout_metadata(
    wrench_sources: set[str],
    selected_wrench_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = sorted(wrench_sources)[0] if len(wrench_sources) == 1 else "selected_wrench_source"
    metadata = selected_wrench_metadata.get(source, {}) if len(wrench_sources) == 1 else {}
    return {
        "source": source,
        "frame": _first_text(metadata.get("frame"), default="declared_by_selected_wrench_source"),
        "provenance": _first_text(metadata.get("source_type"), default="raw_wrench_signal"),
        "verified": metadata.get("verified") is True,
        "approved_for_training": metadata.get("approved_for_training") is True,
    }


def _joint_layout_sources(joint_sources: set[str]) -> tuple[str, str]:
    if joint_sources == {"robot_state_rt.actual_joint_position; robot_state_rt.actual_joint_velocity"}:
        return "robot_state_rt.actual_joint_position", "robot_state_rt.actual_joint_velocity"
    if joint_sources == {"joint_states.position; joint_states.velocity"}:
        return "joint_states.position", "joint_states.velocity"
    return "mixed_joint_position_sources", "mixed_joint_velocity_sources"


def _build_model_state_layout(
    *,
    tcp_frame: str,
    tcp_position_conversion: str,
    tcp_orientation_conversion: str,
    gripper: GripperMetadata,
    wrench_sources: set[str],
    selected_wrench_metadata: dict[str, dict[str, Any]],
    joint_sources: set[str],
) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    for idx, axis in enumerate(["x", "y", "z"]):
        layout.append(
            {
                "index": idx,
                "name": f"tcp_{axis}_m",
                "source": f"robot_state_rt.actual_tcp_position[{idx}]",
                "unit": "m",
                "frame": tcp_frame,
                "order": "xyz",
                "conversion": tcp_position_conversion,
                "provenance": "raw_robot_state_rt",
                "value_kind": "real_or_synthetic_source_measurement",
            }
        )
    for offset, axis in enumerate(["x", "y", "z"], start=3):
        layout.append(
            {
                "index": offset,
                "name": f"tcp_rotvec_{axis}_rad",
                "source": f"robot_state_rt.actual_tcp_position[{offset}]",
                "unit": "rad",
                "frame": tcp_frame,
                "order": "rotation_vector_xyz",
                "conversion": tcp_orientation_conversion,
                "provenance": "raw_robot_state_rt",
                "value_kind": "rotation_vector_radians",
            }
        )
    layout.append(
        {
            "index": 6,
            "name": "gripper_position",
            "source": gripper.state_value_field,
            "unit": gripper.state_unit,
            "frame": "gripper",
            "order": "scalar",
            "conversion": "preserved" if gripper.state_value_field != "fallback_zero" else "fallback_zero",
            "provenance": gripper.provenance,
            "value_kind": "placeholder" if gripper.is_placeholder else "real",
        }
    )
    wrench = _wrench_layout_metadata(wrench_sources, selected_wrench_metadata)
    for idx, name in enumerate(
        [
            "tcp_force_x_n",
            "tcp_force_y_n",
            "tcp_force_z_n",
            "tcp_torque_x_nm",
            "tcp_torque_y_nm",
            "tcp_torque_z_nm",
        ],
        start=7,
    ):
        source_index = idx - 7
        unit = "N" if source_index < 3 else "Nm"
        layout.append(
            {
                "index": idx,
                "name": name,
                "source": f"robot_state_rt.{wrench['source']}[{source_index}]",
                "unit": unit,
                "frame": wrench["frame"],
                "order": "Fx,Fy,Fz,Tx,Ty,Tz",
                "conversion": "preserved",
                "provenance": wrench["provenance"],
                "value_kind": "unverified" if not wrench["verified"] else "declared_verified",
            }
        )
    joint_pos_source, joint_vel_source = _joint_layout_sources(joint_sources)
    for joint_idx in range(6):
        layout.append(
            {
                "index": 13 + joint_idx,
                "name": f"joint_{joint_idx + 1}_position_rad",
                "source": f"{joint_pos_source}[{joint_idx}]",
                "unit": "rad",
                "frame": "joint_space",
                "order": "joint_1_to_joint_6",
                "conversion": "raw_joint_position_units_to_radians",
                "provenance": "raw_robot_or_joint_state",
                "value_kind": "real_or_synthetic_source_measurement",
            }
        )
    for joint_idx in range(6):
        layout.append(
            {
                "index": 19 + joint_idx,
                "name": f"joint_{joint_idx + 1}_velocity_rad_s",
                "source": f"{joint_vel_source}[{joint_idx}]",
                "unit": "rad/s",
                "frame": "joint_space",
                "order": "joint_1_to_joint_6",
                "conversion": "raw_joint_velocity_units_to_radians_per_second",
                "provenance": "raw_robot_or_joint_state",
                "value_kind": "real_or_synthetic_source_measurement",
            }
        )
    return layout


def _build_measured_action_layout(gripper: GripperMetadata) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    for idx, axis in enumerate(["x", "y", "z"]):
        layout.append(
            {
                "index": idx,
                "name": f"delta_tcp_{axis}_m",
                "source": f"processed tcp position[t+1].{axis} - processed tcp position[t].{axis}",
                "unit": "m",
                "frame": "base",
                "delta_convention": "next_minus_current",
                "absolute_or_relative": "relative_delta",
                "rotation_composition_order": "not_applicable",
            }
        )
    for idx, axis in enumerate(["x", "y", "z"], start=3):
        layout.append(
            {
                "index": idx,
                "name": f"delta_tcp_rotvec_{axis}_rad",
                "source": "rotvec(conjugate(tcp_quat_xyzw[t]) * tcp_quat_xyzw[t+1])",
                "unit": "rad",
                "frame": "tcp_t_relative_rotation",
                "delta_convention": "relative_rotation_log_map",
                "absolute_or_relative": "relative_delta",
                "rotation_composition_order": "q_rel = conjugate(q_t) * q_t1; xyzw quaternions",
            }
        )
    layout.append(
        {
            "index": 6,
            "name": "gripper_action",
            "source": f"{gripper.state_value_field}[t+1] - {gripper.state_value_field}[t]",
            "unit": gripper.state_unit,
            "frame": "gripper",
            "delta_convention": "next_minus_current",
            "absolute_or_relative": "relative_delta",
            "rotation_composition_order": "not_applicable",
        }
    )
    return layout


def _terminal_action_policy(frame_count: int) -> dict[str, Any]:
    return {
        "final_observation_retained": True,
        "final_action_padded": True,
        "padding_value": [0.0] * ACTION_DIM,
        "padding_count": 1,
        "terminal_padding_frame_indices": [frame_count - 1],
        "exporters_must_exclude_terminal_padding_rows": True,
        "terminal_padding_rows_valid_for_observation_only_visualization": True,
    }


def _source_stream_verification(streams: dict[str, Any]) -> dict[str, Any]:
    verification: dict[str, Any] = {}
    for stream_name, entry in streams.items():
        if not isinstance(entry, dict):
            continue
        verification[stream_name] = {
            "verified": entry.get("verified") is True,
            "source_name": entry.get("source_name"),
            "source_type": entry.get("source_type"),
        }
    return verification


def _semantic_verification_pending(gripper: GripperMetadata, selected_wrench_metadata: dict[str, dict[str, Any]]) -> list[str]:
    pending: list[str] = []
    if gripper.is_placeholder or not gripper.valid_for_training:
        pending.append("gripper_state_not_validated_for_real_training")
    for source, metadata in selected_wrench_metadata.items():
        if metadata.get("verified") is not True or metadata.get("approved_for_training") is not True:
            pending.append(f"wrench_source_{source}_training_semantics_unverified")
    return pending


def _build_model_state(
    *,
    robot_record: dict[str, Any],
    robot_entry: dict[str, Any],
    metadata: dict[str, Any],
    streams_index: dict[str, Any],
    joint_record: dict[str, Any],
    joint_entry: dict[str, Any],
    gripper_record: dict[str, Any] | None,
    synthetic: bool,
    orientation_policy: OrientationPolicy,
    frame_index: int,
) -> tuple[list[float], list[float], list[float], float, str, str, TcpConversionInfo]:
    tcp = _finite_vector(
        robot_record.get("actual_tcp_position"), 6, f"robot_state_rt record {frame_index} actual_tcp_position"
    )
    position_unit = resolve_tcp_position_unit(
        record=robot_record,
        record_source=f"robot_state_rt record {frame_index}",
        stream_entry=robot_entry,
        stream_source="streams/index.json streams.robot_state_rt",
        metadata=metadata,
        streams_index=streams_index,
        synthetic=synthetic,
        context=f"robot_state_rt record {frame_index} actual_tcp_position[0:3]",
    )
    orientation_unit = resolve_tcp_orientation_unit(
        record=robot_record,
        record_source=f"robot_state_rt record {frame_index}",
        stream_entry=robot_entry,
        stream_source="streams/index.json streams.robot_state_rt",
        metadata=metadata,
        streams_index=streams_index,
        synthetic=synthetic,
        context=f"robot_state_rt record {frame_index} actual_tcp_position[3:6]",
    )
    ee_pos, position_conversion = _convert_tcp_position(
        tcp[:3], position_unit, f"robot_state_rt record {frame_index} actual_tcp_position[0:3]"
    )
    ee_axis_angle, orientation_conversion = _convert_tcp_orientation_rotvec(
        tcp[3:6],
        orientation_unit,
        orientation_policy,
        f"robot_state_rt record {frame_index} actual_tcp_position[3:6]",
    )
    ee_quat = _rotvec_to_quat_xyzw(ee_axis_angle)
    gripper_pos = _select_gripper(gripper_record, frame_index)
    wrench, wrench_source = _select_wrench(robot_record, frame_index)
    joint_pos, joint_vel, joint_source = _select_joint_vectors(
        robot_record, robot_entry, joint_record, joint_entry, synthetic, frame_index
    )

    model_state = ee_pos + ee_axis_angle + [gripper_pos] + wrench + joint_pos + joint_vel
    if len(model_state) != MODEL_STATE_DIM:
        raise ValueError(f"model_state length must be {MODEL_STATE_DIM}, got {len(model_state)}")
    if not all(math.isfinite(value) for value in model_state):
        raise ValueError(f"frame {frame_index}: model_state contains a non-finite value")
    return (
        model_state,
        ee_pos,
        ee_quat,
        gripper_pos,
        wrench_source,
        joint_source,
        TcpConversionInfo(
            position_unit=position_unit,
            orientation_unit=orientation_unit,
            position_conversion=position_conversion,
            orientation_conversion=orientation_conversion,
        ),
    )


def convert_raw_real_to_processed(
    raw_real_episode_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    copy_images: bool = False,
    include_optional_debug: bool = False,
) -> Path:
    """Convert a validated raw_real_v0 episode into a processed episode."""

    raw_root = Path(raw_real_episode_dir)
    output_root = Path(output_dir)

    if _contains_path(raw_root, output_root):
        raise ValueError(
            f"output directory cannot be inside the raw-real episode directory: {output_root}"
        )

    validation = validate_raw_real_episode(raw_root)
    if not validation.ok:
        message = "raw-real episode validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in validation.errors
        )
        raise ValueError(message)

    metadata = _read_json_object(raw_root / "metadata.json")
    recorder_report = _read_json_object(raw_root / "recorder_report.json")
    streams_index = _read_json_object(raw_root / "streams" / "index.json")
    streams = streams_index.get("streams")
    if not isinstance(streams, dict):
        raise ValueError(f"{raw_root / 'streams' / 'index.json'}: streams must be a JSON object")

    orientation_policy = _orientation_policy(metadata, recorder_report, streams_index)
    synthetic = orientation_policy.synthetic

    robot_records = _read_jsonl_objects(_stream_path(raw_root, streams, "robot_state_rt"))
    if not robot_records:
        raise ValueError("robot_state_rt stream must contain at least one record")
    joint_records = _read_jsonl_objects(_stream_path(raw_root, streams, "joint_states"))
    selected_camera_streams, camera_mapping_errors, camera_selection_sources = select_model_camera_streams(
        metadata, streams_index, streams
    )
    if camera_mapping_errors:
        raise ValueError("raw-real camera mapping is not ready for conversion:\n" + "\n".join(camera_mapping_errors))
    external_camera_stream = selected_camera_streams[MODEL_EXTERNAL_IMAGE_KEY]
    tcp_camera_stream = selected_camera_streams[MODEL_TCP_IMAGE_KEY]
    declared_camera_streams = camera_stream_names(streams)
    camera_records_by_stream = {
        stream_name: _load_camera_index(raw_root, streams, stream_name)
        for stream_name in declared_camera_streams
    }

    robot_by_index = _records_by_index(robot_records, "robot_state_rt")
    joint_by_index = _records_by_index(joint_records, "joint_states")
    camera_by_index = {
        stream_name: _records_by_index(records, stream_name)
        for stream_name, records in camera_records_by_stream.items()
    }
    external_camera_by_index = camera_by_index[external_camera_stream]
    tcp_camera_by_index = camera_by_index[tcp_camera_stream]
    primary_indexes = set(robot_by_index)

    _require_aligned_indexes(primary_indexes, joint_by_index, "joint_states")
    for stream_name, records_by_index in camera_by_index.items():
        _require_aligned_indexes(primary_indexes, records_by_index, stream_name)

    gripper_by_index: dict[int, dict[str, Any]] = {}
    gripper_records: list[dict[str, Any]] = []
    if "gripper_state" in streams:
        gripper_records = _read_jsonl_objects(_stream_path(raw_root, streams, "gripper_state"))
        gripper_by_index = _records_by_index(gripper_records, "gripper_state")
        _require_aligned_indexes(primary_indexes, gripper_by_index, "gripper_state")

    command_context_debug: dict[str, Any] | None = None
    if include_optional_debug and "command_context" in streams:
        command_context_records = _read_jsonl_objects(_stream_path(raw_root, streams, "command_context"))
        command_context_debug = {
            "record_count": len(command_context_records),
            "first_record": command_context_records[0] if command_context_records else None,
            "last_record": command_context_records[-1] if command_context_records else None,
            "used_as_action_label": False,
        }

    records_by_stream: dict[str, list[dict[str, Any]]] = {
        "joint_states": joint_records,
        "robot_state_rt": robot_records,
    }
    records_by_stream.update(camera_records_by_stream)
    if gripper_records:
        records_by_stream["gripper_state"] = gripper_records
    calibration_refs = _read_json_object(raw_root / "calibration_refs.json")
    conversion_readiness_errors = raw_real_conversion_readiness_errors(
        metadata,
        recorder_report,
        streams_index,
        streams,
        records_by_stream,
        root_dir=raw_root,
        calibration_refs=calibration_refs,
    )
    if conversion_readiness_errors:
        message = "raw-real episode is not ready for conversion:\n" + "\n".join(
            f"ERROR: {error}" for error in conversion_readiness_errors
        )
        raise ValueError(message)
    selected_wrench_metadata = selected_wrench_metadata_for_model_state(streams, records_by_stream)

    ordered_indexes = sorted(primary_indexes)
    if ordered_indexes != list(range(len(ordered_indexes))):
        raise ValueError("robot_state_rt record_index values must be contiguous from 0 for processed frame_index mapping")
    if len(ordered_indexes) < 2:
        raise ValueError(f"raw-real conversion requires at least 2 aligned records; got {len(ordered_indexes)}")
    ordered_robot_records = [robot_by_index[index] for index in ordered_indexes]
    timestamps = _relative_timestamps(ordered_robot_records)

    if not synthetic:
        _require_non_synthetic_gripper_state(primary_indexes, gripper_by_index)

    _prepare_output(raw_root, output_root, overwrite=overwrite)

    robot_entry = _stream_entry(streams, "robot_state_rt")
    joint_entry = _stream_entry(streams, "joint_states")
    frames: list[dict[str, Any]] = []
    tcp_positions: list[list[float]] = []
    tcp_quats: list[list[float]] = []
    tcp_conversion_infos: list[TcpConversionInfo] = []
    gripper_positions: list[float] = []
    wrench_sources: set[str] = set()
    joint_sources: set[str] = set()

    for frame_index, record_index in enumerate(ordered_indexes):
        model_state, tcp_pos, tcp_quat, gripper_pos, wrench_source, joint_source, conversion_info = _build_model_state(
            robot_record=robot_by_index[record_index],
            robot_entry=robot_entry,
            metadata=metadata,
            streams_index=streams_index,
            joint_record=joint_by_index[record_index],
            joint_entry=joint_entry,
            gripper_record=gripper_by_index.get(record_index),
            synthetic=synthetic,
            orientation_policy=orientation_policy,
            frame_index=frame_index,
        )
        tcp_positions.append(tcp_pos)
        tcp_quats.append(tcp_quat)
        tcp_conversion_infos.append(conversion_info)
        gripper_positions.append(gripper_pos)
        wrench_sources.add(wrench_source)
        joint_sources.add(joint_source)

        frames.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamps[frame_index],
                "external_rgb_path": _frame_image_path(
                    raw_root=raw_root,
                    output_root=output_root,
                    camera_record=external_camera_by_index[record_index],
                    stream_name=external_camera_stream,
                    frame_index=frame_index,
                    copy_images=copy_images,
                ),
                "tcp_rgb_path": _frame_image_path(
                    raw_root=raw_root,
                    output_root=output_root,
                    camera_record=tcp_camera_by_index[record_index],
                    stream_name=tcp_camera_stream,
                    frame_index=frame_index,
                    copy_images=copy_images,
                ),
                "model_state": model_state,
                "measured_action": [0.0] * ACTION_DIM,
                "action_is_terminal_padding": True,
            }
        )

    for frame_index, frame in enumerate(frames):
        if frame_index == len(frames) - 1:
            frame["measured_action"] = [0.0] * ACTION_DIM
            frame["action_is_terminal_padding"] = True
            continue
        frame["measured_action"] = compute_measured_tcp_delta_action(
            tcp_positions[frame_index],
            tcp_quats[frame_index],
            tcp_positions[frame_index + 1],
            tcp_quats[frame_index + 1],
            gripper_t=gripper_positions[frame_index],
            gripper_t1=gripper_positions[frame_index + 1],
        )
        frame["action_is_terminal_padding"] = False

    gripper_metadata = _gripper_metadata(streams, gripper_records, synthetic)
    tcp_conversion_metadata = _tcp_conversion_metadata(tcp_conversion_infos, orientation_policy)
    tcp_position_conversion_text = _conversion_text([info.position_conversion for info in tcp_conversion_infos])
    tcp_orientation_conversion_text = _conversion_text([info.orientation_conversion for info in tcp_conversion_infos])
    tcp_frame = _first_text(robot_entry.get("frame_id"), ordered_robot_records[0].get("frame_id"), default="unspecified")
    model_state_layout = _build_model_state_layout(
        tcp_frame=tcp_frame,
        tcp_position_conversion=tcp_position_conversion_text,
        tcp_orientation_conversion=tcp_orientation_conversion_text,
        gripper=gripper_metadata,
        wrench_sources=wrench_sources,
        selected_wrench_metadata=selected_wrench_metadata,
        joint_sources=joint_sources,
    )
    measured_action_layout = _build_measured_action_layout(gripper_metadata)
    terminal_policy = _terminal_action_policy(len(frames))
    placeholder_fields = []
    if gripper_metadata.is_placeholder:
        placeholder_fields.extend(["model_state[6].gripper_position", "measured_action[6].gripper_action"])

    processed_metadata: dict[str, Any] = {
        "source_raw_episode": str(raw_root.resolve()),
        "processed_metadata_schema_version": PROCESSED_METADATA_SCHEMA_VERSION,
        "dataset_name": DATASET_NAME,
        "robot_type": metadata.get("robot_type"),
        "fps": metadata.get("fps"),
        "quaternion_convention": QUATERNION_CONVENTION,
        "model_state_dim": MODEL_STATE_DIM,
        "model_state_layout_version": MODEL_STATE_LAYOUT_VERSION,
        "model_state_layout_semantics": (
            "tcp position meters, tcp orientation rotation-vector radians, gripper scalar, "
            "TCP wrench, joint positions radians, joint velocities radians_per_second"
        ),
        "model_state_layout": model_state_layout,
        "action_dim": ACTION_DIM,
        "action_label_primary": "measured_tcp_delta",
        "measured_action_layout_version": MEASURED_ACTION_LAYOUT_VERSION,
        "measured_action_layout": measured_action_layout,
        "measured_action_semantics": (
            "measured consecutive TCP delta from frame t to frame t+1; final frame retains observation "
            "and uses terminal zero padding because no t+1 pose exists"
        ),
        "translation_delta_frame": "base",
        "rotation_delta_convention": "relative rotation vector from normalized xyzw quaternions",
        "rotation_composition_order": "q_rel = conjugate(q_t) * q_t1; delta_rotvec = log(q_rel)",
        "gripper_action_semantics": (
            "gripper scalar delta gripper[t+1] - gripper[t] using the same scalar stored at model_state[6]; "
            "terminal padding uses 0.0"
        ),
        "terminal_action_policy": terminal_policy,
        "frame_count": len(frames),
        "task_instruction": metadata.get("task_instruction"),
        "geometry_type": metadata.get("geometry_type"),
        "orientation_type": metadata.get("orientation_type"),
        "collection_method": metadata.get("collection_method"),
        "success": metadata.get("success"),
        "failure_reason": metadata.get("failure_reason"),
        "notes": (
            "raw_real_v0 to processed JSONL conversion. Actions are measured consecutive TCP deltas; "
            "command_context is diagnostic only and terminal action padding is applied."
        ),
        "source_schema_version": metadata.get("schema_version"),
        "source_episode_id": metadata.get("episode_id"),
        "source_collection_method": metadata.get("collection_method"),
        "source_real_hardware_recording": metadata.get("real_hardware_recording"),
        "source_training_ready": metadata.get("training_ready"),
        "source_real_hardware_verified": metadata.get("real_hardware_verified"),
        "source_stream_verification": _source_stream_verification(streams),
        "placeholder_fields": placeholder_fields,
        "semantic_verification_pending": _semantic_verification_pending(gripper_metadata, selected_wrench_metadata),
        "conversion_structurally_valid": True,
        "training_readiness_is_separate_from_structural_validity": True,
        "converter_version": CONVERTER_VERSION,
        "alignment_policy": "record_index equality against robot_state_rt primary timeline",
        "selected_streams": {
            "primary_timeline": "robot_state_rt",
            "joint_states": "record_index aligned fallback only when robot_state_rt joint vectors are unavailable",
            "external_rgb_path": f"{external_camera_stream}.image_path",
            "tcp_rgb_path": f"{tcp_camera_stream}.image_path",
            "gripper_state": gripper_metadata.state_source,
        },
        "camera_mapping": _processed_camera_mapping(selected_camera_streams, camera_selection_sources),
        "raw_camera_streams": _camera_stream_metadata(streams, declared_camera_streams),
        "unit_conversions": {
            "tcp_position": tcp_position_conversion_text,
            "tcp_orientation": tcp_orientation_conversion_text,
            "joint_position": "raw units to radians",
            "joint_velocity": "raw units to radians_per_second",
            "wrench": "preserved from selected raw 6D force/torque signal",
        },
        **tcp_conversion_metadata,
        "wrench_source": sorted(wrench_sources),
        "joint_source": sorted(joint_sources),
        "gripper_state_source": gripper_metadata.state_source,
        "gripper_state_value_field": gripper_metadata.state_value_field,
        "gripper_state_unit": gripper_metadata.state_unit,
        "gripper_state_provenance": gripper_metadata.provenance,
        "gripper_state_is_placeholder": gripper_metadata.is_placeholder,
        "gripper_state_verified": gripper_metadata.verified,
        "gripper_state_valid_for_training": gripper_metadata.valid_for_training,
        "image_copy_policy": "copied into processed images/" if copy_images else "raw-real relative image paths",
        "raw_validation_warnings": validation.warnings,
        "command_context_policy": "diagnostic only; never used as action label",
    }

    if selected_wrench_metadata:
        processed_metadata["wrench_source_metadata"] = {
            source: selected_wrench_metadata[source]
            for source in sorted(wrench_sources)
            if source in selected_wrench_metadata
        }
    if command_context_debug is not None:
        processed_metadata["optional_debug"] = {"command_context": command_context_debug}

    _write_json(output_root / "metadata_processed.json", processed_metadata)
    _write_jsonl(output_root / "frames.jsonl", frames)

    processed_validation = validate_processed_episode(output_root)
    if not processed_validation.ok:
        message = "processed episode validation failed:\n" + "\n".join(
            f"ERROR: {error}" for error in processed_validation.errors
        )
        raise ValueError(message)

    return output_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a raw_real_v0 episode to processed JSONL.")
    parser.add_argument("--raw-real", required=True, help="Raw-real episode directory")
    parser.add_argument("--output", required=True, help="Processed episode output directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into the processed episode")
    parser.add_argument(
        "--include-optional-debug",
        action="store_true",
        help="Copy optional diagnostic metadata such as command_context summary",
    )
    args = parser.parse_args(argv)

    try:
        output_dir = convert_raw_real_to_processed(
            args.raw_real,
            args.output,
            overwrite=args.overwrite,
            copy_images=args.copy_images,
            include_optional_debug=args.include_optional_debug,
        )
    except (OSError, ValueError) as exc:
        print(f"FAILED: could not convert raw-real episode: {args.raw_real}")
        print(str(exc))
        return 1

    print(f"OK: wrote processed episode: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
