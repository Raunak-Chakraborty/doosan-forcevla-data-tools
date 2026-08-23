"""ROS 2 MCAP ingest foundation for recorded Doosan episodes.

This module is intentionally limited to episode discovery, metadata and
sidecar validation, rosbag2 SequentialReader access, and generic ROS message
deserialization. Synchronization and typed field decoding belong to later
converter stages.

Design inspiration: legalaspro/so101-ros-physical-ai
``rosbag_to_lerobot/bag_reader.py`` (Apache-2.0). This implementation is
independent and specialized for the Doosan raw episode contract.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterator


class McapIngestError(RuntimeError):
    """Raised when an episode cannot satisfy the Patch-1 ingest contract."""


@dataclass(frozen=True)
class TopicMetadata:
    name: str
    type_name: str
    serialization_format: str
    message_count: int


@dataclass(frozen=True)
class EpisodeMetadata:
    storage_identifier: str
    relative_file_paths: tuple[str, ...]
    message_count: int
    topics: tuple[TopicMetadata, ...]
    custom_data: dict[str, Any]


@dataclass(frozen=True)
class EpisodeDescriptor:
    episode_dir: Path
    metadata: EpisodeMetadata
    operator: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True)
class BagMessage:
    topic: str
    type_name: str
    bag_timestamp_ns: int
    message: Any


@dataclass(frozen=True)
class EpisodeScanSummary:
    episode_dir: str
    storage_identifier: str
    mcap_files: tuple[str, ...]
    total_messages: int
    topic_counts: dict[str, int]
    topic_types: dict[str, str]
    first_bag_timestamp_ns: dict[str, int]
    last_bag_timestamp_ns: dict[str, int]
    operator_schema_version: str | None
    validation_schema_version: str | None
    validation_passed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_dir": self.episode_dir,
            "storage_identifier": self.storage_identifier,
            "mcap_files": list(self.mcap_files),
            "total_messages": self.total_messages,
            "topic_counts": dict(self.topic_counts),
            "topic_types": dict(self.topic_types),
            "first_bag_timestamp_ns": dict(self.first_bag_timestamp_ns),
            "last_bag_timestamp_ns": dict(self.last_bag_timestamp_ns),
            "operator_schema_version": self.operator_schema_version,
            "validation_schema_version": self.validation_schema_version,
            "validation_passed": self.validation_passed,
        }


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise McapIngestError(f"{context} must be a mapping")
    return value


def _require_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise McapIngestError(
            f"{context} must be a non-negative integer"
        )
    return value


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    # Deliberately lazy. Importing this converter package must not require
    # ROS or the ForceVLA environment to contain ROS-specific dependencies.
    try:
        import yaml
    except Exception as exc:
        raise McapIngestError(
            "PyYAML is required to read rosbag2 metadata.yaml. "
            "For ROS 2 Jazzy use the system ROS Python environment."
        ) from exc

    try:
        data = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise McapIngestError(
            f"failed to read YAML {path}: {exc}"
        ) from exc

    return _require_mapping(data, str(path))


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise McapIngestError(
            f"failed to read JSON {path}: {exc}"
        ) from exc

    return _require_mapping(data, str(path))


def _resolve_required_local_file(
    root: Path,
    relative_name: str,
) -> Path:
    path = root / relative_name

    if not path.is_file():
        raise McapIngestError(
            f"required episode file is missing: {path}"
        )

    root_resolved = root.resolve()
    path_resolved = path.resolve()

    if not path_resolved.is_relative_to(root_resolved):
        raise McapIngestError(
            "episode file resolves outside episode directory: "
            f"{relative_name}"
        )

    return path


def read_episode_metadata(
    episode_dir: str | Path,
) -> EpisodeMetadata:
    """Parse and strictly validate rosbag2 metadata.yaml for an MCAP episode."""

    root = Path(episode_dir)

    if not root.is_dir():
        raise McapIngestError(
            f"episode directory does not exist: {root}"
        )

    metadata_path = _resolve_required_local_file(
        root,
        "metadata.yaml",
    )

    top = _read_yaml_mapping(metadata_path)

    info = _require_mapping(
        top.get("rosbag2_bagfile_information"),
        f"{metadata_path}: rosbag2_bagfile_information",
    )

    storage_identifier = info.get(
        "storage_identifier"
    )

    if storage_identifier != "mcap":
        raise McapIngestError(
            f"{metadata_path}: storage_identifier must be "
            f"'mcap', got {storage_identifier!r}"
        )

    relative_paths_raw = info.get(
        "relative_file_paths"
    )

    if (
        not isinstance(relative_paths_raw, list)
        or not relative_paths_raw
    ):
        raise McapIngestError(
            f"{metadata_path}: relative_file_paths "
            "must be a non-empty list"
        )

    root_resolved = root.resolve()

    relative_paths: list[str] = []
    resolved_mcap_files: list[Path] = []
    seen_relative: set[str] = set()

    for index, value in enumerate(
        relative_paths_raw
    ):
        if not isinstance(value, str) or not value:
            raise McapIngestError(
                f"{metadata_path}: "
                f"relative_file_paths[{index}] "
                "must be a non-empty string"
            )

        if value in seen_relative:
            raise McapIngestError(
                f"{metadata_path}: duplicate "
                f"relative_file_path {value!r}"
            )

        seen_relative.add(value)

        relative = Path(value)

        if (
            relative.is_absolute()
            or relative.suffix.lower() != ".mcap"
        ):
            raise McapIngestError(
                f"{metadata_path}: invalid MCAP "
                f"relative path {value!r}"
            )

        resolved = (
            root / relative
        ).resolve()

        if not resolved.is_relative_to(
            root_resolved
        ):
            raise McapIngestError(
                f"{metadata_path}: MCAP path "
                "escapes episode directory: "
                f"{value!r}"
            )

        if not resolved.is_file():
            raise McapIngestError(
                f"{metadata_path}: referenced "
                f"MCAP file is missing: {value!r}"
            )

        relative_paths.append(value)
        resolved_mcap_files.append(resolved)

    actual_mcap_files = {
        path.resolve()
        for path in root.rglob("*.mcap")
        if path.is_file()
    }

    expected_mcap_files = set(
        resolved_mcap_files
    )

    if actual_mcap_files != expected_mcap_files:
        missing = sorted(
            str(
                path.relative_to(root_resolved)
            )
            for path in (
                expected_mcap_files
                - actual_mcap_files
            )
        )

        extra = sorted(
            str(
                path.relative_to(root_resolved)
            )
            for path in (
                actual_mcap_files
                - expected_mcap_files
            )
        )

        raise McapIngestError(
            f"{metadata_path}: metadata/filesystem "
            "MCAP set mismatch; "
            f"missing={missing}, extra={extra}"
        )

    message_count = _require_nonnegative_int(
        info.get("message_count"),
        f"{metadata_path}: message_count",
    )

    if message_count == 0:
        raise McapIngestError(
            f"{metadata_path}: message_count "
            "must be positive"
        )

    topics_raw = info.get(
        "topics_with_message_count"
    )

    if (
        not isinstance(topics_raw, list)
        or not topics_raw
    ):
        raise McapIngestError(
            f"{metadata_path}: "
            "topics_with_message_count "
            "must be a non-empty list"
        )

    topics: list[TopicMetadata] = []
    seen_topics: set[str] = set()

    for index, entry_raw in enumerate(
        topics_raw
    ):
        entry = _require_mapping(
            entry_raw,
            f"{metadata_path}: "
            f"topics_with_message_count[{index}]",
        )

        topic_meta = _require_mapping(
            entry.get("topic_metadata"),
            f"{metadata_path}: "
            f"topics_with_message_count[{index}]"
            ".topic_metadata",
        )

        name = topic_meta.get("name")
        type_name = topic_meta.get("type")
        serialization = topic_meta.get(
            "serialization_format"
        )

        count = _require_nonnegative_int(
            entry.get("message_count"),
            f"{metadata_path}: "
            f"topic {name!r} message_count",
        )

        if (
            not isinstance(name, str)
            or not name.startswith("/")
        ):
            raise McapIngestError(
                f"{metadata_path}: topic name "
                "must be an absolute ROS name, "
                f"got {name!r}"
            )

        if name in seen_topics:
            raise McapIngestError(
                f"{metadata_path}: duplicate "
                f"topic {name!r}"
            )

        seen_topics.add(name)

        if (
            not isinstance(type_name, str)
            or "/msg/" not in type_name
        ):
            raise McapIngestError(
                f"{metadata_path}: invalid ROS "
                f"message type for {name}: "
                f"{type_name!r}"
            )

        if serialization != "cdr":
            raise McapIngestError(
                f"{metadata_path}: topic {name} "
                "serialization_format must be "
                f"'cdr', got {serialization!r}"
            )

        topics.append(
            TopicMetadata(
                name=name,
                type_name=type_name,
                serialization_format=serialization,
                message_count=count,
            )
        )

    topic_total = sum(
        topic.message_count
        for topic in topics
    )

    if topic_total != message_count:
        raise McapIngestError(
            f"{metadata_path}: message_count "
            f"{message_count} does not equal "
            "sum of per-topic counts "
            f"{topic_total}"
        )

    custom_data_raw = info.get(
        "custom_data",
        {},
    )

    custom_data = _require_mapping(
        custom_data_raw,
        f"{metadata_path}: custom_data",
    )

    return EpisodeMetadata(
        storage_identifier=storage_identifier,
        relative_file_paths=tuple(
            relative_paths
        ),
        message_count=message_count,
        topics=tuple(topics),
        custom_data=dict(custom_data),
    )


def load_episode_descriptor(
    episode_dir: str | Path,
) -> EpisodeDescriptor:
    """Load validated bag metadata plus both required episode sidecars."""

    root = Path(episode_dir)

    metadata = read_episode_metadata(
        root
    )

    operator_path = (
        _resolve_required_local_file(
            root,
            "episode_operator.json",
        )
    )

    validation_path = (
        _resolve_required_local_file(
            root,
            "episode_validation.json",
        )
    )

    return EpisodeDescriptor(
        episode_dir=root.resolve(),
        metadata=metadata,
        operator=_read_json_mapping(
            operator_path
        ),
        validation=_read_json_mapping(
            validation_path
        ),
    )


def discover_episode_dirs(
    root: str | Path,
) -> list[Path]:
    """Find a direct episode or immediate child episode directories."""

    root_path = Path(root)

    if not root_path.is_dir():
        raise McapIngestError(
            "episode search root does not "
            f"exist: {root_path}"
        )

    if (
        root_path / "metadata.yaml"
    ).is_file():
        return [root_path.resolve()]

    return sorted(
        child.resolve()
        for child in root_path.iterdir()
        if (
            child.is_dir()
            and (
                child / "metadata.yaml"
            ).is_file()
        )
    )


def _load_ros2_runtime() -> tuple[
    Any,
    Any,
    Any,
]:
    # Keep ROS imports out of module import time.
    #
    # This is important for this project because
    # ForceVLA uses a separate conda Python, while
    # ROS 2 Jazzy binary modules are built against
    # the workstation's system Python.
    try:
        import rosbag2_py
        from rclpy.serialization import (
            deserialize_message,
        )
        from rosidl_runtime_py.utilities import (
            get_message,
        )
    except Exception as exc:
        raise McapIngestError(
            "ROS 2 Python runtime is unavailable "
            "or ABI-incompatible. "
            f"Interpreter: {sys.executable}. "
            "For ROS 2 Jazzy on this workstation, "
            "source /opt/ros/jazzy/setup.bash and "
            "the lab workspace, then run with "
            "/usr/bin/python3; do not use the "
            "ForceVLA conda interpreter."
        ) from exc

    return (
        rosbag2_py,
        deserialize_message,
        get_message,
    )


def _open_reader(
    descriptor: EpisodeDescriptor,
) -> tuple[Any, Any, Any]:
    (
        rosbag2_py,
        deserialize_message,
        get_message,
    ) = _load_ros2_runtime()

    reader = rosbag2_py.SequentialReader()

    try:
        reader.open(
            rosbag2_py.StorageOptions(
                uri=str(
                    descriptor.episode_dir
                ),
                storage_id=(
                    descriptor
                    .metadata
                    .storage_identifier
                ),
            ),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )
    except Exception as exc:
        raise McapIngestError(
            "failed to open MCAP episode "
            f"{descriptor.episode_dir}: {exc}"
        ) from exc

    return (
        reader,
        deserialize_message,
        get_message,
    )


def _metadata_topic_types(
    metadata: EpisodeMetadata,
) -> dict[str, str]:
    return {
        topic.name: topic.type_name
        for topic in metadata.topics
    }


def _metadata_topic_counts(
    metadata: EpisodeMetadata,
) -> dict[str, int]:
    return {
        topic.name: topic.message_count
        for topic in metadata.topics
    }


def _reader_topic_types(
    reader: Any,
) -> dict[str, str]:
    result: dict[str, str] = {}

    for entry in (
        reader.get_all_topics_and_types()
    ):
        if entry.name in result:
            raise McapIngestError(
                "rosbag2 reader reported "
                "duplicate topic metadata: "
                f"{entry.name}"
            )

        result[entry.name] = entry.type

    return result


def _iter_deserialized_from_descriptor(
    descriptor: EpisodeDescriptor,
) -> Iterator[BagMessage]:
    (
        reader,
        deserialize_message,
        get_message,
    ) = _open_reader(descriptor)

    expected_types = (
        _metadata_topic_types(
            descriptor.metadata
        )
    )

    actual_types = _reader_topic_types(
        reader
    )

    if actual_types != expected_types:
        raise McapIngestError(
            "rosbag2 reader topic/type map "
            "does not match metadata.yaml; "
            f"metadata={expected_types!r}, "
            f"reader={actual_types!r}"
        )

    message_classes: dict[
        str,
        Any,
    ] = {}

    for (
        topic,
        type_name,
    ) in expected_types.items():
        try:
            message_classes[topic] = (
                get_message(type_name)
            )
        except Exception as exc:
            raise McapIngestError(
                "cannot resolve installed ROS "
                "message definition "
                f"{type_name!r} for topic "
                f"{topic!r}: {exc}"
            ) from exc

    while reader.has_next():
        try:
            (
                topic,
                serialized,
                bag_timestamp_ns,
            ) = reader.read_next()
        except Exception as exc:
            raise McapIngestError(
                "failed while reading "
                f"{descriptor.episode_dir}: "
                f"{exc}"
            ) from exc

        if topic not in expected_types:
            raise McapIngestError(
                "reader returned topic not "
                "declared in metadata.yaml: "
                f"{topic!r}"
            )

        try:
            message = deserialize_message(
                serialized,
                message_classes[topic],
            )
        except Exception as exc:
            raise McapIngestError(
                "failed to deserialize topic "
                f"{topic!r} as "
                f"{expected_types[topic]!r} "
                "at bag timestamp "
                f"{bag_timestamp_ns}: {exc}"
            ) from exc

        yield BagMessage(
            topic=topic,
            type_name=expected_types[topic],
            bag_timestamp_ns=int(
                bag_timestamp_ns
            ),
            message=message,
        )


def iter_deserialized_messages(
    episode_dir: str | Path,
) -> Iterator[BagMessage]:
    """Yield every bag record as a generically deserialized ROS message."""

    descriptor = (
        load_episode_descriptor(
            episode_dir
        )
    )

    yield from (
        _iter_deserialized_from_descriptor(
            descriptor
        )
    )


def scan_episode(
    episode_dir: str | Path,
) -> EpisodeScanSummary:
    """Fully deserialize and count an episode.

    The scan is streaming: deserialized image
    payloads are not retained after their
    iteration, so a multi-gigabyte bag is not
    accumulated in memory.
    """

    descriptor = (
        load_episode_descriptor(
            episode_dir
        )
    )

    counts: Counter[str] = Counter()

    first_timestamps: dict[
        str,
        int,
    ] = {}

    last_timestamps: dict[
        str,
        int,
    ] = {}

    previous_global_timestamp: (
        int | None
    ) = None

    for record in (
        _iter_deserialized_from_descriptor(
            descriptor
        )
    ):
        if (
            previous_global_timestamp
            is not None
            and record.bag_timestamp_ns
            < previous_global_timestamp
        ):
            raise McapIngestError(
                "global bag timestamp "
                "regression: "
                f"{record.bag_timestamp_ns} < "
                f"{previous_global_timestamp}"
            )

        previous_global_timestamp = (
            record.bag_timestamp_ns
        )

        counts[record.topic] += 1

        first_timestamps.setdefault(
            record.topic,
            record.bag_timestamp_ns,
        )

        previous_topic_timestamp = (
            last_timestamps.get(
                record.topic
            )
        )

        if (
            previous_topic_timestamp
            is not None
            and record.bag_timestamp_ns
            < previous_topic_timestamp
        ):
            raise McapIngestError(
                "bag timestamp regression on "
                f"topic {record.topic!r}: "
                f"{record.bag_timestamp_ns} < "
                f"{previous_topic_timestamp}"
            )

        last_timestamps[
            record.topic
        ] = record.bag_timestamp_ns

    expected_counts = (
        _metadata_topic_counts(
            descriptor.metadata
        )
    )

    unexpected_topics = (
        set(counts)
        - set(expected_counts)
    )

    if unexpected_topics:
        raise McapIngestError(
            "reader produced topics not "
            "declared in metadata: "
            f"{sorted(unexpected_topics)}"
        )

    actual_counts = {
        topic: counts.get(topic, 0)
        for topic in expected_counts
    }

    if actual_counts != expected_counts:
        raise McapIngestError(
            "observed message counts do not "
            "match metadata.yaml; "
            f"metadata={expected_counts!r}, "
            f"observed={actual_counts!r}"
        )

    total_messages = sum(
        actual_counts.values()
    )

    if (
        total_messages
        != descriptor.metadata.message_count
    ):
        raise McapIngestError(
            f"observed total {total_messages} "
            "does not match metadata total "
            f"{descriptor.metadata.message_count}"
        )

    validation_passed_raw = (
        descriptor.validation.get(
            "passed"
        )
    )

    validation_passed = (
        validation_passed_raw
        if isinstance(
            validation_passed_raw,
            bool,
        )
        else None
    )

    operator_schema_raw = (
        descriptor.operator.get(
            "schema_version"
        )
    )

    validation_schema_raw = (
        descriptor.validation.get(
            "schema_version"
        )
    )

    return EpisodeScanSummary(
        episode_dir=str(
            descriptor.episode_dir
        ),
        storage_identifier=(
            descriptor
            .metadata
            .storage_identifier
        ),
        mcap_files=(
            descriptor
            .metadata
            .relative_file_paths
        ),
        total_messages=total_messages,
        topic_counts=dict(
            sorted(
                actual_counts.items()
            )
        ),
        topic_types=dict(
            sorted(
                _metadata_topic_types(
                    descriptor.metadata
                ).items()
            )
        ),
        first_bag_timestamp_ns=dict(
            sorted(
                first_timestamps.items()
            )
        ),
        last_bag_timestamp_ns=dict(
            sorted(
                last_timestamps.items()
            )
        ),
        operator_schema_version=(
            operator_schema_raw
            if isinstance(
                operator_schema_raw,
                str,
            )
            else None
        ),
        validation_schema_version=(
            validation_schema_raw
            if isinstance(
                validation_schema_raw,
                str,
            )
            else None
        ),
        validation_passed=(
            validation_passed
        ),
    )
