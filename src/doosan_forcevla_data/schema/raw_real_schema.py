"""Constants for future raw real Doosan episode files.

This module intentionally uses only the Python standard library.  It describes
files produced by a future recorder, but it does not import ROS packages and it
does not implement a recorder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RAW_REAL_SCHEMA_VERSION = "raw_real_v0"

REQUIRED_TOP_LEVEL_FILES = [
    "metadata.json",
    "calibration_refs.json",
    "events.jsonl",
    "recorder_report.json",
    "streams/index.json",
]

REQUIRED_STREAM_NAMES = [
    "joint_states",
    "robot_state_rt",
    "tf",
    "tf_static",
    "external_camera",
    "wrist_camera",
]

OPTIONAL_STREAM_NAMES = [
    "command_context",
    "gripper_state",
]

REQUIRED_METADATA_KEYS = [
    "schema_version",
    "episode_id",
    "task_instruction",
    "geometry_type",
    "orientation_type",
    "collection_method",
    "action_label_primary",
    "success",
    "failure_reason",
    "fps",
    "robot_type",
    "recorder_version",
    "source_workspace",
]

COMMON_TIMESTAMP_KEYS = [
    "record_index",
    "source_stamp",
    "receipt_stamp",
    "monotonic_stamp",
]

DEFAULT_STREAM_RELATIVE_PATHS = {
    "joint_states": "streams/joint_states.jsonl",
    "robot_state_rt": "streams/robot_state_rt.jsonl",
    "tf": "streams/tf.jsonl",
    "tf_static": "streams/tf_static.jsonl",
    "external_camera": "streams/external_camera",
    "wrist_camera": "streams/wrist_camera",
    "command_context": "streams/command_context.jsonl",
    "gripper_state": "streams/gripper_state.jsonl",
}


@dataclass(frozen=True)
class RawRealEpisodePaths:
    """Convenience accessors for a raw real episode directory."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root))

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def calibration_refs(self) -> Path:
        return self.root / "calibration_refs.json"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def recorder_report(self) -> Path:
        return self.root / "recorder_report.json"

    @property
    def streams_dir(self) -> Path:
        return self.root / "streams"

    @property
    def streams_index(self) -> Path:
        return self.streams_dir / "index.json"

    def stream_path(self, stream_name: str) -> Path:
        """Return the default path for a known stream name."""

        relative_path = DEFAULT_STREAM_RELATIVE_PATHS.get(
            stream_name, f"streams/{stream_name}.jsonl"
        )
        return self.root / relative_path

    @property
    def joint_states(self) -> Path:
        return self.stream_path("joint_states")

    @property
    def robot_state_rt(self) -> Path:
        return self.stream_path("robot_state_rt")

    @property
    def tf(self) -> Path:
        return self.stream_path("tf")

    @property
    def tf_static(self) -> Path:
        return self.stream_path("tf_static")

    @property
    def command_context(self) -> Path:
        return self.stream_path("command_context")

    @property
    def gripper_state(self) -> Path:
        return self.stream_path("gripper_state")

    @property
    def external_camera(self) -> Path:
        return self.stream_path("external_camera")

    @property
    def wrist_camera(self) -> Path:
        return self.stream_path("wrist_camera")

    @property
    def external_camera_index(self) -> Path:
        return self.external_camera / "index.jsonl"

    @property
    def wrist_camera_index(self) -> Path:
        return self.wrist_camera / "index.jsonl"
