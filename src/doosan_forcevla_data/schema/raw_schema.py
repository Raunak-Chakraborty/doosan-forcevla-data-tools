"""Small standard-library structures for raw episode metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


REQUIRED_METADATA_KEYS = [
    "episode_id",
    "task_instruction",
    "geometry_type",
    "orientation_type",
    "collection_method",
    "action_label_primary",
    "success",
    "failure_reason",
    "fps",
]


@dataclass(frozen=True)
class RawEpisodeMetadata:
    """Metadata that describes one raw collection episode."""

    episode_id: str
    task_instruction: str
    geometry_type: str
    orientation_type: str
    collection_method: str
    action_label_primary: str = "measured_tcp_delta"
    success: bool = False
    failure_reason: str | None = None
    fps: int = 30
    optional_action_streams: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "RawEpisodeMetadata":
        """Build metadata from a JSON dictionary and reject missing required keys."""

        missing = [key for key in REQUIRED_METADATA_KEYS if key not in data]
        if missing:
            raise ValueError(f"metadata missing required keys: {', '.join(missing)}")

        return cls(
            episode_id=str(data["episode_id"]),
            task_instruction=str(data["task_instruction"]),
            geometry_type=str(data["geometry_type"]),
            orientation_type=str(data["orientation_type"]),
            collection_method=str(data["collection_method"]),
            action_label_primary=str(data["action_label_primary"]),
            success=bool(data["success"]),
            failure_reason=(
                None if data["failure_reason"] is None else str(data["failure_reason"])
            ),
            fps=int(data["fps"]),
            optional_action_streams=list(data.get("optional_action_streams", [])),
        )


@dataclass(frozen=True)
class RawEpisodePaths:
    """Relative paths expected in a v0 raw episode directory."""

    metadata: str = "metadata.json"
    tcp_pose: str = "robot/tcp_pose.csv"
    joint_states: str = "robot/joint_states.csv"
    wrench: str = "force/wrench.csv"
    commanded_twist: str = "actions/commanded_twist.csv"
    events: str = "events.csv"
    external_rgb: str = "images/external_rgb"
    tcp_rgb: str = "images/tcp_rgb"


RAW_EPISODE_PATHS = RawEpisodePaths()
