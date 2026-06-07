"""Future action chunk utilities for ForceVLA/OpenPI training samples.

The processed/staged/exported dataset stores the primary measured action as a
single 7D TCP delta per frame.  ForceVLA/OpenPI policies consume an action
horizon during training, for example [50, 7] before ForceVLA pads it to
[50, 32].  This module builds those future chunks without changing the raw or
processed episode schema.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doosan_forcevla_data.schema.processed_schema import ACTION_DIM


DEFAULT_ACTION_HORIZON = 50
PAD_MODES = {"repeat_last", "zero"}


@dataclass(frozen=True)
class ActionChunk:
    actions: list[list[float]]
    valid_mask: list[bool]
    start_index: int
    horizon: int
    action_dim: int
    pad_mode: str
    source_action_count: int
    padded_count: int


def _finite_action_vector(values: Any, action_dim: int, name: str) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    if len(values) != action_dim:
        raise ValueError(f"{name} length must be {action_dim}, got {len(values)}")

    result: list[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}[{idx}] must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{idx}] must be finite")
        result.append(number)
    return result


def build_future_action_chunk(
    actions: list[list[Any]],
    start_index: int,
    horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
    pad_mode: str = "repeat_last",
) -> ActionChunk:
    """Build a future action chunk from a sequence of single-step actions.

    For frame ``i``, the chunk is:

        actions[i], actions[i+1], ..., actions[i+horizon-1]

    If the requested horizon extends past the available sequence, the tail is
    padded.  ``repeat_last`` repeats the final available action; ``zero`` pads
    with zeros.  The valid_mask marks which chunk rows came from real future
    actions and which were padded.
    """

    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError("action_dim must be a positive integer")
    if pad_mode not in PAD_MODES:
        raise ValueError(f"pad_mode must be one of: {', '.join(sorted(PAD_MODES))}")
    if not actions:
        raise ValueError("actions must contain at least one action vector")
    if start_index >= len(actions):
        raise ValueError(f"start_index {start_index} outside action sequence length {len(actions)}")

    normalized = [
        _finite_action_vector(action, action_dim, f"actions[{idx}]")
        for idx, action in enumerate(actions)
    ]

    chunk: list[list[float]] = []
    valid_mask: list[bool] = []

    for offset in range(horizon):
        source_idx = start_index + offset
        if source_idx < len(normalized):
            chunk.append(list(normalized[source_idx]))
            valid_mask.append(True)
        else:
            if pad_mode == "repeat_last":
                chunk.append(list(normalized[-1]))
            else:
                chunk.append([0.0] * action_dim)
            valid_mask.append(False)

    padded_count = sum(1 for value in valid_mask if not value)

    return ActionChunk(
        actions=chunk,
        valid_mask=valid_mask,
        start_index=start_index,
        horizon=horizon,
        action_dim=action_dim,
        pad_mode=pad_mode,
        source_action_count=len(normalized),
        padded_count=padded_count,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunks_size = info.get("chunks_size", 1000)
    if not isinstance(chunks_size, int) or isinstance(chunks_size, bool) or chunks_size <= 0:
        chunks_size = 1000
    return episode_index // chunks_size


def _format_data_path(info: dict[str, Any], episode_index: int) -> Path:
    template = info.get("data_path")
    if not isinstance(template, str) or not template:
        raise ValueError("meta/info.json must contain a non-empty data_path template")
    return Path(
        template.format(
            episode_chunk=_episode_chunk(info, episode_index),
            episode_index=episode_index,
        )
    )


def _require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required to build action chunks from LeRobot parquet exports. "
            "Run this inside the validated lab ForceVLA environment."
        ) from exc
    return pq


def load_single_step_actions_from_lerobot_export(
    dataset_root: str | Path,
    episode_index: int = 0,
    action_dim: int = ACTION_DIM,
) -> list[list[float]]:
    """Load the 7D single-step ``action`` column from a local LeRobot export."""

    root = Path(dataset_root)
    info = _read_json_object(root / "meta" / "info.json")
    parquet_path = root / _format_data_path(info, episode_index)
    if parquet_path.suffix != ".parquet":
        raise ValueError(f"expected parquet data path, got: {parquet_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    pq = _require_pyarrow_parquet()
    table = pq.read_table(parquet_path, columns=["action"])
    data = table.to_pydict()
    raw_actions = data.get("action")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError(f"{parquet_path}: action column is missing or empty")

    return [
        _finite_action_vector(action, action_dim, f"action row {idx}")
        for idx, action in enumerate(raw_actions)
    ]


def build_future_action_chunk_from_lerobot_export(
    dataset_root: str | Path,
    episode_index: int,
    row_index: int,
    horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
    pad_mode: str = "repeat_last",
) -> ActionChunk:
    """Load single-step actions from a local export and build one future chunk."""

    actions = load_single_step_actions_from_lerobot_export(
        dataset_root=dataset_root,
        episode_index=episode_index,
        action_dim=action_dim,
    )
    return build_future_action_chunk(
        actions=actions,
        start_index=row_index,
        horizon=horizon,
        action_dim=action_dim,
        pad_mode=pad_mode,
    )
