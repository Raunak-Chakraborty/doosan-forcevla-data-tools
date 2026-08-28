"""Golden Episode-10 acceptance contract for the production Doosan pipeline.

Patch 9 deliberately adds no new dataset semantics.  It freezes one known-good
real episode as an end-to-end acceptance target for the layers already owned by
Patches 1--8: raw MCAP identity, Patch-4 synchronization, the Patch-5/6 25D
state, the Patch-7 measured 7D action, the Patch-8 two-camera processed/export
layout, the pinned LeRobot loader, and the ForceVLA Doosan adapter.

Heavy ForceVLA/LeRobot dependencies are imported lazily so this module remains
importable in the ROS/Jazzy Python used for raw MCAP conversion.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from doosan_forcevla_data.convert.doosan_processed_episode_v1 import (
    CAMERA_SPECS,
    EXTERNAL_CAMERA_KEY,
    FPS,
    PROCESSED_SCHEMA_ID,
    SYNTHETIC_MODEL_SLOT,
    TCP_CAMERA_KEY,
)
from doosan_forcevla_data.convert.doosan_processed_to_lerobot_v21 import (
    EXTERNAL_VIDEO_KEY,
    EXPECTED_DLIMP_COMMIT,
    EXPECTED_FORCEVLA_COMMIT,
    EXPECTED_LEROBOT_COMMIT,
    EXPORT_SCHEMA_ID,
    TCP_VIDEO_KEY,
)
from doosan_forcevla_data.ingest.doosan_raw_v1 import (
    EXTERNAL_CAMERA_INFO_TOPIC,
    EXTERNAL_IMAGE_TOPIC,
    GRIPPER_STATE_TOPIC,
    JOINT_STATE_TOPIC,
    JOY_TOPIC,
    ROBOT_STATE_RT_TOPIC,
    SPEEDL_STREAM_TOPIC,
    TCP_CAMERA_INFO_TOPIC,
    TCP_IMAGE_TOPIC,
    TF_STATIC_TOPIC,
    TF_TOPIC,
    validate_doosan_raw_v1_episode,
)
from doosan_forcevla_data.validate.validate_doosan_lerobot_v21 import (
    validate_doosan_lerobot_v21,
)
from doosan_forcevla_data.validate.validate_doosan_processed_episode_v1 import (
    validate_doosan_processed_episode_v1,
)


GOLDEN_SCHEMA_ID = "doosan_episode10_golden_acceptance_v1"
GOLDEN_EPISODE_INDEX = 10
GOLDEN_TASK = "real_robot_demonstration"
GOLDEN_MCAP_FILENAME = "episode_000010_0.mcap"
GOLDEN_MCAP_SHA256 = "075365f21d7e6a5cbb6d42cb4ad099b0b46fa71ce4eea4f069bb9e1231936d57"
GOLDEN_MESSAGE_COUNT = 16_702
GOLDEN_SYNCHRONIZED_STATE_COUNT = 1009
GOLDEN_FRAME_COUNT = 1008
GOLDEN_TERMINAL_REFERENCE_INDEX = 1008
GOLDEN_RELEASE_SOURCE_REFERENCE_INDEX = 955
GOLDEN_RELEASE_TARGET_REFERENCE_INDEX = 956
GOLDEN_HELD_ACTION_TARGET_COUNT = 955
GOLDEN_RELEASED_ACTION_TARGET_COUNT = 53
GOLDEN_HELD_STATE_ROW_COUNT = 956
GOLDEN_RELEASED_STATE_ROW_COUNT = 52
GOLDEN_ACTION_HORIZON = 50
GOLDEN_INTERNAL_ACTION_DIM = 32
GOLDEN_FRAMES_CANONICAL_SHA256 = (
    "3ab89835fde8a8a03c780d81d1a498f04e49868fffa7b5a5b2458db0bed4629f"
)

FROZEN_FORCEVLA_COMMIT = "9b61abef116f207d587d10aaf30170b73757c3e0"
FROZEN_LEROBOT_COMMIT = "e7aea92dd833f83d163820dcf2e58250307697a4"
FROZEN_DLIMP_COMMIT = "5edaa4691567873d495633f2708982b42edf1972"
FROZEN_LEROBOT_URL = "https://github.com/Raunak-Chakraborty/lerobot.git"

FROZEN_PYTHON_MAJOR_MINOR = (3, 11)
FROZEN_RUNTIME_VERSIONS = {
    "datasets": "4.8.5",
    "numpy": "1.26.4",
    "torch": "2.12.0+cu130",
    "torchvision": "0.27.0+cu130",
    "jax": "0.5.3",
    "pyarrow": "24.0.0",
    "av": "17.0.1",
}

GOLDEN_TOPIC_COUNTS = {
    TCP_CAMERA_INFO_TOPIC: 1009,
    EXTERNAL_CAMERA_INFO_TOPIC: 2001,
    TCP_IMAGE_TOPIC: 1009,
    TF_STATIC_TOPIC: 1,
    TF_TOPIC: 605,
    GRIPPER_STATE_TOPIC: 1699,
    JOY_TOPIC: 1636,
    SPEEDL_STREAM_TOPIC: 1698,
    EXTERNAL_IMAGE_TOPIC: 2001,
    JOINT_STATE_TOPIC: 3362,
    ROBOT_STATE_RT_TOPIC: 1681,
}

if EXPECTED_FORCEVLA_COMMIT != FROZEN_FORCEVLA_COMMIT:  # pragma: no cover
    raise RuntimeError("Patch-8 ForceVLA provenance drifted from the Patch-9 golden contract")
if EXPECTED_LEROBOT_COMMIT != FROZEN_LEROBOT_COMMIT:  # pragma: no cover
    raise RuntimeError("Patch-8 LeRobot provenance drifted from the Patch-9 golden contract")
if EXPECTED_DLIMP_COMMIT != FROZEN_DLIMP_COMMIT:  # pragma: no cover
    raise RuntimeError("Patch-8 dlimp provenance drifted from the Patch-9 golden contract")


class GoldenEpisode10Error(ValueError):
    """Raised when the frozen Episode-10 end-to-end contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GoldenEpisode10Error(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenEpisode10Error(f"{path}: could not read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GoldenEpisode10Error(f"{path}: expected JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GoldenEpisode10Error(f"{path}: could not read JSONL: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise GoldenEpisode10Error(f"{path}: empty line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenEpisode10Error(
                f"{path}: invalid JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise GoldenEpisode10Error(f"{path}: line {line_number} is not an object")
        rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GoldenEpisode10Error(f"{path}: could not hash file: {exc}") from exc
    return digest.hexdigest()


def _canonical_jsonl_sha256(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        try:
            encoded = (
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise GoldenEpisode10Error(f"could not canonicalize processed row: {exc}") from exc
        digest.update(encoded)
    return digest.hexdigest()


def _run_git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or ""
        suffix = f": {detail.strip()}" if detail.strip() else ""
        raise GoldenEpisode10Error(
            f"git {' '.join(args)} failed for {root}{suffix}"
        ) from exc
    return completed.stdout.strip()


def validate_golden_source_repositories(
    forcevla_root: str | Path,
) -> dict[str, Any]:
    """Require the exact ForceVLA/LeRobot/dlimp commits frozen after Patch 8."""

    forcevla = Path(forcevla_root).resolve()
    lerobot = (forcevla / "lerobot").resolve()
    dlimp = (forcevla / "dlimp").resolve()

    for name, root in (("ForceVLA", forcevla), ("LeRobot", lerobot), ("dlimp", dlimp)):
        _require(root.is_dir(), f"{name} repository does not exist: {root}")
        status = _run_git(root, "status", "--porcelain=v1", "-uall")
        _require(not status, f"{name} repository must be clean for golden acceptance")

    forcevla_head = _run_git(forcevla, "rev-parse", "HEAD")
    lerobot_head = _run_git(lerobot, "rev-parse", "HEAD")
    dlimp_head = _run_git(dlimp, "rev-parse", "HEAD")

    _require(
        forcevla_head == FROZEN_FORCEVLA_COMMIT,
        f"ForceVLA commit mismatch: {forcevla_head}",
    )
    _require(
        lerobot_head == FROZEN_LEROBOT_COMMIT,
        f"LeRobot commit mismatch: {lerobot_head}",
    )
    _require(dlimp_head == FROZEN_DLIMP_COMMIT, f"dlimp commit mismatch: {dlimp_head}")

    gitmodules_url = _run_git(
        forcevla,
        "config",
        "-f",
        ".gitmodules",
        "--get",
        "submodule.lerobot.url",
    )
    _require(
        gitmodules_url == FROZEN_LEROBOT_URL,
        f"ForceVLA .gitmodules LeRobot URL mismatch: {gitmodules_url!r}",
    )

    return {
        "forcevla_root": str(forcevla),
        "forcevla_commit": forcevla_head,
        "lerobot_root": str(lerobot),
        "lerobot_commit": lerobot_head,
        "lerobot_url": gitmodules_url,
        "dlimp_root": str(dlimp),
        "dlimp_commit": dlimp_head,
    }


def validate_golden_raw_episode(episode_dir: str | Path) -> dict[str, Any]:
    """Validate Episode 10 identity before any expensive conversion work."""

    episode = Path(episode_dir).resolve()
    descriptor = validate_doosan_raw_v1_episode(episode)
    metadata = descriptor.metadata

    _require(
        metadata.relative_file_paths == (GOLDEN_MCAP_FILENAME,),
        f"golden episode must contain exactly {GOLDEN_MCAP_FILENAME}",
    )
    _require(
        metadata.message_count == GOLDEN_MESSAGE_COUNT,
        f"golden total message count must be {GOLDEN_MESSAGE_COUNT}",
    )

    topic_counts = {topic.name: topic.message_count for topic in metadata.topics}
    _require(
        topic_counts == GOLDEN_TOPIC_COUNTS,
        f"golden per-topic counts changed: {topic_counts!r}",
    )

    custom = metadata.custom_data
    _require(custom.get("episode_index") == GOLDEN_EPISODE_INDEX, "metadata episode_index must be 10")
    _require(custom.get("task") == GOLDEN_TASK, f"metadata task must be {GOLDEN_TASK!r}")

    operator_task = descriptor.operator.get("task")
    validation_custom = descriptor.validation.get("metadata", {}).get("custom_data", {})
    validation_task = validation_custom.get("task") if isinstance(validation_custom, dict) else None
    validation_index = validation_custom.get("episode_index") if isinstance(validation_custom, dict) else None

    _require(operator_task == GOLDEN_TASK, f"operator task must be {GOLDEN_TASK!r}")
    _require(validation_task == GOLDEN_TASK, f"validation task must be {GOLDEN_TASK!r}")
    _require(validation_index == GOLDEN_EPISODE_INDEX, "validation episode_index must be 10")

    mcap = episode / GOLDEN_MCAP_FILENAME
    _require(mcap.is_file(), f"golden MCAP is missing: {mcap}")
    mcap_sha = _sha256_file(mcap)
    _require(mcap_sha == GOLDEN_MCAP_SHA256, f"golden MCAP SHA256 mismatch: {mcap_sha}")

    return {
        "episode_dir": str(episode),
        "episode_index": GOLDEN_EPISODE_INDEX,
        "task": GOLDEN_TASK,
        "mcap": str(mcap),
        "mcap_sha256": mcap_sha,
        "message_count": metadata.message_count,
        "topic_counts": dict(sorted(topic_counts.items())),
    }


def _binary_counter(values: Sequence[Any], *, context: str) -> Counter[float]:
    result: Counter[float] = Counter()
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GoldenEpisode10Error(f"{context}[{index}] must be numeric binary value")
        converted = float(value)
        if converted not in (0.0, 1.0):
            raise GoldenEpisode10Error(f"{context}[{index}] is not binary: {converted}")
        result[converted] += 1
    return result


def validate_golden_processed_episode(processed_dir: str | Path) -> dict[str, Any]:
    """Assert the exact frozen Episode-10 Patch-8 processed output."""

    root = Path(processed_dir).resolve()
    validation = validate_doosan_processed_episode_v1(root)
    _require(validation.ok, "processed validation failed: " + "; ".join(validation.errors))

    metadata = _read_json(root / "metadata_processed.json")
    rows = _read_jsonl(root / "frames.jsonl")

    _require(metadata.get("schema_version") == PROCESSED_SCHEMA_ID, "processed schema mismatch")
    _require(metadata.get("source_episode_index") == GOLDEN_EPISODE_INDEX, "processed episode index mismatch")
    _require(metadata.get("task") == GOLDEN_TASK, "processed task mismatch")
    _require(metadata.get("fps") == FPS, f"processed fps must be {FPS}")
    _require(metadata.get("synchronized_state_count") == GOLDEN_SYNCHRONIZED_STATE_COUNT, "synchronized state count mismatch")
    _require(metadata.get("frame_count") == GOLDEN_FRAME_COUNT, "processed frame_count mismatch")
    _require(metadata.get("measured_action_count") == GOLDEN_FRAME_COUNT, "measured action count mismatch")
    _require(metadata.get("excluded_terminal_reference_index") == GOLDEN_TERMINAL_REFERENCE_INDEX, "terminal reference mismatch")
    _require(metadata.get("terminal_action_emitted") is False, "terminal action must remain absent")
    _require(metadata.get("state_dim") == 25, "processed state dimension must be 25")
    _require(metadata.get("action_dim") == 7, "processed action dimension must be 7")
    _require(metadata.get("physical_camera_count") == 2, "processed physical camera count must be 2")
    _require(metadata.get("cameras") == CAMERA_SPECS, "processed camera contract changed")
    _require(metadata.get("synthetic_model_slot") == SYNTHETIC_MODEL_SLOT, "synthetic right-wrist contract changed")

    source_raw = metadata.get("source_raw_episode")
    _require(isinstance(source_raw, str) and Path(source_raw).name == "episode_000010", "processed source_raw_episode is not Episode 10")

    _require(len(rows) == GOLDEN_FRAME_COUNT, "processed row count must be exactly 1008")
    digest = _canonical_jsonl_sha256(rows)
    _require(
        digest == GOLDEN_FRAMES_CANONICAL_SHA256,
        f"processed frames canonical SHA256 mismatch: {digest}",
    )

    action_targets: list[float] = []
    state_gripper: list[float] = []
    for index, row in enumerate(rows):
        _require(row.get("frame_index") == index, f"row {index}: frame_index mismatch")
        _require(row.get("reference_index") == index, f"row {index}: reference_index mismatch")
        _require(row.get("action_target_reference_index") == index + 1, f"row {index}: action target mismatch")

        state = row.get("observation_state_25d")
        action = row.get("action_7d")
        _require(isinstance(state, list) and len(state) == 25, f"row {index}: state shape mismatch")
        _require(isinstance(action, list) and len(action) == 7, f"row {index}: action shape mismatch")
        _require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in state), f"row {index}: non-finite state")
        _require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in action), f"row {index}: non-finite action")

        state_gripper.append(float(state[6]))
        action_targets.append(float(action[6]))

    action_counts = _binary_counter(action_targets, context="action gripper target")
    state_counts = _binary_counter(state_gripper, context="state gripper")

    _require(
        action_counts == Counter({0.0: GOLDEN_HELD_ACTION_TARGET_COUNT, 1.0: GOLDEN_RELEASED_ACTION_TARGET_COUNT}),
        f"golden action gripper counts changed: {dict(action_counts)!r}",
    )
    _require(
        state_counts == Counter({0.0: GOLDEN_HELD_STATE_ROW_COUNT, 1.0: GOLDEN_RELEASED_STATE_ROW_COUNT}),
        f"golden processed-state gripper counts changed: {dict(state_counts)!r}",
    )

    release_row = rows[GOLDEN_RELEASE_SOURCE_REFERENCE_INDEX]
    _require(release_row["action_target_reference_index"] == GOLDEN_RELEASE_TARGET_REFERENCE_INDEX, "release target reference must be 956")
    _require(float(release_row["observation_state_25d"][6]) == 0.0, "release action source must still be held")
    _require(float(release_row["action_7d"][6]) == 1.0, "release action must target open/released")
    _require(float(rows[GOLDEN_RELEASE_SOURCE_REFERENCE_INDEX - 1]["action_7d"][6]) == 0.0, "row before release must still target held")
    _require(float(rows[GOLDEN_RELEASE_TARGET_REFERENCE_INDEX]["observation_state_25d"][6]) == 1.0, "release target state must be released")

    _require(rows[-1]["reference_index"] == GOLDEN_FRAME_COUNT - 1, "last source reference must be 1007")
    _require(rows[-1]["action_target_reference_index"] == GOLDEN_TERMINAL_REFERENCE_INDEX, "last action must target terminal reference 1008")

    return {
        "processed_dir": str(root),
        "schema_version": metadata["schema_version"],
        "frame_count": len(rows),
        "synchronized_state_count": metadata["synchronized_state_count"],
        "excluded_terminal_reference_index": metadata["excluded_terminal_reference_index"],
        "frames_canonical_sha256": digest,
        "action_gripper_target_counts": {"held": action_counts[0.0], "released": action_counts[1.0]},
        "processed_state_gripper_counts": {"held": state_counts[0.0], "released": state_counts[1.0]},
        "release_source_reference_index": GOLDEN_RELEASE_SOURCE_REFERENCE_INDEX,
        "release_target_reference_index": GOLDEN_RELEASE_TARGET_REFERENCE_INDEX,
    }


def _compare_parquet_rows(
    processed_rows: Sequence[dict[str, Any]],
    parquet_rows: Sequence[dict[str, Any]],
) -> None:
    _require(len(processed_rows) == GOLDEN_FRAME_COUNT, "processed comparison rows must be 1008")
    _require(len(parquet_rows) == GOLDEN_FRAME_COUNT, "Parquet row count must be 1008")

    for index, (source, target) in enumerate(zip(processed_rows, parquet_rows, strict=True)):
        _require(target.get("observation.state") == source.get("observation_state_25d"), f"Parquet row {index}: state changed during export")
        _require(target.get("action") == source.get("action_7d"), f"Parquet row {index}: action changed during export")
        _require(target.get("frame_index") == index, f"Parquet row {index}: frame_index mismatch")
        _require(target.get("index") == index, f"Parquet row {index}: global index mismatch")
        _require(target.get("episode_index") == 0, f"Parquet row {index}: episode_index mismatch")
        _require(target.get("task_index") == 0, f"Parquet row {index}: task_index mismatch")
        timestamp = target.get("timestamp")
        _require(isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool), f"Parquet row {index}: timestamp must be numeric")
        _require(abs(float(timestamp) - index / FPS) <= 2e-6, f"Parquet row {index}: timestamp mismatch")


def validate_golden_lerobot_dataset(
    processed_dir: str | Path,
    dataset_dir: str | Path,
) -> dict[str, Any]:
    """Validate the exact on-disk LeRobot export and semantic row preservation."""

    processed_root = Path(processed_dir).resolve()
    dataset_root = Path(dataset_dir).resolve()

    validation = validate_doosan_lerobot_v21(dataset_root)
    _require(validation.ok, "LeRobot validation failed: " + "; ".join(validation.errors))
    _require(validation.frame_count == GOLDEN_FRAME_COUNT, "LeRobot validator frame count mismatch")

    processed_rows = _read_jsonl(processed_root / "frames.jsonl")
    info = _read_json(dataset_root / "meta" / "info.json")
    provenance = _read_json(dataset_root / "meta" / "export_provenance.json")
    tasks = _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")

    _require(info.get("total_frames") == GOLDEN_FRAME_COUNT, "LeRobot total_frames must be 1008")
    _require(info.get("total_videos") == 2, "LeRobot total_videos must be 2")
    _require(info.get("fps") == FPS, "LeRobot fps mismatch")
    _require(
        set(info.get("features", {}))
        == {
            "observation.state",
            "action",
            TCP_VIDEO_KEY,
            EXTERNAL_VIDEO_KEY,
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        },
        "LeRobot feature set changed",
    )
    _require(tasks == [{"task": GOLDEN_TASK, "task_index": 0}], "LeRobot task metadata changed")
    _require(episodes == [{"episode_index": 0, "length": GOLDEN_FRAME_COUNT, "tasks": [GOLDEN_TASK]}], "LeRobot episode metadata changed")

    _require(provenance.get("schema_version") == EXPORT_SCHEMA_ID, "export provenance schema mismatch")
    _require(provenance.get("target_forcevla_commit") == FROZEN_FORCEVLA_COMMIT, "exported ForceVLA provenance mismatch")
    _require(provenance.get("target_lerobot_commit") == FROZEN_LEROBOT_COMMIT, "exported LeRobot provenance mismatch")
    _require(provenance.get("target_dlimp_commit") == FROZEN_DLIMP_COMMIT, "exported dlimp provenance mismatch")
    _require(provenance.get("frame_count") == GOLDEN_FRAME_COUNT, "export provenance frame_count mismatch")
    _require(provenance.get("task") == GOLDEN_TASK, "export provenance task mismatch")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GoldenEpisode10Error(
            "PyArrow is required for golden LeRobot acceptance; run this stage in the frozen ForceVLA environment"
        ) from exc

    parquet_path = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    try:
        table = pq.read_table(parquet_path)
        parquet_rows = table.to_pylist()
    except Exception as exc:
        raise GoldenEpisode10Error(f"could not read golden Parquet file {parquet_path}: {exc}") from exc

    _compare_parquet_rows(processed_rows, parquet_rows)

    return {
        "dataset_dir": str(dataset_root),
        "frame_count": len(parquet_rows),
        "physical_video_count": info["total_videos"],
        "task": GOLDEN_TASK,
        "target_forcevla_commit": provenance["target_forcevla_commit"],
        "target_lerobot_commit": provenance["target_lerobot_commit"],
        "target_dlimp_commit": provenance["target_dlimp_commit"],
        "parquet_state_action_exact_match": True,
    }


def _require_module_below(module_file: Any, root: Path, name: str) -> str:
    _require(isinstance(module_file, str) and module_file, f"{name} module has no __file__")
    resolved = Path(module_file).resolve()
    _require(resolved.is_relative_to(root.resolve()), f"{name} loaded from unexpected path: {resolved}")
    return str(resolved)


def validate_golden_forcevla_runtime(
    dataset_dir: str | Path,
    *,
    converter_root: str | Path,
    forcevla_root: str | Path,
) -> dict[str, Any]:
    """Load the golden dataset through pinned LeRobot and the 25D ForceVLA adapter."""

    converter = Path(converter_root).resolve()
    forcevla = Path(forcevla_root).resolve()
    lerobot_root = (forcevla / "lerobot").resolve()
    dataset_root = Path(dataset_dir).resolve()

    try:
        import av
        import datasets
        import jax
        import numpy as np
        import pyarrow
        import torch
        import torchvision
        import doosan_forcevla_data
        import lerobot
        import openpi.policies.forcevla_policy as forcevla_policy_module
        import openpi.transforms as transforms_module
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from openpi.policies.forcevla_policy import DoosanForcevlaInputs
        from openpi.transforms import PromptFromLeRobotTask
    except Exception as exc:  # pragma: no cover - environment dependent
        raise GoldenEpisode10Error(
            f"frozen ForceVLA runtime imports failed under {sys.executable}: {exc}"
        ) from exc

    _require(sys.version_info[:2] == FROZEN_PYTHON_MAJOR_MINOR, f"ForceVLA Python must be 3.11, got {sys.version.split()[0]}")
    _require(not any("/opt/ros/jazzy/lib/python3.12" in value for value in sys.path), "ROS Python 3.12 leaked into ForceVLA sys.path")

    actual_versions = {
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "jax": jax.__version__,
        "pyarrow": pyarrow.__version__,
        "av": av.__version__,
    }
    _require(actual_versions == FROZEN_RUNTIME_VERSIONS, f"ForceVLA runtime version drift: {actual_versions!r}")

    origins = {
        "converter": _require_module_below(
            doosan_forcevla_data.__file__, converter / "src", "doosan_forcevla_data"
        ),
        "lerobot": _require_module_below(lerobot.__file__, lerobot_root, "lerobot"),
        "forcevla_policy": _require_module_below(
            forcevla_policy_module.__file__, forcevla / "src", "openpi.policies.forcevla_policy"
        ),
        "openpi_transforms": _require_module_below(
            transforms_module.__file__, forcevla / "src", "openpi.transforms"
        ),
    }

    dataset = LeRobotDataset(
        "doosan_peg_in_hole_v0",
        root=dataset_root,
        revision="v2.1",
        delta_timestamps={"action": [step / FPS for step in range(GOLDEN_ACTION_HORIZON)]},
        video_backend="pyav",
    )

    _require(len(dataset) == GOLDEN_FRAME_COUNT, "pinned LeRobot loader row count mismatch")
    _require(set(dataset.meta.video_keys) == {TCP_VIDEO_KEY, EXTERNAL_VIDEO_KEY}, "pinned LeRobot video keys changed")
    _require(len(dataset.meta.video_keys) == 2, "pinned LeRobot physical video key count must be 2")
    _require(dataset.meta.total_frames == GOLDEN_FRAME_COUNT, "pinned LeRobot total_frames mismatch")
    _require(dataset.meta.tasks == {0: GOLDEN_TASK}, "pinned LeRobot task mapping changed")

    item0 = dataset[0]
    item_before_release = dataset[GOLDEN_RELEASE_SOURCE_REFERENCE_INDEX - 1]
    item_release = dataset[GOLDEN_RELEASE_SOURCE_REFERENCE_INDEX]
    item_last = dataset[GOLDEN_FRAME_COUNT - 1]

    _require(tuple(item0["observation.state"].shape) == (25,), "loader state shape must be 25")
    _require(tuple(item0["action"].shape) == (GOLDEN_ACTION_HORIZON, 7), "loader action horizon must be 50x7")
    _require(tuple(item0[TCP_VIDEO_KEY].shape) == (3, 480, 640), "D405 loader image shape mismatch")
    _require(tuple(item0[EXTERNAL_VIDEO_KEY].shape) == (3, 480, 848), "D435I loader image shape mismatch")
    _require(float(item_before_release["action"][0, 6]) == 0.0, "row before release must target held")
    _require(float(item_release["action"][0, 6]) == 1.0, "release row must target open")

    padding = item_last.get("action_is_pad")
    _require(tuple(padding.shape) == (GOLDEN_ACTION_HORIZON,), "terminal loader action_is_pad shape mismatch")
    _require(bool(padding[0]) is False, "terminal source action must not be padding")
    _require(bool(padding[1:].all()) is True, "future actions after terminal successor must be loader-side padding")

    prompted = PromptFromLeRobotTask(dataset.meta.tasks)(item0)
    _require(prompted.get("prompt") == GOLDEN_TASK, "LeRobot task did not map to exact ForceVLA prompt")

    adapter = DoosanForcevlaInputs(
        state_dim=25,
        action_dim=GOLDEN_INTERNAL_ACTION_DIM,
        robot_action_dim=7,
    )
    adapted = adapter(
        {
            "images": {
                TCP_CAMERA_KEY: item0[TCP_VIDEO_KEY],
                EXTERNAL_CAMERA_KEY: item0[EXTERNAL_VIDEO_KEY],
            },
            "state": item0["observation.state"],
            "actions": item0["action"],
            "prompt": prompted["prompt"],
        }
    )

    _require(adapted["state"].shape[-1] == 25, "ForceVLA adapter state dimension changed")
    _require(adapted["actions"].shape == (GOLDEN_ACTION_HORIZON, GOLDEN_INTERNAL_ACTION_DIM), "ForceVLA internal action shape must be 50x32")
    _require(adapted["image"]["base_0_rgb"].shape == (480, 848, 3), "D435I must map to base_0_rgb")
    _require(adapted["image"]["left_wrist_0_rgb"].shape == (480, 640, 3), "D405 must map to left_wrist_0_rgb")
    _require(adapted["image"]["right_wrist_0_rgb"].shape == (480, 848, 3), "synthetic right wrist shape mismatch")
    _require(bool(adapted["image_mask"]["base_0_rgb"]), "base image mask must be true")
    _require(bool(adapted["image_mask"]["left_wrist_0_rgb"]), "left-wrist image mask must be true")
    _require(not bool(adapted["image_mask"]["right_wrist_0_rgb"]), "right-wrist image mask must be false")
    _require(np.count_nonzero(adapted["image"]["right_wrist_0_rgb"]) == 0, "right-wrist synthetic image must be all zeros")

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "versions": actual_versions,
        "module_origins": origins,
        "dataset_rows": len(dataset),
        "action_horizon": GOLDEN_ACTION_HORIZON,
        "semantic_action_dim": 7,
        "internal_action_dim": GOLDEN_INTERNAL_ACTION_DIM,
        "tcp_image_shape_chw": [3, 480, 640],
        "external_image_shape_chw": [3, 480, 848],
        "loader_terminal_padding": {"first_is_pad": False, "remaining_are_pad": True},
        "forcevla_mapping": {
            "external_camera_2": "base_0_rgb",
            "tcp_camera": "left_wrist_0_rgb",
            "right_wrist_0_rgb": "synthetic_zero_mask_false",
        },
    }


def write_golden_report(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{os.getpid()}"
    staging.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    staging.replace(target)
    return target


__all__ = [
    "FROZEN_DLIMP_COMMIT",
    "FROZEN_FORCEVLA_COMMIT",
    "FROZEN_LEROBOT_COMMIT",
    "GOLDEN_ACTION_HORIZON",
    "GOLDEN_EPISODE_INDEX",
    "GOLDEN_FRAME_COUNT",
    "GOLDEN_FRAMES_CANONICAL_SHA256",
    "GOLDEN_MCAP_SHA256",
    "GOLDEN_SCHEMA_ID",
    "GOLDEN_TASK",
    "GoldenEpisode10Error",
    "validate_golden_forcevla_runtime",
    "validate_golden_lerobot_dataset",
    "validate_golden_processed_episode",
    "validate_golden_raw_episode",
    "validate_golden_source_repositories",
    "write_golden_report",
]
