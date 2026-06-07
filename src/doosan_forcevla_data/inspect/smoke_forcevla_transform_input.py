"""Smoke-test conversion from exported Doosan dataset sample to ForceVLA input.

This tool validates the bridge:

    local LeRobot-style parquet/videos
    -> PyAV/parquet observation builder
    -> ForceVLA policy input adapter
    -> model-style observation dictionary

It intentionally does not run model inference or restore checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from doosan_forcevla_data.convert.action_chunks import build_future_action_chunk_from_lerobot_export

from doosan_forcevla_data.inspect.smoke_forcevla_observation_builder import build_smoke_observation


def _default_forcevla_root() -> Path:
    return Path.home() / "robotics_thesis" / "forcevla_lab" / "ForceVLA"


def _add_forcevla_to_path(forcevla_root: str | Path) -> Path:
    root = Path(forcevla_root).expanduser().resolve()
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"ForceVLA src directory not found: {src}")
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return root


def _require_forcevla_modules(forcevla_root: str | Path):
    root = _add_forcevla_to_path(forcevla_root)
    try:
        from openpi.models import model as _model
        from openpi.models import pi0_force
        from openpi.policies import forcevla_policy
    except Exception as exc:
        raise RuntimeError(
            f"Could not import local ForceVLA/OpenPI modules from {root}. "
            "Run inside the validated lab ForceVLA environment."
        ) from exc
    return root, _model, pi0_force, forcevla_policy


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(v) for v in shape]


def _dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)


def _safe_bool(value: Any) -> bool | str:
    try:
        return bool(value)
    except Exception:
        return str(value)


def _array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    summary: dict[str, Any] = {
        "shape": _shape(arr),
        "dtype": str(arr.dtype),
    }
    if arr.size:
        if np.issubdtype(arr.dtype, np.number):
            summary["min"] = float(np.nanmin(arr))
            summary["max"] = float(np.nanmax(arr))
            summary["mean"] = float(np.nanmean(arr))
        elif arr.dtype == np.bool_:
            summary["values"] = arr.tolist()
    return summary


def build_forcevla_transform_input_report(
    dataset_root: str | Path,
    forcevla_root: str | Path | None = None,
    episode_index: int = 0,
    row_index: int = 0,
    action_chunk_mode: str = "repeat",
) -> dict[str, Any]:
    """Build and validate one ForceVLA pre-tokenization input sample.

    ``action_chunk_mode=repeat`` converts the single 7D action label into a
    dummy action chunk with the model horizon. This is only for shape testing.
    Real training should use a real action horizon from consecutive frames.
    """

    if action_chunk_mode not in {"single", "repeat", "future"}:
        raise ValueError("action_chunk_mode must be 'single', 'repeat', or 'future'")

    forcevla_root = Path(forcevla_root or os.environ.get("FORCEVLA_ROOT") or _default_forcevla_root())
    forcevla_root, _model, pi0_force, forcevla_policy = _require_forcevla_modules(forcevla_root)

    observation, observation_report = build_smoke_observation(
        dataset_root=dataset_root,
        episode_index=episode_index,
        row_index=row_index,
        expected_state_dim=13,
        expected_action_dim=7,
        strict_video_shape=False,
    )

    errors: list[str] = []
    warnings: list[str] = list(observation_report.get("warnings", []))

    model_config = pi0_force.Pi0_GuidanceConfig(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )

    action_dim = int(model_config.action_dim)
    action_horizon = int(model_config.action_horizon)
    model_type = model_config.model_type

    raw_action = np.asarray(observation["action"], dtype=np.float32)
    action_chunk_report = None
    if action_chunk_mode == "future":
        action_chunk = build_future_action_chunk_from_lerobot_export(
            dataset_root=dataset_root,
            episode_index=episode_index,
            row_index=row_index,
            horizon=action_horizon,
            action_dim=7,
            pad_mode="repeat_last",
        )
        actions = np.asarray(action_chunk.actions, dtype=np.float32)
        action_chunk_report = {
            "mode": "future",
            "horizon": action_chunk.horizon,
            "action_dim": action_chunk.action_dim,
            "source_action_count": action_chunk.source_action_count,
            "valid_count": int(sum(action_chunk.valid_mask)),
            "padded_count": action_chunk.padded_count,
            "pad_mode": action_chunk.pad_mode,
            "valid_mask_first_20": action_chunk.valid_mask[:20],
        }
    elif action_chunk_mode == "repeat":
        actions = np.repeat(raw_action[None, :], action_horizon, axis=0)
        action_chunk_report = {
            "mode": "repeat",
            "horizon": action_horizon,
            "action_dim": int(raw_action.shape[-1]),
            "source_action_count": 1,
            "valid_count": action_horizon,
            "padded_count": 0,
            "pad_mode": "repeat_current_action",
        }
    else:
        actions = raw_action
        action_chunk_report = {
            "mode": "single",
            "horizon": 1,
            "action_dim": int(raw_action.shape[-1]),
            "source_action_count": 1,
            "valid_count": 1,
            "padded_count": 0,
            "pad_mode": "none",
        }

    policy_input = {
        "image": np.asarray(observation["observation.image"]),
        "wrist_image": np.asarray(observation["observation.wrist_image"]),
        "state": np.asarray(observation["observation.state"], dtype=np.float32),
        "actions": actions,
        "prompt": observation["prompt"],
    }

    transform = forcevla_policy.Forcevla_inputs(
        action_dim=action_dim,
        model_type=model_type,
    )
    transformed = transform(policy_input)

    required_top_keys = ["state", "image", "image_mask", "actions", "prompt"]
    for key in required_top_keys:
        if key not in transformed:
            errors.append(f"transformed output missing key: {key}")

    expected_image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    image = transformed.get("image", {})
    image_mask = transformed.get("image_mask", {})

    if not isinstance(image, dict):
        errors.append("transformed image must be a dict")
        image = {}
    if not isinstance(image_mask, dict):
        errors.append("transformed image_mask must be a dict")
        image_mask = {}

    for key in expected_image_keys:
        if key not in image:
            errors.append(f"transformed image missing key: {key}")
        if key not in image_mask:
            errors.append(f"transformed image_mask missing key: {key}")

    state = np.asarray(transformed.get("state"))
    transformed_actions = np.asarray(transformed.get("actions"))

    if state.shape[-1:] != (action_dim,):
        errors.append(f"state last dim should be padded to action_dim={action_dim}, got {state.shape}")
    if transformed_actions.shape[-1:] != (action_dim,):
        errors.append(
            f"actions last dim should be padded to action_dim={action_dim}, got {transformed_actions.shape}"
        )

    if transformed.get("prompt") != observation["prompt"]:
        errors.append("prompt was not preserved through ForceVLA input transform")

    if "right_wrist_0_rgb" in image and "base_0_rgb" in image:
        if image["right_wrist_0_rgb"].shape != image["base_0_rgb"].shape:
            errors.append("right_wrist_0_rgb zero-fill shape does not match base_0_rgb shape")
        if np.any(np.asarray(image["right_wrist_0_rgb"]) != 0):
            errors.append("right_wrist_0_rgb should be zero-filled for this two-camera dataset")

    # Observation.from_dict converts uint8 images to float32 [-1, 1].  It does
    # not require tokenized_prompt, so this is a safe pre-tokenization check.
    #
    # The local OpenPI dataclass type checks against JAX arrays, not NumPy
    # arrays. ForceVLA's adapter emits NumPy arrays, which are fine as an
    # intermediate representation, but before calling Observation.from_dict we
    # convert this validation copy to jax.numpy arrays.
    try:
        import jax.numpy as jnp
    except Exception as exc:
        raise RuntimeError(
            "jax is required for the final Observation.from_dict smoke check. "
            "Run this inside the validated lab ForceVLA environment."
        ) from exc

    model_observation_input = {
        "image": {key: jnp.asarray(value) for key, value in copy.deepcopy(image).items()},
        "image_mask": {key: jnp.asarray(value) for key, value in copy.deepcopy(image_mask).items()},
        "state": jnp.asarray(state, dtype=jnp.float32),
    }
    model_observation = _model.Observation.from_dict(model_observation_input)

    report = {
        "ok": not errors and bool(observation_report.get("ok", False)),
        "dataset_root": str(Path(dataset_root).resolve()),
        "forcevla_root": str(forcevla_root),
        "episode_index": episode_index,
        "row_index": row_index,
        "action_chunk_mode": action_chunk_mode,
        "model_config": {
            "class": type(model_config).__name__,
            "model_type": str(model_type),
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "max_token_len": int(model_config.max_token_len),
            "image_resolution": list(_model.IMAGE_RESOLUTION),
        },
        "source_observation": observation_report.get("forcevla_observation_summary", {}),
        "policy_input": {
            "image": _array_summary(policy_input["image"]),
            "wrist_image": _array_summary(policy_input["wrist_image"]),
            "state": _array_summary(policy_input["state"]),
            "actions": _array_summary(policy_input["actions"]),
            "action_chunk": action_chunk_report,
            "prompt": policy_input["prompt"],
        },
        "transformed": {
            "top_keys": sorted(transformed.keys()),
            "state": _array_summary(transformed.get("state")),
            "actions": _array_summary(transformed.get("actions")),
            "prompt": transformed.get("prompt"),
            "images": {key: _array_summary(value) for key, value in image.items()},
            "image_masks": {key: _safe_bool(value) for key, value in image_mask.items()},
        },
        "model_observation_from_dict": {
            "images": {key: _array_summary(value) for key, value in model_observation.images.items()},
            "image_masks": {key: _safe_bool(value) for key, value in model_observation.image_masks.items()},
            "state": _array_summary(model_observation.state),
            "tokenized_prompt_present": model_observation.tokenized_prompt is not None,
            "tokenized_prompt_mask_present": model_observation.tokenized_prompt_mask is not None,
        },
        "warnings": warnings,
        "errors": errors,
        "notes": [
            "This smoke test stops before tokenization and model inference.",
            "Use --action-chunk-mode future for real future action chunks; repeat/single are debug modes.",
            "Dummy videos may be 16x16 while real exports should be 480x640 before model resizing.",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test Doosan exported sample through local ForceVLA input transform."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("--forcevla-root", default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--action-chunk-mode", choices=["single", "repeat", "future"], default="future")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report = build_forcevla_transform_input_report(
        dataset_root=args.dataset_root,
        forcevla_root=args.forcevla_root,
        episode_index=args.episode_index,
        row_index=args.row_index,
        action_chunk_mode=args.action_chunk_mode,
    )

    text = json.dumps(report, indent=2)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote ForceVLA transform smoke report: {output_path}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
