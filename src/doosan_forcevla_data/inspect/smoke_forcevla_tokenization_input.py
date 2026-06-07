"""Smoke-test ForceVLA model transforms up to prompt tokenization.

This validates:

    local LeRobot-style parquet/videos
    -> PyAV/parquet observation builder
    -> ForceVLA input adapter
    -> OpenPI ModelTransformFactory inputs
    -> resized images + tokenized_prompt/tokenized_prompt_mask
    -> Observation.from_dict

No model checkpoint is loaded and no inference is run.
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

from doosan_forcevla_data.inspect.smoke_forcevla_transform_input import (
    _array_summary,
    _default_forcevla_root,
    _require_forcevla_modules,
    build_forcevla_transform_input_report,
)


def _add_forcevla_to_path(forcevla_root: str | Path) -> Path:
    root = Path(forcevla_root).expanduser().resolve()
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"ForceVLA src directory not found: {src}")
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return root


def _safe_bool(value: Any) -> bool | str:
    try:
        return bool(value)
    except Exception:
        return str(value)


def _jnp_tree_summary(tree: dict[str, Any]) -> dict[str, Any]:
    return {key: _array_summary(value) for key, value in tree.items()}


def build_forcevla_tokenization_input_report(
    dataset_root: str | Path,
    forcevla_root: str | Path | None = None,
    episode_index: int = 0,
    row_index: int = 0,
    action_chunk_mode: str = "repeat",
) -> dict[str, Any]:
    forcevla_root = Path(forcevla_root or os.environ.get("FORCEVLA_ROOT") or _default_forcevla_root())
    forcevla_root, _model, pi0_force, forcevla_policy = _require_forcevla_modules(forcevla_root)
    _add_forcevla_to_path(forcevla_root)

    try:
        import jax.numpy as jnp
        from openpi.training import config as _config
    except Exception as exc:
        raise RuntimeError(
            "jax and local openpi.training.config are required for tokenization smoke testing. "
            "Run this inside the validated lab ForceVLA environment."
        ) from exc

    transform_report = build_forcevla_transform_input_report(
        dataset_root=dataset_root,
        forcevla_root=forcevla_root,
        episode_index=episode_index,
        row_index=row_index,
        action_chunk_mode=action_chunk_mode,
    )

    errors: list[str] = []
    warnings: list[str] = list(transform_report.get("warnings", []))

    model_config = pi0_force.Pi0_GuidanceConfig(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )

    # Rebuild the same pre-tokenization ForceVLA input, but keep the actual
    # nested dict so ModelTransformFactory can operate on it.
    from doosan_forcevla_data.inspect.smoke_forcevla_observation_builder import build_smoke_observation

    observation, _ = build_smoke_observation(
        dataset_root=dataset_root,
        episode_index=episode_index,
        row_index=row_index,
        expected_state_dim=13,
        expected_action_dim=7,
        strict_video_shape=False,
    )

    raw_action = np.asarray(observation["action"], dtype=np.float32)
    if action_chunk_mode == "repeat":
        actions = np.repeat(raw_action[None, :], int(model_config.action_horizon), axis=0)
    else:
        actions = raw_action

    policy_input = {
        "image": np.asarray(observation["observation.image"]),
        "wrist_image": np.asarray(observation["observation.wrist_image"]),
        "state": np.asarray(observation["observation.state"], dtype=np.float32),
        "actions": actions,
        "prompt": observation["prompt"],
    }

    forcevla_input_transform = forcevla_policy.Forcevla_inputs(
        action_dim=int(model_config.action_dim),
        model_type=model_config.model_type,
    )
    pre_model = forcevla_input_transform(policy_input)

    model_transforms = _config.ModelTransformFactory()(model_config)

    # In this local ForceVLA/OpenPI version, ModelTransformFactory returns a
    # transforms.Group object, but Group itself is not directly callable.
    # Apply its input transforms in order: InjectDefaultPrompt -> ResizeImages
    # -> TokenizePrompt.
    post_model = pre_model
    for transform in getattr(model_transforms, "inputs", []):
        post_model = transform(post_model)

    required_post_keys = [
        "image",
        "image_mask",
        "state",
        "actions",
        "tokenized_prompt",
        "tokenized_prompt_mask",
    ]
    for key in required_post_keys:
        if key not in post_model:
            errors.append(f"post-model-transform output missing key: {key}")

    if "prompt" in post_model:
        errors.append("prompt should be consumed by TokenizePrompt and removed from post_model output")

    tokenized_prompt = np.asarray(post_model.get("tokenized_prompt"))
    tokenized_prompt_mask = np.asarray(post_model.get("tokenized_prompt_mask"))

    if tokenized_prompt.shape != (int(model_config.max_token_len),):
        errors.append(
            f"tokenized_prompt shape should be ({model_config.max_token_len},), got {tokenized_prompt.shape}"
        )
    if tokenized_prompt_mask.shape != (int(model_config.max_token_len),):
        errors.append(
            "tokenized_prompt_mask shape should be "
            f"({model_config.max_token_len},), got {tokenized_prompt_mask.shape}"
        )
    if tokenized_prompt_mask.dtype != np.bool_:
        errors.append(f"tokenized_prompt_mask dtype should be bool, got {tokenized_prompt_mask.dtype}")
    if tokenized_prompt.size and not np.issubdtype(tokenized_prompt.dtype, np.integer):
        errors.append(f"tokenized_prompt dtype should be integer, got {tokenized_prompt.dtype}")

    image = post_model.get("image", {})
    if isinstance(image, dict):
        for key, value in image.items():
            arr = np.asarray(value)
            if list(arr.shape[-3:]) != [224, 224, 3]:
                errors.append(f"{key} should be resized to 224x224x3, got {arr.shape}")
    else:
        errors.append("post_model image must be a dict")

    # Convert to JAX arrays before OpenPI's typed Observation check.
    observation_input = {
        "image": {key: jnp.asarray(value) for key, value in copy.deepcopy(post_model["image"]).items()},
        "image_mask": {key: jnp.asarray(value) for key, value in copy.deepcopy(post_model["image_mask"]).items()},
        "state": jnp.asarray(post_model["state"], dtype=jnp.float32),
        "tokenized_prompt": jnp.asarray(post_model["tokenized_prompt"], dtype=jnp.int32),
        "tokenized_prompt_mask": jnp.asarray(post_model["tokenized_prompt_mask"]),
    }
    model_observation = _model.Observation.from_dict(observation_input)

    report = {
        "ok": not errors and bool(transform_report.get("ok", False)),
        "dataset_root": str(Path(dataset_root).resolve()),
        "forcevla_root": str(forcevla_root),
        "episode_index": episode_index,
        "row_index": row_index,
        "action_chunk_mode": action_chunk_mode,
        "model_config": {
            "class": type(model_config).__name__,
            "model_type": str(model_config.model_type),
            "action_dim": int(model_config.action_dim),
            "action_horizon": int(model_config.action_horizon),
            "max_token_len": int(model_config.max_token_len),
            "image_resolution": list(_model.IMAGE_RESOLUTION),
        },
        "pre_model_transform": {
            "top_keys": sorted(pre_model.keys()),
            "images": _jnp_tree_summary(pre_model["image"]),
            "image_masks": {key: _safe_bool(value) for key, value in pre_model["image_mask"].items()},
            "state": _array_summary(pre_model["state"]),
            "actions": _array_summary(pre_model["actions"]),
            "prompt": pre_model.get("prompt"),
        },
        "post_model_transform": {
            "top_keys": sorted(post_model.keys()),
            "images": _jnp_tree_summary(post_model["image"]),
            "image_masks": {key: _safe_bool(value) for key, value in post_model["image_mask"].items()},
            "state": _array_summary(post_model["state"]),
            "actions": _array_summary(post_model["actions"]),
            "tokenized_prompt": _array_summary(post_model["tokenized_prompt"]),
            "tokenized_prompt_mask": _array_summary(post_model["tokenized_prompt_mask"]),
            "tokenized_prompt_mask_true_count": int(np.asarray(post_model["tokenized_prompt_mask"]).sum()),
            "first_12_tokens": np.asarray(post_model["tokenized_prompt"])[:12].astype(int).tolist(),
        },
        "model_observation_from_dict": {
            "images": {key: _array_summary(value) for key, value in model_observation.images.items()},
            "image_masks": {key: _safe_bool(value) for key, value in model_observation.image_masks.items()},
            "state": _array_summary(model_observation.state),
            "tokenized_prompt": _array_summary(model_observation.tokenized_prompt),
            "tokenized_prompt_mask": _array_summary(model_observation.tokenized_prompt_mask),
            "tokenized_prompt_present": model_observation.tokenized_prompt is not None,
            "tokenized_prompt_mask_present": model_observation.tokenized_prompt_mask is not None,
        },
        "upstream_transform_report_ok": bool(transform_report.get("ok", False)),
        "warnings": warnings,
        "errors": errors,
        "notes": [
            "This smoke test stops after model transforms and Observation.from_dict.",
            "No checkpoint is loaded and no model inference is run.",
            "Repeated actions are only for shape validation; real training needs real action chunks.",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test Doosan exported sample through ForceVLA model transforms/tokenization."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("--forcevla-root", default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--action-chunk-mode", choices=["single", "repeat"], default="repeat")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report = build_forcevla_tokenization_input_report(
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
        print(f"Wrote ForceVLA tokenization smoke report: {output_path}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
