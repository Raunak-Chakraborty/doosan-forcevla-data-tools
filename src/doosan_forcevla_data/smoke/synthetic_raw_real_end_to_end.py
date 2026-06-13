"""Offline synthetic raw-real end-to-end smoke pipeline.

The smoke stays laptop-only and file-based. It generates a synthetic raw-real
episode, converts it to processed JSONL, stages the existing LeRobot/ForceVLA
dry-run export path, and writes a machine-readable report.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from doosan_forcevla_data.convert.plan_lerobot_export import (
    VALID_PROFILES,
    write_lerobot_export_plan,
)
from doosan_forcevla_data.convert.raw_real_to_processed import convert_raw_real_to_processed
from doosan_forcevla_data.convert.stage_lerobot_export import stage_lerobot_export
from doosan_forcevla_data.convert.write_lerobot_skeleton import write_lerobot_skeleton
from doosan_forcevla_data.convert.write_real_lerobot_export import write_real_lerobot_export
from doosan_forcevla_data.dummy.make_synthetic_raw_real_episode import make_synthetic_raw_real_episode
from doosan_forcevla_data.inspect.preflight_real_export import (
    preflight_real_export,
    write_preflight_report,
)
from doosan_forcevla_data.validate.validate_export_plan import validate_export_plan
from doosan_forcevla_data.validate.validate_lerobot_skeleton import validate_lerobot_skeleton
from doosan_forcevla_data.validate.validate_processed_episode import validate_processed_episode
from doosan_forcevla_data.validate.validate_raw_real_episode import validate_raw_real_episode
from doosan_forcevla_data.validate.validate_real_lerobot_export_attempt import (
    validate_real_lerobot_export_attempt,
)
from doosan_forcevla_data.validate.validate_staged_export import validate_staged_export


REPORT_NAME = "synthetic_raw_real_end_to_end_report.json"
DEFAULT_DATASET_NAME = "doosan_peg_in_hole_v0"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}: line {line_number} must be a JSON object")
        records.append(record)
    return records


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _contains_path(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() or output_root.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output_root already exists: {output_root}")
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError(f"output_root exists and is not a directory: {output_root}")
        if output_root.resolve() == Path.cwd().resolve() or _contains_path(output_root, Path.cwd()):
            raise ValueError(f"refusing to overwrite a directory that contains the repository: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)


def _validation_dict(result: Any) -> dict[str, Any]:
    data = {"ok": bool(result.ok), "errors": list(result.errors)}
    warnings = getattr(result, "warnings", None)
    if warnings is not None:
        data["warnings"] = list(warnings)
    return data


def _step(name: str, status: str, message: str, path: str | Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, "status": status, "message": message}
    if path is not None:
        record["path"] = str(path)
    return record


def _require_validation_ok(name: str, validation: Any) -> None:
    if validation.ok:
        return
    errors = "\n".join(f"ERROR: {error}" for error in validation.errors)
    raise ValueError(f"{name} failed:\n{errors}")


def _summarize_plan(plan_path: Path) -> dict[str, Any]:
    plan = _read_json(plan_path)
    return {
        "path": str(plan_path),
        "profile": plan.get("profile"),
        "input_frame_count": plan.get("input_frame_count"),
        "exported_frame_count": plan.get("exported_frame_count"),
        "excluded_terminal_padding_frame_count": plan.get("excluded_terminal_padding_frame_count"),
        "observation_state_dim": plan.get("observation_state_dim"),
        "action_dim": plan.get("action_dim"),
        "image_availability": plan.get("image_availability"),
    }


def _summarize_staged(staged_dir: Path) -> dict[str, Any]:
    metadata = _read_json(staged_dir / "metadata_staged.json")
    frames = _read_jsonl(staged_dir / "frames.jsonl")
    return {
        "path": str(staged_dir),
        "profile": metadata.get("profile"),
        "exported_frame_count": metadata.get("exported_frame_count"),
        "observation_state_dim": metadata.get("observation_state_dim"),
        "action_dim": metadata.get("action_dim"),
        "frame_record_count": len(frames),
    }


def _summarize_skeleton(skeleton_dir: Path) -> dict[str, Any]:
    info = _read_json(skeleton_dir / "meta" / "info.json")
    notes = info.get("notes") if isinstance(info.get("notes"), dict) else {}
    return {
        "path": str(skeleton_dir),
        "profile": info.get("export_profile"),
        "total_frames": info.get("total_frames"),
        "image_mode": notes.get("image_mode"),
        "skeleton_only": notes.get("skeleton_only"),
        "parquet_written": notes.get("parquet_written"),
        "videos_encoded": notes.get("videos_encoded"),
    }


def _summarize_preflight(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        "path": str(report_path),
        "skeleton_valid": report.get("skeleton_valid"),
        "schema_valid": report.get("schema_valid"),
        "profile": report.get("profile"),
        "total_frames": report.get("total_frames"),
        "state_dim": report.get("state_dim"),
        "action_dim": report.get("action_dim"),
        "image_staging_complete": report.get("image_staging_complete"),
        "parquet_ready": report.get("parquet_ready"),
        "video_ready": report.get("video_ready"),
        "real_export_ready": report.get("real_export_ready"),
        "video_export_ready": report.get("video_export_ready"),
        "warnings": report.get("warnings", []),
        "errors": report.get("errors", []),
    }


def _summarize_real_export_attempt(report_path: Path) -> dict[str, Any]:
    report = _read_json(report_path)
    return {
        "path": str(report_path),
        "output_dir": report.get("output_dir"),
        "mode": report.get("mode"),
        "profile": report.get("profile"),
        "total_frames": report.get("total_frames"),
        "state_dim": report.get("state_dim"),
        "action_dim": report.get("action_dim"),
        "parquet_ready": report.get("parquet_ready"),
        "video_ready": report.get("video_ready"),
        "parquet_written": report.get("parquet_written"),
        "videos_written": report.get("videos_written"),
        "metadata_written": report.get("metadata_written"),
        "skipped_reasons": report.get("skipped_reasons", []),
    }


def _report_path_for(output_root: Path, requested_path: str | Path | None) -> Path:
    if requested_path is None:
        return output_root / "reports" / REPORT_NAME
    requested = Path(requested_path)
    if not requested.is_absolute():
        requested = output_root / requested
    try:
        requested.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError("--json-report must resolve inside output_root") from exc
    return requested


def run_synthetic_raw_real_end_to_end_smoke(
    output_root: str | Path,
    episode_id: str = "episode_000000",
    frame_count: int = 8,
    fps: float = 30.0,
    include_optional_streams: bool = True,
    overwrite: bool = False,
    copy_images: bool = False,
    profile: str = "forcevla_13d",
    run_export_preflight: bool = True,
    json_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the offline synthetic raw-real smoke and return a JSON-serializable report."""

    if profile not in VALID_PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(sorted(VALID_PROFILES))}")
    if frame_count <= 1:
        raise ValueError("frame_count must be greater than 1")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not episode_id or Path(episode_id).name != episode_id or ".." in Path(episode_id).parts:
        raise ValueError("episode_id must be a safe single path component")

    root = Path(output_root)
    _prepare_output_root(root, overwrite=overwrite)

    raw_real_dir = root / "raw_real" / episode_id
    processed_dir = root / "processed" / episode_id
    reports_dir = root / "reports"
    plan_path = reports_dir / f"export_plan_{profile}.json"
    staged_dir = root / "staged" / profile / episode_id
    skeleton_dir = root / "lerobot" / profile / DEFAULT_DATASET_NAME
    preflight_report_path = reports_dir / "real_export_preflight_report.json"
    real_export_dir = root / "real_lerobot" / profile / DEFAULT_DATASET_NAME
    report_path = _report_path_for(root, json_report_path)

    steps: list[dict[str, Any]] = []
    skips: list[str] = []
    warnings: list[str] = []

    make_synthetic_raw_real_episode(
        raw_real_dir,
        episode_id=episode_id,
        frame_count=frame_count,
        fps=fps,
        include_optional_streams=include_optional_streams,
    )
    steps.append(_step("generate_synthetic_raw_real", "ok", "synthetic raw-real episode generated", raw_real_dir))

    raw_validation = validate_raw_real_episode(raw_real_dir)
    _require_validation_ok("raw-real validation", raw_validation)
    warnings.extend(raw_validation.warnings)
    steps.append(_step("validate_raw_real", "ok", "raw-real episode validated", raw_real_dir))

    convert_raw_real_to_processed(
        raw_real_dir,
        processed_dir,
        copy_images=copy_images,
        include_optional_debug=include_optional_streams,
    )
    steps.append(_step("convert_raw_real_to_processed", "ok", "processed episode written", processed_dir))

    processed_validation = validate_processed_episode(processed_dir)
    _require_validation_ok("processed validation", processed_validation)
    steps.append(_step("validate_processed", "ok", "processed episode validated", processed_dir))

    write_lerobot_export_plan(processed_dir, profile, plan_path)
    plan_validation = validate_export_plan(plan_path)
    _require_validation_ok("export plan validation", plan_validation)
    steps.append(_step("plan_lerobot_export", "ok", "dry-run LeRobot export plan written", plan_path))

    stage_lerobot_export(processed_dir, plan_path, staged_dir)
    staged_validation = validate_staged_export(staged_dir)
    _require_validation_ok("staged export validation", staged_validation)
    steps.append(_step("stage_lerobot_export", "ok", "LeRobot dry-run export staged", staged_dir))

    skeleton_validation = None
    export_preflight_summary = None
    real_export_attempt_validation = None
    real_export_dry_run_summary = None
    skeleton_summary = None

    if run_export_preflight:
        write_lerobot_skeleton(
            staged_dir,
            skeleton_dir,
            episode_index=0,
            task_index=0,
            profile=profile,
            image_mode="symlink",
        )
        skeleton_validation = validate_lerobot_skeleton(skeleton_dir)
        _require_validation_ok("LeRobot skeleton validation", skeleton_validation)
        skeleton_summary = _summarize_skeleton(skeleton_dir)
        steps.append(_step("write_lerobot_skeleton", "ok", "local LeRobot skeleton written", skeleton_dir))

        preflight_report = preflight_real_export(skeleton_dir)
        write_preflight_report(preflight_report, preflight_report_path)
        export_preflight_summary = _summarize_preflight(preflight_report, preflight_report_path)
        warnings.extend(str(warning) for warning in preflight_report.get("warnings", []))
        steps.append(_step("preflight_real_export", "ok", "dependency-safe preflight report written", preflight_report_path))

        real_export_report_path = write_real_lerobot_export(skeleton_dir, real_export_dir, mode="dry-run")
        real_export_attempt_validation = validate_real_lerobot_export_attempt(real_export_dir)
        _require_validation_ok("real export dry-run validation", real_export_attempt_validation)
        real_export_dry_run_summary = _summarize_real_export_attempt(real_export_report_path)
        steps.append(_step("real_export_dry_run", "ok", "dependency-optional real-export dry-run report written", real_export_report_path))
    else:
        message = "run_export_preflight=False"
        skips.append(message)
        steps.append(_step("write_lerobot_skeleton", "skipped", message, skeleton_dir))
        steps.append(_step("preflight_real_export", "skipped", message, preflight_report_path))
        steps.append(_step("real_export_dry_run", "skipped", message, real_export_dir))

    processed_metadata = _read_json(processed_dir / "metadata_processed.json")
    processed_frames = _read_jsonl(processed_dir / "frames.jsonl")

    report: dict[str, Any] = {
        "status": "ok",
        "output_root": str(root.resolve()),
        "episode_id": episode_id,
        "frame_count": frame_count,
        "fps": float(fps),
        "profile": profile,
        "include_optional_streams": include_optional_streams,
        "copy_images": copy_images,
        "run_export_preflight": run_export_preflight,
        "steps": steps,
        "paths": {
            "raw_real_episode": str(raw_real_dir),
            "processed_episode": str(processed_dir),
            "export_plan": str(plan_path),
            "staged_export": str(staged_dir),
            "lerobot_skeleton": str(skeleton_dir),
            "preflight_report": str(preflight_report_path),
            "real_export_dry_run": str(real_export_dir),
            "json_report": str(report_path),
        },
        "warnings": warnings,
        "skips": skips,
        "raw_real_validation": _validation_dict(raw_validation),
        "processed_validation": _validation_dict(processed_validation),
        "export_plan_validation": _validation_dict(plan_validation),
        "staged_validation": _validation_dict(staged_validation),
        "lerobot_skeleton_validation": _validation_dict(skeleton_validation) if skeleton_validation else None,
        "real_export_attempt_validation": (
            _validation_dict(real_export_attempt_validation) if real_export_attempt_validation else None
        ),
        "plan_summary": _summarize_plan(plan_path),
        "staging_summary": _summarize_staged(staged_dir),
        "skeleton_summary": skeleton_summary,
        "export_preflight_summary": export_preflight_summary,
        "real_export_dry_run_summary": real_export_dry_run_summary,
        "processed_summary": {
            "metadata_path": str(processed_dir / "metadata_processed.json"),
            "frames_path": str(processed_dir / "frames.jsonl"),
            "frame_count": processed_metadata.get("frame_count"),
            "action_label_primary": processed_metadata.get("action_label_primary"),
            "command_context_policy": processed_metadata.get("command_context_policy"),
            "processed_frame_records": len(processed_frames),
        },
    }

    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline synthetic raw-real end-to-end smoke pipeline.")
    parser.add_argument("--output-root", required=True, help="Smoke output root directory")
    parser.add_argument("--episode-id", default="episode_000000")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--profile", choices=sorted(VALID_PROFILES), default="forcevla_13d")
    optional_group = parser.add_mutually_exclusive_group()
    optional_group.add_argument("--include-optional-streams", dest="include_optional_streams", action="store_true")
    optional_group.add_argument("--no-optional-streams", dest="include_optional_streams", action="store_false")
    parser.set_defaults(include_optional_streams=True)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output root")
    parser.add_argument("--no-export-preflight", dest="run_export_preflight", action="store_false")
    parser.set_defaults(run_export_preflight=True)
    parser.add_argument("--json-report", help="Optional report path, resolved inside output root if relative")
    args = parser.parse_args(argv)

    try:
        report = run_synthetic_raw_real_end_to_end_smoke(
            args.output_root,
            episode_id=args.episode_id,
            frame_count=args.frames,
            fps=args.fps,
            include_optional_streams=args.include_optional_streams,
            overwrite=args.overwrite,
            copy_images=args.copy_images,
            profile=args.profile,
            run_export_preflight=args.run_export_preflight,
            json_report_path=args.json_report,
        )
    except (OSError, ValueError) as exc:
        print(f"FAILED: synthetic raw-real end-to-end smoke failed: {args.output_root}")
        print(str(exc))
        return 1

    print("OK: synthetic raw-real end-to-end smoke passed")
    print(f"output_root: {report['output_root']}")
    print(f"json_report: {report['paths']['json_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
