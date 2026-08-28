"""One-command Patch-9 golden Episode-10 end-to-end acceptance workflow.

Run the public ``run`` subcommand with ROS 2 Jazzy's system Python.  The module
builds and validates the Patch-8 processed episode in that runtime, then launches
one isolated ForceVLA Python subprocess for LeRobot export, pinned-loader video
decoding, and the 25D/32D adapter acceptance checks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from doosan_forcevla_data.convert.doosan_processed_episode_v1 import (
    build_doosan_processed_episode,
)
from doosan_forcevla_data.convert.doosan_processed_to_lerobot_v21 import (
    export_doosan_processed_to_lerobot_v21,
)
from doosan_forcevla_data.validate.golden_episode10_v1 import (
    FROZEN_DLIMP_COMMIT,
    FROZEN_FORCEVLA_COMMIT,
    FROZEN_LEROBOT_COMMIT,
    GOLDEN_SCHEMA_ID,
    GoldenEpisode10Error,
    validate_golden_forcevla_runtime,
    validate_golden_lerobot_dataset,
    validate_golden_processed_episode,
    validate_golden_raw_episode,
    validate_golden_source_repositories,
    write_golden_report,
)


PROCESSED_DIRNAME = "processed_episode_000010"
DATASET_DIRNAME = "lerobot_doosan_episode10"
FORCEVLA_FRAGMENT_NAME = "forcevla_acceptance.json"
FINAL_REPORT_NAME = "golden_episode10_acceptance.json"
FORCEVLA_LOG_NAME = "forcevla_stage.log"


class GoldenPipelineError(RuntimeError):
    """Raised when Patch-9 orchestration cannot complete safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GoldenPipelineError(message)


def _source_pythonpath(converter_root: Path, forcevla_root: Path) -> str:
    paths = (
        converter_root / "src",
        forcevla_root / "src",
        forcevla_root / "lerobot",
        forcevla_root / "dlimp",
    )
    for path in paths:
        _require(path.is_dir(), f"required Python source path does not exist: {path}")
    return os.pathsep.join(str(path.resolve()) for path in paths)


def _forcevla_environment(converter_root: Path, forcevla_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = _source_pythonpath(converter_root, forcevla_root)
    return env


def _protected_path_conflict(output: Path, protected: Path) -> bool:
    return (
        output == protected
        or output.is_relative_to(protected)
        or protected.is_relative_to(output)
    )


def _prepare_output_root(
    output_root: str | Path,
    *,
    overwrite: bool,
    protected_paths: tuple[Path, ...],
) -> Path:
    candidate = Path(output_root).expanduser()
    if candidate.is_symlink():
        raise GoldenPipelineError(f"refusing symlink output root: {candidate}")
    output = candidate.resolve()
    _require(output != Path(output.anchor), "refusing to use filesystem root as golden output")

    for protected in protected_paths:
        resolved = protected.resolve()
        _require(
            not _protected_path_conflict(output, resolved),
            f"golden output must be separate from protected input/source path: {resolved}",
        )

    if output.exists():
        _require(output.is_dir(), f"output exists and is not a directory: {output}")
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)

    output.mkdir(parents=True)
    return output


def _run_forcevla_stage(
    *,
    forcevla_python: Path,
    converter_root: Path,
    forcevla_root: Path,
    processed_root: Path,
    dataset_root: Path,
    fragment_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    _require(forcevla_python.is_file(), f"ForceVLA Python does not exist: {forcevla_python}")
    _require(os.access(forcevla_python, os.X_OK), f"ForceVLA Python is not executable: {forcevla_python}")

    command = [
        str(forcevla_python),
        "-m",
        "doosan_forcevla_data.smoke.golden_episode10_end_to_end",
        "_forcevla-stage",
        "--processed",
        str(processed_root),
        "--dataset",
        str(dataset_root),
        "--fragment",
        str(fragment_path),
        "--converter-root",
        str(converter_root),
        "--forcevla-root",
        str(forcevla_root),
    ]

    completed = subprocess.run(
        command,
        env=_forcevla_environment(converter_root, forcevla_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")

    if completed.returncode != 0:
        raise GoldenPipelineError(
            f"ForceVLA golden stage failed with exit code {completed.returncode}; log={log_path}"
        )
    if not fragment_path.is_file():
        raise GoldenPipelineError(f"ForceVLA stage did not write acceptance fragment: {fragment_path}")

    try:
        payload = json.loads(fragment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenPipelineError(f"could not read ForceVLA acceptance fragment: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("accepted") is not True:
        raise GoldenPipelineError("ForceVLA acceptance fragment does not report accepted=true")
    return payload


def run_golden_episode10_pipeline(
    episode_dir: str | Path,
    output_root: str | Path,
    *,
    converter_root: str | Path,
    forcevla_root: str | Path,
    forcevla_python: str | Path,
    overwrite: bool = False,
) -> Path:
    """Run the complete golden Episode-10 conversion/export/runtime acceptance."""

    episode = Path(episode_dir).resolve()
    converter = Path(converter_root).resolve()
    forcevla = Path(forcevla_root).resolve()
    fv_python = Path(forcevla_python).expanduser().resolve()

    _require(converter.is_dir(), f"converter repository does not exist: {converter}")
    _require(forcevla.is_dir(), f"ForceVLA repository does not exist: {forcevla}")
    _require(episode.is_dir(), f"raw episode does not exist: {episode}")

    source_report = validate_golden_source_repositories(forcevla)
    raw_report = validate_golden_raw_episode(episode)

    output = _prepare_output_root(
        output_root,
        overwrite=overwrite,
        protected_paths=(episode, converter, forcevla),
    )
    processed = output / PROCESSED_DIRNAME
    dataset = output / DATASET_DIRNAME
    fragment = output / FORCEVLA_FRAGMENT_NAME
    forcevla_log = output / FORCEVLA_LOG_NAME

    try:
        build_doosan_processed_episode(episode, processed, overwrite=False)
        processed_report = validate_golden_processed_episode(processed)

        forcevla_report = _run_forcevla_stage(
            forcevla_python=fv_python,
            converter_root=converter,
            forcevla_root=forcevla,
            processed_root=processed,
            dataset_root=dataset,
            fragment_path=fragment,
            log_path=forcevla_log,
        )

        final_report = {
            "schema_version": GOLDEN_SCHEMA_ID,
            "accepted": True,
            "frozen_dependencies": {
                "forcevla_commit": FROZEN_FORCEVLA_COMMIT,
                "lerobot_commit": FROZEN_LEROBOT_COMMIT,
                "dlimp_commit": FROZEN_DLIMP_COMMIT,
            },
            "sources": source_report,
            "raw_episode": raw_report,
            "processed_episode": processed_report,
            "forcevla_stage": forcevla_report,
            "artifacts": {
                "output_root": str(output),
                "processed_episode": str(processed),
                "lerobot_dataset": str(dataset),
                "forcevla_stage_log": str(forcevla_log),
            },
        }
        report_path = write_golden_report(output / FINAL_REPORT_NAME, final_report)
    except Exception:
        # Preserve a failed output tree for diagnosis.  The public CLI never
        # silently deletes evidence from a failed multi-gigabyte conversion.
        raise

    return report_path


def _forcevla_stage(
    *,
    processed: Path,
    dataset: Path,
    fragment: Path,
    converter_root: Path,
    forcevla_root: Path,
) -> Path:
    source_report = validate_golden_source_repositories(forcevla_root)
    processed_report = validate_golden_processed_episode(processed)

    export_doosan_processed_to_lerobot_v21(
        processed,
        dataset,
        overwrite=False,
    )
    dataset_report = validate_golden_lerobot_dataset(processed, dataset)
    runtime_report = validate_golden_forcevla_runtime(
        dataset,
        converter_root=converter_root,
        forcevla_root=forcevla_root,
    )

    return write_golden_report(
        fragment,
        {
            "schema_version": GOLDEN_SCHEMA_ID,
            "accepted": True,
            "sources": source_report,
            "processed_episode": processed_report,
            "lerobot_dataset": dataset_report,
            "forcevla_runtime": runtime_report,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Patch-9 golden Episode-10 end-to-end production acceptance workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{run}")

    public = subparsers.add_parser(
        "run",
        help="build Episode 10 through Patch 8 and validate pinned LeRobot/ForceVLA runtime compatibility",
    )
    public.add_argument("episode_dir")
    public.add_argument("output_root")
    public.add_argument("--converter-root", required=True)
    public.add_argument("--forcevla-root", required=True)
    public.add_argument("--forcevla-python", required=True)
    public.add_argument("--overwrite", action="store_true")

    child = subparsers.add_parser("_forcevla-stage", add_help=False)
    child.add_argument("--processed", required=True)
    child.add_argument("--dataset", required=True)
    child.add_argument("--fragment", required=True)
    child.add_argument("--converter-root", required=True)
    child.add_argument("--forcevla-root", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            report = run_golden_episode10_pipeline(
                args.episode_dir,
                args.output_root,
                converter_root=args.converter_root,
                forcevla_root=args.forcevla_root,
                forcevla_python=args.forcevla_python,
                overwrite=args.overwrite,
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            artifacts = payload["artifacts"]
            print("GOLDEN_EPISODE10_ACCEPTANCE=PASS")
            print(f"report={report}")
            print(f"processed={artifacts['processed_episode']}")
            print(f"dataset={artifacts['lerobot_dataset']}")
            return 0

        if args.command == "_forcevla-stage":
            fragment = _forcevla_stage(
                processed=Path(args.processed).resolve(),
                dataset=Path(args.dataset).resolve(),
                fragment=Path(args.fragment).resolve(),
                converter_root=Path(args.converter_root).resolve(),
                forcevla_root=Path(args.forcevla_root).resolve(),
            )
            print("GOLDEN_EPISODE10_FORCEVLA_STAGE=PASS")
            print(f"fragment={fragment}")
            return 0

        raise GoldenPipelineError(f"unsupported command: {args.command}")  # pragma: no cover
    except (
        FileExistsError,
        GoldenEpisode10Error,
        GoldenPipelineError,
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"FAILED: {exc}")
        return 1


__all__ = [
    "DATASET_DIRNAME",
    "FINAL_REPORT_NAME",
    "FORCEVLA_FRAGMENT_NAME",
    "PROCESSED_DIRNAME",
    "GoldenPipelineError",
    "run_golden_episode10_pipeline",
]


if __name__ == "__main__":
    raise SystemExit(main())
