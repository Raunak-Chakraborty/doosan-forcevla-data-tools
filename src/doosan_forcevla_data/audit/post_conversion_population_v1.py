"""Read-only post-conversion population audit for Doosan LeRobot exports.

The audit is intentionally independent from the production exporter.  It reads
already-written LeRobot v2.1 datasets and checks that all eight thesis model
state profiles describe the same physical episode while preserving the exact
7D action, indexing, task, camera videos, and non-orientation state channels.

Input layout
------------
A *bundle* is a directory containing exactly one dataset directory for every
orientation/state-mode profile::

    bundle/
      rotvec_principal__no_wrench/
      rotvec_principal__full/
      rotvec_continuous__no_wrench/
      rotvec_continuous__full/
      quaternion__no_wrench/
      quaternion__full/
      rotation6d__no_wrench/
      rotation6d__full/

For a population, pass a parent whose direct children are bundles.  The audit
never modifies the exports it reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from doosan_forcevla_data.convert.model_state_profile_v1 import (
    ModelStateLayout,
    StateMode,
    model_state_layout,
)
from doosan_forcevla_data.convert.orientation_representation_v1 import (
    OrientationRepresentation,
    decode_orientation,
)
from doosan_forcevla_data.validate.validate_doosan_lerobot_v21 import (
    validate_doosan_lerobot_v21,
)


AUDIT_SCHEMA = "doosan_post_conversion_population_audit_v1"
ORIENTATION_ATOL = 1e-9


@dataclass(frozen=True)
class ProfileSpec:
    directory_name: str
    layout: ModelStateLayout


PROFILE_SPECS: tuple[ProfileSpec, ...] = tuple(
    ProfileSpec(
        f"{representation.value}__{mode.value}",
        model_state_layout(representation, mode),
    )
    for representation in OrientationRepresentation
    for mode in (StateMode.NO_WRENCH, StateMode.FULL)
)
PROFILE_BY_NAME = {spec.directory_name: spec for spec in PROFILE_SPECS}
EXPECTED_PROFILE_NAMES = tuple(spec.directory_name for spec in PROFILE_SPECS)
REFERENCE_PROFILE = "rotvec_principal__full"
VIDEO_KEYS = (
    "observation.images.external_camera_2",
    "observation.images.tcp_camera",
)
SCALAR_COLUMNS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


class PostConversionAuditError(ValueError):
    """Raised when audit input cannot be interpreted unambiguously."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostConversionAuditError(f"{path}: could not read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PostConversionAuditError(f"{path}: expected JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PostConversionAuditError(f"{path}: could not read JSONL: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise PostConversionAuditError(f"{path}: empty line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PostConversionAuditError(
                f"{path}: invalid JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise PostConversionAuditError(
                f"{path}: line {line_number} is not an object"
            )
        rows.append(value)
    return rows


def _import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyArrow is required for post-conversion population audit; "
            "run inside data_conv_thesis_env."
        ) from exc
    return pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_matrix_error(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> float:
    return max(
        abs(float(left[row][col]) - float(right[row][col]))
        for row in range(3)
        for col in range(3)
    )


def _finite_vector(value: Any, expected: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != expected:
        raise PostConversionAuditError(
            f"{context}: expected list length {expected}, got "
            f"{len(value) if isinstance(value, list) else type(value).__name__}"
        )
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PostConversionAuditError(
                f"{context}[{index}]: expected finite numeric scalar"
            )
        converted = float(item)
        if not math.isfinite(converted):
            raise PostConversionAuditError(f"{context}[{index}]: non-finite value")
        result.append(converted)
    return tuple(result)


def _profile_metadata_matches(
    provenance: dict[str, Any],
    spec: ProfileSpec,
) -> bool:
    declared = provenance.get("model_state_profile")
    return declared == spec.layout.to_metadata()


def _video_paths(dataset_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key in VIDEO_KEYS:
        path = (
            dataset_root
            / "videos"
            / "chunk-000"
            / key
            / "episode_000000.mp4"
        )
        if not path.is_file():
            raise PostConversionAuditError(f"missing expected video: {path}")
        result[key] = path
    return result


def _load_profile(dataset_root: Path, spec: ProfileSpec) -> dict[str, Any]:
    validation = validate_doosan_lerobot_v21(dataset_root)
    if not validation.ok:
        raise PostConversionAuditError(
            f"{spec.directory_name}: dataset validator failed: "
            + " | ".join(validation.errors)
        )

    info = _read_json(dataset_root / "meta" / "info.json")
    provenance = _read_json(dataset_root / "meta" / "export_provenance.json")
    tasks = _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")

    if not _profile_metadata_matches(provenance, spec):
        raise PostConversionAuditError(
            f"{spec.directory_name}: model_state_profile provenance does not "
            "match directory profile"
        )
    if provenance.get("state_dim") != spec.layout.state_dim:
        raise PostConversionAuditError(
            f"{spec.directory_name}: provenance state_dim mismatch"
        )

    features = info.get("features")
    if not isinstance(features, dict):
        raise PostConversionAuditError(f"{spec.directory_name}: info.features missing")
    expected_state_feature = {
        "dtype": "float64",
        "shape": [spec.layout.state_dim],
        "names": list(spec.layout.state_fields),
    }
    if features.get("observation.state") != expected_state_feature:
        raise PostConversionAuditError(
            f"{spec.directory_name}: observation.state feature metadata mismatch"
        )

    parquet_files = sorted(dataset_root.glob("data/**/*.parquet"))
    if len(parquet_files) != 1:
        raise PostConversionAuditError(
            f"{spec.directory_name}: expected exactly one parquet file, "
            f"found {len(parquet_files)}"
        )
    pq = _import_pyarrow_parquet()
    table = pq.read_table(parquet_files[0])
    rows = table.to_pylist()
    if not rows:
        raise PostConversionAuditError(f"{spec.directory_name}: zero parquet rows")

    states: list[tuple[float, ...]] = []
    actions: list[tuple[float, ...]] = []
    scalars = {name: [] for name in SCALAR_COLUMNS}
    for row_index, row in enumerate(rows):
        states.append(
            _finite_vector(
                row.get("observation.state"),
                spec.layout.state_dim,
                f"{spec.directory_name} row {row_index} state",
            )
        )
        actions.append(
            _finite_vector(
                row.get("action"),
                7,
                f"{spec.directory_name} row {row_index} action",
            )
        )
        for name in SCALAR_COLUMNS:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PostConversionAuditError(
                    f"{spec.directory_name} row {row_index}: invalid scalar {name}"
                )
            converted = float(value)
            if not math.isfinite(converted):
                raise PostConversionAuditError(
                    f"{spec.directory_name} row {row_index}: non-finite {name}"
                )
            scalars[name].append(value)

    videos = _video_paths(dataset_root)
    video_sha256 = {key: _sha256(path) for key, path in videos.items()}

    return {
        "root": str(dataset_root),
        "info": info,
        "provenance": provenance,
        "tasks": tasks,
        "episodes": episodes,
        "row_count": len(rows),
        "states": states,
        "actions": actions,
        "scalars": scalars,
        "video_sha256": video_sha256,
    }


def _slice_tuple(values: Sequence[float], section: slice) -> tuple[float, ...]:
    return tuple(float(item) for item in values[section])


def _nonorientation_signature(
    state: Sequence[float],
    layout: ModelStateLayout,
) -> tuple[tuple[float, ...], float, tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(float(item) for item in state[0:3]),
        float(state[layout.gripper_index]),
        _slice_tuple(state, layout.joint_position_slice),
        _slice_tuple(state, layout.joint_velocity_slice),
    )


def _orientation_matrix(
    state: Sequence[float],
    layout: ModelStateLayout,
) -> tuple[tuple[float, float, float], ...]:
    encoded = _slice_tuple(state, layout.orientation_slice)
    return decode_orientation(encoded, layout.orientation_representation)


def _stable_cross_profile_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    excluded = {"state_dim", "model_state_profile"}
    return {key: value for key, value in provenance.items() if key not in excluded}


def _audit_bundle(bundle_root: Path) -> dict[str, Any]:
    missing = [name for name in EXPECTED_PROFILE_NAMES if not (bundle_root / name).is_dir()]
    unexpected = sorted(
        path.name
        for path in bundle_root.iterdir()
        if path.is_dir() and "__" in path.name and path.name not in PROFILE_BY_NAME
    ) if bundle_root.is_dir() else []
    if missing:
        raise PostConversionAuditError(
            f"{bundle_root}: missing profiles: {', '.join(missing)}"
        )
    if unexpected:
        raise PostConversionAuditError(
            f"{bundle_root}: unexpected profile directories: {', '.join(unexpected)}"
        )

    loaded = {
        spec.directory_name: _load_profile(bundle_root / spec.directory_name, spec)
        for spec in PROFILE_SPECS
    }
    reference = loaded[REFERENCE_PROFILE]
    reference_spec = PROFILE_BY_NAME[REFERENCE_PROFILE]

    source_processed = reference["provenance"].get("source_processed_episode")
    source_raw = reference["provenance"].get("source_raw_episode")
    source_episode_index = reference["provenance"].get("source_episode_index")
    task = reference["provenance"].get("task")
    row_count = reference["row_count"]
    max_orientation_error = 0.0

    reference_provenance = _stable_cross_profile_provenance(reference["provenance"])
    for profile_name, payload in loaded.items():
        spec = PROFILE_BY_NAME[profile_name]
        if payload["row_count"] != row_count:
            raise PostConversionAuditError(
                f"{bundle_root}: {profile_name} row count {payload['row_count']} "
                f"!= {row_count}"
            )
        if payload["actions"] != reference["actions"]:
            raise PostConversionAuditError(
                f"{bundle_root}: exact action mismatch in {profile_name}"
            )
        for column in SCALAR_COLUMNS:
            if payload["scalars"][column] != reference["scalars"][column]:
                raise PostConversionAuditError(
                    f"{bundle_root}: exact {column} mismatch in {profile_name}"
                )
        if payload["tasks"] != reference["tasks"]:
            raise PostConversionAuditError(
                f"{bundle_root}: tasks.jsonl mismatch in {profile_name}"
            )
        if payload["episodes"] != reference["episodes"]:
            raise PostConversionAuditError(
                f"{bundle_root}: episodes.jsonl mismatch in {profile_name}"
            )
        if _stable_cross_profile_provenance(payload["provenance"]) != reference_provenance:
            raise PostConversionAuditError(
                f"{bundle_root}: cross-profile provenance mismatch in {profile_name}"
            )
        if payload["video_sha256"] != reference["video_sha256"]:
            raise PostConversionAuditError(
                f"{bundle_root}: physical video bytes differ in {profile_name}"
            )

        for row_index, (state, reference_state) in enumerate(
            zip(payload["states"], reference["states"], strict=True)
        ):
            if _nonorientation_signature(state, spec.layout) != _nonorientation_signature(
                reference_state, reference_spec.layout
            ):
                raise PostConversionAuditError(
                    f"{bundle_root}: non-orientation state mismatch in "
                    f"{profile_name} row {row_index}"
                )
            matrix = _orientation_matrix(state, spec.layout)
            reference_matrix = _orientation_matrix(
                reference_state, reference_spec.layout
            )
            error = _max_matrix_error(matrix, reference_matrix)
            max_orientation_error = max(max_orientation_error, error)
            if error > ORIENTATION_ATOL:
                raise PostConversionAuditError(
                    f"{bundle_root}: physical orientation mismatch in "
                    f"{profile_name} row {row_index}: max_error={error:.3e}"
                )

    full_profiles = [
        spec.directory_name
        for spec in PROFILE_SPECS
        if spec.layout.state_mode is StateMode.FULL
    ]
    reference_wrench = [
        _slice_tuple(state, reference_spec.layout.wrench_slice)  # type: ignore[arg-type]
        for state in reference["states"]
    ]
    for profile_name in full_profiles:
        spec = PROFILE_BY_NAME[profile_name]
        assert spec.layout.wrench_slice is not None
        wrench = [
            _slice_tuple(state, spec.layout.wrench_slice)
            for state in loaded[profile_name]["states"]
        ]
        if wrench != reference_wrench:
            raise PostConversionAuditError(
                f"{bundle_root}: exact wrench mismatch in {profile_name}"
            )

    return {
        "bundle_root": str(bundle_root.resolve()),
        "source_processed_episode": source_processed,
        "source_raw_episode": source_raw,
        "source_episode_index": source_episode_index,
        "task": task,
        "row_count": row_count,
        "profile_count": len(loaded),
        "max_orientation_matrix_abs_error": max_orientation_error,
        "video_sha256": reference["video_sha256"],
        "profiles": {
            name: {
                "state_dim": PROFILE_BY_NAME[name].layout.state_dim,
                "row_count": loaded[name]["row_count"],
                "dataset_root": loaded[name]["root"],
            }
            for name in EXPECTED_PROFILE_NAMES
        },
    }


def discover_bundle_roots(input_root: str | Path) -> list[Path]:
    root = Path(input_root)
    if not root.is_dir():
        raise PostConversionAuditError(f"input root is not a directory: {root}")
    if all((root / name).is_dir() for name in EXPECTED_PROFILE_NAMES):
        return [root]
    bundles = sorted(
        child
        for child in root.iterdir()
        if child.is_dir()
        and all((child / name).is_dir() for name in EXPECTED_PROFILE_NAMES)
    )
    if not bundles:
        raise PostConversionAuditError(
            f"{root}: found neither an 8-profile bundle nor direct child bundles"
        )
    return bundles


def _read_expected_sources(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PostConversionAuditError(
            f"could not read expected sources file {path}: {exc}"
        ) from exc
    values = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if len(values) != len(set(values)):
        raise PostConversionAuditError("expected sources file contains duplicates")
    return values


def audit_post_conversion_population(
    input_root: str | Path,
    *,
    expected_bundles: int | None = None,
    expected_sources_file: str | Path | None = None,
) -> dict[str, Any]:
    bundles = discover_bundle_roots(input_root)
    failures: list[dict[str, str]] = []
    successes: list[dict[str, Any]] = []

    for bundle in bundles:
        try:
            successes.append(_audit_bundle(bundle))
        except (PostConversionAuditError, OSError, ValueError, RuntimeError) as exc:
            failures.append({"bundle_root": str(bundle), "error": str(exc)})

    population_issues: list[str] = []
    if expected_bundles is not None and len(bundles) != expected_bundles:
        population_issues.append(
            f"expected {expected_bundles} bundles, discovered {len(bundles)}"
        )

    successful_sources = [item.get("source_processed_episode") for item in successes]
    nonempty_sources = [value for value in successful_sources if isinstance(value, str) and value]
    if len(nonempty_sources) != len(set(nonempty_sources)):
        population_issues.append("duplicate source_processed_episode values across bundles")

    if expected_sources_file is not None:
        expected_sources = _read_expected_sources(Path(expected_sources_file))
        actual_sources = sorted(nonempty_sources)
        if sorted(expected_sources) != actual_sources:
            missing = sorted(set(expected_sources) - set(actual_sources))
            unexpected = sorted(set(actual_sources) - set(expected_sources))
            population_issues.append(
                "source population mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    gate = "PASS" if not failures and not population_issues else "FAIL"
    return {
        "schema_version": AUDIT_SCHEMA,
        "input_root": str(Path(input_root).resolve()),
        "profile_names": list(EXPECTED_PROFILE_NAMES),
        "expected_profile_count": len(EXPECTED_PROFILE_NAMES),
        "discovered_bundle_count": len(bundles),
        "successful_bundle_count": len(successes),
        "failed_bundle_count": len(failures),
        "total_rows_per_profile": sum(item["row_count"] for item in successes),
        "max_orientation_matrix_abs_error": max(
            (item["max_orientation_matrix_abs_error"] for item in successes),
            default=0.0,
        ),
        "population_issues": population_issues,
        "failures": failures,
        "bundles": successes,
        "audit_gate": gate,
    }


def write_audit_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "post_conversion_audit.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    failures = report.get("failures", [])
    failure_lines = [
        f"{item.get('bundle_root')}\t{item.get('error')}"
        for item in failures
    ]
    (output / "failures.txt").write_text(
        ("\n".join(failure_lines) + "\n") if failure_lines else "",
        encoding="utf-8",
    )

    lines = [
        "============================================================",
        "DOOSAN POST-CONVERSION POPULATION AUDIT",
        "============================================================",
        f"input_root={report['input_root']}",
        f"discovered_bundle_count={report['discovered_bundle_count']}",
        f"successful_bundle_count={report['successful_bundle_count']}",
        f"failed_bundle_count={report['failed_bundle_count']}",
        f"total_rows_per_profile={report['total_rows_per_profile']}",
        f"max_orientation_matrix_abs_error={report['max_orientation_matrix_abs_error']:.3e}",
        f"population_issue_count={len(report['population_issues'])}",
        "",
        f"POSTCONVERSION_AUDIT_GATE={report['audit_gate']}",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one or more complete 8-profile Doosan LeRobot v2.1 "
            "post-conversion bundles without modifying them."
        )
    )
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-bundles", type=int, default=None)
    parser.add_argument(
        "--expected-sources-file",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited exact source_processed_episode values; "
            "use this after the accepted population is frozen."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = audit_post_conversion_population(
            args.input_root,
            expected_bundles=args.expected_bundles,
            expected_sources_file=args.expected_sources_file,
        )
        write_audit_report(report, args.output_dir)
    except (PostConversionAuditError, OSError, ValueError, RuntimeError) as exc:
        print(f"POSTCONVERSION_AUDIT_SETUP=FAIL: {exc}")
        return 2

    print(f"bundle_count={report['discovered_bundle_count']}")
    print(f"successful_bundle_count={report['successful_bundle_count']}")
    print(f"failed_bundle_count={report['failed_bundle_count']}")
    print(f"total_rows_per_profile={report['total_rows_per_profile']}")
    print(
        "max_orientation_matrix_abs_error="
        f"{report['max_orientation_matrix_abs_error']:.3e}"
    )
    print(f"POSTCONVERSION_AUDIT_GATE={report['audit_gate']}")
    return 0 if report["audit_gate"] == "PASS" else 1


__all__ = [
    "AUDIT_SCHEMA",
    "EXPECTED_PROFILE_NAMES",
    "ORIENTATION_ATOL",
    "PostConversionAuditError",
    "audit_post_conversion_population",
    "discover_bundle_roots",
    "write_audit_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
