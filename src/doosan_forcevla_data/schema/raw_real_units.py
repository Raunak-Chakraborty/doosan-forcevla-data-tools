"""Unit-resolution helpers for raw_real_v0 TCP pose fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


METER_UNITS = {"m", "meter", "meters", "metre", "metres"}
MILLIMETER_UNITS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}
DEGREE_UNITS = {"deg", "degree", "degrees"}
RADIAN_UNITS = {"rad", "radian", "radians"}

POSITION_UNIT_KEYS = [
    "tcp_position",
    "tcp_position_unit",
    "tcp_position_units",
    "tcp_translation_unit",
    "tcp_translation_units",
    "actual_tcp_position_unit",
    "actual_tcp_position_translation_unit",
]
ORIENTATION_UNIT_KEYS = [
    "tcp_orientation",
    "tcp_orientation_unit",
    "tcp_orientation_units",
    "actual_tcp_orientation_unit",
    "actual_tcp_position_orientation_unit",
]


@dataclass(frozen=True)
class UnitDeclaration:
    """One explicit raw unit declaration after normalization."""

    source: str
    raw_unit: str
    unit: str

    def as_metadata(self) -> dict[str, str]:
        return {"source": self.source, "raw_unit": self.raw_unit, "unit": self.unit}


@dataclass(frozen=True)
class UnitResolution:
    """Resolved unit for one semantic TCP field."""

    unit: str | None
    raw_unit: str | None
    source: str | None
    declarations: tuple[UnitDeclaration, ...]
    legacy_fallback: bool
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors and self.unit is not None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "raw_unit": self.raw_unit,
            "source": self.source,
            "declarations": [declaration.as_metadata() for declaration in self.declarations],
            "legacy_fallback": self.legacy_fallback,
        }


def normalize_unit(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace(" ", "_")


def canonical_tcp_position_unit(value: Any) -> str | None:
    unit = normalize_unit(value)
    if unit in METER_UNITS:
        return "m"
    if unit in MILLIMETER_UNITS:
        return "mm"
    return None


def canonical_tcp_orientation_unit(value: Any) -> str | None:
    unit = normalize_unit(value)
    if unit in RADIAN_UNITS:
        return "rad"
    if unit in DEGREE_UNITS:
        return "deg"
    return None


def _container_unit_candidates(
    container: dict[str, Any] | None,
    container_source: str,
    keys: list[str],
) -> list[tuple[str, Any]]:
    if not isinstance(container, dict):
        return []
    candidates: list[tuple[str, Any]] = []

    units = container.get("units")
    if isinstance(units, dict):
        for key in keys:
            if key in units:
                candidates.append((f"{container_source}.units.{key}", units[key]))

    for key in keys:
        if key in container:
            candidates.append((f"{container_source}.{key}", container[key]))
    return candidates


def _actual_tcp_position_units_candidate(
    container: dict[str, Any] | None,
    container_source: str,
    kind: str,
) -> tuple[str, Any] | None:
    if not isinstance(container, dict) or "actual_tcp_position_units" not in container:
        return None
    value = container["actual_tcp_position_units"]
    if isinstance(value, dict):
        is_position = kind in {"position", "tcp_position"}
        keys = ["position", "translation", "tcp_position"] if is_position else ["orientation", "tcp_orientation"]
        for key in keys:
            if key in value:
                return (f"{container_source}.actual_tcp_position_units.{key}", value[key])
        return None
    if isinstance(value, list):
        if kind in {"position", "tcp_position"}:
            return (f"{container_source}.actual_tcp_position_units[0:3]", value[:3])
        return (f"{container_source}.actual_tcp_position_units[3:6]", value[3:6])
    return (f"{container_source}.actual_tcp_position_units", value)


def _add_candidate_declarations(
    declarations: list[UnitDeclaration],
    errors: list[str],
    *,
    source: str,
    value: Any,
    canonicalizer: Callable[[Any], str | None],
    kind: str,
    context: str,
) -> None:
    values = value if isinstance(value, list) else [value]
    if not values:
        return

    candidate_declarations: list[UnitDeclaration] = []
    for item in values:
        normalized = normalize_unit(item)
        if normalized is None:
            errors.append(f"{context}: unsupported or missing {kind} unit: {item!r} at {source}")
            continue
        canonical = canonicalizer(normalized)
        if canonical is None:
            errors.append(f"{context}: unsupported or missing {kind} unit: {item!r} at {source}")
            continue
        candidate_declarations.append(UnitDeclaration(source=source, raw_unit=str(item), unit=canonical))

    if not candidate_declarations:
        return
    units = {declaration.unit for declaration in candidate_declarations}
    if len(units) != 1:
        raw_units = [declaration.raw_unit for declaration in candidate_declarations]
        errors.append(f"{context}: conflicting {kind} component units at {source}: {raw_units!r}")
        return
    declarations.append(candidate_declarations[0])


def _resolve_unit(
    *,
    record: dict[str, Any] | None,
    record_source: str,
    stream_entry: dict[str, Any] | None,
    stream_source: str,
    metadata: dict[str, Any] | None,
    metadata_source: str,
    streams_index: dict[str, Any] | None,
    streams_source: str,
    keys: list[str],
    kind: str,
    canonicalizer: Callable[[Any], str | None],
    synthetic: bool,
    legacy_unit: str,
    context: str,
) -> UnitResolution:
    candidates: list[tuple[str, Any]] = []
    candidates.extend(_container_unit_candidates(record, record_source, keys))
    candidates.extend(_container_unit_candidates(stream_entry, stream_source, keys))
    for container, source in [
        (record, record_source),
        (stream_entry, stream_source),
        (metadata, metadata_source),
        (streams_index, streams_source),
    ]:
        candidate = _actual_tcp_position_units_candidate(container, source, kind)
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(_container_unit_candidates(metadata, metadata_source, keys))
    candidates.extend(_container_unit_candidates(streams_index, streams_source, keys))

    declarations: list[UnitDeclaration] = []
    errors: list[str] = []
    for source, value in candidates:
        _add_candidate_declarations(
            declarations,
            errors,
            source=source,
            value=value,
            canonicalizer=canonicalizer,
            kind=kind,
            context=context,
        )

    if errors:
        return UnitResolution(None, None, None, tuple(declarations), False, tuple(errors))

    units = {declaration.unit for declaration in declarations}
    if len(units) > 1:
        detail = ", ".join(
            f"{declaration.source}={declaration.raw_unit!r}" for declaration in declarations
        )
        return UnitResolution(
            None,
            None,
            None,
            tuple(declarations),
            False,
            (f"{context}: conflicting {kind} unit declarations: {detail}",),
        )

    if declarations:
        selected = declarations[0]
        return UnitResolution(
            selected.unit,
            selected.raw_unit,
            selected.source,
            tuple(declarations),
            False,
            (),
        )

    if synthetic:
        return UnitResolution(
            legacy_unit,
            legacy_unit,
            "legacy synthetic raw_real_v0 compatibility fallback",
            (),
            True,
            (),
        )

    return UnitResolution(
        None,
        None,
        None,
        (),
        False,
        (f"{context}: unsupported or missing {kind} unit: None",),
    )


def resolve_tcp_position_unit(
    *,
    record: dict[str, Any] | None,
    record_source: str,
    stream_entry: dict[str, Any] | None,
    stream_source: str,
    metadata: dict[str, Any] | None,
    metadata_source: str = "metadata.json",
    streams_index: dict[str, Any] | None = None,
    streams_source: str = "streams/index.json",
    synthetic: bool,
    context: str,
) -> UnitResolution:
    return _resolve_unit(
        record=record,
        record_source=record_source,
        stream_entry=stream_entry,
        stream_source=stream_source,
        metadata=metadata,
        metadata_source=metadata_source,
        streams_index=streams_index,
        streams_source=streams_source,
        keys=POSITION_UNIT_KEYS,
        kind="tcp_position",
        canonicalizer=canonical_tcp_position_unit,
        synthetic=synthetic,
        legacy_unit="mm",
        context=context,
    )


def resolve_tcp_orientation_unit(
    *,
    record: dict[str, Any] | None,
    record_source: str,
    stream_entry: dict[str, Any] | None,
    stream_source: str,
    metadata: dict[str, Any] | None,
    metadata_source: str = "metadata.json",
    streams_index: dict[str, Any] | None = None,
    streams_source: str = "streams/index.json",
    synthetic: bool,
    context: str,
) -> UnitResolution:
    return _resolve_unit(
        record=record,
        record_source=record_source,
        stream_entry=stream_entry,
        stream_source=stream_source,
        metadata=metadata,
        metadata_source=metadata_source,
        streams_index=streams_index,
        streams_source=streams_source,
        keys=ORIENTATION_UNIT_KEYS,
        kind="tcp_orientation",
        canonicalizer=canonical_tcp_orientation_unit,
        synthetic=synthetic,
        legacy_unit="deg",
        context=context,
    )
