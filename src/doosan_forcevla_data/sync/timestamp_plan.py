"""Deterministic timestamp-only synchronization planning.

Patch 3 intentionally plans associations using timestamps and source indices only.
It never stores image payloads, performs interpolation of physical values, chooses
Doosan-specific clock policy, or applies task-specific freshness thresholds.
Those responsibilities belong to later conversion stages.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping


class SynchronizationError(ValueError):
    """Raised when a timeline or synchronization specification is invalid."""


class ClockDomain(str, Enum):
    """Explicit clock domains supported by the generic synchronization layer."""

    BAG = "bag"
    HEADER = "header"
    CONTROLLER = "controller"


class SyncMethod(str, Enum):
    """Association strategies supported by the generic planner."""

    REFERENCE = "reference"
    NEAREST = "nearest"
    CAUSAL_HOLD = "causal_hold"
    LINEAR = "linear"


class MissReason(str, Enum):
    """Reason a reference timestamp did not obtain a usable source selection."""

    MISSING = "missing"
    STALE = "stale"
    NO_BRACKET = "no_bracket"
    BRACKET_TOO_WIDE = "bracket_too_wide"


@dataclass(frozen=True)
class TimestampTimeline:
    """A strictly increasing sequence of timestamps from one explicit clock."""

    name: str
    clock_domain: ClockDomain
    timestamps_ns: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SynchronizationError("timeline name must be a non-empty string")
        if not isinstance(self.clock_domain, ClockDomain):
            raise SynchronizationError(
                "clock_domain must be a ClockDomain value, "
                f"got {self.clock_domain!r}"
            )

        try:
            normalized_timestamps = tuple(self.timestamps_ns)
        except TypeError as exc:
            raise SynchronizationError(
                f"{self.name}: timestamps_ns must be iterable"
            ) from exc
        object.__setattr__(self, "timestamps_ns", normalized_timestamps)

        previous: int | None = None
        for index, value in enumerate(self.timestamps_ns):
            if isinstance(value, bool) or not isinstance(value, int):
                raise SynchronizationError(
                    f"{self.name}: timestamp[{index}] must be an integer nanosecond value"
                )
            if value < 0:
                raise SynchronizationError(
                    f"{self.name}: timestamp[{index}] must be non-negative"
                )
            if previous is not None:
                if value == previous:
                    raise SynchronizationError(
                        f"{self.name}: duplicate timestamp at indices "
                        f"{index - 1} and {index}: {value}"
                    )
                if value < previous:
                    raise SynchronizationError(
                        f"{self.name}: timestamp regression at index {index}: "
                        f"{value} < {previous}"
                    )
            previous = value

    @classmethod
    def from_timestamps(
        cls,
        name: str,
        clock_domain: ClockDomain,
        timestamps_ns: Iterable[int],
    ) -> "TimestampTimeline":
        """Normalize an iterable to an immutable validated timeline."""

        return cls(
            name=name,
            clock_domain=clock_domain,
            timestamps_ns=tuple(timestamps_ns),
        )


@dataclass(frozen=True)
class SyncSpec:
    """Generic association policy for one source timeline.

    ``max_age_ns`` is inclusive. For nearest and causal-hold it bounds the
    selected sample's absolute age from the reference. For linear interpolation
    it bounds *both* endpoint distances independently. ``max_bracket_span_ns``
    is only meaningful for linear interpolation and is also inclusive.
    """

    method: SyncMethod
    required: bool = True
    max_age_ns: int | None = None
    max_bracket_span_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, SyncMethod):
            raise SynchronizationError(
                f"method must be a SyncMethod value, got {self.method!r}"
            )
        if not isinstance(self.required, bool):
            raise SynchronizationError("required must be bool")

        for name, value in (
            ("max_age_ns", self.max_age_ns),
            ("max_bracket_span_ns", self.max_bracket_span_ns),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SynchronizationError(
                    f"{name} must be None or a non-negative integer, got {value!r}"
                )

        if self.method is SyncMethod.REFERENCE:
            if self.max_age_ns is not None or self.max_bracket_span_ns is not None:
                raise SynchronizationError(
                    "reference synchronization does not accept freshness limits"
                )
        elif self.method in (SyncMethod.NEAREST, SyncMethod.CAUSAL_HOLD):
            if self.max_bracket_span_ns is not None:
                raise SynchronizationError(
                    f"{self.method.value} does not accept max_bracket_span_ns"
                )


@dataclass(frozen=True)
class SyncSelection:
    """Provenance for a successful source association."""

    source_indices: tuple[int, ...]
    source_timestamps_ns: tuple[int, ...]
    signed_skews_ns: tuple[int, ...]
    alpha: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_indices": list(self.source_indices),
            "source_timestamps_ns": list(self.source_timestamps_ns),
            "signed_skews_ns": list(self.signed_skews_ns),
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class SyncDecision:
    """Association result for one reference timestamp."""

    reference_index: int
    reference_timestamp_ns: int
    selection: SyncSelection | None
    miss_reason: MissReason | None

    def __post_init__(self) -> None:
        matched = self.selection is not None
        missed = self.miss_reason is not None
        if matched == missed:
            raise SynchronizationError(
                "SyncDecision must contain exactly one of selection or miss_reason"
            )

    @property
    def matched(self) -> bool:
        return self.selection is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_index": self.reference_index,
            "reference_timestamp_ns": self.reference_timestamp_ns,
            "selection": (
                self.selection.to_dict() if self.selection is not None else None
            ),
            "miss_reason": (
                self.miss_reason.value if self.miss_reason is not None else None
            ),
        }


@dataclass(frozen=True)
class NumericSummary:
    """Exact deterministic descriptive statistics with JSON-safe missing values."""

    count: int
    minimum: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    maximum: float | None
    mean: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "min": self.minimum,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.maximum,
            "mean": self.mean,
        }


@dataclass(frozen=True)
class SourcePlanSummary:
    attempted: int
    matched: int
    missing: int
    stale: int
    no_bracket: int
    bracket_too_wide: int
    exact_matches: int
    interpolated_matches: int
    match_rate: float
    signed_skew_ns: NumericSummary
    absolute_skew_ns: NumericSummary
    left_age_ns: NumericSummary
    right_age_ns: NumericSummary
    bracket_span_ns: NumericSummary
    alpha: NumericSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "matched": self.matched,
            "missing": self.missing,
            "stale": self.stale,
            "no_bracket": self.no_bracket,
            "bracket_too_wide": self.bracket_too_wide,
            "exact_matches": self.exact_matches,
            "interpolated_matches": self.interpolated_matches,
            "match_rate": self.match_rate,
            "signed_skew_ns": self.signed_skew_ns.to_dict(),
            "absolute_skew_ns": self.absolute_skew_ns.to_dict(),
            "left_age_ns": self.left_age_ns.to_dict(),
            "right_age_ns": self.right_age_ns.to_dict(),
            "bracket_span_ns": self.bracket_span_ns.to_dict(),
            "alpha": self.alpha.to_dict(),
        }


@dataclass(frozen=True)
class SourcePlan:
    """Complete association plan for one source timeline."""

    source_name: str
    clock_domain: ClockDomain
    spec: SyncSpec
    decisions: tuple[SyncDecision, ...]
    summary: SourcePlanSummary

    def to_dict(self, *, include_decisions: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "source_name": self.source_name,
            "clock_domain": self.clock_domain.value,
            "spec": {
                "method": self.spec.method.value,
                "required": self.spec.required,
                "max_age_ns": self.spec.max_age_ns,
                "max_bracket_span_ns": self.spec.max_bracket_span_ns,
            },
            "summary": self.summary.to_dict(),
        }
        if include_decisions:
            result["decisions"] = [decision.to_dict() for decision in self.decisions]
        return result


@dataclass(frozen=True)
class SynchronizationPlan:
    """Multi-source timestamp/index plan anchored to one reference timeline."""

    reference: TimestampTimeline
    source_plans: tuple[tuple[str, SourcePlan], ...]
    complete_reference_indices: tuple[int, ...]
    dropped_reference_indices: tuple[int, ...]

    def source_plan(self, key: str) -> SourcePlan:
        for candidate_key, plan in self.source_plans:
            if candidate_key == key:
                return plan
        raise KeyError(key)

    def to_dict(self, *, include_decisions: bool = True) -> dict[str, object]:
        return {
            "reference": {
                "name": self.reference.name,
                "clock_domain": self.reference.clock_domain.value,
                "count": len(self.reference.timestamps_ns),
                "timestamps_ns": (
                    list(self.reference.timestamps_ns) if include_decisions else None
                ),
            },
            "sources": {
                key: plan.to_dict(include_decisions=include_decisions)
                for key, plan in self.source_plans
            },
            "complete_reference_indices": list(self.complete_reference_indices),
            "dropped_reference_indices": list(self.dropped_reference_indices),
        }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise SynchronizationError(f"percentile q must be in [0, 1], got {q}")
    if not sorted_values:
        raise SynchronizationError("cannot compute percentile of empty input")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def summarize_numeric(values: Iterable[int | float]) -> NumericSummary:
    """Return exact p50/p95/p99 statistics without NaN sentinels."""

    numeric = [float(value) for value in values]
    if not numeric:
        return NumericSummary(0, None, None, None, None, None, None)
    if not all(math.isfinite(value) for value in numeric):
        raise SynchronizationError("statistics input contains non-finite values")

    ordered = sorted(numeric)
    return NumericSummary(
        count=len(ordered),
        minimum=ordered[0],
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
        mean=math.fsum(ordered) / len(ordered),
    )


def _matched_decision(
    reference_index: int,
    reference_timestamp_ns: int,
    source_indices: tuple[int, ...],
    source_timestamps_ns: tuple[int, ...],
    *,
    alpha: float | None,
) -> SyncDecision:
    return SyncDecision(
        reference_index=reference_index,
        reference_timestamp_ns=reference_timestamp_ns,
        selection=SyncSelection(
            source_indices=source_indices,
            source_timestamps_ns=source_timestamps_ns,
            signed_skews_ns=tuple(
                timestamp - reference_timestamp_ns
                for timestamp in source_timestamps_ns
            ),
            alpha=alpha,
        ),
        miss_reason=None,
    )


def _missed_decision(
    reference_index: int,
    reference_timestamp_ns: int,
    reason: MissReason,
) -> SyncDecision:
    return SyncDecision(
        reference_index=reference_index,
        reference_timestamp_ns=reference_timestamp_ns,
        selection=None,
        miss_reason=reason,
    )


def _validate_clock_compatibility(
    reference: TimestampTimeline,
    source: TimestampTimeline,
) -> None:
    if reference.clock_domain is not source.clock_domain:
        raise SynchronizationError(
            "cannot synchronize different clock domains: "
            f"reference={reference.clock_domain.value!r}, "
            f"source={source.clock_domain.value!r}"
        )


def _reference_decisions(
    reference: TimestampTimeline,
    source: TimestampTimeline,
) -> tuple[SyncDecision, ...]:
    if source.timestamps_ns != reference.timestamps_ns:
        raise SynchronizationError(
            "reference method requires source timestamps to exactly equal "
            "the reference timeline"
        )

    return tuple(
        _matched_decision(
            index,
            timestamp,
            (index,),
            (timestamp,),
            alpha=None,
        )
        for index, timestamp in enumerate(reference.timestamps_ns)
    )


def _nearest_decisions(
    reference: TimestampTimeline,
    source: TimestampTimeline,
    spec: SyncSpec,
) -> tuple[SyncDecision, ...]:
    source_ts = source.timestamps_ns
    decisions: list[SyncDecision] = []

    for reference_index, ref in enumerate(reference.timestamps_ns):
        if not source_ts:
            decisions.append(
                _missed_decision(reference_index, ref, MissReason.MISSING)
            )
            continue

        insertion = bisect_left(source_ts, ref)
        candidates: list[int] = []
        if insertion > 0:
            candidates.append(insertion - 1)
        if insertion < len(source_ts):
            candidates.append(insertion)

        # Deterministic tie policy: minimum absolute skew, then the earlier
        # source timestamp, then the lower source index.
        chosen_index = min(
            candidates,
            key=lambda index: (
                abs(source_ts[index] - ref),
                source_ts[index],
                index,
            ),
        )
        chosen = source_ts[chosen_index]
        age = abs(chosen - ref)

        if spec.max_age_ns is not None and age > spec.max_age_ns:
            decisions.append(
                _missed_decision(reference_index, ref, MissReason.STALE)
            )
            continue

        decisions.append(
            _matched_decision(
                reference_index,
                ref,
                (chosen_index,),
                (chosen,),
                alpha=None,
            )
        )

    return tuple(decisions)


def _causal_hold_decisions(
    reference: TimestampTimeline,
    source: TimestampTimeline,
    spec: SyncSpec,
) -> tuple[SyncDecision, ...]:
    source_ts = source.timestamps_ns
    decisions: list[SyncDecision] = []

    for reference_index, ref in enumerate(reference.timestamps_ns):
        chosen_index = bisect_right(source_ts, ref) - 1
        if chosen_index < 0:
            decisions.append(
                _missed_decision(reference_index, ref, MissReason.MISSING)
            )
            continue

        chosen = source_ts[chosen_index]
        age = ref - chosen
        if age < 0:
            raise SynchronizationError("causal-hold selected a future sample")

        if spec.max_age_ns is not None and age > spec.max_age_ns:
            decisions.append(
                _missed_decision(reference_index, ref, MissReason.STALE)
            )
            continue

        decisions.append(
            _matched_decision(
                reference_index,
                ref,
                (chosen_index,),
                (chosen,),
                alpha=None,
            )
        )

    return tuple(decisions)


def _linear_decisions(
    reference: TimestampTimeline,
    source: TimestampTimeline,
    spec: SyncSpec,
) -> tuple[SyncDecision, ...]:
    source_ts = source.timestamps_ns
    decisions: list[SyncDecision] = []

    for reference_index, ref in enumerate(reference.timestamps_ns):
        if not source_ts:
            decisions.append(
                _missed_decision(reference_index, ref, MissReason.MISSING)
            )
            continue

        right_index = bisect_left(source_ts, ref)

        if right_index < len(source_ts) and source_ts[right_index] == ref:
            decisions.append(
                _matched_decision(
                    reference_index,
                    ref,
                    (right_index,),
                    (ref,),
                    alpha=0.0,
                )
            )
            continue

        left_index = right_index - 1
        if left_index < 0 or right_index >= len(source_ts):
            decisions.append(
                _missed_decision(reference_index, ref, MissReason.NO_BRACKET)
            )
            continue

        left = source_ts[left_index]
        right = source_ts[right_index]
        if not left < ref < right:
            raise SynchronizationError(
                f"invalid interpolation bracket: left={left}, ref={ref}, right={right}"
            )

        left_age = ref - left
        right_age = right - ref
        span = right - left

        if (
            spec.max_age_ns is not None
            and (left_age > spec.max_age_ns or right_age > spec.max_age_ns)
        ):
            decisions.append(
                _missed_decision(reference_index, ref, MissReason.STALE)
            )
            continue

        if (
            spec.max_bracket_span_ns is not None
            and span > spec.max_bracket_span_ns
        ):
            decisions.append(
                _missed_decision(
                    reference_index,
                    ref,
                    MissReason.BRACKET_TOO_WIDE,
                )
            )
            continue

        alpha = left_age / span
        if not 0.0 < alpha < 1.0:
            raise SynchronizationError(f"invalid interpolation alpha {alpha}")

        decisions.append(
            _matched_decision(
                reference_index,
                ref,
                (left_index, right_index),
                (left, right),
                alpha=alpha,
            )
        )

    return tuple(decisions)


def _summarize_decisions(decisions: tuple[SyncDecision, ...]) -> SourcePlanSummary:
    matched = [decision for decision in decisions if decision.matched]
    misses = [decision for decision in decisions if not decision.matched]

    signed_single: list[int] = []
    absolute_single: list[int] = []
    left_ages: list[int] = []
    right_ages: list[int] = []
    bracket_spans: list[int] = []
    alphas: list[float] = []
    exact_matches = 0
    interpolated_matches = 0

    for decision in matched:
        selection = decision.selection
        if selection is None:
            raise SynchronizationError("matched decision unexpectedly lacks selection")

        if len(selection.source_timestamps_ns) == 1:
            skew = selection.signed_skews_ns[0]
            signed_single.append(skew)
            absolute_single.append(abs(skew))
            if skew == 0:
                exact_matches += 1
        elif len(selection.source_timestamps_ns) == 2:
            left, right = selection.source_timestamps_ns
            ref = decision.reference_timestamp_ns
            left_age = ref - left
            right_age = right - ref
            if left_age <= 0 or right_age <= 0:
                raise SynchronizationError("interpolation provenance is not a strict bracket")
            left_ages.append(left_age)
            right_ages.append(right_age)
            bracket_spans.append(right - left)
            if selection.alpha is None or not math.isfinite(selection.alpha):
                raise SynchronizationError("interpolation selection lacks finite alpha")
            alphas.append(selection.alpha)
            interpolated_matches += 1
        else:
            raise SynchronizationError(
                "a source selection must contain one sample or two interpolation endpoints"
            )

    reason_counts = {
        reason: sum(1 for decision in misses if decision.miss_reason is reason)
        for reason in MissReason
    }

    attempted = len(decisions)
    return SourcePlanSummary(
        attempted=attempted,
        matched=len(matched),
        missing=reason_counts[MissReason.MISSING],
        stale=reason_counts[MissReason.STALE],
        no_bracket=reason_counts[MissReason.NO_BRACKET],
        bracket_too_wide=reason_counts[MissReason.BRACKET_TOO_WIDE],
        exact_matches=exact_matches,
        interpolated_matches=interpolated_matches,
        match_rate=(len(matched) / attempted if attempted else 0.0),
        signed_skew_ns=summarize_numeric(signed_single),
        absolute_skew_ns=summarize_numeric(absolute_single),
        left_age_ns=summarize_numeric(left_ages),
        right_age_ns=summarize_numeric(right_ages),
        bracket_span_ns=summarize_numeric(bracket_spans),
        alpha=summarize_numeric(alphas),
    )


def build_source_plan(
    reference: TimestampTimeline,
    source: TimestampTimeline,
    spec: SyncSpec,
) -> SourcePlan:
    """Build a deterministic source-index plan for one source timeline."""

    if not reference.timestamps_ns:
        raise SynchronizationError("reference timeline must not be empty")

    _validate_clock_compatibility(reference, source)

    if spec.method is SyncMethod.REFERENCE:
        decisions = _reference_decisions(reference, source)
    elif spec.method is SyncMethod.NEAREST:
        decisions = _nearest_decisions(reference, source, spec)
    elif spec.method is SyncMethod.CAUSAL_HOLD:
        decisions = _causal_hold_decisions(reference, source, spec)
    elif spec.method is SyncMethod.LINEAR:
        decisions = _linear_decisions(reference, source, spec)
    else:  # pragma: no cover - Enum exhaustiveness guard.
        raise SynchronizationError(f"unsupported synchronization method {spec.method}")

    return SourcePlan(
        source_name=source.name,
        clock_domain=source.clock_domain,
        spec=spec,
        decisions=decisions,
        summary=_summarize_decisions(decisions),
    )


def build_synchronization_plan(
    reference: TimestampTimeline,
    source_timelines: Mapping[str, TimestampTimeline],
    specs: Mapping[str, SyncSpec],
) -> SynchronizationPlan:
    """Build a deterministic multi-source synchronization plan.

    The two mappings must contain exactly the same keys. A reference index is
    considered complete only when every source whose ``SyncSpec.required`` is
    true has a successful selection. Optional sources never force a frame drop.
    """

    if not reference.timestamps_ns:
        raise SynchronizationError("reference timeline must not be empty")

    for mapping_name, mapping in (
        ("source_timelines", source_timelines),
        ("specs", specs),
    ):
        invalid_keys = [
            key
            for key in mapping
            if not isinstance(key, str) or not key
        ]
        if invalid_keys:
            raise SynchronizationError(
                f"{mapping_name} keys must be non-empty strings, "
                f"got {invalid_keys!r}"
            )

    source_keys = set(source_timelines)
    spec_keys = set(specs)
    if source_keys != spec_keys:
        raise SynchronizationError(
            "source/spec key mismatch; "
            f"missing_specs={sorted(source_keys - spec_keys)}, "
            f"missing_sources={sorted(spec_keys - source_keys)}"
        )

    plans: list[tuple[str, SourcePlan]] = []
    for key in sorted(source_timelines):
        plans.append(
            (
                key,
                build_source_plan(
                    reference,
                    source_timelines[key],
                    specs[key],
                ),
            )
        )

    complete: list[int] = []
    dropped: list[int] = []

    for reference_index in range(len(reference.timestamps_ns)):
        required_ok = all(
            plan.decisions[reference_index].matched
            for _, plan in plans
            if plan.spec.required
        )
        if required_ok:
            complete.append(reference_index)
        else:
            dropped.append(reference_index)

    return SynchronizationPlan(
        reference=reference,
        source_plans=tuple(plans),
        complete_reference_indices=tuple(complete),
        dropped_reference_indices=tuple(dropped),
    )
