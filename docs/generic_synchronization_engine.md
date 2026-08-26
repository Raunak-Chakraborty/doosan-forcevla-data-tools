# Generic synchronization engine

Patch 3 adds a deterministic timestamp/index association layer between typed raw
records and later Doosan-specific synchronization policy.

## Scope

The engine is intentionally payload-free. It consumes only validated timestamp
sequences and produces source indices plus timing provenance. It does not retain
camera images, ROS messages, NumPy arrays, or decoded physical values.

Supported association methods are:

- `reference`: exact one-to-one reference timeline identity
- `nearest`: closest source timestamp, with deterministic earlier-sample tie break
- `causal_hold`: latest source timestamp not later than the reference
- `linear`: exact match or strict two-sided interpolation bracket

The engine also provides exact deterministic timing statistics and a multi-source
frame-completeness plan.

## Explicit clock domains

Every timeline carries one `ClockDomain`. Patch 3 introduced `bag`, `header`,
and `controller`; Patch 4 adds `ros` so production policy can distinguish a
timestamp's raw source field from the clock epoch it belongs to.

Synchronization across different domains is rejected. The engine never falls
back from a missing header timestamp to bag time. A policy may place explicitly
validated header- and bag-sourced timestamps into the same `ros` epoch only
after proving that relationship at the episode boundary.

Patch 4 owns the Doosan timestamp-source, clock-epoch, synchronization-method,
and freshness policy. The generic planner remains payload-free.

## Freshness semantics

Freshness limits are optional in Patch 3 because real per-stream thresholds are
owned by Patch 4.

When supplied, limits are inclusive:

- nearest: `abs(source - reference) <= max_age_ns`
- causal hold: `reference - source <= max_age_ns`
- linear: both endpoint distances must be `<= max_age_ns`
- linear: optional bracket span must be `<= max_bracket_span_ns`

Linear interpolation never extrapolates.

## Duplicate and out-of-order timestamps

Input timelines must be strictly increasing. Duplicate timestamps and timestamp
regressions fail closed rather than being silently discarded or ambiguously
resolved.

For nearest-neighbor matching, an equal-distance tie selects the earlier source
timestamp, then the lower source index. This rule is deterministic.

## Provenance

Every successful association records:

- reference index and timestamp
- source index or interpolation endpoint indices
- source timestamp or endpoint timestamps
- signed skew(s), defined as `source_timestamp_ns - reference_timestamp_ns`
- interpolation alpha when applicable

Every miss records an explicit reason: `missing`, `stale`, `no_bracket`, or
`bracket_too_wide`.

## Statistics

Statistics are exact over all matched associations. The implementation does not
use a bounded/reservoir percentile approximation. Empty statistics are encoded
with JSON-safe `None` values rather than NaN.

## Difference from the SO101 reference implementation

Design inspiration was taken from the Apache-2.0
`legalaspro/so101-ros-physical-ai` rosbag-to-LeRobot converter, frozen during
Patch-3 preflight at commit:

`58318c905a2c61289fa907de85cb8473322fbe68`

The reference implementation uses a single-item `LastBuffer`, emits a frame as
soon as the reference message arrives, and performs causal as-of sampling with
a freshness window. Patch 3 deliberately improves that design for this thesis:

- timestamp/index plans rather than payload buffers
- symmetric nearest-neighbor and two-sided linear planning
- no future-data limitation imposed by sequential immediate emission
- explicit clock-domain compatibility instead of header-to-bag fallback
- strict duplicate/regression rejection
- deterministic tie behavior
- exact percentile statistics instead of a bounded reservoir
- JSON-safe missing statistics instead of NaN
- explicit per-association provenance and miss reasons
- independent `required`/optional source semantics

The SO101 default freshness values are not copied. Patch 4 will use the measured
Doosan timing distributions to define the production synchronization policy.

## Deferred to Patch 4

Patch 3 does not decide:

- which Doosan stream is the production reference clock/timeline
- bag vs header timestamp policy for each stream
- per-stream freshness thresholds
- which streams use nearest, causal hold, or linear interpolation
- whether a missing optional diagnostic stream should be exported
- any physical-value interpolation implementation
- image decoding or image payload retrieval
- ForceVLA state/action construction
