import json
import unittest

from doosan_forcevla_data.sync.timestamp_plan import (
    ClockDomain,
    MissReason,
    SyncMethod,
    SyncSpec,
    SynchronizationError,
    TimestampTimeline,
    build_source_plan,
    build_synchronization_plan,
    summarize_numeric,
)


def timeline(name, values, domain=ClockDomain.BAG):
    return TimestampTimeline.from_timestamps(name, domain, values)


class TimestampPlanTests(unittest.TestCase):
    def test_timeline_rejects_duplicate_timestamps(self):
        with self.assertRaisesRegex(SynchronizationError, "duplicate timestamp"):
            timeline("source", [1, 2, 2, 3])

    def test_timeline_rejects_regression(self):
        with self.assertRaisesRegex(SynchronizationError, "timestamp regression"):
            timeline("source", [1, 3, 2])

    def test_timeline_normalizes_direct_list_input_to_tuple(self):
        value = TimestampTimeline("source", ClockDomain.BAG, [1, 2, 3])
        self.assertEqual(value.timestamps_ns, (1, 2, 3))
        self.assertIsInstance(value.timestamps_ns, tuple)

    def test_timeline_rejects_negative_and_boolean_timestamp(self):
        with self.assertRaises(SynchronizationError):
            timeline("source", [-1])
        with self.assertRaises(SynchronizationError):
            timeline("source", [True])

    def test_different_clock_domains_fail_closed(self):
        reference = timeline("reference", [10], ClockDomain.BAG)
        source = timeline("source", [10], ClockDomain.HEADER)
        with self.assertRaisesRegex(SynchronizationError, "different clock domains"):
            build_source_plan(reference, source, SyncSpec(SyncMethod.NEAREST))

    def test_reference_method_requires_exact_same_timeline(self):
        reference = timeline("reference", [10, 20, 30])
        plan = build_source_plan(
            reference,
            timeline("reference-copy", [10, 20, 30]),
            SyncSpec(SyncMethod.REFERENCE),
        )
        self.assertEqual(plan.summary.matched, 3)
        self.assertEqual(plan.summary.exact_matches, 3)

        with self.assertRaisesRegex(SynchronizationError, "exactly equal"):
            build_source_plan(
                reference,
                timeline("wrong", [10, 21, 30]),
                SyncSpec(SyncMethod.REFERENCE),
            )

    def test_nearest_chooses_past_future_and_earlier_on_tie(self):
        reference = timeline("reference", [10, 20, 30])
        source = timeline("source", [7, 15, 25, 35])
        plan = build_source_plan(
            reference,
            source,
            SyncSpec(SyncMethod.NEAREST),
        )

        selections = [decision.selection for decision in plan.decisions]
        self.assertEqual(selections[0].source_timestamps_ns, (7,))
        self.assertEqual(selections[1].source_timestamps_ns, (15,))
        self.assertEqual(selections[2].source_timestamps_ns, (25,))
        self.assertEqual(selections[1].signed_skews_ns, (-5,))
        self.assertEqual(selections[2].signed_skews_ns, (-5,))

    def test_nearest_freshness_boundary_is_inclusive(self):
        reference = timeline("reference", [10, 20])
        source = timeline("source", [5, 14])
        plan = build_source_plan(
            reference,
            source,
            SyncSpec(SyncMethod.NEAREST, max_age_ns=5),
        )
        self.assertTrue(plan.decisions[0].matched)
        self.assertFalse(plan.decisions[1].matched)
        self.assertIs(plan.decisions[1].miss_reason, MissReason.STALE)

    def test_causal_hold_never_selects_future(self):
        reference = timeline("reference", [5, 10, 15])
        source = timeline("source", [7, 12])
        plan = build_source_plan(
            reference,
            source,
            SyncSpec(SyncMethod.CAUSAL_HOLD),
        )
        self.assertIs(plan.decisions[0].miss_reason, MissReason.MISSING)
        self.assertEqual(plan.decisions[1].selection.source_timestamps_ns, (7,))
        self.assertEqual(plan.decisions[2].selection.source_timestamps_ns, (12,))
        self.assertTrue(
            all(
                skew <= 0
                for decision in plan.decisions
                if decision.selection is not None
                for skew in decision.selection.signed_skews_ns
            )
        )

    def test_causal_hold_freshness_boundary_is_inclusive(self):
        reference = timeline("reference", [10, 11])
        source = timeline("source", [5])
        plan = build_source_plan(
            reference,
            source,
            SyncSpec(SyncMethod.CAUSAL_HOLD, max_age_ns=5),
        )
        self.assertTrue(plan.decisions[0].matched)
        self.assertIs(plan.decisions[1].miss_reason, MissReason.STALE)

    def test_linear_exact_bracket_and_no_extrapolation(self):
        reference = timeline("reference", [5, 10, 15, 20, 25])
        source = timeline("source", [10, 20])
        plan = build_source_plan(
            reference,
            source,
            SyncSpec(SyncMethod.LINEAR),
        )

        self.assertIs(plan.decisions[0].miss_reason, MissReason.NO_BRACKET)
        self.assertEqual(plan.decisions[1].selection.source_indices, (0,))
        self.assertEqual(plan.decisions[1].selection.alpha, 0.0)
        self.assertEqual(plan.decisions[2].selection.source_indices, (0, 1))
        self.assertAlmostEqual(plan.decisions[2].selection.alpha, 0.5)
        self.assertEqual(plan.decisions[3].selection.source_indices, (1,))
        self.assertIs(plan.decisions[4].miss_reason, MissReason.NO_BRACKET)
        self.assertEqual(plan.summary.exact_matches, 2)
        self.assertEqual(plan.summary.interpolated_matches, 1)

    def test_linear_endpoint_age_and_span_limits_are_independent(self):
        reference = timeline("reference", [10])

        stale = build_source_plan(
            reference,
            timeline("source", [0, 12]),
            SyncSpec(SyncMethod.LINEAR, max_age_ns=9),
        )
        self.assertIs(stale.decisions[0].miss_reason, MissReason.STALE)

        wide = build_source_plan(
            reference,
            timeline("source", [5, 15]),
            SyncSpec(
                SyncMethod.LINEAR,
                max_age_ns=5,
                max_bracket_span_ns=9,
            ),
        )
        self.assertIs(
            wide.decisions[0].miss_reason,
            MissReason.BRACKET_TOO_WIDE,
        )

        accepted = build_source_plan(
            reference,
            timeline("source", [5, 15]),
            SyncSpec(
                SyncMethod.LINEAR,
                max_age_ns=5,
                max_bracket_span_ns=10,
            ),
        )
        self.assertTrue(accepted.decisions[0].matched)

    def test_empty_source_reports_missing_without_crashing(self):
        reference = timeline("reference", [10, 20])
        source = timeline("source", [])
        for method in (
            SyncMethod.NEAREST,
            SyncMethod.CAUSAL_HOLD,
            SyncMethod.LINEAR,
        ):
            plan = build_source_plan(reference, source, SyncSpec(method))
            self.assertEqual(plan.summary.missing, 2)
            self.assertEqual(plan.summary.matched, 0)
            self.assertEqual(plan.summary.match_rate, 0.0)

    def test_multi_source_required_and_optional_frame_completion(self):
        reference = timeline("reference", [10, 20, 30])
        plan = build_synchronization_plan(
            reference,
            {
                "required": timeline("required", [9, 21]),
                "optional": timeline("optional", []),
            },
            {
                "required": SyncSpec(
                    SyncMethod.NEAREST,
                    required=True,
                    max_age_ns=2,
                ),
                "optional": SyncSpec(
                    SyncMethod.NEAREST,
                    required=False,
                ),
            },
        )
        self.assertEqual(plan.complete_reference_indices, (0, 1))
        self.assertEqual(plan.dropped_reference_indices, (2,))

    def test_source_and_spec_keys_must_match_exactly(self):
        reference = timeline("reference", [10])
        with self.assertRaisesRegex(SynchronizationError, "source/spec key mismatch"):
            build_synchronization_plan(
                reference,
                {"a": timeline("a", [10])},
                {"b": SyncSpec(SyncMethod.NEAREST)},
            )

    def test_mapping_keys_must_be_nonempty_strings(self):
        reference = timeline("reference", [10])
        with self.assertRaisesRegex(SynchronizationError, "keys must be non-empty strings"):
            build_synchronization_plan(
                reference,
                {1: timeline("a", [10])},
                {1: SyncSpec(SyncMethod.NEAREST)},
            )

    def test_statistics_are_exact_deterministic_and_json_safe(self):
        summary = summarize_numeric([1, 2, 3, 4, 5])
        self.assertEqual(summary.minimum, 1.0)
        self.assertEqual(summary.p50, 3.0)
        self.assertEqual(summary.maximum, 5.0)

        empty = summarize_numeric([])
        payload = json.dumps(empty.to_dict(), allow_nan=False, sort_keys=True)
        self.assertIn('"mean": null', payload)

        reference = timeline("reference", [10, 20, 30])
        source = timeline("source", [9, 21, 29])
        first = build_source_plan(reference, source, SyncSpec(SyncMethod.NEAREST))
        second = build_source_plan(reference, source, SyncSpec(SyncMethod.NEAREST))
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first.to_dict(), allow_nan=False, sort_keys=True),
            json.dumps(second.to_dict(), allow_nan=False, sort_keys=True),
        )

    def test_invalid_spec_combinations_fail_closed(self):
        with self.assertRaises(SynchronizationError):
            SyncSpec(SyncMethod.REFERENCE, max_age_ns=1)
        with self.assertRaises(SynchronizationError):
            SyncSpec(SyncMethod.NEAREST, max_bracket_span_ns=1)
        with self.assertRaises(SynchronizationError):
            SyncSpec(SyncMethod.LINEAR, max_age_ns=-1)

    def test_plan_contains_indices_not_payloads(self):
        reference = timeline("reference", [10])
        source = timeline("source", [9, 11])
        plan = build_source_plan(reference, source, SyncSpec(SyncMethod.NEAREST))
        selection = plan.decisions[0].selection
        self.assertEqual(selection.source_indices, (0,))
        self.assertFalse(hasattr(selection, "value"))
        self.assertFalse(hasattr(selection, "payload"))


if __name__ == "__main__":
    unittest.main()
