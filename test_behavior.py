"""Analytical fixtures for behavior semantics, independent of dataset labels."""
import unittest
import numpy as np
from behavior_features import (BLOCKS, LATENCY_BINS, autocorrelation, consistency,
    drift, entropy, extract_blocks, extract_model_features, interruption_recovery,
    relative_recovery, response_series, silence_recovery)


class BehaviorTests(unittest.TestCase):
    def test_latency_stress_shifts_distribution_without_changing_durations(self):
        from stress_latency import shift_latency_features, shift_caller_turns
        turns = {"turns": [{"channel": 1, "start": 0, "end": 2},
                            {"channel": 0, "start": 4, "end": 5},
                            {"channel": 1, "start": 8, "end": 9},
                            {"channel": 0, "start": 12, "end": 14}]}
        old = extract_model_features(turns)
        new = shift_latency_features(turns, (), 1.5)
        self.assertEqual(new["lat_med"], old["lat_med"] - 1.5)
        self.assertEqual(new["lat_std"], old["lat_std"])
        self.assertEqual(new["cdur_mean"], old["cdur_mean"])
        shifted = shift_caller_turns(turns, 1.5)
        self.assertEqual([t for t in shifted["turns"] if t["channel"] == 1],
                         [t for t in turns["turns"] if t["channel"] == 1])
        self.assertEqual(shifted["turns"][1]["start"], 2.5)

    def test_interruption_then_resume(self):
        f = interruption_recovery([[1, 3], [6, 7]], [[2, 5], [10, 11]])
        self.assertEqual(f["recovery_stop_within_10"], 1)
        self.assertEqual(f["recovery_stop_before_agent_end"], 1)
        self.assertEqual(f["recovery_resume_rate"], 1)
        self.assertEqual(f["recovery_resume_wait_med"], 3)
        self.assertEqual(f["recovery_resume_from_agent_end_med"], 1)

    def test_next_agent_turn_closes_resume_window(self):
        f = interruption_recovery([[1, 3], [8, 9]], [[2, 5], [7, 7.5]])
        self.assertEqual(f["recovery_resume_rate"], 0)
        self.assertTrue(np.isnan(f["recovery_resume_wait_med"]))

    def test_censored_interruption_is_not_failure(self):
        f = interruption_recovery([[1, 5]], [[2, 3]])
        self.assertTrue(np.isnan(f["recovery_resume_rate"]))

    def test_silence_clips_filler_at_agent_return(self):
        f = silence_recovery([[4, 9]], [[0, 2], [8, 10]])
        self.assertEqual(f["silence_fill_wait_med"], 2)
        self.assertEqual(f["silence_fill_duration_med"], 4)
        self.assertEqual(f["silence_fill_whole_turn_med"], 5)
        self.assertAlmostEqual(f["silence_occupied_fraction_med"], 4 / 6)

    def test_no_silence_opportunities_is_missing(self):
        self.assertTrue(all(np.isnan(v) for v in silence_recovery([[2, 3]], [[0, 1], [4, 5]]).values()))

    def test_linear_drift_uses_original_caller_index(self):
        f = drift(np.array([1., 3., 5.]), np.array([1., 2., 3.]))
        self.assertAlmostEqual(f["lat_trend_slope"], .5)
        self.assertAlmostEqual(f["lat_trend_r2"], 1)

    def test_autocorrelation_constant_and_alternating(self):
        self.assertTrue(np.isnan(autocorrelation(np.ones(6))["lat_autocorr_1"]))
        f = autocorrelation(np.array([0., 1., 0., 1., 0., 1.]))
        self.assertAlmostEqual(f["lat_autocorr_1"], -1)
        self.assertAlmostEqual(f["lat_autocorr_2"], 1)

    def test_latency_matches_baseline_overlap_and_filter(self):
        index, value = response_series([[1, 1.5], [4, 5], [20, 21]], [[0, 2]])
        np.testing.assert_array_equal(index, [0, 1])
        np.testing.assert_array_equal(value, [-1, 2])

    def test_entropy_and_feature_names(self):
        self.assertEqual(entropy([2, 2, 2], LATENCY_BINS), 0)
        self.assertTrue(np.isnan(entropy([], LATENCY_BINS)))
        turns = [{"channel": 1, "start": 0, "end": 1}, {"channel": 0, "start": 2, "end": 3}]
        base = extract_model_features(turns)
        extra = extract_blocks(turns)
        self.assertFalse(set(base) & set(extra))
        self.assertEqual(len(extract_model_features(turns, BLOCKS)), len(base) + len(extra))
        with self.assertRaises(ValueError):
            extract_blocks(turns, ["typo"])

    def test_relative_recovery_ignores_global_response_speed(self):
        caller = [[4, 5], [9, 10], [15, 16], [22, 23]]
        agent = [[0, 2], [6, 7], [11, 13], [18, 20], [25, 26]]
        original = relative_recovery(caller, agent)
        faster = relative_recovery([[start - .7, end - .7] for start, end in caller], agent)
        for key in ("relative_latency_mad", "relative_latency_diff_mad",
                    "relative_latency_repeat_100ms", "relative_latency_entropy"):
            self.assertAlmostEqual(original[key], faster[key])

    def test_relative_event_median_is_reliability_weighted(self):
        caller = [[3, 4], [8, 9]]
        agent = [[0, 1], [5, 6], [12, 13]]
        result = relative_recovery(caller, agent)
        self.assertAlmostEqual(result["relative_silence_reliability"], 2 / 5)
        self.assertTrue(np.isfinite(result["relative_silence_delta"]))

    def test_relative_missing_event_is_explicit_zero_reliability(self):
        result = relative_recovery([[2, 3]], [[0, 1], [4, 5]])
        self.assertEqual(result["relative_silence_reliability"], 0)
        self.assertEqual(result["relative_silence_delta"], 0)


if __name__ == "__main__":
    unittest.main()
