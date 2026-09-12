import unittest
from unittest.mock import patch

import numpy as np

from phase4_robust import (EVALUATION_SHIFTS, TRAIN_SHIFT_INTERVALS,
                           TRAIN_TRUNCATION_INTERVALS_S, calibration_diagnostic, candidate_gates,
                           audio_promotion_gate, expected_calibration_error, jitter_turns, sample_training_shift,
                           truncate_turns)


class RobustnessTests(unittest.TestCase):
    def test_training_shifts_hold_out_fixed_evaluation_neighborhoods(self):
        rng = np.random.default_rng(7)
        values = [sample_training_shift(rng) for _ in range(5000)]
        self.assertTrue(all(any(left <= value <= right for left, right in TRAIN_SHIFT_INTERVALS)
                            for value in values))
        for fixed in EVALUATION_SHIFTS[1:]:
            self.assertTrue(all(abs(value - fixed) >= .1 - 1e-12 for value in values))

    def test_jitter_preserves_channels_and_positive_intervals(self):
        payload = {"turns": [{"channel": 1, "start": 0, "end": 1},
                             {"channel": 0, "start": 2, "end": 2.1}]}
        result = jitter_turns(payload, np.random.default_rng(9))
        self.assertEqual(sorted(turn["channel"] for turn in result["turns"]), [0, 1])
        self.assertTrue(all(turn["start"] >= 0 and turn["end"] > turn["start"]
                            for turn in result["turns"]))

    def test_training_truncations_hold_out_evaluation_neighborhoods(self):
        for left, right in TRAIN_TRUNCATION_INTERVALS_S:
            self.assertFalse(left <= 30 <= right)
            self.assertFalse(left <= 60 <= right)
        self.assertLessEqual(TRAIN_TRUNCATION_INTERVALS_S[0][1], 28)
        self.assertGreaterEqual(TRAIN_TRUNCATION_INTERVALS_S[1][0], 32)
        self.assertLessEqual(TRAIN_TRUNCATION_INTERVALS_S[1][1], 55)
        self.assertGreaterEqual(TRAIN_TRUNCATION_INTERVALS_S[2][0], 65)

    def test_calibration_diagnostic_is_finite_and_explicitly_non_deployment(self):
        result = calibration_diagnostic([0, 0, 1, 1], [.1, .3, .7, .9])
        self.assertEqual(result["status"], "diagnostic_only")
        self.assertTrue(np.isfinite(result["platt_in_sample"]["log_loss"]))
        self.assertIn("not used", result["limitation"])

    def test_ece_perfect_probabilities(self):
        self.assertEqual(expected_calibration_error([0, 1], [0, 1]), 0)

    def test_truncate_turns_drops_future_and_clips_crossing_turn(self):
        payload = {"turns": [{"channel": 1, "start": 10, "end": 31},
                             {"channel": 0, "start": 29, "end": 35},
                             {"channel": 1, "start": 30, "end": 32}]}
        result = truncate_turns(payload, 30)
        self.assertEqual(result, {"turns": [
            {"channel": 1, "start": 10., "end": 30},
            {"channel": 0, "start": 29., "end": 30},
        ]})

    def test_candidate_must_balance_every_scenario(self):
        champion = [
            {"scenario": "clean", "correct": 69},
            {"scenario": "acceleration_1s", "correct": 45},
            {"scenario": "truncate_30s", "correct": 59},
        ]
        candidate = [
            {"scenario": "clean", "correct": 69},
            {"scenario": "acceleration_1s", "correct": 66},
            {"scenario": "truncate_30s", "correct": 48},
        ]
        gates = candidate_gates(candidate, champion)
        self.assertTrue(gates["clean_gate_passed"])
        self.assertTrue(gates["timing_robustness_gate_passed"])
        self.assertFalse(gates["balance_gate_passed"])

    def test_audio_gate_rejects_two_call_regression(self):
        champion = {"status": "complete", "model_sha256": "champion", "scenarios": {
            "clean": {"correct": 69}, "truncate_30s": {"correct": 59}}}
        candidate = {"status": "complete", "model_sha256": "candidate", "scenarios": {
            "clean": {"correct": 69}, "truncate_30s": {"correct": 57}}}
        mock_paths = {
            "reports/audio_robustness_champion.json": champion,
            "reports/audio_robustness_candidate.json": candidate,
        }

        class MemoryPath:
            def __init__(self, name):
                self.name = name

            def exists(self):
                return self.name in mock_paths

            def read_text(self):
                import json
                return json.dumps(mock_paths[self.name])

            def __str__(self):
                return self.name

        with patch("phase4_robust.Path", MemoryPath):
            result = audio_promotion_gate({"artifact_sha256": "candidate"}, "champion")
        self.assertFalse(result["passed"])
        self.assertEqual(result["scenario_correct_deltas"]["truncate_30s"], -2)


if __name__ == "__main__":
    unittest.main()
