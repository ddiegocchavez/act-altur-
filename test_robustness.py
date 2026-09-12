import unittest

import numpy as np

from phase4_robust import (EVALUATION_SHIFTS, TRAIN_SHIFT_INTERVALS, calibration_diagnostic,
                           expected_calibration_error, jitter_turns, sample_training_shift)


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

    def test_calibration_diagnostic_is_finite_and_explicitly_non_deployment(self):
        result = calibration_diagnostic([0, 0, 1, 1], [.1, .3, .7, .9])
        self.assertEqual(result["status"], "diagnostic_only")
        self.assertTrue(np.isfinite(result["platt_in_sample"]["log_loss"]))
        self.assertIn("not used", result["limitation"])

    def test_ece_perfect_probabilities(self):
        self.assertEqual(expected_calibration_error([0, 1], [0, 1]), 0)


if __name__ == "__main__":
    unittest.main()
