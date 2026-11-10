"""Scientific protocol invariants, rather than checks of a particular model score."""

import unittest
from unittest.mock import patch

from river_lab.evaluation import evaluate, match_drifts


class ProtocolTests(unittest.TestCase):
    def test_delay_matching_keeps_misses_and_extra_alarms(self):
        matches, extras = match_drifts([10, 101, 103, 999], [100, 500], 20)
        self.assertEqual(matches[0]["delay"], 1)
        self.assertIsNone(matches[1]["delay"])
        self.assertEqual(extras, 3)

    def test_prediction_precedes_current_label_and_reset(self):
        calls = []

        class Learner:
            def __init__(self, generation):
                self.value = 0
                self.generation = generation

            def predict_one(self, x):
                calls.append(("predict", self.generation, self.value))
                return self.value

            def learn_one(self, x, y):
                calls.append(("learn", self.generation, y))
                self.value = y

        class Detector:
            n = 0
            drift_detected = False

            def update(self, signal):
                self.n += 1
                self.drift_detected = self.n == 2

        generations = []

        def factory(*args):
            learner = Learner(len(generations))
            generations.append(learner)
            return learner

        config = {"rolling_window": 2, "checkpoint_every": 1, "detector_warmup": 0, "drift_match_window": 2}
        item = {"task": "regression", "dataset": "test", "n_samples": 3, "known_drift_positions": []}
        with patch("river_lab.evaluation.make_model", side_effect=factory), patch("river_lab.evaluation.make_detector", return_value=Detector()):
            result = evaluate([({"x": 0}, 10), ({"x": 1}, 20), ({"x": 2}, 30)], item, "HoeffdingTree", 42, config, "ADWIN")
        self.assertEqual(calls, [("predict", 0, 0), ("learn", 0, 10), ("predict", 0, 10), ("learn", 1, 20), ("predict", 1, 20), ("learn", 1, 30)])
        self.assertEqual(result["summary"]["mae"], 10)
        self.assertEqual(result["events"][0]["position"], 1)
        self.assertEqual(result["summary"]["n_resets"], 1)


if __name__ == "__main__":
    unittest.main()
