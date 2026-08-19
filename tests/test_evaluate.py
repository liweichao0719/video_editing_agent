import unittest

from evaluate import calculate_metrics, interval_iou, match_intervals


class IntervalMetricTests(unittest.TestCase):
    def test_interval_iou(self):
        self.assertAlmostEqual(
            interval_iou({"start": 1.0, "end": 3.0}, {"start": 2.0, "end": 4.0}),
            1 / 3,
        )

    def test_match_reports_complete_coverage(self):
        matches = match_intervals(
            [{"start": 5.0, "end": 6.0}],
            [{"start": 4.5, "end": 6.5}],
        )
        self.assertTrue(matches[0]["complete_coverage"])


class SummaryMetricTests(unittest.TestCase):
    def test_classification_and_temporal_metrics(self):
        results = [
            {
                "status": "ok",
                "expected": True,
                "predicted": True,
                "events": [{"start": 5.0, "end": 6.0}],
                "predictions": [{"start": 4.5, "end": 6.5}],
            },
            {
                "status": "ok",
                "expected": False,
                "predicted": False,
                "events": [],
                "predictions": [],
            },
        ]
        metrics = calculate_metrics(results)
        self.assertEqual(metrics["classification"]["accuracy"], 1.0)
        self.assertEqual(metrics["classification"]["precision"], 1.0)
        self.assertEqual(metrics["temporal"]["complete_coverage_rate"], 1.0)
        self.assertEqual(metrics["temporal"]["mean_iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
