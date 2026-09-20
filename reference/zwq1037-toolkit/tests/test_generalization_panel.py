"""Check alignment, time subsets, pooled cosine, gates and Pareto semantics."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generalization_panel as panel
import numpy as np
import pandas as pd


class PanelTests(unittest.TestCase):
    def setUp(self):
        months = np.repeat(np.arange(50, 71), 3)
        self.frame = pd.DataFrame(dict(sample_id=np.arange(len(months)),
                                     month=months, target=np.tile([1., 2., -1.], 21),
                                     prediction=np.tile([1., 1., -1.], 21)))

    def test_cosine_numeric_and_zero(self):
        self.assertAlmostEqual(panel.cosine([1, 0], [1, 1]), 1 / np.sqrt(2))
        self.assertEqual(panel.cosine([1, 0], [0, 0]), 0)
        with self.assertRaises(ValueError):
            panel.cosine([1], [float("nan")])

    def test_alignment_and_mutation(self):
        original = self.frame.copy(deep=True)
        summary, _, _ = panel.evaluate_pair(self.frame, self.frame.iloc[::-1])
        self.assertEqual(summary["cosine_50_59_delta"], 0)
        pd.testing.assert_frame_equal(original, self.frame)
        for column in ("sample_id", "month", "target"):
            bad = self.frame.copy()
            bad.loc[0, column] += 1000
            with self.assertRaises(ValueError):
                panel.evaluate_pair(self.frame, bad)
        with self.assertRaises(ValueError):
            panel.evaluate_pair(self.frame, pd.concat([self.frame, self.frame.iloc[:1]]))

    def test_exact_no66_and_leave_one(self):
        bad = self.frame.copy()
        bad.loc[bad.month == 66, "prediction"] *= -100
        summary, monthly, loo = panel.evaluate_pair(self.frame, bad)
        self.assertEqual(summary["cosine_60_70_no66_delta"], 0)
        self.assertLess(summary["cosine_60_70_delta"], 0)
        months = {row["month"] for row in monthly if row["panel"] == "cosine_60_70_no66"}
        self.assertIn(60, months)
        self.assertIn(61, months)
        self.assertNotIn(66, months)
        gap_months = {row["month"] for row in monthly if row["panel"] == "cosine_62_70_no66"}
        self.assertEqual(gap_months, {62, 63, 64, 65, 67, 68, 69, 70})
        deleted66 = [row for row in loo if row["panel"] == "cosine_60_70" and row["excluded_month"] == 66][0]
        self.assertEqual(deleted66["delta"], 0)
        remaining = bad.loc[(bad.month >= 60) & (bad.month != 61)]
        deleted61 = [row for row in loo if row["panel"] == "cosine_60_70" and row["excluded_month"] == 61][0]
        self.assertAlmostEqual(deleted61["candidate"], panel.cosine(remaining.target, remaining.prediction))

    def test_missing_month_blocks_gate(self):
        short = self.frame.loc[self.frame.month != 60]
        summary, _, _ = panel.evaluate_pair(short, short)
        self.assertIsNone(summary["cosine_60_70_no66"])
        self.assertIsNotNone(summary["cosine_67_70"])
        self.assertFalse(panel.apply_gates(summary, {})["passes_gates"])

    def test_pooled_score_is_not_month_mean(self):
        candidate = self.frame.copy()
        candidate.loc[candidate.month == 50, "prediction"] *= -100
        summary, monthly, _ = panel.evaluate_pair(self.frame, candidate)
        expected = candidate.loc[candidate.month.between(50, 59)]
        self.assertAlmostEqual(summary["cosine_50_59"], panel.cosine(expected.target, expected.prediction))
        macro = np.mean([row["candidate"] for row in monthly if row["panel"] == "cosine_50_59"])
        self.assertNotAlmostEqual(summary["cosine_50_59"], macro)

    def test_gates_and_pareto_tradeoff(self):
        policy = dict(minimum_gain_one_primary=.001, maximum_loo_regression=.001)
        summary = {}
        for key in panel.PRIMARY:
            summary.update({key: .15, key + "_delta": .002, key + "_loo_min_delta": 0})
        self.assertTrue(panel.apply_gates(summary, policy)["passes_gates"])
        summary[panel.PRIMARY[0] + "_delta"] = -.00001
        self.assertFalse(panel.apply_gates(summary, policy)["passes_gates"])
        def row(name, values, group="a"):
            return dict(name=name, group=group, complete=True, passes_gates=True, **dict(zip(panel.PRIMARY, values)))
        rows = [row("base", [1, 1, 1, 1]), row("better", [2, 1, 1, 1]),
                row("tradeoff", [0, 2, 1, 1]), row("other_seed", [100, 100, 100, 100], "b")]
        panel.pareto_labels(rows)
        self.assertEqual(rows[0]["dominated_by"], ["better"])
        self.assertTrue(rows[1]["pareto_nondominated"])
        self.assertTrue(rows[2]["pareto_nondominated"])


if __name__ == "__main__":
    unittest.main()


