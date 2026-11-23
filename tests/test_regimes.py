"""Verify the training boundary, regime edge cases, and descriptive EDA."""

import unittest

import numpy as np
import pandas as pd

from src.regimes import _eda_tables, _sensitivity, _training_panel, classify


START = "2015-01-01"
END = "2015-01-05"


def make_panel() -> pd.DataFrame:
    examples = {
        ("CA_1", "stable"): [0, 0, 3, 3, 3],
        ("CA_1", "volatile"): [1, 3, 0, 0, 0],
        ("CA_1", "cv_edge"): [0, 0, 1, 3, 0],
        ("CA_1", "single"): [0, 0, 0, 0, 4],
        ("CA_1", "none"): [0, 0, 0, 0, 0],
        ("TX_1", "stable"): [8, 1, 1, 1, 1],
    }
    rows = []
    for (store, sku), sales in examples.items():
        for date, units in zip(pd.date_range(START, END), sales, strict=True):
            rows.append(
                {
                    "store_id": store,
                    "item_id": sku,
                    "id": f"{sku}_{store}_validation",
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "state_id": store[:2],
                    "date": date,
                    "sales": units,
                    "price_missing": units == 0,
                    "weekday": date.weekday(),
                    "snap": date.day % 2 == 0,
                    "has_event": date.day == 2,
                }
            )
    return pd.DataFrame(rows)


class RegimeTests(unittest.TestCase):
    def test_rules_and_boundary_conditions(self) -> None:
        summary = classify(make_panel(), start=START, end=END, cv2_threshold=0.5)
        summary = summary.set_index(["store_id", "item_id"])
        stable = summary.loc[("CA_1", "stable")]
        self.assertEqual(stable["regime"], "Stable")
        self.assertEqual(stable["training_days"], 5)
        self.assertEqual(stable["first_positive_date"], pd.Timestamp("2015-01-03"))
        self.assertEqual(stable["active_days"], 3)
        self.assertAlmostEqual(stable["full_zero_share"], 0.4)
        self.assertEqual(stable["active_zero_share"], 0)
        self.assertEqual(stable["adi"], 1)

        zero_edge = summary.loc[("CA_1", "volatile")]
        self.assertAlmostEqual(zero_edge["active_zero_share"], 0.6)
        self.assertAlmostEqual(zero_edge["adi"], 2.5)
        self.assertEqual(zero_edge["regime"], "Intermittent")
        self.assertFalse(zero_edge["insufficient_history"])

        cv_edge = summary.loc[("CA_1", "cv_edge")]
        self.assertAlmostEqual(cv_edge["positive_mean"], 2)
        self.assertAlmostEqual(cv_edge["positive_cv2"], 0.5)
        self.assertEqual(cv_edge["regime"], "Volatile")
        self.assertEqual(summary.loc[("TX_1", "stable"), "regime"], "Volatile")

        single = summary.loc[("CA_1", "single")]
        self.assertEqual(single["regime"], "Intermittent")
        self.assertTrue(single["insufficient_history"])
        self.assertEqual(single["active_zero_share"], 0)
        self.assertTrue(pd.isna(single["positive_cv2"]))

        none = summary.loc[("CA_1", "none")]
        self.assertEqual(none["regime"], "Intermittent")
        self.assertEqual(none["active_days"], 0)
        self.assertEqual(none["full_zero_share"], 1)
        self.assertTrue(pd.isna(none["active_zero_share"]))
        self.assertTrue(pd.isna(none["adi"]))

    def test_changes_outside_training_cannot_change_regimes_or_eda(self) -> None:
        training = make_panel()
        other_rows = training.copy()
        other_rows["date"] += pd.Timedelta(days=7)
        original = pd.concat([training, other_rows], ignore_index=True)
        perturbed = original.copy()
        future = perturbed["date"].gt(pd.Timestamp(END))
        for column in ("store_id", "item_id", "id", "dept_id", "cat_id", "state_id"):
            perturbed.loc[future, column] = "__changed__"
        perturbed.loc[future, "sales"] = -10000
        perturbed.loc[future, "weekday"] = 99
        for column in ("price_missing", "snap", "has_event"):
            perturbed.loc[future, column] = ~perturbed.loc[future, column]
        perturbed.loc[future, "date"] += pd.Timedelta(days=2)
        self.assertTrue(perturbed.loc[future, "date"].gt(pd.Timestamp(END)).all())

        pd.testing.assert_frame_equal(
            classify(original, start=START, end=END),
            classify(perturbed, start=START, end=END),
        )
        start, end = pd.Timestamp(START), pd.Timestamp(END)
        original_eda = _eda_tables(_training_panel(original, start, end))
        perturbed_eda = _eda_tables(_training_panel(perturbed, start, end))
        for label in original_eda:
            pd.testing.assert_frame_equal(original_eda[label], perturbed_eda[label])

    def test_validation_rejects_gaps_duplicates_and_bad_sales(self) -> None:
        panel = make_panel()
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            classify(panel.iloc[1:], start=START, end=END)
        twice = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            classify(twice, start=START, end=END)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            classify(panel.assign(sales=-1), start=START, end=END)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            classify(panel.assign(sales=np.nan), start=START, end=END)
        with self.assertRaisesRegex(ValueError, "zero_share_threshold"):
            classify(panel, start=START, end=END, zero_share_threshold=1.1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            classify(panel, start=START, end=END, min_positive_days=1)

    def test_threshold_grid_preserves_every_sku_and_eda_tracks_price_gaps(self) -> None:
        panel = make_panel()
        panel.loc[panel["sales"].eq(4), "price_missing"] = True
        regimes = classify(panel, start=START, end=END)
        sensitivity = _sensitivity(regimes, [0.60, 0.80], [0.40, 0.49])
        sums = sensitivity.groupby(["store_id", "zero_share_threshold", "cv2_threshold"])["sku_count"].sum()
        self.assertEqual(len(sensitivity), 2 * 2 * 2 * 3)
        self.assertTrue(sums.loc["CA_1"].eq(5).all())
        self.assertTrue(sums.loc["TX_1"].eq(1).all())
        for group, counts in sensitivity.groupby(["store_id", "zero_share_threshold", "cv2_threshold"]):
            if group[1:] == (0.60, 0.49):
                expected = regimes.loc[regimes["store_id"].eq(group[0]), "regime"].value_counts()
                for row in counts.itertuples(index=False):
                    self.assertEqual(row.sku_count, expected.get(row.regime, 0))

        eda = _eda_tables(panel)
        ca = eda["store"].set_index("store_id").loc["CA_1"]
        self.assertEqual(ca["item_days"], 25)
        self.assertEqual(ca["positive_missing_price_rows"], 1)
        self.assertAlmostEqual(ca["missing_price_share"], (ca["zero_rows"] + 1) / 25)
        self.assertEqual(eda["context"]["item_days"].sum(), len(panel))


if __name__ == "__main__":
    unittest.main()
