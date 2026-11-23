import unittest

import numpy as np
import pandas as pd

from src.inference import snapshot
from src.inventory_sim import Candidate, frontier, interpolate, opening_state, reference_prices, simulate, stock_target


class InventoryTests(unittest.TestCase):
    def test_one_day_receipt_is_not_same_day_and_units_conserve(self):
        opening = opening_state([0], 1)
        totals, close, ledger = simulate([[5], [3], [8]], [[6], [7], [8]], opening, capture=True)
        self.assertEqual([day["receipts"].item() for day in ledger], [0, 6, 1])
        self.assertEqual([day["ordered"].item() for day in ledger], [6, 1, 4])
        self.assertEqual(totals["lost"].item(), 9)
        self.assertEqual(close.inbound.item(), 4)
        self.assertEqual(opening.inbound.item(), 0)

    def test_seven_day_receipt_and_cutover(self):
        opening = opening_state([0], 7)
        demand = np.zeros((9, 1), dtype=int)
        demand[7] = 12
        targets = np.full((9, 1), 10.0)
        continuous, _, detail = simulate(demand, targets, opening, capture=True)
        self.assertEqual([entry["receipts"].item() for entry in detail], [0] * 7 + [10, 0])
        self.assertEqual(continuous["lost"].item(), 2)
        before, state, _ = simulate(demand[:4], targets[:4], opening)
        after, final, _ = simulate(demand[4:], targets[4:], state)
        for key in continuous:
            np.testing.assert_array_equal(before[key] + after[key], continuous[key])
        self.assertEqual(final.inbound.sum(), 10)
        self.assertEqual(state.inbound.sum(), 10)

    def test_units_targets_and_receipt_contract(self):
        state = opening_state([1, 2], 1)
        for demand, target in [([[0, -1]], [[1, 2]]), ([[0, 0.5]], [[1, 2]]),
                               ([[0, 2]], [[1, np.nan]]), ([[0, 2]], [[1]])]:
            with self.assertRaises(ValueError):
                simulate(demand, target, state)
        _, _, ledger = simulate([[1, 0]], [[1.1, 2.0]], state, capture=True)
        self.assertEqual(ledger[0]["ordered"].tolist(), [1, 0])

    def test_critical_levels_buffers_and_pareto(self):
        alphas = [.5, .75, .9, .95]
        grid = np.array([[1.0, 3, 6, 8], [3.0, 5, 6, 8]])
        self.assertEqual(interpolate(.999, alphas, grid).tolist(), [8, 8])
        self.assertEqual(interpolate(.3, alphas, grid).tolist(), [1, 3])
        np.testing.assert_allclose(interpolate(.625, alphas, grid), [2, 4])
        out = stock_target(Candidate("naive", "own_buffer", .5), alphas, np.array([1, 2]), grid, grid - 4)
        self.assertEqual(out.tolist(), [1, 2])
        ranked = pd.DataFrame({"fill_rate": [.99, .99, .98, .96], "cost": [11, 12, 15, 8]})
        self.assertEqual(frontier(ranked).index.tolist(), [0, 3])

    def test_reference_prices_never_use_future_and_report_fallback(self):
        cohort = pd.DataFrame({"store_id": "CA_1", "item_id": ["a", "b", "c"], "dept_id": ["A", "A", "C"]})
        prices = pd.DataFrame({"store_id": "CA_1", "item_id": ["a", "a", "b"], "sell_price": [3., 500., 500.],
                               "week_end": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-12"])})
        result = reference_prices(prices, cohort, "2024-01-06", 3)
        self.assertEqual(result["reference_price"].tolist(), [3., 3., 3.])
        self.assertEqual(result["price_source"].tolist(), ["item_median", "department_median", "store_median"])

    def test_batch_snapshot_freshness_and_review_time(self):
        stock = pd.DataFrame({"store_id": "CA_1", "item_id": ["a", "b"], "as_of": "2024-01-01", "on_hand": [2, 0]})
        firm = pd.DataFrame({"store_id": "CA_1", "item_id": ["a", "a"], "as_of": "2024-01-01",
                             "due_date": ["2024-01-02", "2024-01-06"], "units": [3, 4]})
        inputs = snapshot(stock, firm, ["a", "b"], "CA_1", "2024-01-01", 7)
        self.assertEqual(inputs["due_on_review"].tolist(), [3, 0])
        self.assertEqual(inputs["outstanding_after_review_receipts"].tolist(), [4, 0])
        with self.assertRaisesRegex(ValueError, "late receipts"):
            snapshot(stock, firm, ["a", "b"], "CA_1", "2024-01-01", 1)
        with self.assertRaisesRegex(ValueError, "as-of"):
            snapshot(stock, firm, ["a", "b"], "CA_1", "2024-01-02", 7)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            snapshot(stock.iloc[:1], firm, ["a", "b"], "CA_1", "2024-01-01", 7)


if __name__ == "__main__":
    unittest.main()

