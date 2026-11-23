import unittest

import numpy as np
import pandas as pd

from src.features import OriginStore, model_features, phase_origins
from src.metrics import enforce_quantiles, evaluate, pinball, quantile_columns, training_scales
from src.models import calibrate, fit_quantiles, predict_quantiles, prediction_table


CONFIG = {
    "forecast": {
        "feature_settings": {"lags": [0, 1, 7, 14, 28], "rolling_windows": [7, 14, 28], "zero_share_window": 28},
        "baselines": {"average_window_days": 28, "seasonal_period_days": 7, "croston_smoothing": 0.10},
    },
}
ALPHAS = [0.50, 0.75, 0.90, 0.95]


def fixture():
    days = pd.date_range("2024-01-01", periods=70)
    calendar = pd.DataFrame({
        "date": days, "weekday": days.weekday, "month": days.month,
        "event_name_1": pd.Categorical(np.where(days.day == 5, "Event", None)), "event_name_2": None,
        "snap_CA": days.day <= 10, "snap_TX": False, "snap_WI": False,
    })
    panel = pd.DataFrame({
        "store_id": "CA_1", "item_id": "SKU_1", "dept_id": "FOODS_1",
        "date": days[:60], "sales": np.arange(60) % 10,
    })
    prices = pd.DataFrame({
        "store_id": "CA_1", "item_id": "SKU_1", "wm_yr_wk": [1, 2, 3],
        "week_end": days[[27, 34, 41]], "sell_price": [2.0, 3.0, 5.0],
    })
    return panel, calendar, prices, days


class ForecastingTest(unittest.TestCase):
    def test_sparse_origins_and_future_sales_price_perturbation(self):
        panel, calendar, prices, days = fixture()
        wanted = days[[31, 33]]
        original = OriginStore(panel, calendar, prices, CONFIG)
        small = original.origins(wanted)
        full = original.origins(days[28:38]).loc[lambda f: f["origin"].isin(wanted)].reset_index(drop=True)
        pd.testing.assert_frame_equal(small, full)
        self.assertEqual(small["lag_0"].tolist(), [1, 3])
        self.assertEqual(small["lag_7"].tolist(), [4, 6])
        self.assertEqual(small["asof_price"].tolist(), [2, 2])
        self.assertEqual(small["price_age_days"].tolist(), [4, 6])
        self.assertEqual(original.origins(days[[34]]).iloc[0]["relative_price_change"], 0.5)
        panel.loc[panel["date"] > wanted[-1], "sales"] = 99999
        prices.loc[prices["week_end"] > wanted[-1], "sell_price"] = 100
        changed = OriginStore(panel, calendar, prices, CONFIG)
        pd.testing.assert_frame_equal(small, changed.origins(wanted))
        for family in ("daily", "cumulative"):
            a = original.design(small, 8, family=family)
            b = changed.design(changed.origins(wanted), 8, family=family)
            pd.testing.assert_frame_equal(a, b)
            self.assertNotIn("price_week_end", model_features(a, use_price=True))
            self.assertNotIn("asof_price", model_features(a, use_price=False))
            self.assertNotIn("actual", a.columns)

    def test_complete_windows_and_replay_tail(self):
        panel, calendar, prices, days = fixture()
        store = OriginStore(panel, calendar, prices, CONFIG)
        block = {"start": days[51], "end": days[59]}
        self.assertEqual(len(phase_origins(block, 8)), 2)
        origins = store.origins(phase_origins(block))
        complete = store.design(origins, 8, family="cumulative", labels=block)
        replay = store.design(origins, 8, family="cumulative", labels=block, keep_unlabeled=True)
        self.assertEqual(complete["actual"].tolist(), [36, 44])
        self.assertEqual(int(replay["actual"].notna().sum()), 2)
        self.assertEqual(replay["target_date"].max(), days[66])
        daily = store.design(origins, 2, family="daily", labels=block)
        self.assertEqual(daily.iloc[0]["actual"], 2)
        with self.assertRaisesRegex(ValueError, "unique"):
            OriginStore(pd.concat([panel, panel.iloc[:1]]), calendar, prices, CONFIG)

    def test_no_future_scales_and_undefined_scale(self):
        panel, calendar, prices, days = fixture()
        panel["sales"] = 2
        phase = {"start": days[10], "end": days[25]}
        store = OriginStore(panel, calendar, prices, CONFIG)
        original = training_scales(store, phase, [2, 8])
        self.assertEqual(original["daily", 1]["sq_scale"].item(), 0)
        panel.loc[panel["date"] > phase["end"], "sales"] = 1000
        changed = training_scales(OriginStore(panel, calendar, prices, CONFIG), phase, [2, 8])
        for key in original:
            pd.testing.assert_frame_equal(original[key], changed[key])

    def test_quantile_training_calibration_and_metrics(self):
        panel, calendar, prices, days = fixture()
        store = OriginStore(panel, calendar, prices, CONFIG)
        phase = {"start": days[30], "end": days[50]}
        design = store.design(store.origins(phase_origins(phase, 2)), 2, family="cumulative", labels=phase)
        design["regime"] = "Stable"
        models = fit_quantiles(design, ALPHAS, {
            "num_boost_round": 2, "num_threads": 1, "min_data_in_leaf": 2,
        }, use_price=True, seed=42)
        raw = predict_quantiles(design, models)
        self.assertEqual(raw.shape, (20, 4))
        calibration = calibrate(design, raw, ALPHAS)
        raw_for_test = raw[:, ::-1].copy()
        forecast = prediction_table(design, "lightgbm_with_price", ALPHAS, raw=raw_for_test, calibration=calibration, family="cumulative")
        np.testing.assert_array_equal(forecast[quantile_columns(ALPHAS, "raw")].to_numpy(), raw_for_test.astype(np.float32))
        self.assertTrue((np.diff(forecast[quantile_columns(ALPHAS)], axis=1) >= 0).all())
        scored = evaluate(forecast.assign(abs_scale=0.0, sq_scale=0.0), ALPHAS)
        self.assertTrue(scored["rmsse"].isna().all())
        self.assertTrue(scored["spl"].isna().all())
        self.assertTrue(scored["spl_undefined_series"].eq(1).all())
        self.assertTrue(scored["coverage_strict"].le(scored["coverage_inclusive"]).all())
        np.testing.assert_allclose(pinball([0, 3], [2, 2], 0.75), [0.5, 0.75])
        np.testing.assert_array_equal(enforce_quantiles([[3, -1, 2, 0]]), [[0, 0, 2, 3]])


if __name__ == "__main__":
    unittest.main()

