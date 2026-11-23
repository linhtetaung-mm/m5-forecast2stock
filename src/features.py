"""Build direct forecasts from the last completed sales day."""

from __future__ import annotations

from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

from src.data_audit import STORES

KEY = ["store_id", "item_id", "origin"]
DAY = pd.Timedelta(days=1)
CALENDAR = ["date", "weekday", "month", "event_name_1", "event_name_2", "snap_CA", "snap_TX", "snap_WI"]
PRICE_FEATURES = ["asof_price", "relative_price_change", "price_age_days", "price_missing"]
BASELINES = ("naive", "seasonal_naive", "moving_average", "croston", "sba")
DEPARTMENTS = ("FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2")


def _unique(frame, columns, name):
    if frame[columns].isna().any().any() or frame.duplicated(columns).any():
        raise ValueError(f"{name} requires unique non-null keys {columns}.")


def _date(values):
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("Use timezone-free, whole calendar dates.")
    return dates.as_unit("ns")


def phase_origins(phase, label_horizon=1):
    """Give origins whose entire protection window fits inside the label phase."""
    if type(label_horizon) is not int or label_horizon < 1:
        raise ValueError("label_horizon must be a positive integer.")
    start, end = _date([phase["start"], phase["end"]])
    return pd.date_range(start - DAY, end - DAY * label_horizon)


def _croston(sales, gamma):
    forecasts = np.zeros_like(sales, dtype=np.float32)
    for item, row in enumerate(sales):
        positive = np.flatnonzero(row)
        if not positive.size:
            continue
        last = positive[0]
        size = float(row[last])
        interval = float(last + 1)
        for date in positive[1:]:
            forecasts[item, last:date] = size / interval
            interval += gamma * (date - last - interval)
            size += gamma * (float(row[date]) - size)
            last = date
        forecasts[item, last:] = size / interval
    return forecasts


class OriginStore:
    """One store's dense sales history, calendar and optional price vintages."""

    def __init__(self, panel, calendar, prices, config, *, event_codes=None):
        panel = panel.loc[:, ["store_id", "item_id", "dept_id", "date", "sales"]].copy()
        calendar = calendar.loc[:, CALENDAR].copy()
        prices = prices.loc[:, ["store_id", "item_id", "wm_yr_wk", "week_end", "sell_price"]].copy()
        panel["item_id"] = panel["item_id"].astype("str")
        prices["item_id"] = prices["item_id"].astype("str")
        panel["date"] = _date(panel["date"])
        calendar["date"] = _date(calendar["date"])
        prices["week_end"] = _date(prices["week_end"])
        _unique(panel, ["store_id", "item_id", "date"], "Sales")
        _unique(calendar, ["date"], "Calendar")
        _unique(prices, ["store_id", "item_id", "wm_yr_wk"], "Price")
        _unique(prices, ["store_id", "item_id", "week_end"], "Price")
        stores = panel["store_id"].unique()
        if len(stores) != 1 or stores[0] not in STORES:
            raise ValueError("Load one recognized M5 store per OriginStore.")
        self.store = stores[0]
        if not prices["store_id"].eq(self.store).all():
            raise ValueError("Weekly prices belong to a different store.")
        panel = panel.sort_values(["item_id", "date"], ignore_index=True)
        self.items = panel["item_id"].drop_duplicates().to_numpy()
        self.days = pd.date_range(panel["date"].iloc[0], panel["date"].max())
        shape = (len(self.items), len(self.days))
        if len(panel) != shape[0] * shape[1]:
            raise ValueError("A sales date is missing for at least one item.")
        date_matrix = panel["date"].to_numpy().reshape(shape)
        if not np.array_equal(date_matrix, np.broadcast_to(self.days.to_numpy().astype(date_matrix.dtype), shape)):
            raise ValueError("Each item must have the same complete calendar of sales dates.")
        self.sales = panel["sales"].to_numpy(dtype=np.float32).reshape(shape)
        if not np.isfinite(self.sales).all() or (self.sales < 0).any() or (self.sales % 1 != 0).any():
            raise ValueError("Sales must be finite nonnegative integer units.")
        self.depts = panel["dept_id"].to_numpy().reshape(shape)
        if not (self.depts == self.depts[:, :1]).all():
            raise ValueError("Each item must have a stable department.")
        self.depts = np.array([DEPARTMENTS.index(dept) + 1 if dept in DEPARTMENTS else 0 for dept in self.depts[:, 0]])
        self.calendar = calendar.set_index("date").sort_index()
        event_labels = set(calendar["event_name_1"].dropna()) | set(calendar["event_name_2"].dropna())
        self.event_codes = (event_codes if event_codes is not None else
                            {name: index for index, name in enumerate(sorted(event_labels), start=1)})
        for index in (1, 2):
            self.calendar[f"event_code_{index}"] = self.calendar[f"event_name_{index}"].astype("str").map(self.event_codes).fillna(0).astype("int32")
        if not np.isfinite(prices["sell_price"].to_numpy(dtype=float)).all() or prices["sell_price"].le(0).any():
            raise ValueError("Weekly price records require finite positive prices; absent weeks are absent rows.")
        if not prices["item_id"].isin(self.items).all():
            raise ValueError("Weekly prices contain unknown items.")
        self.prices = prices.sort_values(["item_id", "week_end"], ignore_index=True)
        prior_price = self.prices.groupby("item_id")["sell_price"].shift()
        self.prices["relative_price_change"] = self.prices["sell_price"] / prior_price - 1
        self.prices = (
            self.prices.rename(columns={"week_end": "price_week_end", "wm_yr_wk": "price_source_week", "sell_price": "asof_price"})
            .sort_values("price_week_end").drop(columns="store_id")
        )
        self.options = config["forecast"]
        self.cumsum = self._prefix(self.sales)
        self.square_sum = self._prefix(self.sales.astype(np.float64) ** 2)
        self.zero_sum = self._prefix(self.sales == 0)
        positive_date = np.where(self.sales > 0, np.arange(shape[1], dtype=np.int32), -1)
        self.last_sale = np.maximum.accumulate(positive_date, axis=1)
        self.croston = _croston(self.sales, self.options["baselines"]["croston_smoothing"])

    @classmethod
    def from_processed(cls, path: Path, store: str, config, *, event_codes=None):
        return cls(
            pd.read_parquet(path / "panel" / f"{store}.parquet", columns=["store_id", "item_id", "dept_id", "date", "sales"]),
            pd.read_parquet(path / "calendar.parquet", columns=CALENDAR),
            pd.read_parquet(path / "price_history" / f"{store}.parquet"),
            config, event_codes=event_codes,
        )

    @staticmethod
    def _prefix(data):
        sums = np.zeros((data.shape[0], data.shape[1] + 1), dtype=np.float64)
        np.cumsum(data, axis=1, dtype=np.float64, out=sums[:, 1:])
        return sums

    def _window(self, prefix, item, date, width):
        start = date + 1 - width
        value = prefix[item, date + 1] - prefix[item, start.clip(min=0)]
        return np.where(start >= 0, value, np.nan)

    def _lag(self, item, date, lag):
        offset = date - lag
        return np.where(offset >= 0, self.sales[item, offset.clip(min=0)], np.nan)

    def origins(self, dates, *, training_stride=1):
        """Keyed sales/price features, independent of target values and horizon."""
        origin = _date(dates)
        if origin.has_duplicates or not origin.is_monotonic_increasing or not len(origin):
            raise ValueError("Supply nonempty, unique, increasing forecast origins.")
        if type(training_stride) is not int or training_stride < 1:
            raise ValueError("training_stride must be a positive integer.")
        date_indexes = self.days.get_indexer(origin)
        if (date_indexes < 0).any():
            raise ValueError("Every origin requires completed sales for its date.")
        item = np.repeat(np.arange(len(self.items), dtype=np.int32), len(origin))
        date = np.tile(date_indexes.astype(np.int32), len(self.items))
        if training_stride != 1:
            offsets = np.array([crc32(f"{self.store}/{sku}".encode()) % training_stride for sku in self.items])
            sample = (date + offsets[item]) % training_stride == 0
            item, date = item[sample], date[sample]
        settings = self.options["feature_settings"]
        fields = {
            "store_id": self.store, "item_id": self.items[item], "origin": self.days[date],
            "_item": item, "_day": date, "store_code": STORES.index(self.store) + 1,
            "dept_code": self.depts[item],
        }
        for lag in settings["lags"]:
            fields[f"lag_{lag}"] = self._lag(item, date, lag)
        for width in settings["rolling_windows"]:
            total = self._window(self.cumsum, item, date, width)
            squared = self._window(self.square_sum, item, date, width)
            fields[f"mean_{width}"] = (total / width).astype(np.float32)
            fields[f"sd_{width}"] = np.sqrt(np.maximum(0, squared - total**2 / width) / max(1, width - 1)).astype(np.float32)
        zero_width = settings["zero_share_window"]
        fields[f"zero_share_{zero_width}"] = (self._window(self.zero_sum, item, date, zero_width) / zero_width).astype(np.float32)
        last = self.last_sale[item, date]
        fields["days_since_sale"] = np.where(last >= 0, date - last, np.nan).astype(np.float32)
        fields["has_prior_sale"] = last >= 0
        base = pd.DataFrame(fields)
        base = pd.merge_asof(
            base.sort_values("origin"), self.prices,
            left_on="origin", right_on="price_week_end", by="item_id", allow_exact_matches=True,
        ).sort_values(["_item", "_day"], ignore_index=True)
        if (base["price_week_end"] > base["origin"]).any():
            raise AssertionError("A price was attached before its eligibility date.")
        base["price_age_days"] = (base["origin"] - base["price_week_end"]).dt.days
        base["price_missing"] = base["price_week_end"].isna()
        return base

    def design(self, base, horizon, *, family, labels=None, keep_unlabeled=False):
        """Attach known calendar, baselines and optional targets without shifting on sparse origins."""
        if family not in {"daily", "cumulative"} or type(horizon) is not int or horizon < 1:
            raise ValueError("Supply a valid forecast family and positive integer horizon.")
        _unique(base, KEY, "Origins")
        result = base.copy()
        if labels is not None:
            first, last = _date([labels["start"], labels["end"]])
            first_target = result["origin"] + DAY * (horizon if family == "daily" else 1)
            last_target = result["origin"] + DAY * horizon
            eligible = first_target.ge(first) & last_target.le(last)
            if (last_target.loc[eligible] > self.days[-1]).any():
                raise ValueError("The label block extends beyond observed sales.")
            if not keep_unlabeled:
                result = result.loc[eligible].copy()
        dates = pd.DatetimeIndex(result["origin"])
        result["target_date"] = dates + DAY * horizon
        result["horizon"] = horizon
        endpoint = self._target_calendar(result["target_date"])
        for name in ("weekday", "month", "event_code_1", "event_code_2"):
            result[f"target_{name}"] = endpoint[name].to_numpy()
        snap = f"snap_{self.store[:2]}"
        result["target_snap"] = endpoint[snap].to_numpy()
        result["target_event"] = endpoint["event_name_1"].notna().to_numpy() | endpoint["event_name_2"].notna().to_numpy()
        item, date = result["_item"].to_numpy(), result["_day"].to_numpy()
        steps = range(horizon) if family == "cumulative" else [horizon - 1]
        count = len(steps)
        result["naive"] = self.sales[item, date] * count
        average_width = self.options["baselines"]["average_window_days"]
        result["moving_average"] = self._window(self.cumsum, item, date, average_width) * (count / average_width)
        period = self.options["baselines"]["seasonal_period_days"]
        result["seasonal_naive"] = sum(self._lag(item, date, period - 1 - step % period) for step in steps)
        result["croston"] = self.croston[item, date] * count
        result["sba"] = result["croston"] * (1 - self.options["baselines"]["croston_smoothing"] / 2)
        if family == "cumulative":
            path = [self._target_calendar(dates + DAY * (step + 1)) for step in steps]
            result["window_snap_days"] = sum(day[snap].to_numpy(dtype=int) for day in path)
            result["window_weekend_days"] = sum(day["weekday"].ge(5).to_numpy(dtype=int) for day in path)
            result["window_event_days"] = sum(day[["event_name_1", "event_name_2"]].notna().any(axis=1).to_numpy(dtype=int) for day in path)
        if labels is not None:
            valid = eligible.reindex(result.index).to_numpy()
            actual = np.full(len(result), np.nan, dtype=np.float32)
            i, d = item[valid], date[valid]
            actual[valid] = (
                self.sales[i, d + horizon] if family == "daily"
                else self.cumsum[i, d + horizon + 1] - self.cumsum[i, d + 1]
            )
            result["actual"] = actual
        return result.drop(columns=["_item", "_day"]).reset_index(drop=True)

    def _target_calendar(self, dates):
        index = pd.DatetimeIndex(dates)
        if not index.isin(self.calendar.index).all():
            raise ValueError("The known calendar does not cover every requested target day.")
        return self.calendar.reindex(index)


def model_features(design, *, use_price):
    """No label, raw key, or price provenance fields may enter the estimator."""
    candidates = design.columns
    columns = [
        column for column in candidates
        if column.startswith(("lag_", "mean_", "sd_", "zero_share_", "target_", "window_"))
        and column != "target_date"
        or column in {"store_code", "dept_code", "days_since_sale", "has_prior_sale", "horizon"}
    ]
    if use_price:
        columns.extend(PRICE_FEATURES)
    return columns



def export_features(config_path, phase_name, output_dir):
    """Materialize a label-free Parquet snapshot for a configured split."""
    from tempfile import TemporaryDirectory

    from src.data_audit import load_config

    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    processed = (config_path.parent.parent / config["data"]["processed_dir"]).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() or destination.is_relative_to(processed) or processed.is_relative_to(destination):
        raise ValueError("Choose a new destination separate from the Part 1 data.")
    families = {family: config["forecast"][f"{family}_horizons"] for family in ("daily", "cumulative")}
    origins = phase_origins(config["splits"][phase_name], max(map(max, families.values())))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".features-", dir=destination.parent) as scratch:
        staging = Path(scratch) / "snapshot"
        for name in config["data"]["stores"]:
            store = OriginStore.from_processed(processed, name, config)
            base = store.origins(origins)
            for family, horizons in families.items():
                folder = staging / name / family
                folder.mkdir(parents=True)
                for horizon in horizons:
                    design = store.design(base, horizon, family=family)
                    design.to_parquet(folder / f"horizon_{horizon:02d}.parquet", index=False)
        staging.rename(destination)
    return destination


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--phase", required=True, choices=["training", "development", "calibration", "policy_validation", "warmup", "final_test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(export_features(args.config, args.phase, args.output_dir))


if __name__ == "__main__":
    main()
