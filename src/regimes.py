"""Characterize observed M5 sales using the configured training dates only."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


SKU_KEYS = ("store_id", "item_id")
SKU_METADATA = ("id", "dept_id", "cat_id", "state_id")
REGIME_LABELS = ("Stable", "Volatile", "Intermittent")
EDA_GROUPS = {
    "store": ("store_id",),
    "category": ("store_id", "cat_id"),
    "weekday": ("store_id", "weekday"),
    "context": ("store_id", "state_id", "snap", "has_event"),
}
EDA_INPUTS = ("cat_id", "state_id", "weekday", "snap", "has_event", "price_missing")


def _training_bounds(start: str | pd.Timestamp, end: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_day, end_day = pd.Timestamp(start), pd.Timestamp(end)
    if any(pd.isna(day) or day.tzinfo is not None or day != day.normalize() for day in (start_day, end_day)):
        raise ValueError("Training bounds must be valid timezone-free calendar dates.")
    if start_day > end_day:
        raise ValueError("Training start must be on or before training end.")
    return start_day, end_day


def _training_panel(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = {*SKU_KEYS, "date", "sales"}
    if missing := required.difference(panel.columns):
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")

    dates = pd.to_datetime(panel["date"])
    selected = dates.between(start, end)
    training = panel.loc[selected].copy()
    if training.empty:
        raise ValueError("Panel has no rows in the requested training dates.")
    training["date"] = dates.loc[selected]
    if not training["date"].eq(training["date"].dt.normalize()).all():
        raise ValueError("Training rows must have calendar dates at midnight.")
    if training[list(SKU_KEYS)].isna().any().any():
        raise ValueError("Training rows must have a store_id and an item_id.")
    if training.duplicated([*SKU_KEYS, "date"]).any():
        raise ValueError("Duplicate store, item, and date in training rows.")
    counts = training.groupby(list(SKU_KEYS), sort=False, observed=True).size()
    if not counts.eq((end - start).days + 1).all():
        raise ValueError("Every training SKU must have exactly one row per training calendar day.")

    sales = pd.to_numeric(training["sales"], errors="raise")
    values = sales.to_numpy(dtype=np.float64, na_value=np.nan)
    if not (np.isfinite(values) & (values >= 0) & (values == np.floor(values))).all():
        raise ValueError("Training sales must be finite, nonnegative whole units.")
    training["sales"] = sales
    return training


def _validate_cutoffs(zero_share_threshold: float, cv2_threshold: float) -> None:
    if not np.isfinite(zero_share_threshold) or not 0 <= zero_share_threshold <= 1:
        raise ValueError("zero_share_threshold must be finite and in [0, 1].")
    if not np.isfinite(cv2_threshold) or cv2_threshold < 0:
        raise ValueError("cv2_threshold must be finite and nonnegative.")


def _assign_regime(
    summary: pd.DataFrame, zero_share_threshold: float, cv2_threshold: float
) -> np.ndarray:
    intermittent = summary["insufficient_history"] | summary["active_zero_share"].ge(zero_share_threshold)
    volatile = summary["positive_cv2"].ge(cv2_threshold)
    return np.select([intermittent, volatile], ["Intermittent", "Volatile"], default="Stable")


def _classify_training(
    training: pd.DataFrame,
    end: pd.Timestamp,
    zero_share_threshold: float,
    cv2_threshold: float,
    min_positive_days: int,
) -> pd.DataFrame:
    _validate_cutoffs(zero_share_threshold, cv2_threshold)
    if (
        isinstance(min_positive_days, (bool, np.bool_))
        or not isinstance(min_positive_days, (int, np.integer))
        or min_positive_days < 2
    ):
        raise ValueError("min_positive_days must be an integer of at least two.")

    key_columns = list(SKU_KEYS)
    grouped = training.groupby(key_columns, sort=False, observed=True)
    metadata = [column for column in SKU_METADATA if column in training]
    if metadata and grouped[metadata].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Training SKU metadata must remain constant within each store and item.")
    series = grouped.size().rename("training_days").to_frame()
    if metadata:
        series = grouped[metadata].first().join(series)
    positive_sales = training.loc[training["sales"].gt(0), [*SKU_KEYS, "date", "sales"]]
    positives = positive_sales.groupby(key_columns, sort=False, observed=True).agg(
        positive_days=("sales", "size"),
        first_positive_date=("date", "min"),
        positive_mean=("sales", "mean"),
        positive_variance=("sales", "var"),
    )
    series = series.join(positives)
    series["positive_days"] = series["positive_days"].fillna(0).astype("int64")
    positive_days = series["positive_days"]
    active_days = ((end - series["first_positive_date"]).dt.days + 1).fillna(0).astype("int64")
    series["active_days"] = active_days
    series["full_zero_share"] = (series["training_days"] - positive_days) / series["training_days"]
    series["active_zero_share"] = (active_days - positive_days) / active_days.where(active_days.gt(0))
    series["adi"] = active_days / positive_days.where(positive_days.gt(0))
    series["positive_cv2"] = series.pop("positive_variance") / series["positive_mean"].pow(2)
    series["insufficient_history"] = positive_days.lt(min_positive_days)
    series["regime"] = _assign_regime(series, zero_share_threshold, cv2_threshold)
    return series.reset_index().sort_values(key_columns, ignore_index=True)


def classify(
    panel: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    zero_share_threshold: float = 0.60,
    cv2_threshold: float = 0.49,
    min_positive_days: int = 2,
) -> pd.DataFrame:
    """Return one regime row per training store and item over inclusive bounds.

    A full panel may be supplied. Sales and metadata outside the training dates
    do not contribute to validation, estimates, or labels.
    """
    start_day, end_day = _training_bounds(start, end)
    training = _training_panel(panel, start_day, end_day)
    return _classify_training(training, end_day, zero_share_threshold, cv2_threshold, min_positive_days)


def _eda_tables(training: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if missing := set(EDA_INPUTS).difference(training.columns):
        raise ValueError(f"Panel is missing training EDA columns: {sorted(missing)}")
    for flag in ("snap", "has_event", "price_missing"):
        if not training[flag].isin([True, False]).all():
            raise ValueError(f"Training {flag} values must be true or false.")
    if not training["weekday"].eq(training["date"].dt.weekday).all():
        raise ValueError("Training weekday must match date, with Monday equal to 0.")
    if training[["cat_id", "state_id"]].isna().any().any():
        raise ValueError("Training categorization requires cat_id and state_id on every row.")

    columns = ["store_id", "cat_id", "state_id", "weekday", "snap", "has_event", "sales", "price_missing"]
    data = training.loc[:, columns].copy()
    data["zero"] = data["sales"].eq(0)
    data["positive_units"] = data["sales"].where(~data["zero"])
    data["positive_missing_price"] = ~data["zero"] & data["price_missing"].astype(bool)
    tables = {}
    for level, keys in EDA_GROUPS.items():
        groups = data.groupby(list(keys), sort=True, observed=True)
        metrics = groups.agg(
            item_days=("sales", "size"),
            sales_units=("sales", "sum"),
            mean_units=("sales", "mean"),
            sales_skewness=("sales", "skew"),
            zero_rows=("zero", "sum"),
            positive_rows=("positive_units", "count"),
            positive_mean=("positive_units", "mean"),
            positive_median=("positive_units", "median"),
            positive_skewness=("positive_units", "skew"),
            missing_price_rows=("price_missing", "sum"),
            positive_missing_price_rows=("positive_missing_price", "sum"),
            distinct_sales_sizes=("sales", "nunique"),
            distinct_positive_sizes=("positive_units", "nunique"),
        )
        metrics["sales_skewness"] = metrics["sales_skewness"].where(metrics.pop("distinct_sales_sizes").gt(1))
        metrics["positive_skewness"] = metrics["positive_skewness"].where(
            metrics.pop("distinct_positive_sizes").gt(1)
        )
        metrics["zero_share"] = metrics["zero_rows"] / metrics["item_days"]
        metrics["missing_price_share"] = metrics["missing_price_rows"] / metrics["item_days"]
        tables[level] = metrics.reset_index()
    return tables


def _sensitivity(
    regimes: pd.DataFrame, zero_thresholds: list[float], cv2_thresholds: list[float]
) -> pd.DataFrame:
    if not zero_thresholds or not cv2_thresholds:
        raise ValueError("Both sensitivity threshold lists must be nonempty.")
    series_index = pd.MultiIndex.from_product(
        [sorted(regimes["store_id"].unique()), REGIME_LABELS], names=["store_id", "regime"]
    )
    results = []
    for zero_threshold, cv2_threshold in product(sorted(set(zero_thresholds)), sorted(set(cv2_thresholds))):
        _validate_cutoffs(zero_threshold, cv2_threshold)
        labels = _assign_regime(regimes, zero_threshold, cv2_threshold)
        counts = (
            regimes[["store_id"]].assign(regime=labels)
            .groupby(["store_id", "regime"], observed=True)
            .size()
            .reindex(series_index, fill_value=0)
            .rename("sku_count")
            .reset_index()
        )
        results.append(counts.assign(zero_share_threshold=zero_threshold, cv2_threshold=cv2_threshold))
    columns = ["store_id", "zero_share_threshold", "cv2_threshold", "regime", "sku_count"]
    return pd.concat(results, ignore_index=True).loc[:, columns].sort_values(columns[:3], ignore_index=True)


def run_regimes(
    config_path: str | Path,
    *,
    processed_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Write training-only regime and EDA artifacts from ingested store panels."""
    from src.data_audit import load_config

    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    settings = config["regimes"]
    processed = Path(processed_dir) if processed_dir is not None else root / config["data"]["processed_dir"]
    output = Path(output_dir) if output_dir is not None else root / settings["output_dir"]
    stores = config["data"]["stores"]
    start, end = _training_bounds(**config["splits"]["training"])
    columns = sorted({*SKU_KEYS, *SKU_METADATA, *EDA_INPUTS, "date", "sales"})
    series = []
    tables = {level: [] for level in EDA_GROUPS}

    for store in stores:
        panel_path = processed / "panel" / f"{store}.parquet"
        if not panel_path.is_file():
            raise FileNotFoundError(f"Run the data audit first. Store panel is missing: {panel_path}")
        panel = pd.read_parquet(
            panel_path,
            columns=columns,
            filters=[("date", ">=", start), ("date", "<=", end)],
        )
        training = _training_panel(panel, start, end)
        if not training["store_id"].eq(store).all():
            raise ValueError(f"Training panel contains rows from another store: {panel_path}")
        store_regimes = _classify_training(
            training,
            end,
            settings["zero_share_threshold"],
            settings["cv2_threshold"],
            settings["min_positive_days"],
        )
        expected_series = config["data"]["expected_series_per_store"]
        if len(store_regimes) != expected_series:
            raise ValueError(f"Expected {expected_series} training items for {store}, found {len(store_regimes)}.")
        series.append(store_regimes)
        for level, table in _eda_tables(training).items():
            tables[level].append(table)
        del panel, training

    sku_regimes = pd.concat(series, ignore_index=True)
    sensitivity = _sensitivity(
        sku_regimes, settings["sensitivity_zero_thresholds"], settings["sensitivity_cv2_thresholds"]
    )
    store_metrics = pd.concat(tables["store"], ignore_index=True)
    item_days = int(store_metrics["item_days"].sum())
    zero_rows = int(store_metrics["zero_rows"].sum())
    missing_price_rows = int(store_metrics["missing_price_rows"].sum())
    summary = {
        "training_start": start.date().isoformat(),
        "training_end": end.date().isoformat(),
        "stores": list(stores),
        "series_count": len(sku_regimes),
        "item_days": item_days,
        "sales_units": int(store_metrics["sales_units"].sum()),
        "zero_rows": zero_rows,
        "zero_share": zero_rows / item_days,
        "missing_price_rows": missing_price_rows,
        "missing_price_share": missing_price_rows / item_days,
        "positive_missing_price_rows": int(store_metrics["positive_missing_price_rows"].sum()),
        "regime_counts": {
            label: int(sku_regimes["regime"].eq(label).sum()) for label in REGIME_LABELS
        },
        "insufficient_history_series": int(sku_regimes["insufficient_history"].sum()),
        "zero_share_threshold": settings["zero_share_threshold"],
        "cv2_threshold": settings["cv2_threshold"],
        "min_positive_days": settings["min_positive_days"],
        "eda_population": "All training item-days, including sales zeros before the first positive sale.",
        "interpretation": "These are observed sales associations. True demand and on-shelf availability are unobserved.",
    }

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary_dir:
        staging = Path(temporary_dir)
        sku_regimes.to_parquet(staging / "sku_regimes.parquet", index=False)
        sensitivity.to_csv(staging / "threshold_sensitivity.csv", index=False)
        for level, table_parts in tables.items():
            pd.concat(table_parts, ignore_index=True).to_csv(staging / f"eda_by_{level}.csv", index=False)
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for path in sorted(staging.iterdir(), key=lambda path: path.name == "summary.json"):
            path.replace(output / path.name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_regimes(args.config, processed_dir=args.processed_dir, output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
