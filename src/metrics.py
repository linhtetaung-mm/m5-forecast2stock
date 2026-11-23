"""Forecast and quantile diagnostics with scales fixed on the training block."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_columns(alphas, prefix="q"):
    names = [f"{prefix}_{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "") for alpha in alphas]
    if len(set(names)) != len(names):
        raise ValueError("Quantiles have colliding names at six-decimal precision.")
    return names


def enforce_quantiles(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Quantile predictions must be finite.")
    return np.sort(np.maximum(values, 0), axis=-1)


def pinball(actual, prediction, alpha):
    residual = np.asarray(actual) - np.asarray(prediction)
    return np.maximum(alpha * residual, (alpha - 1) * residual)


def training_scales(store, phase, cumulative_horizons):
    dates = store.days.get_indexer(pd.date_range(phase["start"], phase["end"]))
    if len(dates) < 2 or (dates < 0).any():
        raise ValueError("Training scales require a contiguous observed window.")
    sales = store.sales[:, dates].astype(float)
    first = np.where((sales > 0).any(axis=1), (sales > 0).argmax(axis=1), sales.shape[1])
    prefix = store._prefix(sales)
    scales = {}
    for family, horizons in (("daily", [1]), ("cumulative", cumulative_horizons)):
        for horizon in horizons:
            totals = prefix[:, horizon:] - prefix[:, :-horizon]
            differences = np.diff(totals, axis=1)
            # Do not use a complete target window that begins before the first positive daily sale.
            eligible = np.arange(differences.shape[1])[None, :] >= first[:, None]
            count = eligible.sum(axis=1)
            absolute = np.full(len(first), np.nan)
            squared = np.full(len(first), np.nan)
            np.divide((np.abs(differences) * eligible).sum(axis=1), count, out=absolute, where=count > 0)
            np.divide((differences**2 * eligible).sum(axis=1), count, out=squared, where=count > 0)
            scales[(family, horizon)] = pd.DataFrame({
                "store_id": store.store, "item_id": store.items,
                "abs_scale": absolute, "sq_scale": squared,
            })
    return scales


def _summarize_group(frame, alphas, regime):
    error = frame["actual"] - frame["point"]
    scored = frame.assign(absolute_error=error.abs(), squared_error=error**2)
    series_key = ["store_id", "item_id"]
    keys = scored.groupby(series_key, observed=True, sort=False)
    per_series_squared = keys["squared_error"].mean()
    square_scale = keys["sq_scale"].first().where(lambda s: s > 0)
    rmsse = np.sqrt(per_series_squared / square_scale)
    absolute_scale = keys["abs_scale"].first().where(lambda s: s > 0)
    raw = frame[quantile_columns(alphas, "raw")].to_numpy()
    crossing_rate = float((np.diff(raw, axis=1) < 0).any(axis=1).mean())
    records = []
    for alpha, column in zip(alphas, quantile_columns(alphas), strict=True):
        prediction, actual = frame[column], frame["actual"]
        losses = pinball(actual, prediction, alpha)
        per_series_pinball = frame.assign(loss=losses).groupby(series_key, observed=True, sort=False)["loss"].mean()
        records.append({
            "regime": regime, "alpha": alpha,
            "rows": len(frame), "series": len(absolute_scale), "origins": frame["origin"].nunique(),
            "mae": float(error.abs().mean()), "rmse": float(np.sqrt((error**2).mean())), "rmsse": rmsse.mean(),
            "rmsse_undefined_series": int(square_scale.isna().sum()),
            "pinball": float(losses.mean()), "spl": (per_series_pinball / absolute_scale).mean(),
            "spl_undefined_series": int(absolute_scale.isna().sum()),
            "coverage_strict": float(actual.lt(prediction).mean()), "coverage_inclusive": float(actual.le(prediction).mean()),
            "raw_crossing_share": crossing_rate,
        })
    return records


def evaluate(frame, alphas):
    """Micro unscaled metrics and macro SKU-scaled metrics, overall and by frozen regime."""
    sample = frame.loc[frame["actual"].notna()]
    if sample.empty:
        return pd.DataFrame()
    if sample["regime"].isna().any() or not np.isfinite(sample[["point", *quantile_columns(alphas)]].to_numpy()).all():
        raise ValueError("Evaluation requires finite predictions and a frozen regime for every observation.")
    if sample.duplicated(["model", "store_id", "item_id", "origin"]).any():
        raise ValueError("Duplicate evaluation row within a model and horizon.")
    records = []
    for model, modeled in sample.groupby("model", sort=True, observed=True):
        for regime, group in [("All", modeled), *modeled.groupby("regime", observed=True, sort=True)]:
            records.extend({"model": model, **row} for row in _summarize_group(group, alphas, regime))
    return pd.DataFrame(records)

