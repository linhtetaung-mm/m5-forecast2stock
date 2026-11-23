"""Daily lost-sales replenishment, fixed-path economics and validation-only policy selection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data_audit import load_config
from src.features import DAY
from src.metrics import enforce_quantiles, quantile_columns
from src.models import load_bundle

SKU = ["store_id", "item_id"]
POLICIES = ("direct_quantile", "own_buffer", "shared_buffer")
INCUMBENT = "moving_average"


def _whole(values, name):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError(f"{name} must contain finite nonnegative whole units.")
    if (values > 2**40).any():
        raise ValueError(f"{name} exceeds the supported unit count.")
    return values.astype(np.int64)


def _targets(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 2**40).any():
        raise ValueError("Order-up-to targets must be finite and nonnegative.")
    return values


@dataclass
class StockState:
    """On-hand stock and an ordered queue, due today first."""

    on_hand: np.ndarray
    inbound: np.ndarray

    def copy(self):
        return StockState(self.on_hand.copy(), self.inbound.copy())


def opening_state(on_hand, lead):
    on_hand = _whole(on_hand, "Opening stock")
    if on_hand.ndim != 1 or not len(on_hand) or type(lead) is not int or lead < 1:
        raise ValueError("Opening stock must be a nonempty item vector and lead time a positive integer.")
    return StockState(on_hand, np.zeros((lead, len(on_hand)), dtype=np.int64))


def order_quantity(target, available, outstanding):
    """Whole-unit order suggestion before the current day's demand is observed."""
    target = _targets(target)
    available = _whole(available, "Available stock")
    outstanding = _whole(outstanding, "Outstanding stock")
    if target.shape != available.shape or target.shape != outstanding.shape:
        raise ValueError("Each order target needs a matching stock and pipeline position.")
    return np.ceil(np.maximum(0, target - available - outstanding)).astype(np.int64)


def _step(state, demand, target):
    start = state.on_hand
    receipt = state.inbound[0].copy()
    available = start + receipt
    outstanding = state.inbound[1:].sum(axis=0)
    position = available + outstanding
    ordered = np.ceil(np.maximum(0, target - position)).astype(np.int64)
    filled = np.minimum(available, demand)
    lost = demand - filled
    ending = available - filled
    state.inbound[:-1] = state.inbound[1:]
    state.inbound[-1] = ordered
    state.on_hand = ending
    return {
        "start": start, "receipts": receipt, "available": available, "position": position,
        "target": target, "ordered": ordered, "demand": demand,
        "filled": filled, "lost": lost, "ending": ending,
        "pipeline": outstanding + ordered,
    }


def simulate(demand, targets, opening, *, capture=False):
    """Replay all items in parallel, returning item totals and a reusable closing state."""
    demand, targets = _whole(demand, "Demand"), _targets(targets)
    on_hand = _whole(opening.on_hand, "Opening stock")
    inbound = _whole(opening.inbound, "Opening pipeline")
    if (demand.ndim != 2 or not demand.shape[0] or targets.shape != demand.shape
            or on_hand.shape != (demand.shape[1],) or inbound.ndim != 2
            or inbound.shape[1] != demand.shape[1] or not inbound.shape[0]):
        raise ValueError("Demand and targets need the same nonempty day × item shape and a matching opening pipeline.")
    state = StockState(on_hand.copy(), inbound.copy())
    totals = {field: np.zeros(len(on_hand), dtype=np.int64) for field in
              ("demand", "filled", "lost", "stockout_days", "stock_days", "ordered", "order_lines", "receipts")}
    detail = []
    for units, target in zip(demand, targets, strict=True):
        day = _step(state, units, target)
        for field in ("demand", "filled", "lost", "ordered", "receipts"):
            totals[field] += day[field]
        totals["stockout_days"] += day["lost"] > 0
        totals["stock_days"] += day["ending"]
        totals["order_lines"] += day["ordered"] > 0
        if capture:
            detail.append(day)
    np.testing.assert_array_equal(totals["demand"], totals["filled"] + totals["lost"])
    np.testing.assert_array_equal(state.on_hand, on_hand + totals["receipts"] - totals["filled"])
    np.testing.assert_array_equal(state.inbound.sum(axis=0), inbound.sum(axis=0) + totals["ordered"] - totals["receipts"])
    return totals, state, detail


@dataclass(frozen=True)
class CostScenario:
    name: str
    holding_per_price: float
    shortage_per_price: float
    order_fee: float
    margin: float
    kind: str

    def fractile(self, periods):
        return self.shortage_per_price / (self.shortage_per_price + periods * self.holding_per_price)


def cost_scenarios(settings):
    """One-at-a-time dollar scenarios plus normalized holding/shortage ratios."""
    base = settings["base"]
    axes = {
        "annual_carry_rate": settings["annual_carry_rates"],
        "margin_fraction": settings["margin_fractions"],
        "stockout_multiplier": settings["stockout_multipliers"],
        "positive_item_order_fee": settings["positive_item_order_fees"],
    }

    def price_anchored(name, params):
        margin = params["margin_fraction"]
        return CostScenario(name, params["annual_carry_rate"] * (1 - margin) / settings["days_per_year"],
                            margin * params["stockout_multiplier"], params["positive_item_order_fee"], margin, "price_anchored")

    scenarios = [price_anchored("base", base)]
    for axis, values in axes.items():
        scenarios.extend(price_anchored(f"{axis}_{value:g}", {**base, axis: value})
                         for value in values if value != base[axis])
    scenarios.extend(CostScenario(f"ratio_{ratio:g}", ratio, 1.0, 0.0, base["margin_fraction"], "normalized")
                     for ratio in settings["relative_daily_holding_to_shortage"])
    for scenario in scenarios:
        if not (np.isfinite([scenario.holding_per_price, scenario.shortage_per_price, scenario.order_fee]).all()
                and scenario.holding_per_price > 0 and scenario.shortage_per_price > 0
                and scenario.order_fee >= 0 and 0 < scenario.margin < 1):
            raise ValueError(f"Invalid scenario economics: {scenario.name}")
    return scenarios


def reference_prices(prices, cohort, cutoff, weeks):
    """Median listed price across completed weeks, with transparent hierarchy fallback."""
    cutoff = pd.Timestamp(cutoff)
    if type(weeks) is not int or weeks < 1:
        raise ValueError("Reference price lookback must be a positive number of weeks.")
    price_days = pd.to_datetime(prices["week_end"])
    period = prices.loc[price_days.gt(cutoff - DAY * 7 * weeks) & price_days.le(cutoff)].copy()
    if not np.isfinite(period["sell_price"].astype(float)).all() or period["sell_price"].le(0).any():
        raise ValueError("Reference price rows must have positive finite selling prices.")
    direct = period.groupby(SKU, as_index=False, observed=True).agg(
        reference_price=("sell_price", "median"), observed_weeks=("sell_price", "size"))
    keys = cohort.reset_index(drop=True).merge(direct, how="left", on=SKU, validate="one_to_one")
    observed = keys.loc[keys["reference_price"].notna()]
    dept = observed.groupby(["store_id", "dept_id"], as_index=False, observed=True)["reference_price"].median()
    dept = dept.rename(columns={"reference_price": "dept_price"})
    store = observed.groupby("store_id", as_index=False, observed=True)["reference_price"].median()
    store = store.rename(columns={"reference_price": "store_price"})
    keys = keys.merge(dept, how="left", on=["store_id", "dept_id"], validate="many_to_one")
    keys = keys.merge(store, how="left", on="store_id", validate="many_to_one")
    keys["price_source"] = np.select(
        [keys["reference_price"].notna(), keys["dept_price"].notna()], ["item_median", "department_median"],
        default="store_median",
    )
    keys["reference_price"] = keys["reference_price"].fillna(keys["dept_price"]).fillna(keys["store_price"])
    keys["observed_weeks"] = keys["observed_weeks"].fillna(0).astype(int)
    if keys["reference_price"].isna().any():
        raise ValueError("A store has no pre-validation reference prices.")
    return keys.drop(columns=["dept_price", "store_price"])


def interpolate(level, alphas, grid):
    """Interpolate inside the trained levels; use the nearest trained endpoint outside."""
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("The requested service quantile must be between zero and one.")
    levels = np.asarray(alphas, dtype=float)
    upper = int(np.searchsorted(levels, level))
    if upper == 0:
        return grid[..., 0]
    if upper >= len(levels):
        return grid[..., -1]
    weight = (level - levels[upper - 1]) / (levels[upper] - levels[upper - 1])
    return grid[..., upper - 1] * (1 - weight) + grid[..., upper] * weight


def _level(alpha):
    return float(np.clip(round(alpha, 12), 1e-12, 1 - 1e-12))


@dataclass(frozen=True)
class Candidate:
    model: str
    rule: str
    alpha: float

    @property
    def key(self):
        return f"{self.model}|{self.rule}|{self.alpha:g}" if self.alpha in (.5, .75, .9, .95) else f"{self.model}|{self.rule}|{self.alpha:.12g}"


def point_center(table, model, alphas):
    if model.startswith("lightgbm"):
        return enforce_quantiles(table[quantile_columns(alphas, "raw")].to_numpy())[:, alphas.index(0.5)]
    return table["point"].to_numpy(dtype=float)


def stock_target(candidate, alphas, center, quantiles, residual_grid):
    """A forecast quantile or a point forecast plus a nonnegative error buffer."""
    if candidate.rule == "direct_quantile":
        return interpolate(candidate.alpha, alphas, quantiles)
    if candidate.rule not in {"own_buffer", "shared_buffer"}:
        raise ValueError(f"Unknown stocking rule: {candidate.rule}")
    return center + np.maximum(0, interpolate(candidate.alpha, alphas, residual_grid))


def frontier(table):
    """Return all cost/fill combinations on the weak Pareto frontier."""
    table = table.dropna(subset=["fill_rate", "cost"])
    if table.empty:
        return table
    ranked = table.sort_values(["fill_rate", "cost"], ascending=[False, True], kind="stable")
    preceding = ranked["cost"].astype(float).cummin().shift(fill_value=np.inf)
    return ranked.loc[ranked["cost"].lt(preceding)].copy()


def _choose(frame, fill_floor):
    feasible = frame.loc[frame["fill_rate"].ge(fill_floor)]
    if feasible.empty:
        return frame.sort_values(["fill_rate", "cost", "candidate"], ascending=[False, True, True]).iloc[0], False
    return feasible.sort_values(["cost", "fill_rate", "candidate"], ascending=[True, False, True]).iloc[0], True


def _hash(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _dates(phase):
    return pd.date_range(phase["start"], phase["end"])


def _cohort(root, config, bundle):
    processed = root / config["data"]["processed_dir"]
    regimes = pd.read_parquet(bundle / "regimes.parquet")
    items = []
    for store in config["data"]["stores"]:
        panel = pd.read_parquet(processed / "panel" / f"{store}.parquet", columns=[*SKU, "dept_id"])
        items.append(panel.drop_duplicates())
    result = pd.concat(items, ignore_index=True).merge(regimes, validate="one_to_one", on=SKU)
    if len(result) != len(regimes):
        raise ValueError("Replay cohort differs from the frozen forecast cohort.")
    return result.sort_values(SKU, ignore_index=True)


def _load_sales(processed, cohort, dates, stores):
    filters = [("date", ">=", dates.min()), ("date", "<=", dates.max())]
    data = pd.concat([pd.read_parquet(processed / "panel" / f"{store}.parquet",
                                     columns=[*SKU, "date", "sales"], filters=filters) for store in stores],
                     ignore_index=True)
    return _matrix(data, cohort, dates, "date", ["sales"])[..., 0]


def _matrix(table, cohort, dates, date_column, values):
    keys = [date_column, *SKU]
    expected = pd.MultiIndex.from_product([dates, range(len(cohort))], names=[date_column, "sku_index"])
    index = cohort[SKU].assign(sku_index=np.arange(len(cohort)))
    ordered = table[keys + values].merge(index, on=SKU, how="left", validate="many_to_one")
    if ordered["sku_index"].isna().any() or ordered.duplicated([date_column, "sku_index"]).any():
        raise ValueError(f"Unknown or duplicate item-date rows in {values}.")
    if len(ordered) != len(expected):
        raise ValueError(f"Missing item-date rows in {values}.")
    ordered = ordered.set_index([date_column, "sku_index"])[values].reindex(expected)
    matrix = ordered.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Missing or nonfinite item-date values in {values}.")
    return matrix.reshape(len(dates), len(cohort), len(values))



def residual_tables(forecast_dir, cohort, validation, horizon, alphas):
    table = pd.read_parquet(forecast_dir / "calibration_cumulative.parquet",
                            filters=[("horizon", "==", horizon)])
    dates = pd.DatetimeIndex(sorted(table["origin"].unique()))
    if not len(dates) or dates[-1] + DAY * horizon >= validation[0]:
        raise ValueError("Cumulative calibration must end before policy validation.")
    results = {}
    for name, rows in table.groupby("model", sort=True):
        rows = rows.assign(error=rows["actual"] - point_center(rows, name, alphas))
        errors = _matrix(rows, cohort, dates, "origin", ["error"])[..., 0]
        results[name] = np.quantile(errors, alphas, axis=0, method="higher").T
    if INCUMBENT not in results:
        raise ValueError("The common safety buffer requires moving-average calibration.")
    return results, len(dates)


def _forecast_block(directory, cohort, dates, horizon, alphas, phase):
    table = pd.read_parquet(directory / f"{phase}_cumulative.parquet", filters=[("horizon", "==", horizon)])
    if not pd.to_datetime(table["target_date"]).eq(pd.to_datetime(table["origin"]) + horizon * DAY).all():
        raise ValueError("An eight- or two-day forecast has the wrong target endpoint.")
    outputs = {}
    columns = quantile_columns(alphas)
    for model, rows in table.groupby("model", sort=True):
        rows = rows.assign(center=point_center(rows, model, alphas))
        stacked = _matrix(rows, cohort, dates - DAY, "origin", ["center", *columns])
        if (stacked < 0).any() or (np.diff(stacked[..., 1:], axis=-1) < 0).any():
            raise ValueError("Replay requires nonnegative noncrossing forecasts.")
        outputs[model] = stacked
    return outputs


def _candidate_grid(models, scenarios, alphas, horizon):
    economic = [
        {"scenario": scenario.name, "protection": horizon, "formula": label,
         "requested_alpha": scenario.fractile(period)}
        for scenario in scenarios for label, period in (("one_day_holding", 1), ("protection_holding", horizon))
    ]
    for row in economic:
        row["used_alpha"] = _level(np.clip(row["requested_alpha"], alphas[0], alphas[-1]))
        row["clipped"] = not alphas[0] <= row["requested_alpha"] <= alphas[-1]
    levels = sorted(set(alphas) | {row["used_alpha"] for row in economic})
    candidates = [Candidate(model, rule, alpha) for model in sorted(models) for rule in POLICIES for alpha in levels]
    return candidates, economic


def _target(block, candidate, residual, alphas):
    features = block[candidate.model]
    buffer_model = INCUMBENT if candidate.rule == "shared_buffer" else candidate.model
    return stock_target(candidate, alphas, features[..., 0], features[..., 1:], residual[buffer_model])


def _summaries(cohort, candidate, before, after, totals, days, phase, lead):
    prices = cohort["reference_price"].to_numpy(dtype=float)
    rows = []
    groups = {"All": np.ones(len(prices), dtype=bool)}
    for regime in sorted(cohort["regime"].unique()):
        groups[regime] = cohort["regime"].eq(regime).to_numpy()
    for group, mask in groups.items():
        numbers = {name: int(values[mask].sum()) for name, values in totals.items()}
        demand = numbers["demand"]
        sku_days = int(mask.sum() * days)
        rows.append({
            "phase": phase, "lead": lead, "protection": lead + 1, "candidate": candidate.key,
            "model": candidate.model, "rule": candidate.rule, "alpha": candidate.alpha,
            "regime": group, "days": days, "sku_days": sku_days, **numbers,
            "fill_rate": numbers["filled"] / demand if demand else np.nan,
            "stockout_rate": numbers["stockout_days"] / sku_days,
            "mean_on_hand": numbers["stock_days"] / sku_days,
            "opening_stock": int(before.on_hand[mask].sum()),
            "opening_pipeline": int(before.inbound[:, mask].sum()),
            "ending_stock": int(after.on_hand[mask].sum()),
            "ending_pipeline": int(after.inbound[:, mask].sum()),
            "price_stock_days": float(prices[mask] @ totals["stock_days"][mask]),
            "price_lost": float(prices[mask] @ totals["lost"][mask]),
        })
    return rows


def _cost(table, scenario):
    return table.assign(
        scenario=scenario.name,
        valuation=scenario.kind,
        holding_cost=table["price_stock_days"] * scenario.holding_per_price,
        stockout_cost=table["price_lost"] * scenario.shortage_per_price,
        ordering_cost=table["order_lines"] * scenario.order_fee,
        cost=table["price_stock_days"] * scenario.holding_per_price
             + table["price_lost"] * scenario.shortage_per_price + table["order_lines"] * scenario.order_fee,
        mean_acquisition_value=table["price_stock_days"] * (1 - scenario.margin) / table["sku_days"],
    )


def _parameters(row, feasible, *, selection_mode, benchmark_fill=None):
    entry = {
        "scenario": row["scenario"], "lead": int(row["lead"]), "candidate": row["candidate"],
        "model": row["model"], "rule": row["rule"], "alpha": float(row["alpha"]),
        "validation_fill_rate": float(row["fill_rate"]), "validation_cost": float(row["cost"]),
        "validation_feasible": feasible, "selection_mode": selection_mode,
    }
    if benchmark_fill is not None:
        entry["validation_incumbent_fill"] = float(benchmark_fill)
    return entry


def _select(validation, scenarios, alphas, fractiles, config):
    selections = []
    valid = validation.loc[validation["regime"].eq("All")]
    fill_floor = config["fill_floor"]
    for scenario in scenarios:
        alpha = set(alphas) | {row["used_alpha"] for row in fractiles if row["scenario"] == scenario.name}
        priced = _cost(valid.loc[valid["alpha"].isin(alpha)], scenario)
        incumbent, feasible = _choose(priced.loc[priced["model"].eq(INCUMBENT)], fill_floor)
        for _, model in priced.groupby("model", sort=True):
            chosen, floor_ok = _choose(model, fill_floor)
            selections.append(_parameters(chosen, floor_ok, selection_mode="fill_floor"))
            if scenario.name == "base" and feasible:
                gap = (model["fill_rate"] - incumbent["fill_rate"]).abs()
                compatible = model.loc[gap.le(config["service_comparability_tolerance"]) & model["fill_rate"].ge(fill_floor)]
                if not compatible.empty:
                    paired, _ = _choose(compatible, fill_floor)
                    selections.append(_parameters(paired, True, selection_mode="matched_validation_service",
                                                  benchmark_fill=incumbent["fill_rate"]))
    return selections


def _opening(data, cohort, dates, names, lead):
    history = pd.date_range(dates[0] - DAY * 28, dates[0] - DAY)
    trailing = _load_sales(data, cohort, history, names)
    return opening_state(np.ceil(trailing.mean(axis=0) * (lead + 1)), lead)


def _break_even(validation, chosen, scenarios):
    comparisons, stress = [], set()
    for lead, group in validation.loc[validation["regime"].eq("All")].groupby("lead"):
        selections = [entry for entry in chosen if entry["lead"] == lead and entry["scenario"] == "base"
                      and entry["selection_mode"] == "fill_floor"]
        lookup = group.set_index("candidate")
        incumbent = lookup.loc[next(item["candidate"] for item in selections if item["model"] == INCUMBENT)]
        for entry in selections:
            if entry["model"] == INCUMBENT:
                continue
            run = lookup.loc[entry["candidate"]]
            delta_stock = run["price_stock_days"] - incumbent["price_stock_days"]
            delta_short = run["price_lost"] - incumbent["price_lost"]
            ratio = -delta_short / delta_stock if abs(delta_stock) > 1e-9 else np.nan
            positive = np.isfinite(ratio) and ratio > 0
            comparisons.append({"lead": lead, "model": entry["model"], "candidate": entry["candidate"],
                                "incumbent": incumbent.name, "delta_price_stock_days": delta_stock,
                                "delta_price_lost": delta_short, "positive_break_even": ratio if positive else np.nan})
            if positive:
                stress.update([round(ratio * .5, 10), round(ratio * 2, 10)])
    ratios = {scenario.holding_per_price for scenario in scenarios if scenario.kind == "normalized"}
    stress.difference_update(ratios)
    return pd.DataFrame(comparisons), sorted(value for value in stress if 0 < value <= 100)


def _frontiers(trajectories):
    data = trajectories.loc[trajectories["phase"].isin(["policy_validation", "final_cutover"])]
    return pd.concat([frontier(group) for _, group in data.groupby(["phase", "lead", "regime"], sort=True)],
                     ignore_index=True)


def _comparisons(results, metrics, tolerance):
    base = results.loc[results["scenario"].eq("base") & results["phase"].eq("final_cutover")].copy()
    by_key = ["lead", "regime", "selection_mode"]
    incumbent = base.loc[base["model"].eq(INCUMBENT), by_key + ["cost", "fill_rate", "candidate"]]
    joined = base.merge(incumbent, on=by_key, suffixes=("", "_incumbent"), validate="many_to_one")
    joined["cost_delta"] = joined["cost"] - joined["cost_incumbent"]
    joined["fill_delta"] = joined["fill_rate"] - joined["fill_rate_incumbent"]
    joined["comparable_service"] = joined["fill_delta"].abs().le(tolerance)
    scored = metrics.loc[
        metrics["phase"].eq("final_test") & metrics["family"].eq("cumulative") & metrics["alpha"].eq(.90),
        ["model", "horizon", "regime", "spl", "origins"],
    ].rename(columns={"horizon": "protection", "spl": "q90_spl", "origins": "forecast_origins"})
    joined = joined.merge(scored, on=["model", "protection", "regime"], validate="many_to_one")
    reference = scored.loc[scored["model"].eq(INCUMBENT), ["protection", "regime", "q90_spl"]]
    joined = joined.merge(reference.rename(columns={"q90_spl": "incumbent_q90_spl"}),
                          on=["protection", "regime"], validate="many_to_one")
    joined["q90_spl_delta"] = joined["q90_spl"] - joined["incumbent_q90_spl"]
    return joined


def run_inventory(config_path, *, output_dir=None):
    """Run frozen validation selection and one April replay for each starting-state design."""
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    process = root / config["data"]["processed_dir"]
    forecast = root / config["forecast"]["output_dir"]
    forecasts = forecast / "forecasts"
    bundle = forecast / "bundle"
    inventory = config["inventory"]
    splits = config["splits"]
    settings = inventory["costs"]
    if inventory["review_days"] != 1 or not inventory["lost_sales"] or inventory["terminal_adjustment"] != "none":
        raise ValueError("Inventory supports daily lost sales with a disclosed ending stock and no terminal charge.")
    if not 0 < inventory["fill_floor"] <= 1 or not 0 <= inventory["service_comparability_tolerance"] <= 1:
        raise ValueError("Fill floor and service tolerance must be valid service fractions.")
    destination = Path(output_dir).resolve() if output_dir else root / config["artifacts"]["report_dir"] / "inventory"
    if destination.exists() or any(destination.is_relative_to(folder.resolve()) or folder.resolve().is_relative_to(destination)
                                   for folder in [forecast, process]):
        raise ValueError(f"Choose a new inventory output directory separate from canonical inputs: {destination}")
    meta = load_bundle(bundle)
    if (meta["splits"] != splits or meta["stores"] != config["data"]["stores"]
            or meta["forecast_settings"] != config["forecast"]):
        raise ValueError("Inventory configuration does not match the frozen forecast run.")
    checksums = json.loads((forecast / "run_manifest.json").read_text())["files_sha256"]
    for name in ["metrics.csv", "bundle/metadata.json", *[f"forecasts/{phase}_cumulative.parquet"
                for phase in ("calibration", "policy_validation", "warmup", "final_test")]]:
        if _hash(forecast / name) != checksums.get(name):
            raise ValueError(f"Frozen forecasting output changed: {name}")
    alphas = meta["alphas"]
    names = meta["stores"]
    cohorts = _cohort(root, config, bundle)
    cut = pd.Timestamp(settings["reference_price_cutoff"])
    phases = {phase: _dates(splits[phase]) for phase in ("policy_validation", "warmup", "final_test")}
    if cut >= phases["policy_validation"][0]:
        raise ValueError("The selling-price reference cutoff must precede policy validation.")
    prices = pd.concat([pd.read_parquet(process / "price_history" / f"{store}.parquet") for store in names],
                       ignore_index=True)
    cohort = reference_prices(prices, cohorts, cut, settings["reference_price_lookback_weeks"])
    del prices
    scenarios = cost_scenarios(settings)
    leads = inventory["lead_times"]
    if [lead + 1 for lead in leads] != inventory["protection_periods"] or any(
        horizon not in meta["horizons"]["cumulative"] for horizon in inventory["protection_periods"]
    ):
        raise ValueError("Protection windows must match the frozen lead-plus-one forecasts.")
    validation_dates = phases["policy_validation"]
    validation_sales = _load_sales(process, cohort, validation_dates, names)
    data = {}
    validation_rows, fractiles, selection, stored_residuals = [], [], [], []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".inventory-", dir=destination.parent) as temp:
        stage = Path(temp) / "run"
        stage.mkdir()
        for lead in leads:
            horizon = lead + 1
            residuals, count = residual_tables(forecasts, cohort, validation_dates, horizon, alphas)
            for model, values in residuals.items():
                stored = cohort[SKU].copy()
                stored[quantile_columns(alphas, "residual")] = values
                stored_residuals.append(stored.assign(model=model, horizon=horizon))
            block = _forecast_block(forecasts, cohort, validation_dates, horizon, alphas, "policy_validation")
            opening = _opening(process, cohort, validation_dates, names, lead)
            candidates, economic = _candidate_grid(block, scenarios, alphas, horizon)
            if count < 2:
                raise ValueError("Insufficient complete pre-validation residual windows.")
            rows, closing = [], {}
            for candidate in candidates:
                total, state, _ = simulate(validation_sales, _target(block, candidate, residuals, alphas), opening)
                rows.extend(_summaries(cohort, candidate, opening, state, total, len(validation_dates), "policy_validation", lead))
                closing[candidate.key] = state
            frame = pd.DataFrame(rows)
            validation_rows.append(frame)
            selection.extend(_select(frame, scenarios, alphas, economic, inventory))
            fractiles.extend(economic)
            data[lead] = candidates, residuals, closing
            print(f"Lead {lead}: assessed {len(candidates)} candidates on validation, {count} calibration windows per SKU.", flush=True)
        validation = pd.concat(validation_rows, ignore_index=True)
        break_even, stress = _break_even(validation, selection, scenarios)
        stress_scenarios = [CostScenario(f"break_even_probe_{index}", ratio, 1.0, 0.0, scenarios[0].margin, "normalized")
                            for index, ratio in enumerate(stress, start=1)]
        winners = {}
        for lead in leads:
            entries = [entry for entry in selection if entry["scenario"] == "base" and entry["lead"] == lead
                       and entry["selection_mode"] == "fill_floor"]
            feasible = [entry for entry in entries if entry["validation_feasible"]]
            if feasible:
                winners[str(lead)] = min(feasible, key=lambda item: (item["validation_cost"], item["candidate"]))
        cohort.to_parquet(stage / "reference_prices.parquet", index=False)
        pd.concat(stored_residuals, ignore_index=True).to_parquet(stage / "residual_buffers.parquet", index=False)
        cost_table = [scenario.__dict__ for scenario in scenarios + stress_scenarios]
        _write_json(stage / "policy.json", {
            "schema_version": 1, "bundle_metadata_sha256": _hash(bundle / "metadata.json"),
            "selection_dates": splits["policy_validation"], "warmup_dates": splits["warmup"],
            "test_dates": splits["final_test"], "fill_floor": inventory["fill_floor"],
            "service_tolerance": inventory["service_comparability_tolerance"], "cases": selection, "winners": winners,
            "reference_prices_sha256": _hash(stage / "reference_prices.parquet"),
            "residual_buffers_sha256": _hash(stage / "residual_buffers.parquet"),
            "alphas": alphas, "scenarios": cost_table,
            "incumbent": {"model": INCUMBENT, "rule": "own_buffer", "alpha": max(alphas)},
        })
        final_rows = []
        warmup_dates, final_dates = phases["warmup"], phases["final_test"]
        warmup_sales = _load_sales(process, cohort, warmup_dates, names)
        final_sales = _load_sales(process, cohort, final_dates, names)
        for lead in leads:
            horizon = lead + 1
            candidates, residuals, closing = data[lead]
            warm = _forecast_block(forecasts, cohort, warmup_dates, horizon, alphas, "warmup")
            block = _forecast_block(forecasts, cohort, final_dates, horizon, alphas, "final_test")
            incumbent = Candidate(INCUMBENT, "own_buffer", max(alphas))
            _, shared, _ = simulate(warmup_sales, _target(warm, incumbent, residuals, alphas), closing[incumbent.key])
            for candidate in candidates:
                _, individual, _ = simulate(warmup_sales, _target(warm, candidate, residuals, alphas), closing[candidate.key])
                choices = [("final_continuous", individual)]
                if inventory["include_common_state_cutover"]:
                    choices.append(("final_cutover", shared))
                target = _target(block, candidate, residuals, alphas)
                for phase, opening in choices:
                    total, settled, _ = simulate(final_sales, target, opening)
                    final_rows.extend(_summaries(cohort, candidate, opening, settled, total, len(final_dates), phase, lead))
            print(f"Lead {lead}: replayed {len(final_dates)} April dates with the policy settings fixed.", flush=True)
        raw = pd.concat([validation, pd.DataFrame(final_rows)], ignore_index=True)
        all_output = pd.concat([_cost(raw, scenario) for scenario in scenarios], ignore_index=True)
        base_output = all_output.loc[all_output["scenario"].eq("base")]
        frozen = pd.DataFrame(selection)
        selected = all_output.merge(frozen[["scenario", "lead", "candidate", "selection_mode", "validation_feasible"]],
                                    on=["scenario", "lead", "candidate"], validate="many_to_many")
        selected["phase_fill_floor_met"] = selected["fill_rate"].ge(inventory["fill_floor"])
        base_picks = frozen.loc[frozen["scenario"].eq("base") & frozen["selection_mode"].eq("fill_floor"), ["lead", "candidate"]]
        fixed = raw.merge(base_picks, on=["lead", "candidate"], validate="many_to_one")
        repriced = pd.concat([_cost(fixed, scenario) for scenario in scenarios + stress_scenarios], ignore_index=True)
        all_output.to_parquet(stage / "scenario_trajectories.parquet", index=False)
        _frontiers(base_output).to_csv(stage / "base_frontiers.csv", index=False)
        selected.to_csv(stage / "selected_policies.csv", index=False)
        repriced.to_csv(stage / "fixed_path_repricing.csv", index=False)
        pd.DataFrame(fractiles).to_csv(stage / "critical_fractiles.csv", index=False)
        break_even.to_csv(stage / "validation_break_even.csv", index=False)
        metrics = pd.read_csv(forecast / "metrics.csv")
        _comparisons(selected, metrics, inventory["service_comparability_tolerance"]).to_csv(
            stage / "forecast_inventory_pairs.csv", index=False)
        _write_json(stage / "run_manifest.json", {
            "config_sha256": _hash(config_path), "forecast_manifest_sha256": _hash(forecast / "run_manifest.json"),
            "files_sha256": {file.name: _hash(file) for file in sorted(stage.iterdir())},
        })
        stage.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(run_inventory(args.config, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
