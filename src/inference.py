"""Validate as-of retail inputs and produce dated purchase-order recommendations."""

from __future__ import annotations

from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_audit import load_config
from src.features import DAY, OriginStore
from src.inventory_sim import Candidate, INCUMBENT, SKU, _whole, order_quantity, point_center, stock_target
from src.metrics import quantile_columns
from src.models import load_bundle, score_bundle


def _read(path):
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input files must be CSV or Parquet.")


def _require_columns(frame, columns, name):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(sorted(missing))}.")


def _date(values, name):
    dates = pd.to_datetime(values, errors="raise")
    index = pd.DatetimeIndex(dates)
    if index.hasnans or index.tz is not None or not index.equals(index.normalize()):
        raise ValueError(f"{name} requires non-null, timezone-free whole calendar dates.")
    return index.as_unit("ns")


def _snapshot_day(value):
    return _date([value], "As-of date")[0]


def _hash(path):
    with Path(path).open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _read_store(history, prices, calendar, store_id, origin, config, meta, new_sales=None):
    columns = [*SKU, "dept_id", "date", "sales"]
    panel = _read(history)
    _require_columns(panel, columns, "Sales history")
    panel = panel.loc[panel["store_id"].eq(store_id), columns].copy()
    origin = _snapshot_day(origin)
    panel["date"] = _date(panel["date"], "Sales history")
    panel = panel.loc[panel["date"].le(origin)]
    if new_sales is not None:
        additional = _read(new_sales)
        _require_columns(additional, columns, "New sales")
        additional = additional.loc[additional["store_id"].eq(store_id), columns].copy()
        additional["date"] = _date(additional["date"], "New sales")
        if additional.empty or additional["date"].gt(origin).any() or (
            not panel.empty and additional["date"].min() <= panel["date"].max()
        ):
            raise ValueError("New sales must add later completed days after history and through the origin.")
        panel = pd.concat([panel, additional], ignore_index=True)
    first = pd.Timestamp(config["data"]["panel_start"])
    panel = panel.loc[panel["date"].ge(first)]
    if panel.empty or panel["date"].min() != first or panel["date"].max() != origin:
        raise ValueError("Sales must cover the trained history start through the as-of date.")
    _whole(panel["sales"], "Sales")
    schedule = _read(calendar)
    _require_columns(schedule, ["date", "event_name_1", "event_name_2", "snap_CA", "snap_TX", "snap_WI"], "Calendar")
    schedule["date"] = _date(schedule["date"], "Calendar")
    schedule["weekday"] = schedule["date"].dt.dayofweek
    schedule["month"] = schedule["date"].dt.month
    for flag in ("snap_CA", "snap_TX", "snap_WI"):
        if schedule[flag].isna().any() or not schedule[flag].isin([0, 1, False, True]).all():
            raise ValueError(f"Calendar {flag} requires only 0/1 flags.")
    weekly = _read(prices)
    _require_columns(weekly, [*SKU, "wm_yr_wk", "sell_price"], "Prices")
    weekly = weekly.loc[weekly["store_id"].eq(store_id)].copy()
    if "week_end" not in weekly:
        _require_columns(schedule, ["wm_yr_wk"], "Calendar used to date weekly prices")
        # Raw prices can be paired with a complete M5 retail-week calendar.
        weeks = schedule.groupby("wm_yr_wk")["date"].agg(["max", "size"])
        weekly["week_end"] = weekly["wm_yr_wk"].map(weeks["max"])
        if weekly["week_end"].isna().any():
            raise ValueError("Price input contains retail weeks absent from the calendar.")
        # A truncated scheduled edge week has no completed-week publication date.
        weekly = weekly.loc[weekly["wm_yr_wk"].map(weeks["size"]).eq(7)].copy()
    weekly["week_end"] = _date(weekly["week_end"], "Price week end")
    weekly = weekly.loc[weekly["week_end"].le(origin)]
    return OriginStore(panel, schedule, weekly, config, event_codes=meta["event_codes"])


def snapshot(inventory, inbound, items, store, as_of, lead):
    """Compose the first review position from close-of-origin stock and dated firm receipts."""
    _require_columns(inventory, [*SKU, "as_of", "on_hand"], "Inventory")
    _require_columns(inbound, [*SKU, "as_of", "due_date", "units"], "Inbound")
    origin = _snapshot_day(as_of)
    inventory = inventory.loc[inventory["store_id"].eq(store)].copy()
    inbound = inbound.loc[inbound["store_id"].eq(store)].copy()
    if not (_date(inventory["as_of"], "Inventory snapshot") == origin).all() or not (
        _date(inbound["as_of"], "Inbound snapshot") == origin
    ).all():
        raise ValueError("Inventory and inbound must both be from the requested as-of date.")
    if inventory["item_id"].isna().any() or inventory["item_id"].duplicated().any() or set(inventory["item_id"]) != set(items):
        raise ValueError("Inventory requires exactly one stock count for each SKU in the trained store.")
    inventory["on_hand"] = _whole(inventory["on_hand"], "On-hand stock")
    if not inbound["item_id"].isin(items).all():
        raise ValueError("Inbound contains an unknown item.")
    due = _date(inbound["due_date"], "Inbound due date")
    tomorrow = origin + DAY
    if (due < tomorrow).any() or (due > origin + DAY * lead).any():
        raise ValueError("Inbound has late receipts or dates beyond the configured fixed lead-time pipeline.")
    inbound["due_date"] = due
    if inbound.duplicated(["item_id", "due_date"]).any():
        raise ValueError("Aggregate inbound to one row per SKU and due date before scoring.")
    inbound["units"] = _whole(inbound["units"], "Inbound units")
    due_now = inbound.loc[inbound["due_date"].eq(tomorrow)].groupby("item_id")["units"].sum()
    later = inbound.loc[inbound["due_date"].gt(tomorrow)].groupby("item_id")["units"].sum()
    return pd.DataFrame({
        "item_id": items,
        "on_hand": inventory.set_index("item_id")["on_hand"].reindex(items).to_numpy(),
        "due_on_review": due_now.reindex(items, fill_value=0).to_numpy(),
        "outstanding_after_review_receipts": later.reindex(items, fill_value=0).to_numpy(),
    })


def load_policy(path, bundle, lead, store, model=None):
    path = Path(path)
    bundle = Path(bundle)
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy["schema_version"] != 1 or policy["bundle_metadata_sha256"] != _hash(bundle / "metadata.json"):
        raise ValueError("Stocking policy does not belong to the supplied forecast bundle.")
    for name, checksum in [("residual_buffers", policy["residual_buffers_sha256"]),
                           ("reference_prices", policy["reference_prices_sha256"])]:
        if _hash(path.with_name(f"{name}.parquet")) != checksum:
            raise ValueError(f"Policy input changed: {name}.")
    if model is None:
        selection = policy["winners"].get(str(lead))
    else:
        candidates = [item for item in policy["cases"] if item["lead"] == lead and item["model"] == model
                      and item["scenario"] == "base" and item["selection_mode"] == "fill_floor"]
        selection = candidates[0] if len(candidates) == 1 else None
    if not selection or not selection["validation_feasible"]:
        raise ValueError(f"No base policy met the validation service floor for lead {lead} and model {model or 'default'}.")
    metadata = load_bundle(bundle)
    if store not in metadata["stores"] or lead + 1 not in metadata["horizons"]["cumulative"]:
        raise ValueError("Requested store and lead must be represented in the forecast bundle.")
    if metadata["alphas"] != policy["alphas"]:
        raise ValueError("Policy and model quantiles differ.")
    return policy, selection, metadata


def recommend_order_table(config_path, *, history, prices, calendar, inventory, inbound,
                          bundle, policy, as_of, store, lead, new_sales=None, model=None):
    """Score one complete store snapshot into traceable order-line suggestions."""
    if type(lead) is not int or lead < 1:
        raise ValueError("Lead time must be a positive integer.")
    origin = _snapshot_day(as_of)
    config = load_config(config_path)
    if lead not in config["inventory"]["lead_times"]:
        raise ValueError("Lead time is outside the configured study.")
    selected_policy, choice, metadata = load_policy(policy, bundle, lead, store, model)
    if origin < pd.Timestamp(selected_policy["selection_dates"]["end"]):
        raise ValueError("The policy decision predates the completed selection period.")
    dataset = _read_store(history, prices, calendar, store, origin, config, metadata, new_sales)
    stock = snapshot(_read(inventory), _read(inbound), dataset.items, store, origin, lead)
    predictions = next(
        table for table in score_bundle(dataset, [origin], bundle)
        if table["horizon"].iat[0] == lead + 1 and table["model"].iat[0] == choice["model"]
    )
    alphas = metadata["alphas"]
    candidate = Candidate(choice["model"], choice["rule"], choice["alpha"])
    buffer_model = INCUMBENT if candidate.rule == "shared_buffer" else candidate.model
    buffer = pd.read_parquet(Path(policy).with_name("residual_buffers.parquet"),
                             filters=[("horizon", "==", lead + 1), ("model", "==", buffer_model), ("store_id", "==", store)])
    rows = predictions.merge(stock, on="item_id", validate="one_to_one")
    residuals = buffer.set_index("item_id")[quantile_columns(alphas, "residual")].reindex(rows["item_id"]).to_numpy()
    if not np.isfinite(residuals).all():
        raise ValueError("Required calibration rows are missing for this cohort.")
    target = stock_target(candidate, alphas, point_center(rows, candidate.model, alphas),
                          rows[quantile_columns(alphas)].to_numpy(), residuals)
    available = rows["on_hand"] + rows["due_on_review"]
    outstanding = rows["outstanding_after_review_receipts"]
    rows["inventory_position"] = available + outstanding
    rows["target_units"] = target
    rows["recommended_units"] = order_quantity(target, available, outstanding)
    rows["review_date"] = origin + DAY
    rows["order_due_date"] = origin + DAY * (lead + 1)
    rows["rule"], rows["policy_alpha"], rows["lead_days"] = candidate.rule, candidate.alpha, lead
    rows["validation_fill_rate"] = choice["validation_fill_rate"]
    rows["policy_sha256"] = _hash(policy)
    rows["bundle_metadata_sha256"] = _hash(Path(bundle) / "metadata.json")
    rows = rows.merge(dataset.origins([origin])[["item_id", "asof_price", "price_week_end", "price_age_days"]],
                      validate="one_to_one", on="item_id")
    columns = [
        *SKU, "origin", "review_date", "order_due_date", "lead_days", "horizon", "regime", "model", "rule",
        "policy_alpha", "validation_fill_rate", *quantile_columns(alphas), "on_hand", "due_on_review",
        "outstanding_after_review_receipts", "inventory_position", "target_units", "recommended_units",
        "asof_price", "price_week_end", "price_age_days", "policy_sha256", "bundle_metadata_sha256",
    ]
    return rows[columns].sort_values(SKU, ignore_index=True)


def monitoring_report(dataset, orders, training, *, ledger=None):
    """Compare sales and price availability with training, and report realized service if provided."""
    horizon = min(28, dataset.sales.shape[1])
    if horizon < 28:
        raise ValueError("Drift monitoring requires at least 28 observed days.")
    end = dataset.days.get_indexer([pd.Timestamp(orders["origin"].iat[0])])[0]
    baseline = dataset.days.get_indexer(pd.date_range(training["start"], training["end"]))
    if end < 27 or (baseline < 0).any():
        raise ValueError("The supplied history lacks the monitoring reference or trailing window.")
    recent = dataset.sales[:, end - 27:end + 1]
    reference = dataset.sales[:, baseline]
    regime = orders.set_index("item_id")["regime"].reindex(dataset.items)
    result = []
    for name in ["All", *sorted(regime.unique())]:
        mask = np.ones(len(regime), dtype=bool) if name == "All" else regime.eq(name).to_numpy()
        group = orders if name == "All" else orders.loc[orders["regime"].eq(name)]
        prior_zero, current_zero = float((reference[mask] == 0).mean()), float((recent[mask] == 0).mean())
        result.append({
            "regime": name, "train_zero_share": prior_zero, "recent_zero_share": current_zero,
            "zero_share_change": current_zero - prior_zero, "train_units_per_item_day": float(reference[mask].mean()),
            "recent_units_per_item_day": float(recent[mask].mean()), "missing_asof_price_share": float(group["asof_price"].isna().mean()),
            "positive_order_lines": int(group["recommended_units"].gt(0).sum()),
        })
    service = None
    if ledger is not None:
        _require_columns(ledger, [*SKU, "date", "demand", "fulfilled", "lost"], "Service ledger")
        ledger = ledger.loc[ledger["store_id"].eq(dataset.store)].copy()
        ledger["date"] = _date(ledger["date"], "Service ledger dates")
        if ledger.empty or ledger.duplicated([*SKU, "date"]).any() or not ledger["item_id"].isin(dataset.items).all():
            raise ValueError("Service ledger contains no eligible rows, or duplicate or unknown item-days.")
        if ledger["date"].gt(dataset.days[end]).any():
            raise ValueError("Service ledger may only use dates through the scoring origin.")
        demand, fulfilled, lost = (_whole(ledger[field], field) for field in ("demand", "fulfilled", "lost"))
        if not np.array_equal(demand, fulfilled + lost):
            raise ValueError("Service ledger requires demand = fulfilled + lost.")
        service = {"days": int(ledger["date"].nunique()), "observed_item_days": len(ledger),
                   "demand": int(demand.sum()), "fulfilled": int(fulfilled.sum()), "lost": int(lost.sum()),
                   "fill_rate": float(fulfilled.sum() / demand.sum()) if demand.sum() else None,
                   "stockout_item_day_rate": float((lost > 0).mean())}
    return {"trailing_days": 28, "drift": result, "service": service}


def monitor_from_files(config_path, *, orders, history, prices, calendar, bundle, store, new_sales=None, ledger=None):
    config = load_config(config_path)
    origin = orders["origin"].iat[0]
    dataset = _read_store(history, prices, calendar, store, origin, config, load_bundle(bundle), new_sales)
    return monitoring_report(dataset, orders, config["splits"]["training"], ledger=_read(ledger) if ledger else None)

