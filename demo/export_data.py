"""Export a small, frozen replay and verified aggregate results for the static demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dist/study-data.js")
    args = parser.parse_args()
    root = args.project.resolve()
    sys.path.insert(0, str(root))
    from src.data_audit import load_config
    from src.inventory_sim import (
        Candidate, INCUMBENT, _forecast_block, _hash, _load_sales, _opening, _target,
        simulate,
    )
    from src.metrics import quantile_columns

    config_file = root / "configs/config.yaml"
    config = load_config(config_file)
    inventory = root / config["artifacts"]["report_dir"] / "inventory"
    forecasting = root / config["forecast"]["output_dir"]
    processed = root / config["data"]["processed_dir"]
    manifest = json.loads((inventory / "run_manifest.json").read_text())
    forecast_manifest = json.loads((forecasting / "run_manifest.json").read_text())
    if manifest["config_sha256"] != _hash(config_file) or manifest["forecast_manifest_sha256"] != _hash(forecasting / "run_manifest.json"):
        raise ValueError("The inventory study and forecast run do not match this project configuration.")
    for name in ("policy.json", "reference_prices.parquet", "residual_buffers.parquet", "selected_policies.csv", "base_frontiers.csv", "forecast_inventory_pairs.csv"):
        if _hash(inventory / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Inventory artifact changed: {name}")
    for phase in ("policy_validation", "warmup", "final_test"):
        name = f"forecasts/{phase}_cumulative.parquet"
        if _hash(forecasting / name) != forecast_manifest["files_sha256"][name]:
            raise ValueError(f"Forecast artifact changed: {name}")

    meta = json.loads((inventory / "policy.json").read_text())
    cohort = pd.read_parquet(inventory / "reference_prices.parquet").sort_values(["store_id", "item_id"], ignore_index=True)
    stores = cohort["store_id"].unique().tolist()
    if stores != ["CA_1"]:
        raise ValueError("This classroom demo is scoped to the CA_1 study.")
    grouped = pd.read_parquet(inventory / "residual_buffers.parquet")
    alphas = meta["alphas"]
    date_sets = {name: pd.date_range(**config["splits"][name]) for name in ("policy_validation", "warmup", "final_test")}
    sales = {name: _load_sales(processed, cohort, dates, stores) for name, dates in date_sets.items()}
    last_days = date_sets["warmup"][-14:]
    prior_sales = sales["warmup"][-len(last_days):]
    chosen_ml = next(row["model"] for row in meta["cases"] if row["model"].startswith("lightgbm"))
    cases = [row for row in meta["cases"] if row["scenario"] == "base" and row["selection_mode"] == "fill_floor"]

    def residuals(horizon):
        return {
            name: cohort[["store_id", "item_id"]]
            .merge(rows, on=["store_id", "item_id"], validate="one_to_one")[quantile_columns(alphas, "residual")]
            .to_numpy()
            for name, rows in grouped.loc[grouped["horizon"].eq(horizon)].groupby("model")
        }

    opening = {}
    forecasts = {}
    for lead in config["inventory"]["lead_times"]:
        horizon = lead + 1
        residual = residuals(horizon)
        blocks = {
            phase: _forecast_block(forecasting / "forecasts", cohort, dates, horizon, alphas, phase)
            for phase, dates in date_sets.items()
        }
        incumbent = Candidate(INCUMBENT, "own_buffer", max(alphas))
        state = _opening(processed, cohort, date_sets["policy_validation"], stores, lead)
        for phase in ("policy_validation", "warmup"):
            _, state, _ = simulate(sales[phase], _target(blocks[phase], incumbent, residual, alphas), state)
        opening[lead] = state
        forecasts[lead] = {}
        for case in cases:
            if case["lead"] != lead:
                continue
            candidate = Candidate(case["model"], case["rule"], case["alpha"])
            forecasts[lead][case["model"]] = {
                "target": _target(blocks["final_test"], candidate, residual, alphas),
                "median": blocks["final_test"][case["model"]][..., 1],
                "rule": case["rule"],
                "alpha": case["alpha"],
                "feasible": case["validation_feasible"],
            }

    # A representative item per regime, with a suggested first-day order to demonstrate.
    cards = []
    for regime, goal in [("Stable", 5), ("Volatile", 6), ("Intermittent", 2)]:
        state = opening[7]
        positions = state.on_hand + state.inbound.sum(axis=0)
        first = np.ceil(np.maximum(0, forecasts[7][chosen_ml]["target"][0] - positions))
        eligible = cohort["regime"].eq(regime).to_numpy() & (first >= 2) & (first < 18) & (prior_sales.sum(axis=0) > 0)
        values = np.flatnonzero(eligible)
        if not len(values):
            raise ValueError(f"No teaching example available for {regime}.")
        mean = prior_sales.mean(axis=0)
        position = min(values, key=lambda index: (abs(mean[index] - goal) + abs(first[index] - 7) * .35, cohort["item_id"].iat[index]))
        row = cohort.iloc[position]
        card = {
            "id": row["item_id"], "regime": regime, "department": row["dept_id"],
            "demand": sales["final_test"][:, position].astype(int).tolist(),
            "history": prior_sales[:, position].astype(int).tolist(),
            "lead": {},
        }
        for lead, state in opening.items():
            details = {}
            for name, prediction in forecasts[lead].items():
                targets = prediction["target"][:, [position]]
                just_one = type(state)(state.on_hand[[position]], state.inbound[:, [position]])
                total, ending, _ = simulate(sales["final_test"][:, [position]], targets, just_one)
                details[name] = {
                    "target": targets[:, 0].tolist(), "median": prediction["median"][:, position].tolist(),
                    "rule": prediction["rule"], "alpha": prediction["alpha"], "feasible": prediction["feasible"],
                    "golden": {**{key: int(total[key].item()) for key in ("demand", "filled", "lost", "ordered")},
                               "stock": int(ending.on_hand.item()), "pipeline": int(ending.inbound.sum())},
                }
            card["lead"][lead] = {"on_hand": int(state.on_hand[position]), "inbound": state.inbound[:, position].tolist(), "models": details}
        cards.append(card)

    columns = ["phase", "lead", "regime", "model", "rule", "alpha", "cost", "fill_rate", "mean_on_hand",
               "lost", "demand", "stockout_rate", "ending_pipeline"]
    results = pd.read_csv(inventory / "selected_policies.csv")
    selected = results.loc[results["scenario"].eq("base") & results["selection_mode"].eq("fill_floor"),
                           [*columns, "validation_feasible"]]
    frontier = pd.read_csv(inventory / "base_frontiers.csv", usecols=columns)
    extra = pd.read_csv(inventory / "forecast_inventory_pairs.csv")
    evidence = extra.loc[extra["scenario"].eq("base") & extra["selection_mode"].eq("fill_floor"),
                         ["lead", "regime", "model", "q90_spl", "forecast_origins"]]
    payload = {
        "store": "CA_1", "model": chosen_ml, "fill_floor": meta["fill_floor"],
        "dates": date_sets["final_test"].strftime("%Y-%m-%d").tolist(),
        "history_dates": last_days.strftime("%Y-%m-%d").tolist(),
        "cases": cards, "results": selected.to_dict("records"), "frontier": frontier.to_dict("records"),
        "evidence": evidence.to_dict("records"),
        "validation_winners": {lead: row["model"] for lead, row in meta["winners"].items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("globalThis.STOCKDESK = " + json.dumps(payload, separators=(",", ":"), allow_nan=False) + ";\n")
    print("Exported a 24-day replay for three teaching SKUs, six models, and both lead times.")


if __name__ == "__main__":
    main()

