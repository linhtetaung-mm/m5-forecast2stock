"""Train, calibrate and serialize the direct M5 forecasts and baseline comparisons."""

from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_audit import load_config
from src.features import BASELINES, DAY, KEY, OriginStore, model_features, phase_origins
from src.metrics import enforce_quantiles, evaluate, quantile_columns, training_scales

CATEGORIES = {"store_code", "dept_code", "target_weekday", "target_month", "target_event_code_1", "target_event_code_2"}
VARIANTS = {"lightgbm_no_price": False, "lightgbm_with_price": True}
REPLAY_PHASES = ("policy_validation", "warmup", "final_test")


def _hash(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _float_frame(frame, columns):
    values = frame.loc[:, columns].astype(np.float32)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Feature matrix contains infinite values.")
    return values


def fit_quantiles(frame, alphas, params, *, use_price, seed):
    """One pooled direct model per alpha for the supplied family."""
    columns = model_features(frame, use_price=use_price)
    dataset = lgb.Dataset(
        _float_frame(frame, columns),
        label=frame["actual"].to_numpy(dtype=np.float32),
        categorical_feature=[name for name in columns if name in CATEGORIES],
        params={"max_bin": params.get("max_bin", 63)},
        free_raw_data=True,
    )
    rounds = params["num_boost_round"]
    options = {
        "objective": "quantile", "verbosity": -1, "force_col_wise": True, "seed": seed,
        **{key: value for key, value in params.items() if key not in {"num_boost_round", "refit_on_training_and_development"}},
    }
    return {
        alpha: lgb.train({**options, "alpha": alpha}, dataset, num_boost_round=rounds)
        for alpha in alphas
    }


def predict_quantiles(design, models):
    first = next(iter(models.values()))
    data = _float_frame(design, first.feature_name())
    return np.column_stack([booster.predict(data) for booster in models.values()])


def _reference(design, model):
    source = design[model].to_numpy(dtype=float)
    if model in {"croston", "sba"}:
        source = np.where(design["regime"].eq("Intermittent"), source, design["seasonal_naive"])
    if not np.isfinite(source).all():
        raise ValueError("Insufficient history for the requested baseline.")
    return source


def _lookups(design, groups, table, columns):
    keys = pd.MultiIndex.from_frame(design[groups]) if len(groups) > 1 else pd.Index(design[groups[0]])
    values = table.set_index(groups)[columns].reindex(keys).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Calibration is missing for a required key {groups}.")
    return values


def calibrate(design, raw_ml, alphas):
    """Fit frozen baseline SKU offsets and separate regime-level ML corrections."""
    if design["actual"].isna().any():
        raise ValueError("Calibration requires complete observed labels.")
    columns = quantile_columns(alphas, "offset")
    sku = ["store_id", "item_id"]
    baseline_tables = []
    for model in BASELINES:
        residuals = design[sku].assign(residual=design["actual"] - _reference(design, model))
        offsets = residuals.groupby(sku, sort=True)["residual"].quantile(alphas, interpolation="higher").unstack()
        offsets.columns = columns
        baseline_tables.append(offsets.reset_index().assign(model=model))
    ordered = enforce_quantiles(raw_ml)
    ml = design[["regime"]].copy()
    for index, column in enumerate(columns):
        ml[column] = design["actual"].to_numpy() - ordered[:, index]
    ml_table = pd.concat(
        [ml.groupby("regime")[column].quantile(alpha, interpolation="higher").rename(column)
         for column, alpha in zip(columns, alphas, strict=True)],
        axis=1,
    ).reset_index()
    return pd.concat(baseline_tables, ignore_index=True), ml_table


def prediction_table(design, model, alphas, *, raw=None, calibration=None, family):
    """Use the same point and quantile rules for historical replay and inference."""
    median = list(alphas).index(0.5)
    offsets = quantile_columns(alphas, "offset")
    if model in BASELINES:
        point = _reference(design, model)
        raw = np.broadcast_to(point[:, None], (len(point), len(alphas)))
        if calibration is None:
            corrected = raw
        else:
            table = calibration[0].loc[lambda rows: rows["model"].eq(model)]
            corrected = raw + _lookups(design, ["store_id", "item_id"], table, offsets)
    else:
        ordered = enforce_quantiles(raw)
        corrected = ordered if calibration is None else ordered + _lookups(design, ["regime"], calibration[1], offsets)
        point = enforce_quantiles(corrected)[:, median]
    columns = [*KEY, "target_date", "horizon", "regime"]
    forecast = design[columns + (["actual"] if "actual" in design else [])].copy()
    forecast["family"], forecast["model"], forecast["point"] = family, model, point.astype(np.float32)
    forecast[quantile_columns(alphas, "raw")] = np.asarray(raw, dtype=np.float32)
    forecast[quantile_columns(alphas)] = enforce_quantiles(corrected).astype(np.float32)
    return forecast


def _with_regimes(design, regimes):
    design = design.merge(regimes, on=["store_id", "item_id"], how="left", validate="many_to_one")
    if design["regime"].isna().any():
        raise ValueError("Missing training regime for a forecast row.")
    return design


def _score(frame, alphas, scale, phase, family, horizon, end, longest):
    mask = frame["origin"].le(pd.Timestamp(end) - DAY * longest)
    scored = frame.loc[mask].merge(scale, on=["store_id", "item_id"], validate="many_to_one")
    return evaluate(scored, alphas).assign(phase=phase, family=family, horizon=horizon)


def _phase_base(stores, phase, longest, stride=1):
    dates = phase_origins(phase, longest)
    return [store.origins(dates, training_stride=stride) for store in stores]


def _design(stores, bases, phase, family, horizon, keep_unlabeled=False):
    return pd.concat(
        [store.design(base, horizon, family=family, labels=phase, keep_unlabeled=keep_unlabeled)
         for store, base in zip(stores, bases, strict=True)],
        ignore_index=True,
    )


def _fit_frame(stores, bases, phase, family, horizons, use_price):
    tables = []
    for horizon in horizons:
        design = _design(stores, bases, phase, family, horizon)
        columns = [*model_features(design, use_price=use_price), "actual"]
        tables.append(_float_frame(design, columns))
    return pd.concat(tables, ignore_index=True)


class ForecastWriter:
    """Append horizon slices without buffering an entire forecast file."""

    def __init__(self, path):
        self.path = path
        self.writer = None

    def __enter__(self):
        return self

    def append(self, frame):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)

    def __exit__(self, *exc):
        if self.writer is not None:
            self.writer.close()


def _chronological_inputs(processed, config, regime_path):
    snapshot = json.loads((processed / "audit.json").read_text(encoding="utf-8"))
    for name in ("data", "splits", "regimes"):
        if snapshot["config"][name] != config[name]:
            raise ValueError(f"Part 1 artifacts use different {name} settings. Re-run ingestion and regimes.")
    regimes = pd.read_parquet(regime_path / "sku_regimes.parquet", columns=["store_id", "item_id", "regime"])
    profile = json.loads((regime_path / "summary.json").read_text(encoding="utf-8"))
    if (profile["training_start"], profile["training_end"]) != (config["splits"]["training"]["start"], config["splits"]["training"]["end"]):
        raise ValueError("Regime artifacts were generated using a different training period.")
    for option in ("zero_share_threshold", "cv2_threshold", "min_positive_days"):
        if profile[option] != config["regimes"][option]:
            raise ValueError(f"Regime artifact uses a different {option}.")
    return snapshot, regimes


def run_forecasting(config_path, *, output_dir=None, processed_dir=None):
    """Run development, frozen-model calibration, and independent assessment."""
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    options = config["forecast"]
    phases = config["splits"]
    settings = options["lightgbm"]
    alphas = options["quantiles"]
    if 0.5 not in alphas:
        raise ValueError("The forecast contract requires a median quantile.")
    processed = Path(processed_dir).resolve() if processed_dir else root / config["data"]["processed_dir"]
    regimes_path = processed / "regimes" if processed_dir else root / config["regimes"]["output_dir"]
    destination = Path(output_dir).resolve() if output_dir else root / options["output_dir"]
    if destination.exists() or destination.is_relative_to(processed) or processed.is_relative_to(destination):
        raise ValueError(f"Choose a new forecast output directory, separate from Part 1 data: {destination}")
    snapshot, regimes = _chronological_inputs(processed, config, regimes_path)
    store_names = config["data"]["stores"]
    stores = [OriginStore.from_processed(processed, name, config) for name in store_names]
    expected = sum(len(store.items) for store in stores)
    if len(regimes) != expected or regimes.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Part 1 regime cohort and forecast series do not match.")
    families = {"daily": options["daily_horizons"], "cumulative": options["cumulative_horizons"]}
    longest = max(max(horizons) for horizons in families.values())
    scale_tables = [training_scales(store, phases["training"], families["cumulative"]) for store in stores]
    scales = {key: pd.concat([table[key] for table in scale_tables], ignore_index=True) for key in scale_tables[0]}
    training_stride = options["training_origin_stride"]
    training_base = _phase_base(stores, phases["training"], 1, training_stride)
    dev_base = _phase_base(stores, phases["development"], longest)
    refit_phase = {"start": phases["training"]["start"], "end": phases["development"]["end"]}
    if not settings["refit_on_training_and_development"]:
        refit_phase = phases["training"]
    refit_base = _phase_base(stores, refit_phase, 1, training_stride)
    calibration_base = _phase_base(stores, phases["calibration"], longest)
    seed = config["project"]["random_seed"]
    metrics, selection = [], {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".forecast-", dir=destination.parent) as temp:
        stage = Path(temp) / "run"
        bundle = stage / "bundle"
        forecasts = stage / "forecasts"
        bundle.mkdir(parents=True)
        forecasts.mkdir()
        regimes.to_parquet(bundle / "regimes.parquet", index=False)
        for family, horizons in families.items():
            fit = _fit_frame(stores, training_base, phases["training"], family, horizons, True)
            candidate_metrics = {}
            for model, price in VARIANTS.items():
                if price and not options["feature_settings"]["include_as_of_price"]:
                    continue
                started = perf_counter()
                print(f"{family}: fitting {model} on {len(fit):,} training rows", flush=True)
                boosters = fit_quantiles(fit, alphas, settings, use_price=price, seed=seed)
                records = []
                for horizon in horizons:
                    design = _with_regimes(_design(stores, dev_base, phases["development"], family, horizon), regimes)
                    pred = prediction_table(design, model, alphas, raw=predict_quantiles(design, boosters), family=family)
                    scale = scales[(family, horizon if family == "cumulative" else 1)]
                    records.append(_score(pred, alphas, scale, "development", family, horizon, phases["development"]["end"], longest))
                scores = pd.concat(records, ignore_index=True)
                overall = scores.loc[scores["regime"].eq("All")]
                if overall["spl"].notna().all():
                    criterion = overall["spl"].mean()
                else:
                    criterion = overall["pinball"].mean()
                candidate_metrics[model] = float(criterion)
                metrics.append(scores)
                print(f"{family}: {model} complete in {perf_counter() - started:.1f}s", flush=True)
                del boosters
            del fit
            chosen = min(candidate_metrics, key=candidate_metrics.get)
            refit = _fit_frame(stores, refit_base, refit_phase, family, horizons, VARIANTS[chosen])
            print(f"{family}: refitting {chosen} on {len(refit):,} rows", flush=True)
            boosters = fit_quantiles(refit, alphas, settings, use_price=VARIANTS[chosen], seed=seed)
            selection[family] = {"selected": chosen, "development_scores": candidate_metrics, "refit_rows": len(refit)}
            del refit
            for index, model in enumerate(boosters.values()):
                model.save_model(str(bundle / f"{family}_{index}.txt"))
            corrections = {}
            with ForecastWriter(forecasts / f"calibration_{family}.parquet") as writer:
                for horizon in horizons:
                    design = _with_regimes(_design(stores, calibration_base, phases["calibration"], family, horizon), regimes)
                    raw = predict_quantiles(design, boosters)
                    calibration = calibrate(design, raw, alphas)
                    corrections[horizon] = calibration
                    calibration[0].to_parquet(bundle / f"{family}_{horizon}_baseline_offsets.parquet", index=False)
                    calibration[1].to_parquet(bundle / f"{family}_{horizon}_ml_offsets.parquet", index=False)
                    for name in (*BASELINES, chosen):
                        writer.append(prediction_table(design, name, alphas, raw=raw, family=family))
            for phase in REPLAY_PHASES:
                dates = phases[phase]
                bases = _phase_base(stores, dates, 1)
                with ForecastWriter(forecasts / f"{phase}_{family}.parquet") as writer:
                    for horizon in horizons:
                        design = _with_regimes(_design(stores, bases, dates, family, horizon, True), regimes)
                        raw = predict_quantiles(design, boosters)
                        tables = [
                            prediction_table(design, name, alphas, raw=raw, calibration=corrections[horizon], family=family)
                            for name in (*BASELINES, chosen)
                        ]
                        for table in tables:
                            writer.append(table)
                        if phase != "warmup":
                            scale = scales[(family, horizon if family == "cumulative" else 1)]
                            metrics.append(_score(pd.concat(tables, ignore_index=True), alphas, scale, phase, family, horizon, dates["end"], longest))
                print(f"{family}: scored {phase}", flush=True)
        pd.concat(metrics, ignore_index=True).to_csv(stage / "metrics.csv", index=False)
        pd.concat([table.assign(family=family, scale_horizon=horizon) for (family, horizon), table in scales.items()], ignore_index=True).to_parquet(stage / "training_scales.parquet", index=False)
        metadata = {
            "schema_version": 1, "alphas": alphas, "horizons": families, "selection": selection,
            "forecast_settings": options, "stores": store_names, "event_codes": stores[0].event_codes, "config_sha256": _hash(config_path),
            "sources": snapshot["sources"], "splits": phases, "refit_phase": refit_phase,
            "score_origins": "All horizons scored on common complete longest-window origins within each phase.",
            "baseline_note": "Croston and SBA use seasonal naive outside the Intermittent regime.",
            "quantile_columns": dict(zip(map(str, alphas), quantile_columns(alphas), strict=True)),
            "runtime": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            "members_sha256": {member.name: _hash(member) for member in sorted(bundle.iterdir())},
        }
        _json(bundle / "metadata.json", metadata)
        _json(stage / "run_manifest.json", {"files_sha256": {
            member.relative_to(stage).as_posix(): _hash(member) for member in sorted(stage.rglob("*")) if member.is_file()
        }})
        stage.rename(destination)
    return destination


def load_bundle(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1:
        raise ValueError("Unsupported forecast bundle version.")
    for name, expected in metadata["members_sha256"].items():
        if Path(name).name != name or _hash(directory / name) != expected:
            raise ValueError(f"Forecast bundle checksum does not match: {name}")
    return metadata


def score_bundle(store, origins, directory, *, family="cumulative"):
    """Score new origins from a serialized bundle and available as-of input records."""
    directory = Path(directory)
    metadata = load_bundle(directory)
    if store.store not in metadata["stores"]:
        raise ValueError("The selected store was outside the trained bundle cohort.")
    if store.options != metadata["forecast_settings"] or store.event_codes != metadata["event_codes"]:
        raise ValueError("The forecast feature configuration differs from the trained bundle.")
    regimes = pd.read_parquet(directory / "regimes.parquet")
    alphas = metadata["alphas"]
    models = {alpha: lgb.Booster(model_file=str(directory / f"{family}_{index}.txt")) for index, alpha in enumerate(alphas)}
    chosen = metadata["selection"][family]["selected"]
    base = store.origins(origins)
    for horizon in metadata["horizons"][family]:
        design = _with_regimes(store.design(base, horizon, family=family), regimes)
        corrections = (
            pd.read_parquet(directory / f"{family}_{horizon}_baseline_offsets.parquet"),
            pd.read_parquet(directory / f"{family}_{horizon}_ml_offsets.parquet"),
        )
        raw = predict_quantiles(design, models)
        for name in (*BASELINES, chosen):
            yield prediction_table(design, name, alphas, raw=raw, calibration=corrections, family=family)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--processed-dir", type=Path)
    args = parser.parse_args()
    print(run_forecasting(args.config, output_dir=args.output_dir, processed_dir=args.processed_dir))


if __name__ == "__main__":
    main()

