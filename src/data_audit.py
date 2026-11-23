"""Validate the M5 source files and export a typed, store-partitioned daily panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

SPLIT_NAMES = ("training", "development", "calibration", "policy_validation", "warmup", "final_test")
SALES_KEYS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
PRICE_KEYS = ("store_id", "item_id", "wm_yr_wk")
EVENT_COLUMNS = ("event_name_1", "event_type_1", "event_name_2", "event_type_2")
SNAP_COLUMNS = ("snap_CA", "snap_TX", "snap_WI")
CALENDAR_COLUMNS = ("date", "wm_yr_wk", "weekday", "wday", "month", "year", "d", *EVENT_COLUMNS, *SNAP_COLUMNS)
RAW_FILE_KEYS = ("sales", "calendar", "prices")
STORES = ("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3")
DAY = pd.Timedelta(days=1)


class AuditError(ValueError):
    """An input failed an M5 or project configuration contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _date(value: object, name: str) -> pd.Timestamp:
    _require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
             f"{name} must use quoted YYYY-MM-DD format.")
    try:
        return pd.Timestamp(value)
    except ValueError as error:
        raise AuditError(f"{name} is an invalid date: {value}") from error


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def load_config(config_path: str | Path) -> dict:
    """Load YAML and validate ingestion settings and inclusive target dates."""
    config_path = Path(config_path).expanduser()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AuditError(f"Cannot read configuration {config_path}: {error}") from error

    _require(isinstance(config, dict), "Configuration must be a YAML mapping.")
    for key in ("data", "splits", "forecast"):
        _require(isinstance(config.get(key), dict), f"Missing or invalid configuration field {key}.")
    data, splits, forecast = config["data"], config["splits"], config["forecast"]
    stores = data.get("stores")
    _require(isinstance(stores, list) and len(stores) > 0 and all(store in STORES for store in stores),
             "data.stores must be a nonempty list of M5 store IDs.")
    _require(len(set(stores)) == len(stores), "data.stores contains duplicate store IDs.")
    _require(_positive_integer(data.get("expected_series_per_store")),
             "data.expected_series_per_store must be a positive integer.")
    for key in ("raw_dir", "processed_dir"):
        _require(isinstance(data.get(key), str) and bool(data[key].strip()), f"data.{key} must be a path.")
    files = data.get("files")
    _require(isinstance(files, dict) and set(files) == set(RAW_FILE_KEYS),
             "data.files must name the sales, calendar and prices source files.")
    for key, name in files.items():
        _require(isinstance(name, str) and bool(name.strip()) and Path(name).name == name
                 and name not in {".", ".."}, f"data.files.{key} must be a filename in raw_dir.")
    _require(len(set(files.values())) == len(files), "data.files must refer to three different files.")
    digests = data.get("expected_sha256")
    if digests is not None:
        _require(isinstance(digests, dict) and set(digests) == set(RAW_FILE_KEYS),
                 "data.expected_sha256 must contain a hash for each source file.")
        for key, value in digests.items():
            _require(isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) is not None,
                     f"data.expected_sha256.{key} must be a SHA-256 hexadecimal digest.")

    panel_start = _date(data.get("panel_start"), "data.panel_start")
    panel_end = _date(data.get("panel_end"), "data.panel_end")
    _require(panel_start <= panel_end, "data.panel_start must precede or equal data.panel_end.")

    for key in ("daily_horizons", "cumulative_horizons"):
        horizons = forecast.get(key)
        _require(isinstance(horizons, list) and len(horizons) > 0
                 and all(_positive_integer(value) for value in horizons),
                 f"forecast.{key} must be a nonempty list of positive integers.")
        _require(horizons == sorted(set(horizons)), f"forecast.{key} must increase without duplicates.")
    daily = forecast["daily_horizons"]
    cumulative = forecast["cumulative_horizons"]
    _require(max(cumulative) <= max(daily), "Cumulative horizons must fit the daily forecast horizon.")
    quantiles = forecast.get("quantiles")
    _require(isinstance(quantiles, list) and len(quantiles) > 0
             and all(type(alpha) in (int, float) and 0 < alpha < 1 for alpha in quantiles),
             "forecast.quantiles must be a nonempty list of probabilities strictly between 0 and 1.")
    _require(quantiles == sorted(set(quantiles)), "forecast.quantiles must increase without duplicates.")

    _require(set(splits) == set(SPLIT_NAMES), f"splits must contain {', '.join(SPLIT_NAMES)}.")
    previous_end = None
    for name in SPLIT_NAMES:
        section = splits[name]
        _require(isinstance(section, dict) and set(section) == {"start", "end"},
                 f"splits.{name} must contain start and end dates.")
        start = _date(section["start"], f"splits.{name}.start")
        end = _date(section["end"], f"splits.{name}.end")
        _require(panel_start <= start <= end <= panel_end, f"splits.{name} falls outside the panel or is reversed.")
        if previous_end is not None:
            _require(start == previous_end + DAY, f"splits.{name} must begin immediately after the previous split.")
        if name != "warmup":
            _require((end - start).days + 1 >= max(daily),
                     f"splits.{name} is shorter than the longest forecast horizon.")
        previous_end = end
    features = forecast.get("feature_settings", {})
    _require(isinstance(features, dict), "forecast.feature_settings must be a mapping.")
    lags = features.get("lags", [])
    _require(isinstance(lags, list) and all(type(lag) is int and lag >= 0 for lag in lags),
             "forecast.feature_settings.lags must contain nonnegative integers.")
    windows = features.get("rolling_windows", [])
    _require(isinstance(windows, list) and all(_positive_integer(window) for window in windows),
             "forecast.feature_settings.rolling_windows must contain positive integers.")
    zero_window = features.get("zero_share_window", 0)
    _require("zero_share_window" not in features or _positive_integer(zero_window),
             "forecast.feature_settings.zero_share_window must be a positive integer.")
    # The first training target follows the forecast origin. Lag k also needs k full days before the origin.
    history = max(max(lags, default=-1) + 1, max(windows, default=0), zero_window)
    available = (pd.Timestamp(splits["training"]["start"]) - panel_start).days
    _require(available >= history, f"The panel needs at least {history} pre-training history days for the declared features.")
    return config


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _header(path: Path, required_columns: tuple[str, ...]) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        columns = next(csv.reader(source), [])
    _require(len(columns) == len(set(columns)), f"{path.name} contains duplicate column names.")
    missing = set(required_columns) - set(columns)
    _require(not missing, f"{path.name} is missing columns: {', '.join(sorted(missing))}.")
    return columns


def _read_calendar(path: Path) -> pd.DataFrame:
    _header(path, CALENDAR_COLUMNS)
    integers = ("wm_yr_wk", "wday", "month", "year", *SNAP_COLUMNS)
    types = {column: "int32" if column in integers else "string" for column in CALENDAR_COLUMNS}
    calendar = pd.read_csv(path, usecols=list(CALENDAR_COLUMNS), dtype=types)
    _require(not calendar.empty, f"{path.name} is empty.")
    _require(calendar[["date", "weekday", "d"]].notna().all().all(), "The calendar contains missing dates or day codes.")
    calendar["date"] = pd.to_datetime(calendar["date"], format="%Y-%m-%d")
    _require(calendar["date"].notna().all(), "The calendar contains invalid dates.")
    calendar = calendar.sort_values("date", ignore_index=True)
    expected_days = "d_" + pd.Series(np.arange(1, len(calendar) + 1), dtype="int64").astype("string")
    _require(calendar["d"].equals(expected_days), "Calendar d_* codes are duplicated, reordered or incomplete.")
    _require(calendar["date"].diff().iloc[1:].eq(DAY).all(), "Calendar dates must be unique and consecutive.")
    weekday = calendar["date"].dt.dayofweek
    _require(calendar["weekday"].eq(calendar["date"].dt.day_name()).all(),
             "Calendar weekday names disagree with their dates.")
    _require(calendar["wday"].eq((weekday + 2) % 7 + 1).all(), "Calendar wday must encode Saturday as 1, through Friday as 7.")
    _require(calendar["month"].eq(calendar["date"].dt.month).all() and calendar["year"].eq(calendar["date"].dt.year).all(),
             "Calendar month or year disagrees with its date.")
    _require(calendar[list(SNAP_COLUMNS)].isin((0, 1)).all().all(), "Calendar SNAP flags must be either 0 or 1.")
    for slot in (1, 2):
        _require(calendar[f"event_name_{slot}"].isna().equals(calendar[f"event_type_{slot}"].isna()),
                 f"Calendar event names and types disagree in slot {slot}.")
    calendar["week_end"] = calendar["date"] + pd.to_timedelta(7 - calendar["wday"], unit="D")
    week_pairs = calendar[["wm_yr_wk", "week_end"]].drop_duplicates()
    _require(calendar["wm_yr_wk"].gt(0).all() and week_pairs["wm_yr_wk"].is_unique
             and week_pairs["week_end"].is_unique, "Calendar weeks do not map one-to-one to Saturday–Friday date ranges.")
    calendar["weekday"] = weekday.astype("int8")
    calendar["wday"] = calendar["wday"].astype("int8")
    calendar["month"] = calendar["month"].astype("int8")
    calendar["year"] = calendar["year"].astype("int16")
    for column in SNAP_COLUMNS:
        calendar[column] = calendar[column].astype("int8")
    for column in EVENT_COLUMNS:
        calendar[column] = calendar[column].astype("category")
    return calendar


def _read_sales(path: Path, calendar: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    columns = _header(path, SALES_KEYS)
    raw_days = [name for name in columns if name not in SALES_KEYS]
    _require(len(raw_days) > 0 and raw_days == [f"d_{day}" for day in range(1, len(raw_days) + 1)],
             f"{path.name} sales columns must be consecutive d_1, d_2, … in chronological order.")
    _require(len(raw_days) <= len(calendar), "The calendar does not cover every observed sales day.")
    observed_dates = calendar.iloc[: len(raw_days)]
    panel_days = observed_dates.loc[observed_dates["date"].between(settings["panel_start"], settings["panel_end"])]
    expected_days = (pd.Timestamp(settings["panel_end"]) - pd.Timestamp(settings["panel_start"])).days + 1
    _require(len(panel_days) == expected_days, "The configured panel has dates without observed sales columns.")
    chosen_days = panel_days["d"].tolist()
    stores = settings["stores"]
    # Scan identifiers across all stores without allocating the full all-store wide array.
    identifiers = pd.read_csv(path, usecols=list(SALES_KEYS), dtype="string", skip_blank_lines=False)
    nonempty = identifiers.notna().all().all() and identifiers.ne("").all().all()
    _require(nonempty, f"{path.name} contains blank series identifiers.")
    _require(identifiers["id"].is_unique and not identifiers.duplicated(["store_id", "item_id"]).any(),
             f"{path.name} must have one row per id and per (store_id, item_id).")
    _require(identifiers["store_id"].isin(STORES).all(), f"{path.name} contains unknown M5 stores.")
    _require(identifiers["state_id"].eq(identifiers["store_id"].str.slice(0, 2)).all(),
             f"{path.name} state_id disagrees with store_id.")
    _require(identifiers["id"].eq(identifiers["item_id"] + "_" + identifiers["store_id"] + "_validation").all(),
             f"{path.name} IDs must match item, store and the _validation suffix.")
    taxonomy = identifiers[["item_id", "dept_id", "cat_id"]].drop_duplicates()
    _require(taxonomy["item_id"].is_unique, f"{path.name} maps an item to multiple departments or categories.")
    counts = identifiers["store_id"].value_counts()
    for store in stores:
        count = int(counts.get(store, 0))
        _require(count == settings["expected_series_per_store"],
                 f"{store} has {count} sales series, expected {settings['expected_series_per_store']}.")
    coverage = {
        "start": observed_dates["date"].iloc[0].date().isoformat(),
        "end": observed_dates["date"].iloc[-1].date().isoformat(),
        "days": len(observed_dates),
        "raw_series": len(identifiers),
        "raw_series_by_store": {str(store): int(counts[store]) for store in sorted(counts.index)},
    }
    return identifiers, panel_days.reset_index(drop=True), coverage


def _store_sales(path: Path, identifiers: pd.DataFrame, days: list[str], store: str) -> pd.DataFrame:
    wanted_rows = set(np.flatnonzero(identifiers["store_id"].eq(store)) + 1)
    values = pd.read_csv(
        path, usecols=days, dtype="int64",
        skiprows=lambda line: line != 0 and line not in wanted_rows,
    )
    magnitudes = values.to_numpy(copy=False)
    _require(len(values) == len(wanted_rows), f"Unexpected sales row count after selecting {store}.")
    _require(magnitudes.min() >= 0 and magnitudes.max() <= np.iinfo(np.int32).max,
             f"{path.name} quantities for {store} must be nonnegative integers fitting in int32.")
    metadata = identifiers.loc[identifiers["store_id"].eq(store)].reset_index(drop=True)
    return pd.concat([metadata, values.astype("int32")], axis=1).sort_values("item_id", ignore_index=True)


def _read_prices(path: Path, calendar: pd.DataFrame, identifiers: pd.DataFrame, stores: list[str], folder: Path) -> int:
    _header(path, (*PRICE_KEYS, "sell_price"))
    types = {"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float64"}
    # Only one input chunk is resident while the selected rows are partitioned on disk.
    import pyarrow as pa
    import pyarrow.parquet as pq

    writers = {}
    try:
        for chunk in pd.read_csv(path, usecols=list(types), dtype=types, chunksize=250_000):
            _require(chunk[list(PRICE_KEYS)].notna().all().all(), f"{path.name} contains missing price keys.")
            _require(chunk["store_id"].isin(STORES).all(), f"{path.name} contains unknown M5 stores.")
            for store, block in chunk.loc[chunk["store_id"].isin(stores)].groupby("store_id", sort=False):
                table = pa.Table.from_pandas(block, preserve_index=False)
                if store not in writers:
                    writers[store] = pq.ParquetWriter(folder / f"raw-{store}.parquet", table.schema)
                writers[store].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()

    weeks = calendar[["wm_yr_wk", "week_end"]].drop_duplicates()
    rows = 0
    schema = {"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float64"}
    for store in stores:
        staging_price = folder / f"raw-{store}.parquet"
        if staging_price.is_file():
            prices = pd.read_parquet(staging_price).astype(schema)
            staging_price.unlink()
        else:
            prices = pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in schema.items()})
        _require(not prices.duplicated(list(PRICE_KEYS)).any(),
                 f"{path.name} contains duplicate (store_id, item_id, wm_yr_wk) keys for {store}.")
        _require(np.isfinite(prices["sell_price"]).all() and prices["sell_price"].gt(0).all(),
                 f"{path.name} contains a missing, non-finite or nonpositive recorded price for {store}.")
        source_items = identifiers.loc[identifiers["store_id"].eq(store), "item_id"]
        _require(prices["item_id"].isin(source_items).all(), f"{path.name} contains items absent from {store} sales.")
        _require(prices["wm_yr_wk"].isin(weeks["wm_yr_wk"]).all(), f"{path.name} contains weeks absent from the calendar.")
        history = prices.merge(weeks, on="wm_yr_wk", how="left", validate="many_to_one")
        history = history.sort_values(["item_id", "week_end"], ignore_index=True)
        history.to_parquet(folder / f"{store}.parquet", index=False)
        rows += len(history)
    return rows


def _make_panel(sales: pd.DataFrame, days: pd.DataFrame, prices: pd.DataFrame, store: str) -> pd.DataFrame:
    n_days, n_series = len(days), len(sales)
    panel = {
        key: pd.Categorical(np.repeat(sales[key].to_numpy(), n_days))
        for key in SALES_KEYS
    }
    for key in ("date", "wm_yr_wk", "weekday", "week_end"):
        panel[key] = np.tile(days[key].to_numpy(), n_series)
    for key in ("d", *EVENT_COLUMNS):
        panel[key] = pd.Categorical(np.tile(days[key].to_numpy(), n_series))
    panel["snap"] = np.tile(days[f"snap_{store[:2]}"].to_numpy(), n_series)
    panel["has_event"] = np.tile(days[["event_name_1", "event_name_2"]].notna().any(axis=1).to_numpy(), n_series)
    panel["sales"] = sales[days["d"].tolist()].to_numpy().ravel()

    # The grid is much smaller than a daily string-key join. Unmatched cells stay NaN.
    weeks = pd.Index(days["wm_yr_wk"].unique())
    price_grid = prices.pivot(index="item_id", columns="wm_yr_wk", values="sell_price")
    price_grid = price_grid.reindex(index=sales["item_id"], columns=weeks)
    panel["sell_price"] = price_grid.to_numpy()[:, weeks.get_indexer(days["wm_yr_wk"])].ravel()
    panel["price_missing"] = np.isnan(panel["sell_price"])
    return pd.DataFrame(panel)


def _summarize(panel: pd.DataFrame) -> dict:
    missing = panel["price_missing"]
    positive = panel["sales"].gt(0)
    rows = len(panel)
    return {
        "rows": rows,
        "units": int(panel["sales"].sum()),
        "positive_sales_rows": int(positive.sum()),
        "zero_sales_rows": int((~positive).sum()),
        "missing_price_rows": int(missing.sum()),
        "missing_price_share": float(missing.mean()) if rows else None,
        "missing_price_positive_sales_rows": int((missing & positive).sum()),
        "missing_price_zero_sales_rows": int((missing & ~positive).sum()),
    }


def _phase_coverage(splits: dict, horizons: list[int]) -> dict:
    phases = {}
    for name, dates in splits.items():
        start = pd.Timestamp(dates["start"])
        end = pd.Timestamp(dates["end"])
        n_days = (end - start).days + 1
        phases[name] = {
            **dates,
            "target_days": n_days,
            "complete_windows_by_horizon": {str(horizon): max(n_days - horizon + 1, 0) for horizon in horizons},
        }
    return phases


def _publish(stage: Path, destination: Path) -> None:
    if not destination.exists():
        stage.rename(destination)
        return
    # Existing runs may have regime or other downstream output in the same directory.
    # Publish the audit last so readers can use it as a completion marker.
    for relative in ("panel", "price_history"):
        folder = destination / relative
        _require(not folder.is_symlink(), f"Processed subdirectory must not be a symbolic link: {folder}.")
        folder.mkdir(exist_ok=True)
    for relative in ("panel", "price_history"):
        for artifact in sorted((stage / relative).iterdir()):
            os.replace(artifact, destination / relative / artifact.name)
    os.replace(stage / "calendar.parquet", destination / "calendar.parquet")
    os.replace(stage / "audit.json", destination / "audit.json")


def run_audit(
    config_path: str | Path,
    *,
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Validate the source snapshot, publish processed data and return its audit summary."""
    import pyarrow

    config_path = Path(config_path).expanduser().resolve()
    config_digest = _sha256(config_path) if config_path.is_file() else None
    config = load_config(config_path)
    settings = config["data"]
    root = config_path.parent.parent
    raw = (Path(raw_dir).expanduser() if raw_dir is not None else root / settings["raw_dir"]).resolve()
    destination = (Path(output_dir).expanduser() if output_dir is not None else root / settings["processed_dir"]).resolve()
    _require(not destination.is_relative_to(raw) and not raw.is_relative_to(destination),
             "The raw and processed directories must be separate and cannot contain one another.")
    _require(not destination.exists() or destination.is_dir(), f"Processed path is not a directory: {destination}.")
    paths = {key: raw / name for key, name in settings["files"].items()}
    missing = [path for path in paths.values() if not path.is_file()]
    _require(not missing, "Missing M5 source files: " + ", ".join(str(path) for path in missing))
    sources = {}
    for key, path in paths.items():
        digest = _sha256(path)
        expected = settings.get("expected_sha256", {}).get(key)
        _require(expected is None or digest == expected.lower(),
                 f"{path.name} SHA-256 mismatch: expected {expected}, found {digest}.")
        sources[key] = {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest}

    try:
        calendar = _read_calendar(paths["calendar"])
        identifiers, panel_days, observed = _read_sales(paths["sales"], calendar, settings)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".m5-audit-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "processed"
            (stage / "panel").mkdir(parents=True)
            (stage / "price_history").mkdir()
            calendar.to_parquet(stage / "calendar.parquet", index=False)
            price_rows = _read_prices(paths["prices"], calendar, identifiers, settings["stores"], stage / "price_history")
            audit = {
                "audit_version": 1,
                "config_sha256": config_digest,
                "config": json.loads(json.dumps(config, default=str, allow_nan=False)),
                "sources": sources,
                "runtime": {
                    "python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__, "pyarrow": pyarrow.__version__, "pyyaml": yaml.__version__,
                },
                "observed_sales": observed,
                "calendar": {
                    "rows": len(calendar),
                    "start": calendar["date"].iloc[0].date().isoformat(),
                    "end": calendar["date"].iloc[-1].date().isoformat(),
                    "path": "calendar.parquet",
                },
                "price_history": {"rows_across_selected_stores": price_rows, "availability_proxy": "week_end"},
                "panel": {"start": settings["panel_start"], "end": settings["panel_end"], "days": len(panel_days)},
                "splits": _phase_coverage(config["splits"], config["forecast"]["daily_horizons"]),
                "stores": {},
            }
            for store in settings["stores"]:
                panel_path = Path("panel") / f"{store}.parquet"
                prices_path = Path("price_history") / f"{store}.parquet"
                sales = _store_sales(paths["sales"], identifiers, panel_days["d"].tolist(), store)
                prices = pd.read_parquet(stage / prices_path)
                panel = _make_panel(sales, panel_days, prices, store)
                store_audit = {
                    "series": len(sales),
                    "panel_path": panel_path.as_posix(),
                    "price_history_path": prices_path.as_posix(),
                    "price_rows": len(prices),
                    "summary": _summarize(panel),
                    "splits": {},
                }
                for name, phase in config["splits"].items():
                    period = panel["date"].between(phase["start"], phase["end"])
                    store_audit["splits"][name] = _summarize(panel.loc[period, ["price_missing", "sales"]])
                panel.to_parquet(stage / panel_path, index=False)
                audit["stores"][store] = store_audit
                del panel
            _require(_sha256(config_path) == config_digest, "Configuration changed during ingestion. Run the audit again.")
            for key, path in paths.items():
                _require(_sha256(path) == sources[key]["sha256"], f"{path.name} changed during ingestion. Run the audit again.")
            (stage / "audit.json").write_text(
                json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
            _publish(stage, destination)
    except AuditError:
        raise
    except (OSError, ValueError) as error:
        raise AuditError(f"M5 ingestion failed: {error}") from error
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--raw-dir", type=Path, help="Override the raw CSV directory.")
    parser.add_argument("--output-dir", type=Path, help="Override the processed data directory.")
    arguments = parser.parse_args(argv)
    try:
        audit = run_audit(arguments.config, raw_dir=arguments.raw_dir, output_dir=arguments.output_dir)
    except AuditError as error:
        parser.exit(2, f"Data audit failed: {error}\n")
    for store, report in audit["stores"].items():
        summary = report["summary"]
        print(f"{store}  {report['series']:,} series  {summary['rows']:,} rows  "
              f"{summary['missing_price_rows']:,} rows without a matched weekly price")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
