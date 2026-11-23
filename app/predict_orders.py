"""Write a purchase-order recommendation CSV from a complete as-of snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import monitor_from_files, recommend_order_table


def _write_new(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as output:
        temp = Path(output.name)
        try:
            write(output)
            output.flush()
            os.fsync(output.fileno())
            os.link(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    for name in ("history", "prices", "calendar", "inventory", "inbound", "bundle", "policy", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--as-of", required=True, help="Last completed sales day, YYYY-MM-DD")
    parser.add_argument("--store", required=True)
    parser.add_argument("--lead-time", type=int, required=True)
    parser.add_argument("--new-sales", type=Path, help="Optional long CSV or Parquet with subsequent complete sales days")
    parser.add_argument("--model", help="Use this named model's feasible validation policy instead of the store default")
    parser.add_argument("--monitoring-output", type=Path)
    parser.add_argument("--service-ledger", type=Path, help="Optional actual demand, fulfilled and lost units by item/day")
    args = parser.parse_args()
    out = args.output.resolve()
    monitoring = args.monitoring_output.resolve() if args.monitoring_output else None
    if out.suffix.lower() != ".csv" or (monitoring and monitoring.suffix.lower() != ".json"):
        parser.error("Use .csv for orders and .json for monitoring.")
    if out.exists() or (monitoring and monitoring.exists()):
        parser.error("Choose new output files to preserve existing recommendations.")
    if args.service_ledger and not monitoring:
        parser.error("--service-ledger requires --monitoring-output.")
    input_paths = {
        "history": args.history, "prices": args.prices, "calendar": args.calendar,
        "bundle": args.bundle, "store": args.store, "new_sales": args.new_sales,
    }
    try:
        rows = recommend_order_table(
            args.config, **input_paths, inventory=args.inventory, inbound=args.inbound,
            policy=args.policy, as_of=args.as_of, lead=args.lead_time, model=args.model,
        )
        report = monitor_from_files(args.config, **input_paths, orders=rows, ledger=args.service_ledger) if monitoring else None
        if monitoring:
            _write_new(monitoring, lambda output: json.dump(report, output, indent=2, allow_nan=False))
        _write_new(out, lambda output: rows.to_csv(output, index=False, date_format="%Y-%m-%d"))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Scoring failed: {error}\n")
    print(f"{len(rows):,} checked items, {(rows['recommended_units'] > 0).sum():,} positive order lines. Saved {out}.")


if __name__ == "__main__":
    main()

