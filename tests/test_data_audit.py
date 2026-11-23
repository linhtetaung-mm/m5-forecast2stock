"""Small source files exercising the Part 1 M5 ingestion contract."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.data_audit import AuditError, SPLIT_NAMES, load_config, run_audit


def _write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


class DataAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "data" / "raw"
        self.raw.mkdir(parents=True)
        self.processed = self.root / "data" / "processed"
        self.path = self.root / "configs" / "config.yaml"
        self.path.parent.mkdir()
        self.dates = pd.date_range("2011-01-29", periods=42)
        sales_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        # Interleave stores to verify selection works even with unsorted raw series.
        sales_rows = [
            ("FOODS_1_001_CA_1_validation", "FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA", *([1, 0] + [0] * 37)),
            ("FOODS_1_001_TX_1_validation", "FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX", *([3] * 39)),
            ("FOODS_1_002_CA_1_validation", "FOODS_1_002", "FOODS_1", "FOODS", "CA_1", "CA", *([2] * 39)),
            ("FOODS_1_002_TX_1_validation", "FOODS_1_002", "FOODS_1", "FOODS", "TX_1", "TX", *([1] * 39)),
        ]
        _write_csv(self.raw / "sales.csv", [*sales_columns, *(f"d_{i}" for i in range(1, 40))], sales_rows)
        calendar_columns = [
            "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ]
        calendar_rows = [
            (
                day.date(), 11101 + i // 7, day.day_name(), i % 7 + 1, day.month, day.year,
                f"d_{i+1}", "Holiday" if i == 2 else "", "National" if i == 2 else "",
                "", "", int(i % 2 == 0), int(i % 2 == 1), 0,
            )
            for i, day in enumerate(self.dates)
        ]
        _write_csv(self.raw / "calendar.csv", calendar_columns, calendar_rows)
        self.price_rows = [
            ("CA_1", "FOODS_1_001", 11102, 4.0),
            ("CA_1", "FOODS_1_002", 11101, 3.0),
            ("TX_1", "FOODS_1_001", 11101, 5.0),
            ("TX_1", "FOODS_1_002", 11106, 7.0),
        ]
        self.write_prices(self.price_rows)
        self.config = {
            "data": {
                "raw_dir": "data/raw", "processed_dir": "data/processed", "stores": ["CA_1", "TX_1"],
                "panel_start": str(self.dates[0].date()), "panel_end": str(self.dates[37].date()),
                "expected_series_per_store": 2,
                "files": {"sales": "sales.csv", "calendar": "calendar.csv", "prices": "prices.csv"},
            },
            "splits": {
                name: {"start": str(self.dates[2 + i * 6].date()), "end": str(self.dates[7 + i * 6].date())}
                for i, name in enumerate(SPLIT_NAMES)
            },
            "forecast": {"daily_horizons": [1, 2], "cumulative_horizons": [2], "quantiles": [0.5, 0.9]},
        }
        self.write_config()

    def write_prices(self, rows):
        _write_csv(self.raw / "prices.csv", ["store_id", "item_id", "wm_yr_wk", "sell_price"], rows)

    def write_config(self):
        self.path.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")

    def test_panel_preserves_zero_sales_and_missing_prices_as_distinct_values(self):
        audit = run_audit(self.path)
        ca = pd.read_parquet(self.processed / "panel" / "CA_1.parquet")
        tx = pd.read_parquet(self.processed / "panel" / "TX_1.parquet")
        self.assertEqual(len(ca), 76)
        self.assertEqual(len(tx), 76)
        self.assertFalse(ca.duplicated(["id", "date"]).any())
        self.assertEqual(ca["sales"].dtype.name, "int32")
        self.assertEqual(ca["weekday"].dtype.name, "int8")
        self.assertEqual(list(ca.loc[:2, "sales"]), [1, 0, 0])
        self.assertTrue(ca.loc[:1, "sell_price"].isna().all())
        self.assertTrue(ca.loc[:1, "price_missing"].all())
        self.assertEqual(ca.loc[7, "sell_price"], 4.0)
        self.assertEqual(tx.loc[0, "sell_price"], 5.0)
        self.assertEqual(ca.loc[0, "weekday"], 5)
        self.assertEqual(ca.loc[0, "snap"], 1)
        self.assertEqual(tx.loc[0, "snap"], 0)
        self.assertTrue(ca.loc[2, "has_event"])
        self.assertEqual(ca.loc[2, "event_name_1"], "Holiday")
        self.assertEqual(audit["stores"]["CA_1"]["summary"]["missing_price_rows"], 62)
        self.assertEqual(audit["stores"]["CA_1"]["summary"]["missing_price_positive_sales_rows"], 32)
        self.assertEqual(audit["stores"]["CA_1"]["splits"]["training"]["rows"], 12)

        calendar = pd.read_parquet(self.processed / "calendar.parquet")
        tx_history = pd.read_parquet(self.processed / "price_history" / "TX_1.parquet")
        self.assertEqual(calendar["date"].max(), self.dates[-1])
        self.assertEqual(tx_history.loc[1, "week_end"], self.dates[-1])
        self.assertEqual(audit["observed_sales"]["end"], str(self.dates[38].date()))
        self.assertEqual(audit["splits"]["final_test"]["complete_windows_by_horizon"]["2"], 5)
        self.assertEqual(json.loads((self.processed / "audit.json").read_text(encoding="utf-8")), audit)

    def test_bad_price_does_not_overwrite_previous_run(self):
        run_audit(self.path)
        audit_file = self.processed / "audit.json"
        original = audit_file.read_bytes()
        self.write_prices([*self.price_rows, self.price_rows[0]])
        with self.assertRaisesRegex(AuditError, "duplicate .* keys for CA_1"):
            run_audit(self.path)
        self.assertEqual(audit_file.read_bytes(), original)

    def test_raw_hash_mismatch_is_explicit(self):
        self.config["data"]["expected_sha256"] = {
            key: hashlib.sha256((self.raw / name).read_bytes()).hexdigest()
            for key, name in self.config["data"]["files"].items()
        }
        self.config["data"]["expected_sha256"]["prices"] = "a" * 64
        self.write_config()
        with self.assertRaisesRegex(AuditError, "prices.csv SHA-256 mismatch"):
            run_audit(self.path)
        self.assertFalse(self.processed.exists())

    def test_cannot_include_future_calendar_as_observed_sales(self):
        self.config["data"]["panel_end"] = str(self.dates[-1].date())
        self.write_config()
        with self.assertRaisesRegex(AuditError, "dates without observed sales columns"):
            run_audit(self.path)

    def test_overlapping_split_fails_before_input_ingestion(self):
        self.config["splits"]["calibration"]["start"] = self.config["splits"]["development"]["end"]
        self.write_config()
        with self.assertRaisesRegex(AuditError, "calibration must begin immediately after"):
            load_config(self.path)


if __name__ == "__main__":
    unittest.main()
