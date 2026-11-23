# M5 Forecast to Stock

The research question is when forecast improvements lead to lower simulated inventory cost and higher service across different demand regimes. This project runs through Parts 1–3, from the three M5 input CSVs to audited Parquet, training-only demand labels, forecast features, daily and cumulative forecast assessment, inventory replay, and batch order recommendations. The default analysis covers the 3,049 CA_1 store–SKU series.

Read the [Technical Glossary and Mathematical Blueprint](docs/TECHNICAL_BLUEPRINT.md) for the timing conventions, formulas, and assumptions. The [Execution Plan](docs/EXECUTION_PLAN.md) defines the date blocks and the forecast and inventory stages.

## Course submission history

See [what changed since the proposal and progress submissions](docs/PROJECT_EVOLUTION.md). The [original Mark III notebooks](references/course_submissions/) retain their original outputs. [Notebook 00](notebooks/00_proposal_reproduction.ipynb) reproduces the original December baseline after ingestion. The current workflow starts with Notebook 01.

## Reproduce Part 1

1. From this project root, create an isolated environment. These versions were validated on Apple silicon using CPython 3.14.6.

   ```sh
   python3.14 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

2. Check that `data/raw/` contains `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv`. See the [raw data notes](data/raw/README.md) to restore them in another checkout. The audit compares all three SHA-256 hashes to the entries in [the configuration](configs/config.yaml).
3. Run the tests, then choose the CLI or the two notebooks. Both paths call the same functions and generate the same artifacts.

   ```sh
   python -m unittest discover -s tests -v
   python -m src.data_audit --config configs/config.yaml
   python -m src.regimes --config configs/config.yaml
   ```

   For Jupyter, run `python -m jupyterlab` in the activated environment. Use the Python kernel from that environment and run [Notebook 01](notebooks/01_data_audit_and_ingestion.ipynb), then [Notebook 02](notebooks/02_eda_and_demand_regimes.ipynb), from top to bottom. Notebook 01 recreates the configured store outputs. Notebook 02 verifies that altering later sales leaves the training regimes unchanged.

## Outputs to expect

| Artifact | Contents |
| --- | --- |
| `data/processed/audit.json` | Input fingerprints, date and key checks, and join counts |
| `data/processed/panel/CA_1.parquet` | One row per store–SKU–day, with observed sales, the calendar, weekly price, and a missing-price flag |
| `data/processed/calendar.parquet` | Audited daily calendar, including dates beyond the observed sales file |
| `data/processed/price_history/CA_1.parquet` | Weekly price history and week ends for later origin-aware access |
| `data/processed/regimes/sku_regimes.parquet` | Training statistics and one frozen regime per store–SKU |
| `data/processed/regimes/` | JSON summary, threshold sensitivity, and EDA CSVs for store, category, weekday, and SNAP/event context |

The default panel has **3,689,290** rows across **1,210** dates. The **1,884,282** training item-days include **58.34%** zero recorded sales. The configured thresholds classify **1,208 Stable**, **371 Volatile**, and **1,470 Intermittent** series. Of the Intermittent group, **11** have insufficient positive history. The thresholds and date blocks are defined in `configs/config.yaml`. To extend ingestion to additional stores, list their valid IDs under `data.stores`. Use a separate processed and regime output directory to keep the CA_1 study artifacts isolated.

M5 provides recorded sales, calendar flags and listed selling prices. The study uses recorded sales as a demand proxy. Actual customer demand, stockouts, on-hand stock, and Walmart’s operating costs are unobserved.

## Reproduce Part 2

Run the Part 1 pipeline first. Then open [Notebook 03](notebooks/03_feature_engineering_pipeline.ipynb) and [Notebook 04](notebooks/04_probabilistic_forecasting_models.ipynb). Notebook 03 explains the forecast clock and perturbs future values. Notebook 04 can train the run from scratch or read a saved run produced with the same config.

```sh
python -m unittest discover -s tests -v
python -m src.features --phase development --output-dir artifacts/features/development
python -m src.models --config configs/config.yaml --output-dir artifacts/forecasting-reproduction
```

The feature export is optional. Its Parquet partitions use store, family and horizon and contain price provenance. They exclude outcomes. The model CLI builds the same origin features in memory, adds eligible labels for training and assessment, and publishes into a new output directory. An existing output directory is protected from accidental overwrite.

The supplied run is under `artifacts/forecasting/`.

| Artifact | Contents |
| --- | --- |
| `bundle/` | Serialized LightGBM trees, calibration offsets, training regime labels, feature configuration, event encoding and member fingerprints |
| `forecasts/calibration_*.parquet` | Forecasts before residual correction and the complete observed calibration outcomes |
| `forecasts/{policy_validation,warmup,final_test}_*.parquet` | Point forecasts, raw and calibrated quantiles, key dates, and observed outcomes where the target fits inside the phase. Predictions extend to the final replenishment decision, even where a future outcome is unknown. |
| `metrics.csv` | Development price ablation and frozen policy-validation/final forecast metrics, with counts and regime labels |
| `training_scales.parquet` | Training-only scales for RMSSE and SPL |
| `run_manifest.json` | SHA-256 for every run artifact |

Part 2 uses all 3,049 CA_1 series. The fixed laptop fitting budget uses a 12-day per-SKU stride with staggered offsets and 96 boosting rounds. This keeps different weekdays in the fit and reduces the row count. Scoring uses all common complete origins within each assessed phase. Tune model design and price ablation on development only. Calibration uses earlier forecast errors and policy settings can be chosen on policy-validation dates. The final assessment uses 17 common, overlapping origins. The optional all-store configuration increases memory and compute requirements. The forecasting run here was verified on CA_1.

## Reproduce Part 3

Run Part 2 or keep its verified forecast bundle and forecast files in the configured location. Run [Notebook 05](notebooks/05_inventory_simulation_and_economics.ipynb) and [Notebook 06](notebooks/06_deployment_and_batch_inference.ipynb). For a separate command-line replay, use a fresh destination.

```sh
python -m unittest discover -s tests -v
python -m src.inventory_sim --config configs/config.yaml --output-dir artifacts/inventory-reproduction
```

The included replay lives at `artifacts/reports/inventory/`. Its manifest binds the forecast run and configuration. `policy.json` lists the validation-selected rules. `selected_policies.csv` compares validation service with the frozen April results. `base_frontiers.csv` contains cost–fill frontiers by regime and lead time. `fixed_path_repricing.csv` holds orders fixed while costs change. `forecast_inventory_pairs.csv` pairs protection-period forecast loss with simulated inventory outcomes. Opening and ending stock and pipeline are reported for both common-state and continuous April starts.

The [batch deployment guide](docs/BATCH_DEPLOYMENT.md) gives the input contracts, local CLI and Docker commands, and a worked demonstration. Use the three `examples/` files together. They contain synthetic inventory, inbound and service records to accompany the M5 sales replay. The pipeline produces a local CSV for buyer review.

## Interactive demo

[Stockdesk](demo/README.md) starts from one stocking decision and lets you replay three example items. The Study results view shows the frozen CA_1 cost and service results. The demo runs as a static site with no server-side dependencies.

Deploy the static site from GitHub using the [Render or Vercel instructions](docs/GITHUB_HOSTING.md).
