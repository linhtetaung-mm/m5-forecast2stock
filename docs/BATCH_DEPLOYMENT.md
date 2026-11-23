# Run batch order recommendations

The batch runner uses a fitted forecasting bundle and a policy selected on a historical validation block. It takes a dated inventory snapshot and generates a local purchase-order **recommendation** CSV for review. All example inventory, inbound, and service rows are synthetic.

## 1. Build the training and policy artifacts

Use the environment from the [README](../README.md) and the three raw M5 files. Starting at the repository root, run the three earlier steps if their artifact folders are absent.

```sh
python -m src.data_audit --config configs/config.yaml
python -m src.regimes --config configs/config.yaml
python -m src.models --config configs/config.yaml
python -m src.inventory_sim --config configs/config.yaml
```

Keep the `artifacts/forecasting/bundle/` directory together with `policy.json`, `reference_prices.parquet`, and `residual_buffers.parquet` from the inventory run. The policy binds the checksums of its input files and the forecasting bundle. The CLI defaults to the lowest validation-cost policy that cleared the service floor for the requested lead time. `--model moving_average` or another saved name requests that model's selected policy and raises an error if it failed the floor.

## 2. Prepare one complete store snapshot

| Input | Required fields and timing |
| --- | --- |
| History | Long CSV or Parquet with `store_id,item_id,dept_id,date,sales`. Supply daily quantities per trained SKU, back to the configured starting date. On a later operating date append complete new days using `--new-sales`. Every item requires a sales observation, including explicit zeros, through the requested origin. |
| Weekly price | CSV or Parquet with `store_id,item_id,wm_yr_wk,sell_price,week_end`. If `week_end` is omitted, provide the complete dated retail-week calendar with `wm_yr_wk`. Only completed weeks can supply forecast features. |
| Calendar | CSV or Parquet with `date,event_name_1,event_name_2,snap_CA,snap_TX,snap_WI`. Include the known schedule through the longest protection period. |
| Inventory | `store_id,item_id,as_of,on_hand`. One close-of-origin count per trained SKU. Exclude deliveries still outstanding at close. |
| Firm inbound | `store_id,item_id,as_of,due_date,units`. Aggregate to one row per SKU and due date. Tomorrow's confirmed receipts enter available stock before tomorrow's review. An empty confirmed pipeline uses a CSV with headers. The fixed-lead model rejects overdue receipts and receipts beyond the configured pipeline horizon. |
| Optional realized service | `store_id,item_id,date,demand,fulfilled,lost`. Provide only observable dates up to the origin. Use actual demand and fulfillment records when available. |

Choose a new output path for each run or archive the prior result. The scorer verifies the snapshot dates, cohort keys, pipeline timing, and model and policy fingerprints. Supply stock and firm orders from one consistent system snapshot.

## 3. Run the example from the project root

```sh
python -m app.predict_orders \
  --history data/processed/panel/CA_1.parquet \
  --prices data/processed/price_history/CA_1.parquet \
  --calendar data/processed/calendar.parquet \
  --inventory examples/inventory_snapshot.csv \
  --inbound examples/inbound_snapshot.csv \
  --bundle artifacts/forecasting/bundle \
  --policy artifacts/reports/inventory/policy.json \
  --as-of 2016-04-24 --store CA_1 --lead-time 7 \
  --output artifacts/orders/demo.csv \
  --monitoring-output artifacts/orders/demo-monitoring.json \
  --service-ledger examples/service_ledger.csv
```

The origin is April 24. The buyer reviews on April 25 and a purchase has a planned April 25 + 7 days due date. The file contains all SKU decisions, including zero quantities. For a purchase-order import, take the positive rows only after approval and any supplier checks. The JSON monitoring file reports recent sales levels and zero share relative to training, price coverage and service from the supplied ledger rows.

## 4. Use the same batch in Docker

Build on your machine from the repository root. The image installs the pinned Python dependencies and contains software and configuration. Sales, stock, and model files are supplied as mounts, outside the image.

```sh
docker build -t m5-forecast2stock .
mkdir -p artifacts/orders
docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/data/processed,dst=/inputs/data,readonly" \
  --mount "type=bind,src=$PWD/examples,dst=/inputs/examples,readonly" \
  --mount "type=bind,src=$PWD/artifacts/forecasting/bundle,dst=/inputs/bundle,readonly" \
  --mount "type=bind,src=$PWD/artifacts/reports/inventory,dst=/inputs/policy,readonly" \
  --mount "type=bind,src=$PWD/artifacts/orders,dst=/output" \
  m5-forecast2stock \
  --history /inputs/data/panel/CA_1.parquet \
  --prices /inputs/data/price_history/CA_1.parquet \
  --calendar /inputs/data/calendar.parquet \
  --inventory /inputs/examples/inventory_snapshot.csv \
  --inbound /inputs/examples/inbound_snapshot.csv \
  --bundle /inputs/bundle --policy /inputs/policy/policy.json \
  --as-of 2016-04-24 --store CA_1 --lead-time 7 \
  --output /output/docker-demo.csv \
  --monitoring-output /output/docker-demo-monitoring.json \
  --service-ledger /inputs/examples/service_ledger.csv
```

For a repeat operation, replace the dates and snapshot paths with the approved daily inputs and mount the new output folder. The image uses the same command and can run as a scheduled batch wherever those inputs and a secure output location are available.

