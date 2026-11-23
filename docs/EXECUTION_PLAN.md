# Project execution plan

## Split contract

Dates are inclusive **target-day** blocks. The origin is the prior completed sales day before the first target. The observed series in the supplied file end on April 24, 2016. April is reserved for this clean-slate protocol, subject to confirming it was not used to pick models in another analysis. Its 24 observed days permit only 17 complete eight-day origins. Report that sample-size limit prominently. The primary cross-horizon forecast comparison pairs the same 17 origins for both horizons. Inventory can still be scored on all 24 observed days because its last reviews need forecasts and known calendar dates, not realized sales labels after the test. Show the terminal pipeline.

| Stage | Target days | Allowed use |
| --- | --- | --- |
| Prepared history | 2013-01-01–2016-04-24 | Source history and lag initialization. Partition by store. Earlier source weeks remain available for price provenance. |
| Training | 2014-01-01–2015-09-10 | Train candidate models, define scales and fix demand regimes. |
| Development | 2015-09-11–2015-10-08 | Select a fixed model design and tuning budget. One predetermined refit may add this block after selection. Regime thresholds and scales remain training-based. |
| Calibration | 2015-10-09–2016-01-07 | Fit empirical quantile corrections or baseline residual distributions with complete earlier target windows. |
| Policy validation | 2016-01-08–2016-03-03 | Select policy settings and check feasibility at the configured fill floor. |
| Warm-up | 2016-03-04–2016-03-31 | Establish stock and pipeline. Use an incumbent for the shared-state cutover and per-method states for the continuous sensitivity. No further tuning. |
| Held-out final test | 2016-04-01–2016-04-24 | Frozen score and policy comparison, with results scoped to the M5 recorded-sales proxy. |

## Part 1: data contracts and regimes

1. Verify raw files and record SHA-256, shapes, expected headers, join keys, domains, missingness and observed label coverage. Join sales to the calendar by day code and to weekly prices by store, item and week. Stop on contradictions such as duplicate keys or incomplete observed dates. Keep unmatched prices as null and count them separately.
2. Process one selected store at a time and write typed Parquet. The default store is CA_1, but the configuration accepts multiple distinct valid stores. Save an audited calendar, weekly prices and panel outputs, plus a run manifest.
3. Use only the training block for EDA and regime fitting. Show the proportion of zero days, signed skewness, positive sizes, weekly patterns, SNAP/event associations, and missing-price counts. Treat associations as descriptive. Freeze three labels, flag insufficient history, and generate sensitivity to both thresholds.
4. Test with malformed fixtures, future perturbations, and the full CA_1 inputs. Notebooks 01–02 call reusable modules, document the outputs, and are executable top to bottom.

## Part 2: origin features and probabilistic forecasting

1. Implement a keyed feature interface `(store_id, item_id, origin, target_date or horizon)`. Persist price source week and eligibility date. Reject duplicates, non-finite data and missing forecast keys. Test future perturbations, irregular origins, 2/8-day cutoff behavior and off-by-one lags.
2. Run last observation, seasonal naïve, moving average, Croston and SBA. Calibrate their distribution estimates on earlier complete windows. Compare pooled LightGBM quantiles for direct daily and cumulative horizons, with or without an as-of price feature. Consider a time-subsample for development only. Keep daily and cumulative fits separate and cache artifacts by fold and model.
3. Report daily MAE, RMSE and RMSSE, plus pinball and SPL by quantile, horizon and regime. Report counts for undefined scales and coverage. Save original and monotone predictions, and lock the forecast feature/schema and artifact manifest. Do not compare daily aggregate errors to protection-period errors as if equivalent.

## Part 3: inventory, economics and batch scoring

1. Implement a vectorized daily stock ledger with exact receipt timing, whole-unit orders, nonnegative constraints, lost units, and inventory conservation. Test one- and seven-day leads, missing forecasts, invalid values, and common-state cutover.
2. Replay each forecast and policy before selecting candidates on the policy-validation block. Cross forecast and safety-buffer choices. Test fixed-path price repricing separately from policy reselection. Save evaluated frontiers by horizon/regime, fill-floor failures, start and terminal state, and costs under low/base/high assumptions. When comparing relative holding/shortage ratios, cover both sides of observed break-even thresholds on earlier validation.
3. Ship a CLI that reads latest available sales, price and calendar inputs and a dated usable-stock/inbound snapshot. Generate traceable order suggestions and stop on stale or inconsistent inventory rather than defaulting it to zero. Keep batch inference auditable and order submission outside its role. Provide Docker, model hashes, schema checks, and a known request fixture.
4. Run the final holdout once with frozen rules. Regenerate notebooks and report tables from these exact artifacts. Include one worked ledger, feature and cost units, forecast diagnostics by regime, actual achieved service and a limits section. Keep the scale-up check a separate result, with matching rules and no inventory transfers in the primary CA_1 experiment.

The empirical contribution is the controlled comparison of forecast choice, protection-period quantile calibration and stocking policy. It is not a claim of introducing any of these methods or of outperforming Walmart's internal procedures.
