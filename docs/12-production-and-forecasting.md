# Lesson 12 — Production architecture, drift, and forecasting

## Part A — Architecture

### A.1 The reference model: ISO 13374 / OSA-CBM

| Layer | Job | In this repo |
|---|---|---|
| 1 Data acquisition | sensors → timestamped values | `data.simulate_run` stands in |
| 2 Data manipulation | clean, align, resample, features, lags | `lag_features`, `Standardizer` |
| 3 State detection | normal or not? | every detector, `threshold.py` |
| 4 Health assessment | what, where, how bad | contributions, residual argmax, lesson 11 table |
| 5 Prognostics | where is it heading, when will it cross a limit | Part C below |
| 6 Advisory generation | alarm, report, recommended action | alarm payload (lesson 11.5) |

### A.2 Online system

```
DAQ ──► message bus (MQTT / Kafka) ──► stream scorer ──► alarm manager ──► UI / notifications
                                         │  per-stream state:       │  debounce, hysteresis,
                                         │  lag buffer, EWMA,       │  dedupe, escalation
                                         │  CUSUM sums              ▼
                                         ▼                    alarm + feedback store
                                  time-series DB  ◄────────── operator labels (true / false)
                                         │
            offline: training ──► model registry (model + threshold + calibration-data hash)
```

Rules that keep it honest:
- **Causal only.** No centred windows, no per-run medians, no future samples. Replace offline
  tricks with warm-up baselines (first N seconds of a run) and trailing windows.
- **One code path** for features offline and online, or training–serving skew silently breaks detection.
- **Version the triple** model + threshold + calibration set. A new model with the old threshold is a new, untested detector.
- **Shadow mode first.** Run silently next to the old system, compare alarms, then switch.

### A.3 Drift

| Type | What changes | Example | Detect with |
|---|---|---|---|
| Data (covariate) drift | input distribution | new engine build, winter | PSI / KS test on inputs vs training |
| Concept drift | the relationship | wear changes EGT-vs-speed | residual distribution on confirmed-healthy runs |
| Operational drift | alarm behaviour | alarm rate creeping up | alarm rate per run as a monitored metric |

Population Stability Index over bins $b$ (expected share $e_b$, actual $a_b$):
$\text{PSI} = \sum_b (a_b - e_b)\ln(a_b/e_b)$ — rule of thumb: < 0.1 stable, 0.1–0.25 watch, > 0.25 shifted.

**The retraining trap.** Retraining on recent data "fixes" drift — and also teaches the model that
slow wear is normal. Wear is often exactly what you want to see. Keep a **frozen golden-baseline
model** (from a known-healthy period) alongside a rolling one; divergence between them *is* the
degradation signal.

### A.4 Feedback loop

Alarm → operator marks true / false / unsure → label store → (1) real-fault evaluation replaces
synthetic, (2) thresholds re-checked, (3) eventually a supervised fault classifier. Ask operators
first about the alarms the model is *least* sure of (active learning).

## Part B — Forecasting

### B.1 How it relates to detection

- Residual detector = predict sensor $j$ **now** from **other** sensors.
- Forecaster = predict a series' **future** $h$ steps from its own past (plus known future inputs).

Uses: forecast-interval anomaly detection (actual outside the predicted band), early warning,
and prognostics.

### B.2 The ladder

| Level | Model | Core equation / idea |
|---|---|---|
| 0 | naive, seasonal naive | $\hat y_{t+h} = y_t$ (or $y_{t+h-s}$). The bar every model must clear |
| 1 | exponential smoothing (ETS) | $\ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1})$, $b_t = \beta(\ell_t-\ell_{t-1}) + (1-\beta)b_{t-1}$ |
| 2 | ARIMA / SARIMAX | $\phi(B)(1-B)^d y_t = \theta(B)\varepsilon_t$ (+ exogenous $X\gamma$) |
| 3 | gradient boosting on lags | features: lags, rolling stats, calendar, exogenous; direct or recursive multi-step |
| 4 | deep | N-BEATS / N-HiTS (basis expansion), DeepAR (probabilistic RNN), TFT (attention + known futures), PatchTST |
| 5 | foundation models | Chronos, TimesFM, Moirai — zero-shot, useful when data is scarce |

ARIMA in words: **AR($p$)** — today depends linearly on the last $p$ values; **I($d$)** — difference
$d$ times to make it stationary (lesson 2!); **MA($q$)** — depends on the last $q$ forecast errors.
$B$ is the backshift operator, $By_t = y_{t-1}$.

For engines, **SARIMAX-style exogenous inputs matter most**: the planned throttle schedule is
*known in advance*; feed it in.

### B.3 Probabilistic forecasts

A point forecast cannot raise an alarm; an interval can. Train quantiles with the **pinball loss**:

$$L_q(y, \hat y) = \max\big(q(y-\hat y),\ (q-1)(y-\hat y)\big)$$

Minimising it gives the $q$-th conditional quantile (q = 0.05 and 0.95 → a 90% band).
Whole-distribution score: CRPS.

### B.4 Metrics and backtesting

- MAE / RMSE in units; **MASE** = MAE / MAE of the in-sample naive forecast (< 1 beats naive).
- Coverage of intervals (does the 90% band contain 90%?) and pinball loss.
- **Rolling-origin backtest**: train to $T$, forecast $T{+}1..T{+}h$, move $T$ forward, repeat. Never a random split.

## Part C — Prognostics: the second time scale

Two clocks:

1. **Within a run** (seconds): anomaly detection — lessons 1–11.
2. **Across runs** (weeks, months): degradation. Compute one **health indicator** per run (e.g. mean
   EGT residual at a fixed steady-state point, or an "EGT margin"). That gives a *new* time series,
   one point per run.

Fit a trend with uncertainty (linear/exponential regression, Gaussian Process, or a state-space
model) and extrapolate to the limit → a *distribution* of time-to-limit = **remaining useful life**.
Public practice data: NASA C-MAPSS turbofan degradation.

## Part D — When your real data arrives

1. Inventory sensors, rates, units; plot every channel of 5–10 runs.
2. Write down the physics: which sensor drives which, time constants.
3. Pick healthy runs with domain experts → train / calibrate / test **by run**.
4. Build the harness: injector + event metrics + per-type table (this repo's `benchmark.py`).
5. Baselines: limits, robust z, stuck detector.
6. Residual models per sensor (ridge → LightGBM), then CUSUM on residuals.
7. PCA/SPE or autoencoder on residuals or windows for relationship faults.
8. Thresholds from the false-alarm budget (quantile / POT), debounce, hysteresis.
9. Ensemble by coverage; add diagnosis payload.
10. Shadow mode → feedback loop → drift monitoring → prognostics on per-run health indicators.

## Further reading

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (free, otexts.com/fpp3)
- Montgomery, *Introduction to Statistical Quality Control* (EWMA, CUSUM, ARL)
- Aggarwal, *Outlier Analysis*
- Chandola, Banerjee & Kumar, "Anomaly Detection: A Survey" (2009)
- Liu, Ting & Zhou, "Isolation Forest" (2008) · Breunig et al., "LOF" (2000) · Schölkopf et al., "Estimating the Support of a High-Dimensional Distribution" (2001)
- Yeh et al., "Matrix Profile I" (2016) · Killick et al., "PELT" (2012)
- Siffer et al., "Anomaly Detection in Streams with Extreme Value Theory" (KDD 2017)
- Wu & Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed" (2021) · Kim et al., "Towards a Rigorous Evaluation of Time-Series Anomaly Detection" (AAAI 2022)
- Huyen, *Designing Machine Learning Systems* (production, drift)
- ISO 13374 / MIMOSA OSA-CBM (condition monitoring architecture)
