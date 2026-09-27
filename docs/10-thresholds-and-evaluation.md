# Lesson 10 — Thresholds and evaluation without labels

Code: `src/adlab/threshold.py`, `src/adlab/evaluate.py`
Tests: `test_pot_extrapolates_close_to_true_quantile`, `test_persistence_suppresses_isolated_alarms`, `test_point_adjust_inflates`, `test_event_metrics`

## Part A — Score → alarm

### A.1 The false-alarm budget

No labels ⇒ you cannot choose a threshold by maximising F1. Choose it from what normal data
does: "on clean data, at most $\alpha$ of samples may exceed $\tau$":

$$P(s(x) > \tau \mid \text{normal}) = \alpha$$

Derive $\alpha$ from operations, not taste. Example: runs of 2,000 samples, the operator
tolerates one false alarm per 10 runs → $\alpha = 1/20{,}000 = 5\times10^{-5}$ per sample.
With $K$ detectors OR-ed together, give each $\alpha/K$ (Bonferroni).

### A.2 Calibrate on *separate* clean runs

Training scores are optimistically low (the model fitted them). Thresholds from training
scores are too tight → false alarms in production. Keep a **calibration set** of clean runs the
model never saw (`benchmark.py`: train / calibrate / test).

### A.3 Empirical quantile

$\tau = \hat Q_{1-\alpha}$ of calibration scores. Needs roughly $\ge 10/\alpha$ samples to be
trustworthy — and *effective* samples (lesson 2), which autocorrelation shrinks. For
$\alpha = 10^{-5}$ you would need millions of clean samples.

### A.4 Extreme value theory: Peaks-Over-Threshold

**Pickands–Balkema–de Haan theorem:** for almost any distribution, excesses over a high threshold $u$,
$Y = S - u \mid S > u$, converge to a **Generalised Pareto Distribution**:

$$P(Y \le y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}$$

$\xi > 0$ heavy tail, $\xi = 0$ exponential, $\xi < 0$ bounded tail.

**Deriving the threshold.** With $n$ samples and $N_u$ above $u$:

$$P(S > z) = \underbrace{\frac{N_u}{n}}_{P(S>u)}\left(1 + \frac{\xi(z-u)}{\beta}\right)^{-1/\xi} = \alpha
\;\;\Rightarrow\;\; z_\alpha = u + \frac{\beta}{\xi}\left[\left(\frac{\alpha n}{N_u}\right)^{-\xi} - 1\right]$$

Fit $\xi, \beta$ by maximum likelihood on the excesses (`pot_threshold`, $u$ = 98th percentile).
EVT **extrapolates beyond the data**. Demo, 20,000 samples of $\chi^2_1$, $\alpha = 10^{-5}$:
true 19.51 · empirical quantile 16.06 · POT 17.95. (Streaming version: SPOT/DSPOT, Siffer et al. 2017.)

### A.5 Debounce (k-of-n persistence)

Alarm only if $\ge k$ of the last $n$ raw alarms fired. If raw false alarms were independent with
probability $p$:

$$P(\text{debounced false alarm}) \approx \binom{n}{k}p^k \qquad (2\text{-of-}3,\ p=10^{-3}:\ 3\times10^{-6})$$

Autocorrelated scores make raw alarms clump, so the real gain is smaller. Costs: up to $k-1$
samples of delay, and 1-sample events (spikes) are suppressed by design.

Benchmark, same thresholds, debounce on vs off:

| Detector | recall (2-of-3) | FA/10k (2-of-3) | recall (raw) | FA/10k (raw) |
|---|---|---|---|---|
| residual (ridge) | 75% | 0.8 | 80% | 5.5 |
| ENSEMBLE | 80% | 4.0 | 90% | 14.3 |

**Hysteresis** — separate on/off thresholds (alarm on above $\tau_{on}$, clear below $\tau_{off} < \tau_{on}$) —
stops chattering when a score hovers at the line.

## Part B — Evaluation

### B.1 Point-wise metrics mislead

Precision/recall/F1 per sample treat a 200-sample fault as 200 things to find, and are dominated
by long events.

### B.2 Point-adjust is worse

The popular protocol: if any sample of an event is flagged, count the whole event as detected.
Test: one alarm on the last sample of a 200-sample event → point recall 0.005, point-adjusted
recall **1.0**. Kim et al. (AAAI 2022) show random scores reach state-of-the-art point-adjusted F1
on standard benchmarks; Wu & Keogh (2021) show many benchmarks are trivially solvable. Distrust
papers that report only PA-F1.

### B.3 What to report

| Metric | Definition | Operator meaning |
|---|---|---|
| event recall | fraction of events with ≥ 1 alarm in [start, end + tolerance) | "did we catch it?" |
| false alarm episodes per 10k samples (or per run / hour) | alarm runs touching no event | "how often are we bothered for nothing?" |
| detection delay | first alarm − event start | "how early?" |
| per-type recall | recall split by fault kind | "what are we blind to?" |
| AUC-PR on scores | threshold-free ranking quality | compare detectors before thresholds |

### B.4 Evaluating without real faults: injection done right

- Inject **many types** and **many magnitudes**.
- Sweep magnitude → curve of recall vs fault size → the **minimum detectable fault**
  ("we catch 90% of EGT offsets ≥ 15 °C within 10 s"). That sentence is what a top engineer delivers.
- Beware injection artefacts: a detector can learn the injector's quirks. Vary shapes and
  onsets; once real faults arrive, validate on them first.

## Check yourself

1. Runs are 5,000 samples; allow one false alarm per 20 runs across 4 detectors. Per-detector α?
2. Why can't the empirical quantile give you a trustworthy α = 1e-6 from 100k samples?
3. A paper reports PA-F1 = 0.95 and no event-level false-alarm rate. What do you ask for?
