# Lesson 3 — Statistical baselines: z-score, robust MAD, EWMA, CUSUM

Code: `src/adlab/stats.py` · Tests: `tests/test_detectors.py`

These are *hypothesis tests*, not models. Each asks "is this sample (or recent stretch)
consistent with a known normal distribution?". They are only as good as the stationarity of what
you feed them — useless on raw engine data (lesson 2), excellent on residuals (lesson 7).

## 3.1 z-score

$$z_t = \frac{|x_t - \mu|}{\sigma}$$

Under $x \sim \mathcal N(\mu,\sigma^2)$, $Z$ is standard normal, so the false-alarm rate of
threshold $c$ is exact:

| threshold $c$ | $P(\lvert Z\rvert > c)$ | at 1 Hz: one false alarm every … |
|---|---|---|
| 2 | 0.0455 | 22 s |
| 3 | 0.0027 | 6 min |
| 4 | 6.3e-5 | 4.4 h |
| 5 | 5.7e-7 | 20 days |

**Multiple sensors multiply false alarms.** With $d$ independent sensors each at rate $\alpha$:
$P(\text{any alarm}) = 1-(1-\alpha)^d \approx d\alpha$. Bonferroni fix: use $\alpha/d$ per sensor.

**Masking.** Outliers inflate $\hat\sigma$, which shrinks their own z-scores. 5% of points at 1000
on N(0,1) data gives $\hat\sigma \approx 218$: the outliers get $z \approx 4.4$, barely flagged.

## 3.2 Robust z-score (median / MAD)

$$\text{MAD} = \operatorname{median}_i |x_i - \operatorname{median}(x)|, \qquad z^{rob}_t = \frac{|x_t - \operatorname{median}(x)|}{1.4826\cdot \text{MAD}}$$

**Breakdown point** = fraction of arbitrarily bad points an estimator tolerates before it can be
dragged anywhere. Mean: $1/n \to 0$. Median and MAD: **50%**.

**Where 1.4826 comes from.** For $X \sim \mathcal N(\mu,\sigma^2)$, $|X-\mu|$ has median $m$ with
$P(|X-\mu| \le m) = 0.5 \Rightarrow 2\Phi(m/\sigma) - 1 = 0.5 \Rightarrow m = \sigma\,\Phi^{-1}(0.75) = 0.6745\sigma$.
So $\sigma = \text{MAD}/0.6745 = 1.4826\,\text{MAD}$ — a consistent estimator of $\sigma$.

**Rolling / causal version** (`rolling_robust_zscore`): median and MAD over the trailing window
$[t-w, t-1]$, so it can run live and adapts to slow operating changes. Cost: it also adapts to
slow *faults* (a drift becomes the new normal) — the fundamental trade-off of any adaptive baseline.

## 3.3 EWMA

$$z_t = \lambda x_t + (1-\lambda) z_{t-1} = \lambda\sum_{i\ge0}(1-\lambda)^i x_{t-i}$$

A weighted average with exponentially fading memory. Effective memory ≈ $(2-\lambda)/\lambda$ samples.

**Variance derivation** (independent $x$, variance $\sigma^2$):

$$\operatorname{Var}(z_t) = \lambda^2\sigma^2 \sum_{i\ge0}(1-\lambda)^{2i} = \frac{\lambda^2\sigma^2}{1-(1-\lambda)^2} = \sigma^2\frac{\lambda}{2-\lambda}$$

(finite-$t$ version multiplies by $1-(1-\lambda)^{2t}$ — see `ewma_chart`).
With $\lambda = 0.1$ the EWMA's standard deviation is $0.23\sigma$: noise is averaged away, so a
**persistent 1σ shift becomes a 4.4σ shift in EWMA units**. Small $\lambda$ = sensitive to small
persistent shifts, slow to react; large $\lambda$ → approaches the raw z-score.

## 3.4 CUSUM — the optimal detector of a mean shift

**Derivation from the likelihood ratio.** Standardised data. Before the change $x \sim \mathcal N(0,1)$;
after it $x \sim \mathcal N(\delta,1)$. Log-likelihood ratio of one sample:

$$\ell(x) = \log\frac{e^{-(x-\delta)^2/2}}{e^{-x^2/2}} = \delta x - \frac{\delta^2}{2} = \delta\left(x - \frac{\delta}{2}\right)$$

Evidence that a change started at time $\tau$ = $\sum_{i=\tau}^{t}\ell(x_i)$. We do not know $\tau$, so
take the max over all possible starts. Page (1954) showed this is computed by a recursion
(dividing by $\delta$, with $k = \delta/2$):

$$S^+_t = \max\left(0,\; S^+_{t-1} + x_t - k\right), \qquad S^-_t = \max\left(0,\; S^-_{t-1} - x_t - k\right)$$

Alarm when $S > h$.

- In control, each step adds $x - k$ with mean $-k$ → $S$ keeps getting pulled back to 0.
- After a shift of $\delta = 2k$, each step adds on average $+k$ → $S$ climbs linearly and crosses $h$ after ≈ $h/k$ samples.
- The `max(0, ·)` is "forget evidence from before the change could have started".

**Optimality** (Lorden 1971, Moustakides 1986): among all detectors with the same false-alarm
rate, CUSUM minimises the worst-case detection delay for a shift of size $\delta$.

**ARL — the language of control charts.** $\text{ARL}_0$ = average samples until a false alarm;
$\text{ARL}_1$ = average samples to detect a real shift.

| Detector | $\text{ARL}_0$ | $\text{ARL}_1$ for a 1σ shift |
|---|---|---|
| 3σ Shewhart (z-score) | 370 | ≈ 44 |
| CUSUM $k=0.5,\ h=5$ (two-sided) | ≈ 465 | ≈ 10 |

Fewer false alarms *and* four times faster. (Shewhart $\text{ARL}_1 = 1/P(|Z+1|>3) \approx 1/0.0228$.)

**Tuning.** $k$ = half the smallest shift you care about (in σ). $h$ sets the false-alarm rate
(4–5 typical). CUSUM is weak on single spikes — it needs *persistent* change; that is what the
z-score is for. Combine them.

## 3.5 Stuck-sensor (flat-line) detector

Real sensors always have noise. Over a window of $w$ healthy samples,
$(w-1)s^2/\sigma^2 \sim \chi^2_{w-1}$, so the probability that $s$ is essentially 0 is vanishingly small.
Score = $-\log_{10}(s + \epsilon)$ over a rolling window. A stuck value sits *inside* the normal
range, so no distance- or density-based method can see it; a variance test sees it immediately.

## 3.6 What the benchmark says

| detector | raw values | on residuals (lesson 7) |
|---|---|---|
| z-score (global) | 0.03 recall, 3.13 FA/10k | — |
| residual + CUSUM | — | 100% of level shifts and drifts |
| stuck detector | 100% of stuck faults | — |

The tests are right; the input was wrong. Feed them something stationary.

## Check yourself

1. You monitor 40 sensors at 10 Hz with a 3σ rule each. How many false alarms per hour?
2. Why does the rolling robust z-score miss slow drifts?
3. You want to catch a 0.5σ persistent shift. What $k$? What happens to the delay if you keep $h=5$?
