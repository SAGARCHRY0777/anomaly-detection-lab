# Lesson 2 — Time-series foundations: why time changes everything

Code: `src/adlab/data.py`

## 2.1 The i.i.d. assumption breaks

Textbook statistics assumes samples are **independent and identically distributed**.
Time series break *both* halves:

- **Not independent**: $x_t$ is close to $x_{t-1}$ (temperatures do not teleport).
- **Not identically distributed**: the distribution at idle differs from full power.

Nearly every mistake in time-series anomaly detection comes from forgetting one of these.

## 2.2 Autocorrelation — measuring "not independent"

$$\mu = E[x_t], \qquad \gamma(h) = E[(x_t-\mu)(x_{t+h}-\mu)], \qquad \rho(h) = \frac{\gamma(h)}{\gamma(0)} \in [-1, 1]$$

$\rho(h)$ is the correlation of the series with itself shifted by $h$ samples (the ACF).
White noise: $\rho(h) = 0$ for $h>0$. Sensor data at 1 Hz: $\rho(1)$ often above 0.95.

**Consequence 1 — fewer effective samples.** For an AR(1) process with lag-1 correlation $\rho$:

$$n_{\text{eff}} \approx n\,\frac{1-\rho}{1+\rho}$$

10,000 samples with $\rho = 0.95$ carry the information of ~256 independent ones.
Thresholds and confidence intervals computed as if $n = 10{,}000$ are badly overconfident.

**Consequence 2 — alarms cluster.** If $x_t$ crosses the line, $x_{t+1}$ probably does too.
Count alarm *episodes* (events), not samples.

**Consequence 3 — random row splits leak.** Row $t$ in train and $t+1$ in test are near
copies. Split by **run** (or by time block).

## 2.3 Stationarity

A series is **weakly stationary** if mean and autocovariance do not depend on $t$:

$$E[x_t] = \mu \quad\text{and}\quad \operatorname{Cov}(x_t, x_{t+h}) = \gamma(h) \;\;\forall t$$

Machine data is **not** stationary: the operating point moves. A global z-score
$|x-\mu|/\sigma$ uses a $\sigma$ dominated by the *operating range*, not the noise —
so a fault of 10% rpm is invisible while every fast transient looks alarming.
Benchmark evidence: global z-score, 15% event recall at 31.7 false alarms / 10k samples.

The fix is to make the thing you score stationary:
- **condition** on the operating point (regimes), or
- **difference / detrend**, or
- **model and subtract** the operating-point effect → residuals (lesson 7). ← the winner.

## 2.4 A mental decomposition of any sensor

$$x_t = \underbrace{g(u_t)}_{\text{steady-state response to operating point}} + \underbrace{d_t}_{\text{transient dynamics}} + \underbrace{\varepsilon_t}_{\text{noise}} + \underbrace{a_t}_{\text{fault (usually 0)}}$$

Anomaly detection = estimate $g$ and $d$ well enough that what is left is
$\varepsilon_t + a_t$, then test whether $a_t \ne 0$. Every method differs in how much of $g$
and $d$ it removes before scoring.

## 2.5 Dynamics: the first-order lag

Physical quantities approach their target exponentially. Discrete form (used in the simulator):

$$x_t = x_{t-1} + \frac{1}{\tau}(\text{target}_t - x_{t-1})$$

After a unit step, $x_t = 1-(1-1/\tau)^t \approx 1-e^{-t/\tau}$: 63% of the way after $\tau$
samples, 95% after $3\tau$.

Why it matters: during acceleration, speed has moved but temperature has not caught up.
The pair (high speed, still-low EGT) *never* appears in steady state — a static model flags every
transient. Fix: give models **lagged inputs** ($x_{t-1}, \dots, x_{t-L}$) so they see the dynamics.

## 2.6 Regimes — "normal" is a mixture

$$p_{\text{normal}}(x) = \sum_{k} \pi_k\, p(x \mid \text{regime}=k)$$

Idle is tight and quiet; full power is wide and noisy; transients are rare (small $\pi_k$).
Consequences:
- Rare *healthy* regimes (transients) get flagged by density methods.
- One global threshold is too loose for idle and too tight for transients.
- Either model each regime, or model the relationship between sensors (conditioning).

## 2.7 Sampling and alignment

- **Nyquist**: to see a frequency $f$ you need sampling $f_s > 2f$. Vibration at 200 Hz is invisible in 1 Hz data (and can alias into false low-frequency patterns).
- **Align** all channels onto one time grid before any multivariate method.
- **Gaps**: never interpolate across long gaps; mark them.
- **Windows never cross run boundaries** — carry a `run_id` (see `data.stack`).

## 2.8 The simulator (`data.py`)

A hidden throttle $u_t \in [0,1]$ follows idle → steps → decel → idle. Every sensor derives from it:

| Sensor | Equation (plus Gaussian noise) |
|---|---|
| speed | first-order lag of $100u$, $\tau = 6$ |
| fuel_flow | $0.1 + 1.9u^{1.3}$ (leads speed) |
| egt | lag($350 + 1.2T_{amb} + 380n^2 + \text{accel overshoot}$), $\tau=4$ |
| pressure | $1 + 3.2n^2$ |
| vibration | $0.4 + 1.8n + $ small oscillation |
| oil_temp | slow lag ($\tau=120$) of $40 + 0.5T_{amb} + 50n$ |

$n$ = speed/100; $T_{amb}$ = per-run ambient temperature (unobserved — a deliberate realism).

## 2.9 Anomaly taxonomy (Chandola et al., 2009) and the six injected types

| Class | Meaning | Injected types |
|---|---|---|
| **Point** | one value abnormal on its own | `spike` |
| **Contextual** | value normal in general, abnormal *given context* (operating point) | `level_shift`, `drift`, `correlation_break` |
| **Collective** | each value fine, the *sequence* is abnormal | `stuck`, `noise_burst` |

- Point anomalies: simple statistics catch them.
- Contextual: need conditioning (multivariate / residual).
- Collective: need windows (variance, shape, matrix profile).

This table predicts the benchmark's per-type recall almost exactly.

## Check yourself

1. Why does a random 80/20 row split give optimistic results on sensor data?
2. A stuck sensor reads a perfectly normal value. Which class of anomaly is it, and why can no point-wise method catch it?
3. With $\rho(1)=0.99$, how many effective samples are in 1 hour of 1 Hz data?
