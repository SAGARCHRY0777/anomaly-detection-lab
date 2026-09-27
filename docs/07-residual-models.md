# Lesson 7 — Residual models (virtual sensors): the winner

Code: `src/adlab/residual.py` · Test: `test_residual_is_small_on_normal_and_large_on_break`

## 7.1 The idea

For each sensor $j$, train a model that predicts it from the **other** sensors (now and a few
steps back). On healthy data the prediction is good. The anomaly score is how badly reality
disagrees with the prediction.

$$\hat x_{j,t} = f_j\big(x_{-j,t},\, x_{-j,t-1},\,\dots,\,x_{-j,t-L}\big) \qquad r_{j,t} = x_{j,t} - \hat x_{j,t}$$

Recall the decomposition from lesson 2: $x_t = g(u_t) + d_t + \varepsilon_t + a_t$.
If $f_j$ learns $g + d$ (steady-state response and dynamics), then

$$r_{j,t} \approx \varepsilon_t + a_t$$

The residual is **stationary noise plus the fault**, in *every* regime. One threshold now works
at idle, full power and during transients — exactly what raw values could never give us.

This is a label-free trick with a big consequence: **it converts an unsupervised problem into a
supervised regression problem**. Every sensor is its own label. You can tune the model the
normal way — minimise prediction error on held-out clean runs — and a better predictor is
directly a more sensitive detector.

## 7.2 Why the *other* sensors, not the sensor's own past?

If $x_{j,t-1}$ is an input, the best model is nearly "$\hat x_{j,t} = x_{j,t-1}$" (persistence).
Then a level shift produces one big residual at its first sample and nothing afterwards, and a
slow drift produces nothing at all — the fault leaks into the input and the model follows it.

Rule: **inputs should be causes, not the target's own past.** (Own-past models are forecasters —
useful for spikes and for lesson 12, but weak detectors of persistent faults.)

## 7.3 The maths: least squares → ridge

Stack inputs into a design matrix $A$ (rows = time steps) and the target into $b$.

**Ordinary least squares:** minimise $\lVert Aw - b\rVert^2$. Gradient $2A^\top(Aw-b)=0$ gives the
normal equations

$$A^\top A\, w = A^\top b \quad\Rightarrow\quad w = (A^\top A)^{-1}A^\top b$$

**The problem:** lagged features are almost copies of each other ($x_{t}\approx x_{t-1}$), so
$A^\top A$ has near-zero eigenvalues. Inverting it gives huge weights of alternating sign that fit
noise and extrapolate wildly.

**Ridge:** minimise $\lVert Aw - b\rVert^2 + \alpha\lVert w\rVert^2$:

$$w = (A^\top A + \alpha I)^{-1}A^\top b$$

In the eigenbasis of $A^\top A$ (eigenvalues $\mu_i$) the solution's component along direction $i$ is
scaled by $\mu_i/(\mu_i+\alpha)$: well-determined directions ($\mu_i \gg \alpha$) are kept, unstable
ones ($\mu_i \ll \alpha$) are shrunk to zero. Same idea as covariance shrinkage in lesson 4.
Bayesian view: ridge = MAP estimate with a Gaussian prior $w \sim \mathcal N(0, \sigma^2/\alpha\, I)$.

The intercept is not penalised: centre $A$ and $b$, solve, then $b_0 = \bar b - \bar a^\top w$ (`ridge_fit`).

## 7.4 Why lags make a linear model work on dynamics

A first-order lag $y_t = y_{t-1} + (u_t - y_{t-1})/\tau$ unrolls to

$$y_t = \sum_{i\ge0} \tfrac1\tau\left(1-\tfrac1\tau\right)^i u_{t-i}$$

— a **linear** function of the input's past (an FIR filter). So a linear model on lagged inputs
can represent the dynamics exactly, truncated at $L$ lags. System-identification name: an **FIR / ARX
model** (Auto-Regressive with eXogenous inputs; ours uses only the X part). Choose $L$ to cover a few
time constants, or let ridge cope with a shorter window (we use $L=5$).

**Nonlinearity for free.** EGT ∝ speed² is nonlinear in speed, but pressure ∝ speed² too — so EGT is
*linear in pressure*. Other sensors act as ready-made nonlinear features. When that is not enough,
add polynomial features or swap ridge for gradient boosting; nothing else changes.

## 7.5 From residual to score

1. **Standardise** each residual by a *robust* scale from training residuals:
   $\tilde r_j = r_j / (1.4826\,\text{MAD}_j)$ → approximately $\mathcal N(0,1)$ on healthy data.
2. **Combine:** $s_t = \sum_j \tilde r_{j,t}^2$ ≈ $\chi^2_d$ if residuals are independent (`score`).
3. **Accumulate** (optional, lesson 3): run CUSUM on each $\tilde r_j$ to catch small persistent shifts.
4. **Diagnose:** $\arg\max_j |\tilde r_{j}|$ names the sensor.

## 7.6 Pitfalls a top engineer checks

- **Unobserved inputs.** Ambient temperature is not a sensor in our data, so each run has a small
  residual offset (the unit test sees medians up to ~1σ). Fix properly: add ambient as an input.
  Fix pragmatically: subtract a per-run baseline estimated during a warm-up window (the benchmark's
  "residual + CUSUM" subtracts the per-run median — acceptable offline, but it can absorb long faults).
- **A faulty *input* contaminates other residuals.** If speed breaks, every model using speed
  mispredicts. The pattern still diagnoses: the broken sensor has the largest residual, others
  show a spill-over. (Formal version: structured residuals / fault signature matrix.)
- **Extrapolation.** Outside the training envelope, linear models extrapolate linearly (may be
  wrong); trees extrapolate flat (certainly wrong). Flag "input out of range" separately.
- **Too-flexible models reproduce the fault.** If the model can see the target (directly or via a
  near-copy), it predicts the fault too and the residual vanishes. Same risk in autoencoders.
- **Validation** = residual spread on *held-out clean runs*, split by run.

## 7.7 Benchmark

| Detector | Event recall | FA / 10k | Delay |
|---|---|---|---|
| PCA SPE (best of lessons 4–6) | 35% | 0.8 | 41 |
| **residual (ridge)** | **75%** | **0.8** | **4.7** |
| iforest on residuals | 75% | 3.2 | 9.3 |
| residual + CUSUM | 65% | 1.6 | 35 |

Per fault type — residual: spike 100%, level shift 75%, drift 100%, **stuck 33%**, noise 67%, correlation break 75%.
Residual + CUSUM: level shift 100%, drift 100%.

Why stuck is weak: at the moment a sensor freezes its value equals the prediction, so the residual
only grows as the operating point moves away. That is the stuck detector's job (lesson 3) —
hence ensembles (lesson 11).

## 7.8 Upgrade path

- **Gradient boosting** (LightGBM) per sensor: nonlinear, robust, fast.
- **Gaussian Process regression**: gives a predictive σ(x) as well — score $r/\sigma(x)$, looser in
  transients where the model is unsure, tighter at steady state.
- **Sequence models** (LSTM/TCN) for long dynamics.
- **Physics-based models** where equations exist — the best residual generator of all.

## Check yourself

1. Why is a model that predicts EGT from its own previous value a poor drift detector?
2. Doubling α in ridge — what happens to the weights along near-collinear directions?
3. Speed fails. Sketch the residual pattern across all six sensors.
