# Lesson 4 — Multivariate detection: Mahalanobis, PCA, Hotelling T², SPE

Code: `src/adlab/pca.py` · Tests: `test_pca_matches_sklearn_and_catches_correlation_break`, `test_mahalanobis_follows_chi2`

## 4.1 Why one sensor at a time is not enough

Speed and EGT move together. The point (speed = 95%, EGT = 420 °C) has each value inside its
own normal range — per-sensor limits form a **box** and the point is inside it. Jointly it is
impossible: normal data lives in a thin **ellipse** along the diagonal. Multivariate methods
replace the box with the ellipse.

## 4.2 Covariance

Standardise first (units differ: °C vs bar). Then

$$S = \frac{1}{n-1}\sum_i (x_i-\bar x)(x_i-\bar x)^\top, \qquad S_{jk} = \operatorname{cov}(x_j, x_k)$$

$S$ is symmetric positive semi-definite, so it has an eigendecomposition
$S = V\Lambda V^\top$ with orthonormal eigenvectors $v_i$ (the ellipse's axes) and eigenvalues
$\lambda_i \ge 0$ (the variance along each axis).

## 4.3 Mahalanobis distance

$$d^2(x) = (x-\mu)^\top S^{-1}(x-\mu)$$

**What it does geometrically.** Since $S^{-1} = V\Lambda^{-1}V^\top$:

$$d^2(x) = \sum_{i=1}^{p}\frac{\big(v_i^\top (x-\mu)\big)^2}{\lambda_i}$$

1. rotate into the ellipse's axes ($v_i^\top(x-\mu)$),
2. divide each coordinate by that axis's standard deviation $\sqrt{\lambda_i}$,
3. take ordinary squared distance.

This is **whitening**: the ellipse becomes a sphere, and distance means "standard deviations
away, accounting for correlation". Moving 1 unit along a tight axis costs far more than along a
wide axis.

**Distribution.** If $x \sim \mathcal N(\mu, S)$, each term is a squared standard normal, so
$d^2 \sim \chi^2_p$. Threshold for false-alarm rate $\alpha$: `chi2.ppf(1-α, p)`.

**The catch — near-singular $S$.** Strongly correlated sensors ⇒ some $\lambda_i \approx 0$ ⇒
dividing by $\lambda_i$ amplifies tiny noise enormously and inverting $S$ is unstable. Fix with
**shrinkage**: $S_\text{shrunk} = (1-a)S + a\,\frac{\operatorname{tr}S}{p}I$ — lifts every eigenvalue by
a floor. (Ledoit & Wolf choose $a$ optimally; `MahalanobisDetector(shrinkage=0.01)`.)

## 4.4 PCA from first principles

Find the unit direction $w$ along which normal data varies most:

$$\max_w\; w^\top S w \quad \text{s.t. } w^\top w = 1$$

Lagrangian $\mathcal L = w^\top S w - \lambda(w^\top w - 1)$; setting $\nabla_w \mathcal L = 0$ gives

$$S w = \lambda w$$

so $w$ is an eigenvector, and the variance captured is $w^\top S w = \lambda$. The best direction is
the top eigenvector; the next best (orthogonal) is the second; and so on.

**Via SVD** (what the code does, numerically better): centred data $Z = U\Sigma V^\top$ gives
$S = V\frac{\Sigma^2}{n-1}V^\top$ — the right singular vectors are the principal axes and
$\lambda_i = s_i^2/(n-1)$.

**Choosing $k$**: smallest $k$ with $\sum_{i\le k}\lambda_i / \sum_i\lambda_i \ge 0.95$.

## 4.5 Two subspaces, two statistics

With $P = [v_1 \dots v_k]$ ($p \times k$), every standardised sample splits into

$$z = \underbrace{PP^\top z}_{\text{model part}} + \underbrace{(I-PP^\top) z}_{\text{residual part}}$$

| Statistic | Formula | Question it answers |
|---|---|---|
| **Hotelling T²** | $T^2 = \sum_{i\le k} t_i^2/\lambda_i$, $t = P^\top z$ | Is the sample *extreme along the normal modes of variation*? (Mahalanobis inside the model plane) |
| **SPE / Q** | $\text{SPE} = \lVert (I-PP^\top) z\rVert^2$ | Does the sample have variation the normal model *has never seen*? |

**Worked example** (the unit test). Three sensors, normal behaviour $x = [z, 2z, -z]$ + noise.
After standardising, normal points lie on the line $\propto [1, 1, -1]$, so $k=1$,
$P = [1,1,-1]/\sqrt3$.

- Normal point $z=[1,1,-1]$: $P^\top z = \sqrt3$, projection $= z$, **SPE ≈ 0**.
- Broken point $z=[1,1,1]$ — every value individually normal: $P^\top z = 1/\sqrt3$, projection $= \frac13[1,1,-1]$, residual $= [\frac23,\frac23,\frac43]$, **SPE = 24/9 ≈ 2.7**, versus noise-level SPE ≈ 0.001.

That is the correlation break: invisible per sensor, enormous in SPE.

## 4.6 Control limits

**T²** for a new observation, with $n$ training samples:

$$T^2_{\alpha} = \frac{k(n-1)(n+1)}{n(n-k)}\,F_{1-\alpha}(k,\, n-k)$$

**SPE** is a weighted sum of $\chi^2_1$ variables with weights = discarded eigenvalues. Jackson &
Mudholkar (1979) approximate its tail. With $\theta_i = \sum_{j>k}\lambda_j^{\,i}$,
$h_0 = 1 - \frac{2\theta_1\theta_3}{3\theta_2^2}$, $c_\alpha = \Phi^{-1}(1-\alpha)$:

$$Q_\alpha = \theta_1\left[\frac{c_\alpha\sqrt{2\theta_2 h_0^2}}{\theta_1} + 1 + \frac{\theta_2 h_0(h_0-1)}{\theta_1^2}\right]^{1/h_0}$$

Both assume Gaussian, independent samples. Machine data is neither (regimes, autocorrelation),
so in practice calibrate empirically on clean runs (lesson 10) and treat these as a sanity check.

## 4.7 Diagnosis: contribution plots

SPE is a sum over sensors: $\text{SPE} = \sum_j r_j^2$. The largest $r_j^2$ points to the sensor
that broke the relationship (`PCADetector.contributions`). That turns "something is wrong" into
"EGT disagrees with the others" — what an operator can act on.

## 4.8 Limits, and what the benchmark shows

| Detector | Event recall | FA / 10k |
|---|---|---|
| Mahalanobis | 30% | 0.8 |
| PCA SPE | 35% | 0.8 |
| PCA T² | 5% | 2.4 |

Why not better:
- **Linear.** $\text{EGT} \propto n^2$ is a curve; PCA fits a straight line, leaving curvature in SPE as "normal noise" → loose limits.
- **Static.** No lags, so healthy transients look like broken relationships (lesson 2.5).
- **Non-Gaussian, multi-regime** data → theoretical limits are off.
- **T² is nearly useless here** because the healthy operating envelope *already* spans the main modes (idle ↔ full power); faults rarely push along them.

Upgrades: **dynamic PCA** (augment with lagged copies), **kernel PCA** (nonlinear),
**autoencoder** (nonlinear PCA, lesson 9), or model the relationships directly (**residual models**, lesson 7).

## Check yourself

1. Two sensors with correlation 0.99. Along which direction is the Mahalanobis "cost" of moving highest?
2. The engine runs harder than anything in training, but all sensors stay consistent with each other. T² or SPE — which one fires?
3. Why does the covariance become near-singular with 50 sensors and 40 training samples?
