# Lesson 8 — Shape and change: Matrix Profile and change-point detection (PELT)

Code: `src/adlab/matrix_profile.py`, `src/adlab/changepoint.py`
Tests: `test_matrix_profile_matches_brute_force`, `test_pelt_finds_level_shifts`

Lessons 3–7 judge *samples*. This lesson judges *stretches*: "is this shape unusual?" and
"where did the behaviour change?".

## Part A — Matrix Profile (Yeh, Keogh et al., 2016)

### A.1 Definition

Subsequence of length $m$ starting at $i$: $T_{i,m} = [x_i, \dots, x_{i+m-1}]$.
**Matrix profile** $MP[i]$ = distance from $T_{i,m}$ to its nearest neighbour among all other
subsequences (excluding trivial matches). Large $MP[i]$ → this shape has no twin anywhere →
**discord** (anomaly). Small $MP[i]$ → repeated pattern → **motif**.

### A.2 z-normalised distance

Each window is z-normalised first: $\hat a = (a - \mu_a)/\sigma_a$. This compares *shape*,
ignoring level and amplitude. The distance collapses to a correlation:

$$\lVert\hat a - \hat b\rVert^2 = \lVert\hat a\rVert^2 + \lVert\hat b\rVert^2 - 2\,\hat a\cdot\hat b = m + m - 2m\rho_{ab} \;\Rightarrow\; d(a,b) = \sqrt{2m(1-\rho_{ab})}$$

(using $\lVert\hat a\rVert^2 = m$ and $\hat a\cdot\hat b = m\rho$ with population std). And

$$\rho_{ab} = \frac{\sum_k a_k b_k - m\mu_a\mu_b}{m\,\sigma_a\sigma_b}$$

so all we need are dot products $QT_{i,j} = T_{i,m}\cdot T_{j,m}$ plus rolling means/stds.
Range: $d=0$ identical shape, $d=\sqrt{2m}$ uncorrelated, $d=2\sqrt m$ exactly inverted.

### A.3 STOMP: $O(n^2)$ time, $O(n)$ memory

Neighbouring dot products share $m-1$ terms:

$$QT_{i,j} = QT_{i-1,j-1} - x_{i-1}x_{j-1} + x_{i+m-1}x_{j+m-1}$$

Each row of the distance matrix is derived from the previous in $O(n)$; keep only its minimum.
(`matrix_profile` — verified against the brute-force $O(n^2 m)$ definition in the test.)

**Exclusion zone.** $T_{i}$ and $T_{i+1}$ overlap by $m-1$ samples and are trivially similar;
ignore matches within $m/2$ of $i$.

### A.4 Self-join vs AB-join

- **Self-join**: compare the series with itself. Fails on the "twin freak": the same anomaly
  occurring twice — each copy matches the other and both look normal.
- **AB-join** (`ab_join`, used via `discord_score(x, m, reference=...)`): compare each test window
  to the nearest window in a *known-clean reference*. The right mode when clean history exists.
- **Streaming**: a *left* matrix profile (only past neighbours) or DAMP (Lu et al., 2022).

### A.5 Choosing $m$ and the catches

- $m$ ≈ the length of the pattern you care about (one cycle, one manoeuvre). Too short: noise; too long: anomaly diluted.
- z-normalisation **discards level**: level shifts and slow drifts are invisible (benchmark: 0% on both).
- Flat windows have $\sigma=0$ — z-normalisation undefined. The code clamps σ, which makes a stuck
  window look maximally strange: why the benchmark shows 100% on *stuck*. An artefact worth knowing.
- Univariate; multi-dimensional variants exist (mSTAMP).

Demo: sine of period 100 with a 40-sample burst at period 40 at t=1500, $m=100$:
discord at 1507, MP = 6.55 vs median 1.72.

## Part B — Change-point detection with PELT

### B.1 Problem

Split $y_{1:n}$ into segments with constant statistics. Change points are where segments start.
Uses: segment regimes before modelling; locate fault onsets offline (run on residuals).

### B.2 Cost of a segment (Gaussian mean change)

$$C(y_{a:b}) = \sum_{i=a}^{b-1}(y_i - \bar y)^2 = S_2 - \frac{S_1^2}{b-a}$$

where $S_1=\sum y_i$, $S_2=\sum y_i^2$ over the segment — $O(1)$ from cumulative sums.

### B.3 Penalised objective and dynamic programming

$$\min_{\text{segmentations}} \sum_{\text{segments}} C(\text{segment}) + \beta\cdot\#\text{segments}$$

Without $\beta$ the optimum is every point its own segment (cost 0). With it, a split must "pay for itself".

**Optimal partitioning**: $F(t)$ = best cost of $y_{0:t}$:

$$F(t) = \min_{s<t}\big[F(s) + C(y_{s:t}) + \beta\big], \qquad F(0) = -\beta$$

Record the arg-min $s$ and backtrack. $O(n^2)$.

### B.4 PELT pruning (Killick, Fearnhead & Eckley, 2012)

If $F(s) + C(y_{s:t}) \ge F(t)$, then $s$ can never be the optimal last change point for any
future $T > t$ (because $C(y_{s:t}) + C(y_{t:T}) \le C(y_{s:T})$). Drop it. With change points
spread through the data, the candidate set stays small → roughly $O(n)$.

### B.5 Penalty

$\beta = 2\sigma^2\ln n$ (BIC-like). $\sigma$ is estimated robustly from first differences:
$\text{Var}(y_{t}-y_{t-1}) = 2\sigma^2$ for noise, and median/MAD ignore the few jumps:
$\hat\sigma = 1.4826\,\text{MAD}(\Delta y)/\sqrt2$.
Demo: shifts at 300 and 500 → found `[300, 500]`; with $\beta=1$ → 85 spurious change points.

### B.6 Online relatives

CUSUM (lesson 3) is the online, single-change, known-size version. Bayesian Online Change-Point
Detection (Adams & MacKay, 2007) maintains a posterior over "time since last change".
Library: `ruptures` (PELT, binary segmentation, kernel costs).

## Check yourself

1. Why is a matrix profile blind to a +50 °C level shift?
2. The same glitch appears twice in a series. Self-join or AB-join, and why?
3. Why does the empty segmentation need $F(0) = -\beta$?
