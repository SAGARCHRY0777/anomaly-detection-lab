# Lesson 6 — Distance and density: kNN, LOF, One-Class SVM

Code: `src/adlab/neighbors.py` · Test: `test_lof_matches_sklearn_novelty`, `test_knn_distance`

## 6.1 kNN distance

Score = distance from $x$ to its $k$-th nearest neighbour in the (clean) training set, $r_k(x)$.

**Why it is a density estimate.** The ball of radius $r_k(x)$ around $x$ contains $k$ of $n$ points, so

$$\hat p(x) \approx \frac{k}{n \cdot V_d\, r_k(x)^d}$$

($V_d$ = volume of the unit ball in $d$ dimensions). Large $r_k$ ⇔ low density ⇔ anomaly.
Ranking by $r_k$ is ranking by estimated $1/\hat p$ — no parametric assumption at all.

**Choosing $k$.** $k=1$: one noisy training point near a fault hides it. Large $k$: smoother,
but small normal clusters (with fewer than $k$ members) start to look anomalous. 5–20 is typical.

**Curse of dimensionality.** As $d$ grows, all pairwise distances concentrate:
$(\max - \min)/\min \to 0$. Everything is equally far from everything. Standardise features,
keep $d$ modest, or reduce dimension (PCA) first.

## 6.2 The problem with any *global* distance threshold

```
tight cluster (idle)          loose cluster (full power)
   ●●●                          ●    ●     ●
   ●●●     A                       ●    ●   B
   ●●●                          ●     ●    ●
```

A and B are the same distance from their nearest neighbours. A is clearly anomalous (the idle
cluster is tight); B is normal (full power is noisy). One global threshold on $r_k$ cannot get both right.

## 6.3 Local Outlier Factor (Breunig et al., 2000)

Compare each point's density to **its neighbours' density**.

| Quantity | Definition |
|---|---|
| $k\text{-dist}(o)$ | distance from $o$ to its $k$-th nearest neighbour |
| $N_k(p)$ | the $k$ nearest neighbours of $p$ |
| reach-dist$_k(p, o)$ | $\max\{k\text{-dist}(o),\; d(p,o)\}$ |
| lrd$_k(p)$ | $\left(\frac{1}{k}\sum_{o\in N_k(p)} \text{reach-dist}_k(p,o)\right)^{-1}$ |
| LOF$_k(p)$ | $\frac{1}{k}\sum_{o\in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}$ |

- **lrd** (local reachability density) = inverse of the average distance to neighbours.
- **LOF ≈ 1**: as dense as its neighbours → normal, whatever the absolute density.
- **LOF ≫ 1**: much sparser than its neighbours → outlier.

In the picture: A's neighbours (idle points) have very high lrd, A's own lrd is low → LOF large.
B's neighbours are sparse too → LOF ≈ 1.

**Why reach-dist instead of plain distance?** Inside a dense cluster, distances to the very
nearest points are tiny and noisy; dividing by them makes lrd explode randomly. Flooring at
$k\text{-dist}(o)$ smooths this: all points within $o$'s neighbourhood get the same reach distance.

**Outlier vs novelty mode.** *Outlier mode* scores the training points themselves (exclude self as
neighbour). *Novelty mode* (our `LOF.score`, sklearn `novelty=True`) fits on clean data and scores
new points against it — the right mode for monitoring.

**Cost.** Brute force $O(n^2)$ distances (chunked in the code). KD/ball trees help in low $d$;
FAISS / HNSW approximate neighbours for millions of points.

## 6.4 One-Class SVM (Schölkopf et al., 2001)

Map points into a feature space $\phi(x)$ and find a hyperplane that separates the data from
the origin with maximum margin. Points on the origin side are anomalies.

**Primal:**

$$\min_{w,\xi,\rho}\; \frac12\lVert w\rVert^2 + \frac{1}{\nu n}\sum_i \xi_i - \rho \quad\text{s.t.}\quad w\cdot\phi(x_i) \ge \rho - \xi_i,\;\; \xi_i \ge 0$$

- $\rho$: offset of the hyperplane (push it far from the origin = tight boundary),
- $\xi_i$: slack — lets some training points fall outside,
- $\nu \in (0,1]$: the trade-off.

**Dual** (via Lagrange multipliers $\alpha_i$):

$$\min_\alpha\; \frac12\sum_{i,j}\alpha_i\alpha_j K(x_i, x_j) \quad\text{s.t.}\quad 0 \le \alpha_i \le \frac{1}{\nu n},\;\; \sum_i \alpha_i = 1$$

**Decision function:**

$$f(x) = \sum_i \alpha_i K(x_i, x) - \rho \qquad f(x) < 0 \Rightarrow \text{anomaly}$$

Only points with $\alpha_i > 0$ (support vectors) matter.

**The ν-property.** $\nu$ is an upper bound on the fraction of training points outside the boundary
and a lower bound on the fraction of support vectors. So $\nu$ ≈ "fraction of training data I allow
to be outliers" — interpretable, unlike most SVM knobs.

**The kernel trick with RBF.** $K(x,x') = \exp(-\gamma\lVert x-x'\rVert^2) = \phi(x)\cdot\phi(x')$ for an
infinite-dimensional $\phi$ we never compute. Since $K(x,x)=1$, every point lies on the unit sphere
in feature space; the hyperplane cuts off a *cap*, which maps back to a smooth closed boundary
around the data in input space (equivalent to SVDD — smallest enclosing ball — for RBF).

Look at $f(x)$ again: a weighted sum of Gaussian bumps centred on support vectors, minus a
threshold. It is essentially a sparse **kernel density estimate** with a cut-off.

- $\gamma$ too large → every point gets its own tiny bubble (overfit, everything new is anomalous).
- $\gamma$ too small → one big blob (underfit, misses everything).
- Training is $O(n^2)$–$O(n^3)$ → subsample to a few thousand points.

Not implemented from scratch here (it needs a quadratic-programming solver); use
`sklearn.svm.OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")` on standardised data.

## 6.5 What the benchmark says

These rows come from a seeded run of `examples/benchmark.py`; rerun it to reproduce them.

| Detector (raw standardised sensors) | Event recall | FA / 10k |
|---|---|---|
| kNN distance | 0.07 | 2.56 |
| LOF | 0.50 | 4.84 |

**The gap between these two rows is the lesson.** Both read the same raw space, and the only
difference is what they compare a point against. kNN scores an *absolute* distance, so on a
manifold whose density varies by regime it cannot tell "far because faulty" from "far because
this operating point is simply sparse" — 0.07 recall. LOF divides by the local neighbourhood's
own density, which cancels that variation, and it becomes the **strongest raw-space detector on
this benchmark** at 0.50 — ahead of Mahalanobis (0.42), PCA SPE (0.28) and Isolation Forest,
which scores 0.00.

It pays for that in the column nobody quotes: **4.84 false alarms per 10k is the worst rate in
the whole table.** Local normalisation is what lets LOF see a fault in a sparse region, and it is
also what makes an ordinary sparse transient look like one. The comparison with Isolation Forest
is the useful one — axis-parallel cuts cannot represent a curved, varying-density manifold at
all, while a local density ratio can, at the cost of firing on the manifold's own thin patches.

All of these improve sharply on residuals, for the reason lesson 7 gives: residuals collapse
every regime onto one tight ball, so "far from normal" stops depending on which regime you are
in — and absolute distance starts working again.

## Check yourself

1. You double every feature's scale except one. What happens to kNN scores, and why must you standardise?
2. LOF = 1.05 for a point far from everything, in a region where all points are far apart. Normal or anomalous?
3. You set $\nu = 0.05$. Roughly what fraction of *training* points end up flagged?
