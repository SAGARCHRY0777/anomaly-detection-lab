# Lesson 5 — Isolation Forest

Code: `src/adlab/iforest.py` · Test: `test_iforest_ranks_like_sklearn`, `test_c_matches_harmonic_definition`
Paper: Liu, Ting & Zhou, *Isolation Forest*, ICDM 2008.

## 5.1 The idea

Every method so far *models normal* and measures distance from it. Isolation Forest flips it:
it never models normal at all. It asks **how easy is this point to separate from everything else?**

Anomalies are **few** and **different**, so random cuts separate them quickly; normal points sit
in dense crowds and need many cuts.

```
1-D example, values:  ●●●●●●●●●●●● ●●●●●●●●●●                  ✗
                      |←──────────── range R ────────────────────→|
A random cut lands in the empty gap next to ✗ with probability  gap / R  — large.
A random cut separates one ● from its neighbour with probability  (tiny gap) / R  — small.
```

## 5.2 Building one isolation tree (iTree)

1. Draw a **subsample** of $\psi$ points (default 256) without replacement.
2. At a node: pick a random feature $q$; pick a split $p \sim \text{Uniform}(\min x_q, \max x_q)$ over the points in the node.
3. Send points with $x_q < p$ left, the rest right. Recurse.
4. Stop when a node has ≤ 1 point, all its points are identical, or depth reaches $\lceil \log_2 \psi \rceil$.

Path length $h(x)$ = number of edges from root to the leaf where $x$ lands. Grow $t$ trees (default 100)
and average: $E[h(x)]$.

**The depth limit.** We only care whether a point is isolated *early*. Beyond the average depth
of a balanced tree ($\log_2\psi$) every point is "normal", so growth stops. A leaf that still
holds $m$ points stands in for an unbuilt subtree, so its path length is corrected by the
expected extra depth:

$$h(x) = e + c(m)$$

where $e$ = edges traversed. (`_route`: `out[idx] += depth + c(node.size)`.)

## 5.3 Normalising: where $c(n)$ comes from

Raw path lengths grow with sample size, so we need a yardstick: the average path length for
$n$ points. An iTree has the same structure as a **binary search tree**, and isolating a point
corresponds to an *unsuccessful search* in a BST. Its expected length over random BSTs of $n$ nodes is

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}, \qquad H(i) \approx \ln i + \gamma,\quad \gamma = 0.5772\ldots$$

Intuition: in a BST built from keys in random order, key $j$ is an ancestor of key $i$ exactly
when $j$ is inserted first among all keys between them (inclusive) — probability $1/(|i-j|+1)$.
Depth = number of ancestors = $\sum_j 1/(|i-j|+1)$, a sum of harmonic numbers from both sides,
and $\sum_{i\le n} 1/i \approx \ln n$. So average depth ≈ $2\ln n \approx 1.39\log_2 n$.
$c(2) = 1$ and $c(1) = 0$ are base cases.

For $\psi = 256$: $c = 2(\ln 255 + 0.5772) - 2\cdot255/256 = 12.24 - 1.99 \approx 10.24$.

## 5.4 The score

$$s(x,\psi) = 2^{-\,E[h(x)]\,/\,c(\psi)}$$

| $E[h(x)]$ | $s$ | meaning |
|---|---|---|
| → 0 | → 1 | isolated immediately: anomaly |
| $= c(\psi)$ | 0.5 | average: nothing special |
| → $\psi - 1$ | → 0 | very hard to isolate: deep inside a dense crowd |

The $2^{-(\cdot)}$ maps "path length relative to average" onto (0, 1] with 0.5 as the neutral point.

## 5.5 Why subsample (small $\psi$ is a feature, not a shortcut)

- **Swamping:** normal points next to anomalies get isolated early and mislabelled. Fewer points ⇒ clearer gaps.
- **Masking:** a *cluster* of anomalies is dense, so its members are hard to isolate. Subsampling thins the cluster.
- **Cost:** training $O(t\,\psi\log\psi)$ — independent of $n$! Scoring $O(n\,t\log\psi)$. Linear, no distance matrix.

Empirically, path lengths converge by $t \approx 100$ trees, and $\psi = 256$ works across most datasets.

## 5.6 Weaknesses

1. **Axis-parallel cuts.** Every split is perpendicular to one feature, so the score map has
   rectangular artefacts; points along a *diagonal* normal band are poorly modelled (healthy data
   off the band but inside the bounding box can score as normal). **Extended Isolation Forest**
   (Hariri et al., 2019) uses random oblique hyperplanes.
2. **No time.** Each sample is judged alone — no dynamics, no context.
3. **Rare ≠ abnormal.** Healthy transients are rare, so they get isolated fast → false alarms.
   Contextual faults (a shift that lands on *another* healthy operating point) sit in dense
   regions → missed.

## 5.7 The benchmark lesson: representation beats algorithm

These are the two `isolation forest` rows printed by `examples/benchmark.py`,
which you can reproduce exactly — the run is seeded:

| Input | Event recall | FA / 10k |
|---|---|---|
| raw standardised sensors | **0.00** | 0.00 |
| residuals from lesson 7 | **0.82** | 2.28 |

**The raw row is zero, not merely low, and that is the result — not a broken
build.** At the calibrated threshold the detector fires on essentially nothing,
which is why its false-alarm rate is also zero and its mean delay prints as a
dash: there were no detections to measure a delay over. Per anomaly type it
scores 0.00 on all six, including spikes. If you were expecting a bug, note that
`test_iforest_ranks_like_sklearn` checks this same implementation against
scikit-learn on scattered outliers and passes — the algorithm is fine, the
*input representation* is not.

Identical code. In raw space normal data is a long curved manifold with dense and sparse parts;
a +50 °C EGT fault lands where another healthy operating point already lives. In residual space
normal data collapses to a **tight ball around the origin** in every regime, and any fault pushes
the point off it along one axis — exactly what axis-parallel random cuts isolate fastest.

## Check yourself

1. Why does the tree depth stop at $\lceil\log_2\psi\rceil$ and not grow until every point is alone?
2. What score does a point get if $E[h] = 2\,c(\psi)$?
3. A cluster of 30 identical anomalous points in 10,000 samples. Why does $\psi = 256$ help?
