"""
Offline change-point detection: where does the *statistics* of the signal change?

Anomaly detection asks "is this point odd?"; change-point detection asks "did the
regime switch here?". Level shifts and drifts are change points, and so are
healthy throttle steps — which is why segmentation comes before detection.

Implemented: optimal partitioning with a Gaussian mean-change cost and PELT pruning
(Killick, Fearnhead & Eckley, 2012). Production: the `ruptures` library.
Maths in docs/08-matrix-profile-changepoint.md.
"""

from __future__ import annotations

import numpy as np


def pelt(x: np.ndarray, penalty: float | None = None, min_size: int = 5) -> list[int]:
    """
    Minimise  sum_segments cost(segment) + penalty * (#segments)

    cost(y_{a:b}) = sum (y - mean)^2 = S2 - S1^2 / n     (O(1) from cumulative sums)

    F(t) = min_{s<t} [ F(s) + cost(s:t) + penalty ]
    PELT: drop s from future candidates once F(s) + cost(s:t) > F(t) — it can never win again.
    Returns sorted change-point indices (segment starts, excluding 0).
    """
    x = np.asarray(x, float)
    n = len(x)
    if penalty is None:
        sigma = 1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2)   # noise sd, robust to the jumps
        penalty = 2 * np.log(n) * sigma ** 2                          # BIC-style
    S1 = np.concatenate([[0], np.cumsum(x)])
    S2 = np.concatenate([[0], np.cumsum(x ** 2)])

    def cost(s, t):
        m = t - s
        return S2[t] - S2[s] - (S1[t] - S1[s]) ** 2 / m

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(n + 1, int)
    candidates = [0]
    for t in range(min_size, n + 1):
        valid = [s for s in candidates if t - s >= min_size]
        if not valid:
            continue
        vals = [F[s] + cost(s, t) + penalty for s in valid]
        best = int(np.argmin(vals))
        F[t], last[t] = vals[best], valid[best]
        candidates = [s for s, v in zip(valid, vals) if v - penalty <= F[t]] + \
                     [s for s in candidates if t - s < min_size] + [t]
    cps, t = [], n
    while t > 0:
        t = last[t]
        if t > 0:
            cps.append(t)
    return sorted(int(c) for c in cps)
