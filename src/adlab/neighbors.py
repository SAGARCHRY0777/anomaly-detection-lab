"""
Distance- and density-based detectors: k-NN distance and Local Outlier Factor (LOF).

Brute force on purpose — the O(n^2) distance matrix makes the maths visible.
For real sizes use a KD-tree / ball-tree / FAISS index; the formulas do not change.

Maths in docs/06-distance-density-ocsvm.md.
"""

from __future__ import annotations

import numpy as np


def pairwise_sq_dist(A, B):
    """||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (vectorised, clipped at 0 for rounding)."""
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.maximum(d, 0)


class KNNDetector:
    """Score = distance to the k-th nearest *training* neighbour. Simple, strong, and hard to beat."""

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, X):
        self.X_ = X
        return self

    def score(self, X, chunk: int = 2048):
        out = np.empty(len(X))
        for s in range(0, len(X), chunk):           # chunked: the full matrix may not fit in RAM
            D = np.sqrt(pairwise_sq_dist(X[s:s + chunk], self.X_))
            out[s:s + chunk] = np.partition(D, self.k - 1, axis=1)[:, self.k - 1]
        return out


class LOF:
    """
    Local Outlier Factor (Breunig et al., 2000).

      k-dist(o)        = distance from o to its k-th neighbour
      reach-dist(p,o)  = max(k-dist(o), d(p,o))           # smooths out tiny distances
      lrd(p)           = 1 / mean_{o in N_k(p)} reach-dist(p,o)
      LOF(p)           = mean_{o in N_k(p)} lrd(o) / lrd(p)

    LOF ~ 1: as dense as its neighbours. LOF >> 1: in a sparser spot than its neighbours.
    'Local' is the point: it works when normal clusters have different densities
    (idle is tight, full power is noisy) where one global distance threshold fails.
    """

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X):
        k = self.k
        D = np.sqrt(pairwise_sq_dist(X, X))
        np.fill_diagonal(D, np.inf)                     # a point is not its own neighbour
        idx = np.argsort(D, axis=1)[:, :k]
        self.X_ = X
        self.kdist_ = np.take_along_axis(D, idx[:, -1:], axis=1)[:, 0]
        reach = np.maximum(np.take_along_axis(D, idx, axis=1), self.kdist_[idx])
        self.lrd_ = 1.0 / (reach.mean(axis=1) + 1e-12)
        return self

    def score(self, X, chunk: int = 2048):
        """Novelty mode: score new points against the fitted training set."""
        out = np.empty(len(X))
        for s in range(0, len(X), chunk):
            D = np.sqrt(pairwise_sq_dist(X[s:s + chunk], self.X_))
            idx = np.argpartition(D, self.k - 1, axis=1)[:, :self.k]
            reach = np.maximum(np.take_along_axis(D, idx, axis=1), self.kdist_[idx])
            lrd = 1.0 / (reach.mean(axis=1) + 1e-12)
            out[s:s + chunk] = self.lrd_[idx].mean(axis=1) / lrd
        return out
