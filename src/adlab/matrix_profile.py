"""
Matrix Profile (Yeh, Keogh et al., 2016) — shape-based anomaly ("discord") detection.

For every subsequence of length m, find the z-normalised Euclidean distance to its
nearest *non-trivial* match elsewhere in the series. That vector of distances is the
matrix profile. A subsequence whose nearest neighbour is still far away has no twin
anywhere — it is a discord, the shape anomaly.

    d(A, B) = sqrt( 2m (1 - corr(A, B)) )        for z-normalised A, B

This file uses the O(n^2) STOMP-style recurrence so you can read it; `stumpy.stump`
is the production version. Maths in docs/08-matrix-profile-changepoint.md.
"""

from __future__ import annotations

import numpy as np


def _moving_mean_std(x, m):
    w = np.lib.stride_tricks.sliding_window_view(x, m)
    return w.mean(axis=1), w.std(axis=1)


def matrix_profile(x: np.ndarray, m: int, exclusion: int | None = None) -> np.ndarray:
    """Self-join matrix profile. Returns array of length n - m + 1."""
    x = np.asarray(x, float)
    n = len(x) - m + 1
    exclusion = m // 2 if exclusion is None else exclusion
    mu, sd = _moving_mean_std(x, m)
    sd = np.where(sd < 1e-8, 1e-8, sd)   # flat windows: avoid divide-by-zero
    mp = np.full(n, np.inf)

    # QT[j] = dot(T[i:i+m], T[j:j+m]) for the current row i, updated in O(1) per cell:
    #   QT_i[j] = QT_{i-1}[j-1] - x[i-1]x[j-1] + x[i+m-1]x[j+m-1]
    first = np.array([x[0:m] @ x[j:j + m] for j in range(n)])
    qt = first.copy()
    for i in range(n):
        if i > 0:
            qt[1:] = qt[:-1] - x[i - 1] * x[:n - 1] + x[i + m - 1] * x[m:m + n - 1]
            qt[0] = first[i]
        corr = (qt - m * mu[i] * mu) / (m * sd[i] * sd)
        d = np.sqrt(np.maximum(2 * m * (1 - corr), 0))
        lo, hi = max(0, i - exclusion), min(n, i + exclusion + 1)
        d[lo:hi] = np.inf                      # trivial matches: a window matches its own shifted self
        mp[i] = d.min()
    return mp


def discord_score(x: np.ndarray, m: int, reference: np.ndarray | None = None) -> np.ndarray:
    """
    Per-sample score: each sample gets the max matrix-profile value of any window covering it.
    With `reference` (known-normal series) this becomes an AB-join: 'how far is each
    test window from anything seen in normal data' — the right mode when you have clean history.
    """
    if reference is None:
        mp = matrix_profile(x, m)
    else:
        mp = ab_join(x, reference, m)
    out = np.zeros(len(x))
    for i, v in enumerate(mp):
        out[i:i + m] = np.maximum(out[i:i + m], v)
    return out


def ab_join(a: np.ndarray, b: np.ndarray, m: int) -> np.ndarray:
    """For each window of a, distance to the nearest window of b (no exclusion zone needed)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    Wa = np.lib.stride_tricks.sliding_window_view(a, m)
    Wb = np.lib.stride_tricks.sliding_window_view(b, m)
    Za = (Wa - Wa.mean(1, keepdims=True)) / np.maximum(Wa.std(1, keepdims=True), 1e-8)
    Zb = (Wb - Wb.mean(1, keepdims=True)) / np.maximum(Wb.std(1, keepdims=True), 1e-8)
    out = np.empty(len(Za))
    step = 512
    for s in range(0, len(Za), step):                 # chunked to bound memory
        corr = Za[s:s + step] @ Zb.T / m
        out[s:s + step] = np.sqrt(np.maximum(2 * m * (1 - corr.max(axis=1)), 0))
    return out
