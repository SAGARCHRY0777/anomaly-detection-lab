"""
Univariate statistical detectors — the baselines everything else must beat.

Each function takes a 1-D signal and returns a per-sample anomaly *score*
(higher = more anomalous). Turning a score into an alarm is a separate step
(see threshold.py) — keeping the two apart is one of the habits that makes
detectors comparable.

Maths for every function is in docs/03-statistical-detectors.md.
"""

from __future__ import annotations

import numpy as np

MAD_TO_SIGMA = 1.4826  # 1 / Phi^{-1}(0.75): makes MAD estimate sigma for Gaussian data


def zscore(x: np.ndarray, mu: float | None = None, sigma: float | None = None) -> np.ndarray:
    """|x - mu| / sigma. Pass mu/sigma fitted on *normal* data; otherwise they are estimated from x."""
    mu = np.mean(x) if mu is None else mu
    sigma = np.std(x) if sigma is None else sigma
    return np.abs(x - mu) / (sigma + 1e-12)


def robust_zscore(x: np.ndarray, med: float | None = None, mad: float | None = None) -> np.ndarray:
    """|x - median| / (1.4826 * MAD). Breakdown point 50%: outliers cannot drag the estimate."""
    med = np.median(x) if med is None else med
    mad = np.median(np.abs(x - med)) if mad is None else mad
    return np.abs(x - med) / (MAD_TO_SIGMA * mad + 1e-12)


def rolling_robust_zscore(x: np.ndarray, window: int = 101) -> np.ndarray:
    """
    Robust z against a *trailing* window (causal — uses only the past, so it can run live).
    The first `window` samples get score 0: there is no history to judge them by.
    """
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    view = np.lib.stride_tricks.sliding_window_view(x, window)[:-1]  # windows ending at t-1
    med = np.median(view, axis=1)
    mad = np.median(np.abs(view - med[:, None]), axis=1)
    out[window:] = np.abs(x[window:] - med) / (MAD_TO_SIGMA * mad + 1e-12)
    return out


def ewma(x: np.ndarray, lam: float = 0.1, mu0: float | None = None) -> np.ndarray:
    """z_t = lam * x_t + (1 - lam) * z_{t-1}.  Effective memory ~ (2 - lam) / lam samples."""
    z = np.empty(len(x))
    prev = x[0] if mu0 is None else mu0
    for t, v in enumerate(x):
        prev = lam * v + (1 - lam) * prev
        z[t] = prev
    return z


def ewma_chart(x: np.ndarray, mu: float, sigma: float, lam: float = 0.1) -> np.ndarray:
    """
    EWMA control-chart statistic in sigma units of the EWMA itself.

    Var(z_t) = sigma^2 * lam/(2-lam) * (1 - (1-lam)^(2t)) -> sigma^2 * lam/(2-lam).
    Score = |z_t - mu| / sd(z_t). Classic alarm at L ~ 2.7-3.
    """
    z = ewma(x, lam, mu0=mu)
    t = np.arange(1, len(x) + 1)
    sd = sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    return np.abs(z - mu) / (sd + 1e-12)


def cusum(x: np.ndarray, mu: float, sigma: float, k: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-sided tabular CUSUM (Page, 1954), on standardised data.

        S+_t = max(0, S+_{t-1} + z_t - k)
        S-_t = max(0, S-_{t-1} - z_t - k)

    k = half the shift (in sigmas) you want to detect fastest. Alarm when S > h (h ~ 4-5).
    Returns (S_plus, S_minus). Score = max of the two.
    """
    z = (np.asarray(x, float) - mu) / (sigma + 1e-12)
    sp = np.empty(len(z))
    sn = np.empty(len(z))
    a = b = 0.0
    for t, v in enumerate(z):
        a = max(0.0, a + v - k)
        b = max(0.0, b - v - k)
        sp[t], sn[t] = a, b
    return sp, sn


def stuck_score(x: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Flat-line detector: 1 / (rolling std + eps), log-scaled. Real sensors always have noise;
    zero variance over `window` samples is itself the anomaly. No distance-based method sees this,
    because a stuck value sits comfortably inside the normal range.
    """
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    sd = np.lib.stride_tricks.sliding_window_view(x, window).std(axis=1)
    out[window - 1:] = -np.log10(sd + 1e-9)
    return out
