"""
Score -> alarm. The most under-taught part of anomaly detection, and where most
false alarms are born.

Without labels you cannot pick a threshold by maximising F1. You pick it from the
distribution of scores on *normal* data, i.e. from a false-alarm budget:
"on clean data I accept at most alpha of samples above the line".

Maths in docs/10-thresholds-and-evaluation.md.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def quantile_threshold(normal_scores: np.ndarray, alpha: float = 0.001) -> float:
    """Empirical (1 - alpha) quantile. Needs roughly >= 10/alpha normal samples to be trustworthy."""
    return float(np.quantile(normal_scores, 1 - alpha))


def pot_threshold(normal_scores: np.ndarray, alpha: float = 1e-4, init_q: float = 0.98) -> float:
    """
    Peaks-Over-Threshold (extreme value theory; Siffer et al., 'SPOT', KDD 2017).

    Pick a high initial threshold u (98th pct). By the Pickands-Balkema-de Haan theorem the
    excesses Y = s - u | s > u follow a Generalised Pareto Distribution GPD(xi, beta).
    Then for a target tail probability alpha:

        z_alpha = u + beta/xi * ( (alpha * n / N_u) ^ (-xi) - 1 )

    This extrapolates *beyond* the data you have — the reason to use it when alpha is tiny.
    """
    s = np.asarray(normal_scores, float)
    u = np.quantile(s, init_q)
    excess = s[s > u] - u
    xi, _, beta = stats.genpareto.fit(excess, floc=0)
    n, nu = len(s), len(excess)
    if abs(xi) < 1e-6:                            # exponential tail limit
        return float(u - beta * np.log(alpha * n / nu))
    return float(u + beta / xi * ((alpha * n / nu) ** (-xi) - 1))


def persistence(alarm: np.ndarray, k: int = 3, n: int = 5) -> np.ndarray:
    """
    k-of-n debounce: alarm at t only if >= k of the last n raw alarms fired.
    If raw false alarms are independent with prob p, the rate drops to ~ C(n,k) p^k.
    Costs up to (k-1) samples of detection delay — a trade you make on purpose.
    """
    a = np.asarray(alarm, int)
    c = np.convolve(a, np.ones(n, int))[:len(a)]
    return c >= k


def to_alarm(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores) > threshold
