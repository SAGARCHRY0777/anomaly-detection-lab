"""
Residual-based detection: "predict what the sensor *should* read, alarm on the gap".

For each sensor j, fit a regression  x_j(t) ~ f( other sensors at t, t-1, ..., t-L ).
On normal data the residual r_j = x_j - f(.) is small, noise-like, and — this is the
whole trick — *independent of operating point*. Idle or full power, a healthy residual
is ~N(0, sigma_j). So one threshold works everywhere, which no raw-value threshold can.

The model here is ridge regression in closed form; swap in LightGBM or a small
neural net without changing anything else. Maths in docs/07-residual-models.md.
"""

from __future__ import annotations

import numpy as np

from .stats import MAD_TO_SIGMA


def lag_features(X: np.ndarray, lags: int) -> np.ndarray:
    """[x(t), x(t-1), ..., x(t-lags)] per row. First `lags` rows repeat the first sample."""
    pad = np.vstack([np.repeat(X[:1], lags, axis=0), X])
    return np.hstack([pad[lags - L: len(pad) - L] for L in range(lags + 1)])


def ridge_fit(A, b, alpha):
    """w = (A^T A + alpha I)^{-1} A^T b, intercept left unpenalised via centring."""
    am, bm = A.mean(0), b.mean()
    Ac = A - am
    w = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ (b - bm))
    return w, bm - am @ w


class ResidualDetector:
    def __init__(self, lags: int = 5, alpha: float = 1.0):
        self.lags = lags
        self.alpha = alpha

    def _design(self, X, j):
        others = np.delete(X, j, axis=1)
        return lag_features((others - self.mu_[np.arange(X.shape[1]) != j]) /
                            self.sd_[np.arange(X.shape[1]) != j], self.lags)

    def fit(self, X, run_id=None):
        """Fit per sensor. Pass run_id so lags never straddle two runs (they are not continuous in time)."""
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        self.models_, self.res_scale_ = [], []
        for j in range(X.shape[1]):
            A, b = self._blocks(X, j, run_id)
            w, b0 = ridge_fit(A, b, self.alpha)
            r = b - (A @ w + b0)
            self.models_.append((w, b0))
            self.res_scale_.append(MAD_TO_SIGMA * np.median(np.abs(r - np.median(r))) + 1e-12)
        self.res_scale_ = np.array(self.res_scale_)
        return self

    def _blocks(self, X, j, run_id):
        if run_id is None:
            return self._design(X, j), X[:, j]
        parts = [(self._design(X[run_id == r], j), X[run_id == r, j]) for r in np.unique(run_id)]
        return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def residuals(self, X):
        """(T, d) standardised residuals — each column ~N(0,1) on healthy data."""
        R = np.empty_like(X, dtype=float)
        for j, (w, b0) in enumerate(self.models_):
            R[:, j] = (X[:, j] - (self._design(X, j) @ w + b0)) / self.res_scale_[j]
        return R

    def score(self, X):
        """Sum of squared standardised residuals: ~chi^2(d) on healthy data."""
        return np.sum(self.residuals(X) ** 2, axis=1)
