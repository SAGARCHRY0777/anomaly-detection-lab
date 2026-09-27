"""
Multivariate statistical detectors: Mahalanobis distance and PCA (Hotelling T^2 + SPE/Q).

These are the workhorses of industrial process monitoring (chemical plants have
run them since the 1990s). They catch the anomaly univariate methods cannot:
every sensor individually in range, but the *combination* impossible.

Maths in docs/04-multivariate-pca.md.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class Standardizer:
    """x -> (x - mean) / std, fitted on normal data only."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class MahalanobisDetector:
    """
    d^2(x) = (x - mu)^T S^{-1} (x - mu).

    Under a Gaussian model of normal data, d^2 ~ chi^2 with p degrees of freedom,
    so the threshold follows from the distribution instead of from guesswork.
    `shrinkage` blends S toward a scaled identity (Ledoit-Wolf style) so a
    near-singular covariance (correlated sensors!) does not explode.
    """

    def __init__(self, shrinkage: float = 0.01):
        self.shrinkage = shrinkage

    def fit(self, X):
        self.mu_ = X.mean(axis=0)
        S = np.cov(X, rowvar=False)
        p = S.shape[0]
        S = (1 - self.shrinkage) * S + self.shrinkage * np.trace(S) / p * np.eye(p)
        self.prec_ = np.linalg.inv(S)
        self.p_ = p
        return self

    def score(self, X):
        D = X - self.mu_
        return np.einsum("ij,jk,ik->i", D, self.prec_, D)

    def threshold(self, alpha: float = 0.001):
        return stats.chi2.ppf(1 - alpha, df=self.p_)


class PCADetector:
    """
    Split the space into a model subspace (top k principal components) and a residual subspace.

      T^2 = sum_{i<=k} t_i^2 / lambda_i    -> unusual *magnitude* along known modes of variation
      SPE = || x - P P^T x ||^2            -> variation the normal model has never seen
                                              (correlation breaks land here)

    Fit on standardised normal data.
    """

    def __init__(self, n_components: int | float = 0.95):
        self.n_components = n_components  # int = k, float = variance fraction to keep

    def fit(self, X):
        self.scaler_ = Standardizer().fit(X)
        Z = self.scaler_.transform(X)
        n = len(Z)
        # SVD of the data matrix == eigendecomposition of covariance, but numerically nicer.
        _, s, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
        eig = s ** 2 / (n - 1)
        if isinstance(self.n_components, float):
            k = int(np.searchsorted(np.cumsum(eig) / eig.sum(), self.n_components) + 1)
        else:
            k = self.n_components
        self.k_ = k
        self.eigvals_ = eig
        self.P_ = Vt[:k].T                      # (p, k) loadings
        self.lam_ = eig[:k]
        self.n_ = n
        return self

    def transform(self, X):
        return self.scaler_.transform(X) @ self.P_

    def t2(self, X):
        T = self.transform(X)
        return np.sum(T ** 2 / self.lam_, axis=1)

    def spe(self, X):
        Z = self.scaler_.transform(X)
        R = Z - (Z @ self.P_) @ self.P_.T
        return np.sum(R ** 2, axis=1)

    def contributions(self, x):
        """Per-sensor share of SPE for one sample — the 'which sensor?' answer for diagnosis."""
        z = self.scaler_.transform(x[None])[0]
        r = z - self.P_ @ (self.P_.T @ z)
        return r ** 2

    # ---- control limits -------------------------------------------------
    def t2_limit(self, alpha: float = 0.001):
        """Hotelling: T^2 ~ k(n-1)(n+1) / (n(n-k)) * F(k, n-k)."""
        n, k = self.n_, self.k_
        return k * (n - 1) * (n + 1) / (n * (n - k)) * stats.f.ppf(1 - alpha, k, n - k)

    def spe_limit(self, alpha: float = 0.001):
        """Jackson & Mudholkar (1979) approximation using the discarded eigenvalues."""
        rest = self.eigvals_[self.k_:]
        if rest.size == 0:
            return 0.0
        th1, th2, th3 = (np.sum(rest ** i) for i in (1, 2, 3))
        h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
        c = stats.norm.ppf(1 - alpha)
        return th1 * (c * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)
