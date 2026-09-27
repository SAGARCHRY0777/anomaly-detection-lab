"""
A dense autoencoder in pure NumPy with hand-written backprop and Adam.

Why hand-written: so the maths in docs/09-autoencoders.md maps line-by-line to code.
In practice use PyTorch; the architecture and the anomaly logic are identical.

Encoder  h = tanh(W1 x + b1),  z = W2 h + b2          (z: bottleneck, dim << input)
Decoder  g = tanh(W3 z + b3),  x_hat = W4 g + b4
Loss     L = mean ||x - x_hat||^2
Anomaly score = reconstruction error ||x - x_hat||^2.

A linear autoencoder with MSE learns exactly the PCA subspace — so the AE is
"nonlinear PCA", and its reconstruction error is a nonlinear SPE.
"""

from __future__ import annotations

import numpy as np

from .pca import Standardizer


class Autoencoder:
    def __init__(self, hidden: int = 32, bottleneck: int = 3, lr: float = 1e-3,
                 epochs: int = 50, batch: int = 256, seed: int = 0):
        self.hidden, self.bottleneck = hidden, bottleneck
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self.rng = np.random.default_rng(seed)

    def _init(self, d):
        def glorot(i, o):
            return self.rng.normal(0, np.sqrt(2 / (i + o)), (i, o))
        h, k = self.hidden, self.bottleneck
        self.params = {
            "W1": glorot(d, h), "b1": np.zeros(h),
            "W2": glorot(h, k), "b2": np.zeros(k),
            "W3": glorot(k, h), "b3": np.zeros(h),
            "W4": glorot(h, d), "b4": np.zeros(d),
        }
        self.m = {k_: np.zeros_like(v) for k_, v in self.params.items()}
        self.v = {k_: np.zeros_like(v) for k_, v in self.params.items()}
        self.t = 0

    def _forward(self, X):
        p = self.params
        h = np.tanh(X @ p["W1"] + p["b1"])
        z = h @ p["W2"] + p["b2"]
        g = np.tanh(z @ p["W3"] + p["b3"])
        out = g @ p["W4"] + p["b4"]
        return out, (X, h, z, g)

    def _backward(self, cache, out):
        X, h, z, g = cache
        p = self.params
        n = len(X)
        d_out = 2 * (out - X) / (n * X.shape[1])          # dL/dx_hat for mean squared error
        grads = {"W4": g.T @ d_out, "b4": d_out.sum(0)}
        d_g = d_out @ p["W4"].T * (1 - g ** 2)            # tanh'(a) = 1 - tanh(a)^2
        grads["W3"], grads["b3"] = z.T @ d_g, d_g.sum(0)
        d_z = d_g @ p["W3"].T                             # bottleneck is linear
        grads["W2"], grads["b2"] = h.T @ d_z, d_z.sum(0)
        d_h = d_z @ p["W2"].T * (1 - h ** 2)
        grads["W1"], grads["b1"] = X.T @ d_h, d_h.sum(0)
        return grads

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.params[k] -= self.lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, verbose: bool = False):
        self.scaler_ = Standardizer().fit(X)
        Z = self.scaler_.transform(X)
        self._init(Z.shape[1])
        self.history_ = []
        for ep in range(self.epochs):
            order = self.rng.permutation(len(Z))
            for s in range(0, len(Z), self.batch):
                xb = Z[order[s:s + self.batch]]
                out, cache = self._forward(xb)
                self._adam(self._backward(cache, out))
            loss = float(np.mean((self._forward(Z)[0] - Z) ** 2))
            self.history_.append(loss)
            if verbose:
                print(f"epoch {ep:3d}  loss {loss:.5f}")
        return self

    def score(self, X):
        Z = self.scaler_.transform(X)
        return np.sum((self._forward(Z)[0] - Z) ** 2, axis=1)
