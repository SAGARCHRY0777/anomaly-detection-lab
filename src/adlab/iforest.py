"""
Isolation Forest from scratch (Liu, Ting & Zhou, 2008).

Idea in one line: anomalies are *few and different*, so random axis-aligned
cuts isolate them in fewer steps than normal points.

    score(x) = 2 ^ ( -E[h(x)] / c(psi) )

h(x) = path length of x in one tree, averaged over trees; c(psi) = average
path length of an unsuccessful BST search over psi points, used to normalise.
Score near 1 -> anomaly, well below 0.5 -> normal.

Maths in docs/05-isolation-forest.md.
"""

from __future__ import annotations

import numpy as np

EULER_GAMMA = 0.5772156649


def c(n: int | np.ndarray) -> np.ndarray:
    """Average path length of unsuccessful search in a BST of n points."""
    n = np.asarray(n, float)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2 * (np.log(n[big] - 1) + EULER_GAMMA) - 2 * (n[big] - 1) / n[big]
    out[n == 2] = 1.0
    return out


class _Node:
    __slots__ = ("feat", "split", "left", "right", "size")

    def __init__(self, feat=-1, split=0.0, left=None, right=None, size=0):
        self.feat, self.split, self.left, self.right, self.size = feat, split, left, right, size


def _build(X, depth, max_depth, rng):
    n = len(X)
    if depth >= max_depth or n <= 1:
        return _Node(size=n)
    lo, hi = X.min(axis=0), X.max(axis=0)
    usable = np.flatnonzero(hi > lo)
    if usable.size == 0:           # all points identical: cannot split
        return _Node(size=n)
    f = int(rng.choice(usable))
    s = rng.uniform(lo[f], hi[f])
    mask = X[:, f] < s
    return _Node(f, s, _build(X[mask], depth + 1, max_depth, rng),
                 _build(X[~mask], depth + 1, max_depth, rng), n)


def _path_length(node, x, depth=0):
    while node.left is not None:
        node = node.left if x[node.feat] < node.split else node.right
        depth += 1
    # A leaf holding `size` points stands in for a subtree we did not grow: add its expected depth.
    return depth + float(c(node.size))


class IsolationForest:
    def __init__(self, n_trees: int = 100, sample_size: int = 256, seed: int | None = 0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        psi = min(self.sample_size, len(X))
        max_depth = int(np.ceil(np.log2(psi)))   # average depth of a balanced tree: deeper adds nothing
        self.trees_ = [_build(X[rng.choice(len(X), psi, replace=False)], 0, max_depth, rng)
                       for _ in range(self.n_trees)]
        self.c_ = float(c(psi))
        return self

    def path_lengths(self, X):
        """(n,) mean path length over trees, routing all rows through each tree at once."""
        h = np.zeros(len(X))
        for tree in self.trees_:
            _route(tree, X, np.arange(len(X)), 0, h)
        return h / len(self.trees_)

    def score(self, X):
        return 2.0 ** (-self.path_lengths(X) / self.c_)


def _route(node, X, idx, depth, out):
    """Vectorised _path_length: send the whole index set down the tree, split by split."""
    if node.left is None:
        out[idx] += depth + float(c(node.size))
        return
    go_left = X[idx, node.feat] < node.split
    _route(node.left, X, idx[go_left], depth + 1, out)
    _route(node.right, X, idx[~go_left], depth + 1, out)
