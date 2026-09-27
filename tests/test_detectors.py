"""
Every from-scratch algorithm is checked against either a reference library or a
brute-force definition. If these pass, the code in src/ really is the maths in docs/.
"""

import numpy as np
import pytest

from adlab import data, stats
from adlab.changepoint import pelt
from adlab.evaluate import event_metrics, point_adjust, point_metrics, segments
from adlab.iforest import IsolationForest, c
from adlab.matrix_profile import matrix_profile
from adlab.neighbors import LOF, KNNDetector
from adlab.pca import MahalanobisDetector, PCADetector
from adlab.residual import ResidualDetector
from adlab.threshold import persistence, pot_threshold, quantile_threshold

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ data
def test_simulator_is_correlated_and_labels_match_events():
    train, test = data.make_dataset(n_train=2, n_test=2, length=1000, seed=1)
    corr = np.corrcoef(train[0].X, rowvar=False)
    assert corr[0, 2] > 0.9            # speed and EGT move together
    for run in test:
        assert run.y.sum() == sum(e.end - e.start for e in run.events)


# ----------------------------------------------------------------- stats
def test_robust_z_ignores_outliers_but_z_does_not():
    x = rng.normal(0, 1, 1000)
    x[:50] = 1000
    assert robust_sd(x) < 1.2
    assert np.std(x) > 100


def robust_sd(x):
    return stats.MAD_TO_SIGMA * np.median(np.abs(x - np.median(x)))


def test_cusum_detects_small_shift_that_z_misses():
    r = np.random.default_rng(42)
    x = r.normal(0, 1, 400)
    x[200:] += 1.0                                  # 1 sigma: a 3-sigma rule almost never fires
    sp, _ = stats.cusum(x, mu=0, sigma=1, k=0.5)    # k = half the shift to detect
    h = 5                                           # textbook pair (k=0.5, h=5): in-control ARL ~ 930
    assert np.mean(stats.zscore(x[200:], 0, 1) > 3) < 0.05
    assert sp[:200].max() < h
    assert np.argmax(sp[200:] > h) < 30             # caught within ~10-30 samples


def test_stuck_score_flags_flatline():
    x = rng.normal(0, 1, 300)
    x[100:200] = x[100]
    s = stats.stuck_score(x, 20)
    assert s[130:200].min() > s[250:].max()


# -------------------------------------------------------------- pca / mahal
def test_pca_matches_sklearn_and_catches_correlation_break():
    from sklearn.decomposition import PCA
    z = rng.normal(size=(2000, 1))
    X = np.hstack([z, 2 * z, -z]) + rng.normal(0, 0.05, (2000, 3))
    det = PCADetector(n_components=1).fit(X)
    Z = det.scaler_.transform(X)
    ref = PCA(1).fit(Z)
    assert np.allclose(np.abs(det.P_[:, 0]), np.abs(ref.components_[0]), atol=1e-6)
    good = np.array([[1.0, 2.0, -1.0]])
    broken = np.array([[1.0, 2.0, 1.0]])            # each value normal, relation broken
    assert det.spe(broken)[0] > 100 * det.spe(good)[0]
    assert det.spe(broken)[0] > det.spe_limit(0.001)


def test_mahalanobis_follows_chi2():
    X = rng.multivariate_normal([0, 0, 0], [[1, .8, 0], [.8, 1, 0], [0, 0, 1]], 20000)
    det = MahalanobisDetector(shrinkage=0).fit(X)
    frac = np.mean(det.score(X) > det.threshold(0.01))
    assert 0.007 < frac < 0.013


# ---------------------------------------------------------------- iforest
def test_c_matches_harmonic_definition():
    n = 256
    H = np.sum(1 / np.arange(1, n))                 # H(n-1)
    assert abs(float(c(n)) - (2 * H - 2 * (n - 1) / n)) < 0.01


def test_iforest_ranks_like_sklearn():
    from sklearn.ensemble import IsolationForest as SK
    X = np.vstack([rng.normal(0, 1, (500, 2)), rng.uniform(-6, 6, (20, 2))])
    ours = IsolationForest(n_trees=100, seed=0).fit(X).score(X)
    ref = -SK(n_estimators=200, random_state=0).fit(X).score_samples(X)
    assert np.corrcoef(ours, ref)[0, 1] > 0.9
    assert ours[500:].mean() > ours[:500].mean() + 0.1


# ---------------------------------------------------------------- neighbours
def test_lof_matches_sklearn_novelty():
    from sklearn.neighbors import LocalOutlierFactor
    X = np.vstack([rng.normal(0, 0.3, (300, 2)), rng.normal(5, 2, (300, 2))])
    Q = rng.normal(2.5, 2, (100, 2))
    ours = LOF(k=20).fit(X).score(Q)
    ref = -LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X).score_samples(Q)
    assert np.allclose(ours, ref, rtol=1e-6)


def test_knn_distance():
    X = np.array([[0.0], [1.0], [3.0]])
    assert KNNDetector(k=2).fit(X).score(np.array([[0.0]]))[0] == pytest.approx(1.0)


# ---------------------------------------------------------------- residual
def test_residual_is_small_on_normal_and_large_on_break():
    train, _ = data.make_dataset(n_train=4, n_test=0, length=1500, seed=2)
    X, _, _, rid = data.stack(train)
    det = ResidualDetector(lags=3).fit(X, rid)
    run = data.simulate_run(1500, seed=99)
    clean = det.residuals(run.X)
    # Not ~0: ambient temperature is unobserved, so each run carries a small bias.
    # That is realistic — and why real systems feed ambient in as an input.
    assert np.all(np.abs(np.median(clean, axis=0)) < 1.5)
    data.inject(run, "correlation_break", "egt", 600, 150, magnitude=1.0)
    R = det.residuals(run.X)
    assert np.abs(R[620:750, 2]).mean() > 5 * np.abs(R[:500, 2]).mean()


# ---------------------------------------------------------------- matrix profile
def test_matrix_profile_matches_brute_force():
    x = np.sin(np.arange(300) / 5) + rng.normal(0, 0.05, 300)
    x[150:160] += 1.5
    m = 20
    mp = matrix_profile(x, m)
    W = np.lib.stride_tricks.sliding_window_view(x, m)
    Z = (W - W.mean(1, keepdims=True)) / W.std(1, keepdims=True)
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    for i in range(len(D)):
        D[i, max(0, i - m // 2):i + m // 2 + 1] = np.inf
    assert np.allclose(mp, D.min(1), atol=1e-5)
    assert 130 <= int(np.argmax(mp)) <= 160


# ---------------------------------------------------------------- change points
def test_pelt_finds_level_shifts():
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(4, 1, 200), rng.normal(-2, 1, 200)])
    cps = pelt(x)
    assert len(cps) == 2
    assert abs(cps[0] - 200) < 5 and abs(cps[1] - 400) < 5


# ---------------------------------------------------------------- thresholds
def test_pot_extrapolates_close_to_true_quantile():
    s = rng.exponential(1.0, 50000)
    true = -np.log(1e-4)
    assert abs(pot_threshold(s, alpha=1e-4) - true) / true < 0.15
    assert quantile_threshold(s, 0.01) == pytest.approx(-np.log(0.01), rel=0.05)


def test_persistence_suppresses_isolated_alarms():
    a = np.zeros(100, bool)
    a[[10, 40, 70]] = True
    a[50:60] = True
    out = persistence(a, k=3, n=5)
    assert not out[:50].any() and out[52:60].all()


# ---------------------------------------------------------------- evaluation
def test_point_adjust_inflates():
    y = np.zeros(1000, bool)
    y[100:300] = True
    a = np.zeros(1000, bool)
    a[299] = True                                   # one lucky hit at the very end
    assert point_metrics(y, a)["recall"] == pytest.approx(1 / 200)
    assert point_metrics(y, point_adjust(y, a))["recall"] == 1.0


def test_event_metrics():
    y = np.zeros(1000, bool)
    y[100:200] = True
    a = np.zeros(1000, bool)
    a[110] = True
    a[600:605] = True
    m = event_metrics(y, a)
    assert m["event_recall"] == 1 and m["false_alarms"] == 1 and m["mean_delay"] == 10
    assert segments(a) == [(110, 111), (600, 605)]
