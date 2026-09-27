"""
End-to-end: every detector, same data, same thresholding rule, same metrics.

    python examples/benchmark.py            (about 1-2 minutes)
    python examples/benchmark.py --fast     (smaller data, ~20 s)

The protocol is the one a careful engineer uses when there are no real labels:

  1. TRAIN on clean runs                -> learn "normal"
  2. CALIBRATE on *other* clean runs    -> threshold = score quantile at a false-alarm budget
  3. TEST on runs with injected faults  -> event recall, false alarms, delay, per-kind recall

Labels are touched only in step 3. Thresholds never see a single anomaly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adlab import data, stats  # noqa: E402
from adlab.autoencoder import Autoencoder  # noqa: E402
from adlab.evaluate import event_metrics, per_kind_recall, point_adjust, point_metrics  # noqa: E402
from adlab.iforest import IsolationForest  # noqa: E402
from adlab.matrix_profile import discord_score  # noqa: E402
from adlab.neighbors import LOF, KNNDetector  # noqa: E402
from adlab.pca import MahalanobisDetector, PCADetector, Standardizer  # noqa: E402
from adlab.residual import ResidualDetector  # noqa: E402
from adlab.threshold import persistence, quantile_threshold  # noqa: E402


def per_run(runs, fn):
    """Apply a per-run scoring function (anything with memory must not cross run boundaries)."""
    return [fn(r.X) for r in runs]


def build_detectors(train_runs, fast: bool):
    Xtr, _, _, rid = data.stack(train_runs)
    sc = Standardizer().fit(Xtr)
    sub = np.random.default_rng(0).choice(len(Xtr), 3000 if fast else 6000, replace=False)
    Ztr_sub = sc.transform(Xtr)[sub]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    ref = train_runs[:2]

    dets = {}

    # --- 1. univariate baselines -------------------------------------------------
    dets["zscore (global)"] = lambda X: stats.zscore(X, mu, sd).max(1)
    dets["robust z (rolling)"] = lambda X: np.max(
        [stats.rolling_robust_zscore(X[:, j], 101) for j in range(X.shape[1])], axis=0)
    dets["stuck detector"] = lambda X: np.max(
        [stats.stuck_score(X[:, j], 20) + np.log10(sd[j]) for j in range(X.shape[1])], axis=0)

    # --- 2. multivariate statistics ----------------------------------------------
    mah = MahalanobisDetector().fit(sc.transform(Xtr))
    dets["mahalanobis"] = lambda X: mah.score(sc.transform(X))
    pca = PCADetector(n_components=0.95).fit(Xtr)
    dets["pca T2"] = pca.t2
    dets["pca SPE"] = pca.spe

    # --- 3. machine learning, point-wise ------------------------------------------
    iso = IsolationForest(n_trees=50 if fast else 100, seed=0).fit(Ztr_sub)
    dets["isolation forest"] = lambda X: iso.score(sc.transform(X))
    knn = KNNDetector(k=10).fit(Ztr_sub)
    dets["kNN distance"] = lambda X: knn.score(sc.transform(X))
    lof = LOF(k=20).fit(Ztr_sub)
    dets["LOF"] = lambda X: lof.score(sc.transform(X))

    # --- 4. residual (virtual sensor) + CUSUM on the residual ----------------------
    res = ResidualDetector(lags=5, alpha=1.0).fit(Xtr, rid)
    dets["residual (ridge)"] = res.score
    # Same Isolation Forest as above, but fed residuals instead of raw values:
    # the algorithm is unchanged, only the representation improved.
    Rtr = res.residuals(Xtr)[sub]
    iso_r = IsolationForest(n_trees=100, seed=0).fit(Rtr)
    dets["iforest on residuals"] = lambda X: iso_r.score(res.residuals(X))

    def residual_cusum(X):
        R = res.residuals(X)
        R = R - np.median(R, axis=0)          # remove per-run bias (unobserved ambient)
        return np.max([np.maximum(*stats.cusum(R[:, j], 0, 1, k=0.5)) for j in range(R.shape[1])], axis=0)
    dets["residual + CUSUM"] = residual_cusum

    # --- 5. deep ------------------------------------------------------------------
    ae = Autoencoder(hidden=32, bottleneck=3, epochs=15 if fast else 40, lr=2e-3).fit(Xtr)
    dets["autoencoder"] = ae.score

    # --- 6. shape (matrix profile AB-join against clean reference runs) ------------
    m = 40
    refs = [np.concatenate([r.X[:, j] for r in ref]) for j in range(Xtr.shape[1])]
    dets["matrix profile"] = lambda X: np.max(
        [discord_score(X[:, j], m, reference=refs[j]) for j in range(X.shape[1])], axis=0)
    return dets


def rank_ensemble(score_lists_by_det, calib_by_det, members):
    """
    Combine detectors that fail on *different* anomaly types.
    Each member score is mapped to its empirical CDF on clean calibration data, i.e. to
    'how unusual is this versus normal', which puts all detectors on one scale.
    Ensemble score = max over members of -log10(1 - F(s)).
    """
    def to_tail(score, calib):
        c = np.sort(calib)
        F = np.searchsorted(c, score, side="right") / (len(c) + 1)
        return -np.log10(1 - F)
    n_runs = len(score_lists_by_det[members[0]])
    return [np.max([to_tail(score_lists_by_det[d][i], calib_by_det[d]) for d in members], axis=0)
            for i in range(n_runs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-debounce", action="store_true", help="alarm on single samples (keeps spikes)")
    ap.add_argument("--alpha", type=float, default=0.001, help="false-alarm budget per sample")
    args = ap.parse_args()

    L = 1500 if args.fast else 2000
    train, test = data.make_dataset(n_train=8 if args.fast else 16, n_test=10 if args.fast else 20,
                                    length=L, anomalies_per_run=2, seed=7)
    calib_runs = [data.simulate_run(L, seed=10_000 + i) for i in range(4 if args.fast else 6)]

    print(f"train {len(train)} clean runs | calibrate {len(calib_runs)} clean runs | "
          f"test {len(test)} runs, {sum(len(r.events) for r in test)} injected events\n")

    dets = build_detectors(train, args.fast)
    test_scores, calib_scores, alarms_all = {}, {}, {}
    rows = []
    for name, fn in dets.items():
        t0 = time.time()
        calib = np.concatenate(per_run(calib_runs, fn))
        scores = per_run(test, fn)
        calib_scores[name], test_scores[name] = calib, scores
        rows.append((name, time.time() - t0))
    ens_members = ["residual + CUSUM", "pca SPE", "stuck detector", "zscore (global)"]
    test_scores["ENSEMBLE"] = rank_ensemble(test_scores, calib_scores, ens_members)
    calib_scores["ENSEMBLE"] = np.concatenate(rank_ensemble(
        {d: per_run(calib_runs, dets[d]) for d in ens_members}, calib_scores, ens_members))
    rows.append(("ENSEMBLE", 0.0))

    y = np.concatenate([r.y for r in test])
    header = f"{'detector':22s} {'evt recall':>10s} {'FA/10k':>8s} {'delay':>6s} {'pt F1':>6s} {'PA F1':>6s} {'sec':>5s}"
    print(header)
    print("-" * len(header))
    for name, secs in rows:
        thr = quantile_threshold(calib_scores[name], args.alpha)
        alarms = [(s > thr) if args.no_debounce else persistence(s > thr, k=2, n=3)
                  for s in test_scores[name]]
        alarms_all[name] = alarms
        a = np.concatenate(alarms)
        ev = event_metrics(y, a)
        f1 = point_metrics(y, a)["f1"]
        pa = point_metrics(y, point_adjust(y, a))["f1"]
        # delay is undefined when a detector fired on nothing; print a dash
        # rather than a bare nan, which reads as a crash instead of a result
        delay = "-".rjust(6) if np.isnan(ev["mean_delay"]) else f"{ev['mean_delay']:6.1f}"
        print(f"{name:22s} {ev['event_recall']:10.2f} {ev['fa_per_10k']:8.2f} {delay} "
              f"{f1:6.2f} {pa:6.2f} {secs:5.1f}")

    print("\nPer-anomaly-type event recall  (which detector sees what):\n")
    kinds = list(data.ANOMALY_KINDS)
    print(f"{'detector':22s} " + " ".join(f"{k[:11]:>11s}" for k in kinds))
    for name, _ in rows:
        pk = per_kind_recall(test, alarms_all[name])
        print(f"{name:22s} " + " ".join(f"{pk.get(k, float('nan')):11.2f}" for k in kinds))

    print("\nRead it like this:\n"
          "  * No single detector wins every column - that is the case for an ensemble.\n"
          "  * 'PA F1' (point-adjusted) is always >= 'pt F1'; do not trust papers that report only PA.\n"
          "  * FA/10k is what the operator feels. A detector with 100% recall and 50 FA/10k gets muted.\n"
          "  * Spikes are 1 sample long; the 2-of-3 debounce suppresses them on purpose. Try --no-debounce.")


if __name__ == "__main__":
    main()
