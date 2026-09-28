# Anomaly Detection Lab

Unsupervised anomaly detection for multivariate sensor time series, taught from first
principles. Every algorithm is implemented from scratch in NumPy, derived in the notes, checked
against a reference implementation, and benchmarked on the same data with the same metrics.

No real data needed: a physics-style simulator generates correlated engine-like sensors and
injects six kinds of fault whose labels are used **only** for scoring.

## Lessons

| # | Lesson | Code |
|---|---|---|
| 1 | [The problem, and how a top ML engineer thinks](docs/01-the-problem.md) | none |
| 2 | [Time-series foundations](docs/02-time-series-foundations.md) | `data.py` |
| 3 | [Statistical baselines: z, MAD, EWMA, CUSUM](docs/03-statistical-detectors.md) | `stats.py` |
| 4 | [Multivariate: Mahalanobis, PCA, T², SPE](docs/04-multivariate-pca.md) | `pca.py` |
| 5 | [Isolation Forest](docs/05-isolation-forest.md) | `iforest.py` |
| 6 | [kNN, LOF, One-Class SVM](docs/06-distance-density-ocsvm.md) | `neighbors.py` |
| 7 | [Residual models (virtual sensors)](docs/07-residual-models.md) | `residual.py` |
| 8 | [Matrix Profile and change points](docs/08-matrix-profile-changepoint.md) | `matrix_profile.py`, `changepoint.py` |
| 9 | [Autoencoders](docs/09-autoencoders.md) | `autoencoder.py` |
| 10 | [Thresholds and evaluation](docs/10-thresholds-and-evaluation.md) | `threshold.py`, `evaluate.py` |
| 11 | [Ensembles and diagnosis](docs/11-ensembles-and-diagnosis.md) | `examples/benchmark.py` |
| 12 | [Production, drift, forecasting](docs/12-production-and-forecasting.md) | none |

## Headline result

`python examples/benchmark.py --fast` (10 test runs, 20 injected events, threshold at a 0.1%
false-alarm budget on separate clean runs, 2-of-3 debounce):

| Detector | Event recall | False alarms / 10k samples |
|---|---|---|
| z-score (global) | 15% | 31.7 |
| Isolation Forest (raw sensors) | 10% | 4.0 |
| PCA SPE | 35% | 0.8 |
| Autoencoder | 35% | 0.8 |
| **Residual model (ridge + lags)** | **75%** | **0.8** |
| Isolation Forest *on residuals* | 75% | 3.2 |
| Ensemble (residual+CUSUM, SPE, stuck, z) | 80% | 4.0 |

The same Isolation Forest goes from 10% to 75% when fed residuals instead of raw values:
**representation beats algorithm**.

## Quick start

```bash
pip install -e ".[dev]"          # numpy, scipy (+ pytest, scikit-learn, matplotlib for dev)
python -m pytest -q              # 19 tests: algorithms vs sklearn, and the lessons vs the benchmark
python -m pytest -q -m "not slow"   # just the fast ones (skips the benchmark re-run)
python examples/benchmark.py --fast
python examples/benchmark.py --fast --no-debounce
```

## Layout

```
src/adlab/        from-scratch implementations (one algorithm family per file)
docs/             the 12 lessons, with derivations
examples/         benchmark.py — every detector, same harness
tests/            correctness tests against reference implementations
```
