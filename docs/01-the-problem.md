# Lesson 1 — The problem, and how a top ML engineer thinks

## 1.1 What "anomaly" means mathematically

Every sensor reading at time *t* is a vector

$$x_t = [\text{speed}, \text{fuel}, \text{EGT}, \text{pressure}, \text{vibration}, \text{oil temp}] \in \mathbb{R}^d$$

Healthy operation produces vectors from some unknown distribution $p_{\text{normal}}(x)$.
An **anomaly is a point that is unlikely under that distribution**:

$$x \text{ is anomalous} \iff p_{\text{normal}}(x) \text{ is small}$$

Every algorithm in this course is a different estimate of "how unlikely", without ever writing $p$ down:

| Algorithm | Its proxy for "unlikely" |
|---|---|
| z-score | far from the mean, in standard deviations |
| Mahalanobis / PCA | far from the mean, *accounting for correlations* |
| kNN / LOF | few neighbours nearby — low density |
| Isolation Forest | easy to separate from the rest |
| Residual model | the sensor disagrees with what the other sensors predict |
| Autoencoder | cannot be compressed and rebuilt well |
| Matrix Profile | this *shape* has no twin anywhere |

Each outputs a **score** $s(x)$ (higher = more anomalous). Turning a score into an alarm is a
**separate** decision (lesson 10). Keep score and threshold apart.

## 1.2 What "unsupervised" really means

It is not "no assumptions". It is one big assumption:

> **The training data is (mostly) normal.**

Consequences:

1. **Garbage in the training set becomes "normal".** Curate training runs — only runs domain experts sign off as healthy.
2. **You detect "different from training", not "broken".** A new-but-healthy condition (cold day, new profile) *will* alarm. That is a coverage gap, not an algorithm bug.
3. **You still need *some* labels to evaluate.** Not for training — for measuring. With no real faults, create them by **synthetic injection** (`src/adlab/data.py` injects six fault types).

## 1.3 Base rates — the maths that surprises everyone

Recall 0.99, false-alarm rate α = 0.001, fault prevalence 1 in 10,000. When the alarm fires:

$$P(\text{fault}\mid\text{alarm}) = \frac{0.99 \times 0.0001}{0.99\times0.0001 + 0.001\times0.9999} \approx 9\%$$

**91% of alarms are false with a "99% accurate" detector.** With α = 0.0001 it becomes ≈ 50%.
Rare faults mean false alarms dominate; hence the obsession with false-alarm rate, debouncing,
and event-level metrics. Accuracy is meaningless here.

## 1.4 The top-engineer playbook

1. **Physics first** — which sensor drives which. This decides the method (why residual models win).
2. **Audit data before modelling** — sampling, gaps, units, stuck channels, noise. Plot everything.
3. **Define "normal" explicitly** — which runs, regimes, conditions.
4. **Build the evaluation harness before any model** — clean train runs, *separate* clean calibration runs (thresholds), test runs with injected faults. Split **by run**, never randomly by row.
5. **Dumbest baseline first**; every fancy model must beat it on the same harness.
6. **Climb the ladder** — statistics → multivariate → residual → ML → deep. Stop when a step stops helping.
7. **Look at which fault types each detector catches** — that designs the ensemble.
8. **Thresholds from a false-alarm budget on clean data** — never tuned on test faults.
9. **Ship with a feedback loop** — confirmed/rejected alarms become real labels.

> The harness is worth more than any model.

## Check yourself

1. Why can't you set the threshold by maximising F1 here? — *Needs labels; tuning on test faults is leakage.*
2. Healthy engine, first hot day, alarm. Algorithm wrong? — *No: coverage or a conditioning input (ambient) is missing.*
3. Base rate with α = 0.0001? — *≈ 50%.*
