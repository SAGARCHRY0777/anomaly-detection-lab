# Lesson 11 — Ensembles and diagnosis

Code: `rank_ensemble` in `examples/benchmark.py`, `per_kind_recall` in `src/adlab/evaluate.py`

## 11.1 Why ensemble: the per-type table

Per-fault-type event recall (benchmark, `--fast`, 2-of-3 debounce):

| Detector | spike | level shift | drift | stuck | noise burst | corr. break |
|---|---|---|---|---|---|---|
| zscore (global) | 0 | .25 | .33 | 0 | 0 | .25 |
| stuck detector | 0 | 0 | 0 | **1.00** | .33 | 0 |
| PCA SPE | 0 | .25 | .67 | .33 | **.67** | .25 |
| residual (ridge) | **1.00** | .75 | **1.00** | .33 | **.67** | **.75** |
| residual + CUSUM | .33 | **1.00** | **1.00** | .67 | 0 | **.75** |
| matrix profile | 0 | 0 | 0 | **1.00** | 0 | 0 |
| **ENSEMBLE** | .33 | **1.00** | **1.00** | **1.00** | **.67** | **.75** |

No detector wins every column; each column has a specialist. Lesson 2's taxonomy predicted
this: point → statistics, contextual → residual/multivariate, collective → windowed tests.

## 11.2 Putting scores on one scale

Raw scores are incomparable (χ² values, path-length scores, log-std …). Map each through its
**empirical CDF on clean calibration data**, i.e. to a p-value:

$$p_m(x) = 1 - \hat F_m\big(s_m(x)\big) \qquad \text{tail score } = -\log_{10} p_m(x)$$

Tail score 3 means "only 1 in 1,000 clean samples scored this high" — for *any* detector.
z-normalising scores instead fails because score distributions are skewed and heavy-tailed.

## 11.3 Combination rules

| Rule | Formula | Behaviour |
|---|---|---|
| **max** (used here) | $\max_m -\log_{10}p_m$ | fires if *any* specialist is confident — right for complementary detectors |
| Bonferroni on max | alarm if $\min_m p_m < \alpha/K$ | keeps the family-wise false-alarm rate at $\alpha$ |
| Fisher | $-2\sum_m \ln p_m \sim \chi^2_{2K}$ | pools weak evidence; assumes independence (detectors on the same data are not) |
| mean / vote | average tail scores, or $k$-of-$K$ | robust, but dilutes specialists |

**The false-alarm price.** OR-ing $K$ detectors adds their false alarms: ENSEMBLE had
4.0 FA/10k versus 0.8 for its best member. Fix by calibrating the *combined* score on clean data
(what the code does) and giving each member a stricter α.

## 11.4 Choosing members

Not the individually best four — the four that **cover the columns**: a set-cover problem over
fault types. Here: residual + CUSUM (contextual, persistent), PCA SPE (relationships, noise),
stuck detector (collective), z-score (gross point faults). Stacking (learning combination weights)
needs labels; with only synthetic labels it overfits the injector — prefer simple rules.

## 11.5 Diagnosis: from "anomaly" to "what and where"

An alarm without an explanation gets ignored. Every alarm should carry:

- **which detector(s)** fired → *type* of fault
- **which sensor** — argmax residual (lesson 7) or SPE contributions (lesson 4)
- **magnitude** in σ, onset time, duration

Fault-signature table (the start of a rule-based diagnoser):

| Signature | Likely meaning |
|---|---|
| stuck detector on sensor *j* | dead sensor / frozen channel *j* |
| residual CUSUM rising slowly on *j*, others quiet | drift / fouling / calibration creep on *j* |
| large residual on *j* only, sudden | level shift, partial blockage, or recalibration on *j* |
| SPE high, residual on EGT only | EGT decoupled from speed — thermal fault or EGT sensor |
| all residuals moderately up together | faulty *input* sensor (e.g. speed) or unseen operating condition |
| T² high, SPE low | consistent but extreme operation — out of envelope, possibly healthy |

Over time, operator-confirmed alarms turn this table into labelled data for a supervised classifier.

## Check yourself

1. Why is z-normalising raw scores a poor way to combine detectors?
2. Two detectors with 0.8 FA/10k each, OR-ed, no recalibration. Roughly what FA rate?
3. Which two detectors would you pick if limited to two, and why?
