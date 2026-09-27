# Lesson 9 — Autoencoders: nonlinear PCA

Code: `src/adlab/autoencoder.py` (pure NumPy, hand-written backprop and Adam)

## 9.1 Architecture

$$\underbrace{x \in \mathbb R^d \xrightarrow{W_1,\ \tanh} h \in \mathbb R^{32} \xrightarrow{W_2} z \in \mathbb R^{3}}_{\text{encoder}} \;\underbrace{\xrightarrow{W_3,\ \tanh} g \in \mathbb R^{32} \xrightarrow{W_4} \hat x \in \mathbb R^d}_{\text{decoder}}$$

$$L = \frac{1}{nd}\sum_i \lVert x_i - \hat x_i\rVert^2 \qquad \text{anomaly score } s(x) = \lVert x - \hat x\rVert^2$$

The **bottleneck** ($3 < d = 6$) forces the network to compress. It can only learn to rebuild
what it has seen often: the manifold of normal data. Points off that manifold rebuild badly.

## 9.2 Backpropagation, line by line

Chain rule from the loss back to each weight (batch of $n$; $\odot$ = elementwise):

| Step | Formula | Code |
|---|---|---|
| output error | $\delta_{out} = \frac{2}{nd}(\hat x - x)$ | `d_out` |
| layer 4 | $\nabla W_4 = g^\top\delta_{out}$, $\nabla b_4 = \sum\delta_{out}$ | `grads["W4"]` |
| through tanh | $\delta_g = (\delta_{out}W_4^\top)\odot(1-g^2)$ | `d_g` — since $\tanh'(a) = 1-\tanh^2(a)$ |
| layer 3 | $\nabla W_3 = z^\top\delta_g$ | |
| bottleneck (linear) | $\delta_z = \delta_g W_3^\top$ | `d_z` |
| layer 2 | $\nabla W_2 = h^\top\delta_z$ | |
| through tanh | $\delta_h = (\delta_z W_2^\top)\odot(1-h^2)$ | `d_h` |
| layer 1 | $\nabla W_1 = x^\top\delta_h$ | |

Pattern: gradient of a weight = (layer input)ᵀ × (error at layer output); error moves backwards
by multiplying with $W^\top$ and the activation's derivative.

## 9.3 Adam

$$m \leftarrow \beta_1 m + (1-\beta_1)\nabla,\quad v \leftarrow \beta_2 v + (1-\beta_2)\nabla^2,\quad \hat m = \frac{m}{1-\beta_1^t},\ \hat v = \frac{v}{1-\beta_2^t},\quad \theta \leftarrow \theta - \eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

$m$ = momentum (EWMA of gradients — lesson 3's EWMA again), $v$ = EWMA of squared gradients
(per-parameter step size). Bias correction fixes the zero initialisation early on.
Glorot init $\mathcal N(0, 2/(\text{fan}_{in}+\text{fan}_{out}))$ keeps activations in tanh's
non-saturated range.

## 9.4 The key theorem: a linear autoencoder *is* PCA

Baldi & Hornik (1989): with linear activations and MSE, a bottleneck-$k$ autoencoder's global
optimum spans exactly the top-$k$ principal subspace. Its reconstruction error equals PCA's SPE.

So a nonlinear autoencoder = **nonlinear PCA**, and its reconstruction error = **nonlinear SPE**.
It fixes PCA's straight-line problem (EGT ∝ speed² is a curve).

Demo (6 clean training runs, correlation break on EGT):

| | clean mean score | fault mean score | ratio |
|---|---|---|---|
| PCA SPE (k=3) | 0.030 | 0.095 | 3.2× |
| Autoencoder (k=3) | 0.011 | 0.140 | 12.7× |

## 9.5 Failure modes

- **Too much capacity → identity map.** A wide bottleneck or huge network learns to copy anything,
  including faults; the score collapses. Keep the bottleneck narrow; validate that *injected*
  anomalies still reconstruct badly.
- **Contaminated training data** → faults learned as normal (worse than for PCA; nets memorise).
- **Static input** — like PCA, a point-wise AE has no dynamics. Healthy transients look odd.
- Many knobs (width, depth, lr, epochs), results vary with the seed.

## 9.6 Variants

| Variant | Change | Why |
|---|---|---|
| Denoising AE | train on $x+\text{noise}$ → reconstruct $x$ | robust manifold, less identity-copying |
| **LSTM / conv AE on windows** | input = window of $w$ steps | adds dynamics — fixes the static problem |
| VAE | encoder outputs $q(z\mid x)$; maximise ELBO $=\mathbb E_q[\log p(x\mid z)] - \text{KL}(q(z\mid x)\Vert p(z))$ | probabilistic score (reconstruction probability) |
| USAD, Anomaly Transformer, TranAD | adversarial / attention | research SOTA — check with a fair harness |

## 9.7 Benchmark

These rows come from a seeded run of `examples/benchmark.py`; rerun it to reproduce them.

| Detector | Event recall | FA / 10k |
|---|---|---|
| PCA SPE | 0.28 | 0.00 |
| Autoencoder | 0.50 | 1.71 |
| Residual (ridge, lags) | 0.82 | 1.14 |

The nonlinearity did buy something — the autoencoder nearly doubles PCA SPE's recall, which is
what you would hope for from a nonlinear PCA on a curved manifold. It still loses heavily to a
two-line ridge regression, because it had the **same information**: one instant, no lags. Model
capacity can exploit a curve the linear model missed; it cannot recover information the input
never contained. Give the AE windows and it can compete; that is the lesson, not "AEs are bad".

## Check yourself

1. Remove the tanh activations. What does the autoencoder converge to?
2. Bottleneck = 6 = input dimension. What happens to the anomaly score?
3. Why does the benchmark AE lose to ridge regression?
