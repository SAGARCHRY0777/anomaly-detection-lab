"""
Synthetic multivariate sensor data with known, injected anomalies.

You have no data yet, so this module is the stand-in. It is deliberately
*physical* rather than random: every sensor is driven by one hidden control
input (throttle), so the channels are correlated the way real machine sensors
are. That matters, because the most interesting anomalies are the ones that
break a *relationship* while every individual channel still looks normal.

Only the simulator knows the labels. Detectors never see them — they are
used exclusively to *score* the detectors afterwards. That mirrors the real
setting: unsupervised in training, some ground truth (even synthetic) in
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SENSORS = ["speed", "fuel_flow", "egt", "pressure", "vibration", "oil_temp"]

# Regime codes, as a controller would report them.
IDLE, ACCEL, STEADY, DECEL = 0, 1, 2, 3
REGIME_NAMES = {IDLE: "idle", ACCEL: "accel", STEADY: "steady", DECEL: "decel"}

ANOMALY_KINDS = (
    "spike",             # one sample far out: sensor glitch, EMI
    "level_shift",       # sudden offset: calibration jump, partial blockage
    "drift",             # slowly growing offset: fouling, sensor ageing
    "stuck",             # value frozen: dead sensor, frozen DAQ channel
    "noise_burst",       # variance jumps: loose connector, bearing onset
    "correlation_break", # EGT inconsistent with speed but inside global range
)


@dataclass
class Event:
    kind: str
    sensor: str
    start: int
    end: int  # exclusive


@dataclass
class Run:
    X: np.ndarray                 # (T, d) sensor matrix
    y: np.ndarray                 # (T,) bool, True = anomalous sample (eval only!)
    regime: np.ndarray            # (T,) int regime code
    throttle: np.ndarray          # (T,) hidden control input, [0, 1]
    events: list[Event] = field(default_factory=list)
    sensors: list[str] = field(default_factory=lambda: list(SENSORS))


def _first_order_lag(target: np.ndarray, tau: float, x0: float) -> np.ndarray:
    """x[t] = x[t-1] + (target[t] - x[t-1]) / tau  — a discrete RC filter."""
    out = np.empty_like(target)
    x = x0
    for t, v in enumerate(target):
        x += (v - x) / tau
        out[t] = x
    return out


def _throttle_profile(rng: np.random.Generator, length: int):
    """Idle -> (accel -> hold -> maybe step) ... -> decel -> idle."""
    u = np.empty(length)
    regime = np.empty(length, dtype=int)
    t = 0
    level = 0.2

    def segment(n, target, reg_ramp, reg_hold, ramp):
        nonlocal t, level
        n = min(n, length - t)
        r = min(ramp, n)
        u[t:t + r] = np.linspace(level, target, r)
        regime[t:t + r] = reg_ramp
        u[t + r:t + n] = target
        regime[t + r:t + n] = reg_hold
        t += n
        level = target

    segment(int(length * 0.10), 0.2, IDLE, IDLE, 1)
    while t < int(length * 0.80):
        target = rng.uniform(0.55, 1.0)
        reg = ACCEL if target > level else DECEL
        segment(int(rng.integers(length // 10, length // 4)), target, reg, STEADY, int(rng.integers(20, 60)))
    segment(length - t, 0.2, DECEL, IDLE, int(rng.integers(30, 80)))
    return u, regime


def simulate_run(length: int = 2000, seed: int | None = None, ambient: float | None = None) -> Run:
    """One clean run. `ambient` (deg C) shifts temperatures run-to-run, like weather does."""
    rng = np.random.default_rng(seed)
    ambient = rng.uniform(5, 40) if ambient is None else ambient
    u, regime = _throttle_profile(rng, length)

    speed = _first_order_lag(100 * u, tau=6.0, x0=20.0)            # % rpm, lags throttle
    n = speed / 100.0
    fuel = 0.1 + 1.9 * u ** 1.3                                      # leads speed
    egt = 350 + 1.2 * ambient + 380 * n ** 2 + 40 * np.gradient(n) * 50  # overshoot on accel
    egt = _first_order_lag(egt, tau=4.0, x0=egt[0])
    pressure = 1.0 + 3.2 * n ** 2
    vibration = 0.4 + 1.8 * n + 0.15 * np.sin(np.arange(length) * 2 * np.pi * n / 7)
    oil = _first_order_lag(40 + 0.5 * ambient + 50 * n, tau=120.0, x0=40 + 0.5 * ambient + 10)

    X = np.column_stack([speed, fuel, egt, pressure, vibration, oil])
    noise_sd = np.array([0.3, 0.02, 3.0, 0.03, 0.05, 0.2])
    X = X + rng.normal(0, noise_sd, size=X.shape)
    return Run(X=X, y=np.zeros(length, bool), regime=regime, throttle=u)


def inject(run: Run, kind: str, sensor: str, start: int, duration: int,
           magnitude: float = 1.0, rng: np.random.Generator | None = None) -> Run:
    """Inject one anomaly in place. `magnitude` is in units of that sensor's noise-free spread."""
    rng = rng or np.random.default_rng()
    j = run.sensors.index(sensor)
    X = run.X
    end = min(start + duration, len(X))
    seg = slice(start, end)
    scale = X[:, j].std()

    if kind == "spike":
        end = start + 1
        seg = slice(start, end)
        X[seg, j] += rng.choice([-1, 1]) * 4 * magnitude * scale
    elif kind == "level_shift":
        X[seg, j] += 0.4 * magnitude * scale
    elif kind == "drift":
        X[seg, j] += np.linspace(0, 0.8 * magnitude * scale, end - start)
    elif kind == "stuck":
        X[seg, j] = X[start, j]
    elif kind == "noise_burst":
        X[seg, j] += rng.normal(0, 0.25 * magnitude * scale, end - start)
    elif kind == "correlation_break":
        # EGT behaves as if the engine were at a *different* speed. Each value
        # is individually plausible; only the relation to speed is wrong.
        if sensor != "egt":
            raise ValueError("correlation_break is defined for egt")
        n = X[seg, run.sensors.index("speed")] / 100
        X[seg, j] += 380 * ((n + 0.08 * magnitude) ** 2 - n ** 2)
    else:
        raise ValueError(f"unknown anomaly kind {kind!r}")

    run.y[seg] = True
    run.events.append(Event(kind, sensor, start, end))
    return run


def make_dataset(n_train: int = 20, n_test: int = 10, length: int = 2000,
                 anomalies_per_run: int = 2, seed: int = 0):
    """
    Returns (train_runs, test_runs).

    Train runs are clean — the unsupervised assumption "training data is
    (mostly) normal". Test runs carry injected events whose labels are only
    for scoring.
    """
    rng = np.random.default_rng(seed)
    train = [simulate_run(length, seed=int(rng.integers(1 << 31))) for _ in range(n_train)]
    test = []
    # Cycle through a shuffled list of kinds so every anomaly type is represented.
    kinds = [k for _ in range(n_test * anomalies_per_run // len(ANOMALY_KINDS) + 1) for k in ANOMALY_KINDS]
    kinds = list(rng.permutation(kinds))
    for _ in range(n_test):
        run = simulate_run(length, seed=int(rng.integers(1 << 31)))
        taken = np.zeros(length, bool)
        for _ in range(anomalies_per_run):
            kind = str(kinds.pop())
            sensor = "egt" if kind == "correlation_break" else str(rng.choice(SENSORS))
            duration = 1 if kind == "spike" else int(rng.integers(40, 200))
            for _attempt in range(20):
                start = int(rng.integers(int(0.1 * length), length - duration - 1))
                if not taken[max(0, start - 50):start + duration + 50].any():
                    break
            taken[start:start + duration] = True
            inject(run, kind, sensor, start, duration, magnitude=float(rng.uniform(0.8, 1.5)), rng=rng)
        test.append(run)
    return train, test


def stack(runs: list[Run]):
    """Concatenate runs -> (X, y, regime, run_id). Keep run_id: never window across runs."""
    X = np.vstack([r.X for r in runs])
    y = np.concatenate([r.y for r in runs])
    regime = np.concatenate([r.regime for r in runs])
    run_id = np.concatenate([np.full(len(r.X), i) for i, r in enumerate(runs)])
    return X, y, regime, run_id
