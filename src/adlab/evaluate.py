"""
Scoring a detector when (synthetic) labels exist.

Point-wise precision/recall treat a 200-sample fault as 200 separate things to find.
Operators do not: they care whether the *event* was caught, how fast, and how many
times the alarm went off for nothing. So this module reports event-level metrics,
and also computes point-adjusted F1 only to show how badly it flatters a detector.

Maths in docs/10-thresholds-and-evaluation.md.
"""

from __future__ import annotations

import numpy as np


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as [start, end) pairs."""
    m = np.concatenate([[0], np.asarray(mask, int), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def point_metrics(y, alarm):
    y, a = np.asarray(y, bool), np.asarray(alarm, bool)
    tp, fp, fn = np.sum(y & a), np.sum(~y & a), np.sum(y & ~a)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def point_adjust(y, alarm):
    """The (flawed) protocol: if any point of an event is flagged, mark the whole event detected."""
    a = np.asarray(alarm, bool).copy()
    for s, e in segments(y):
        if a[s:e].any():
            a[s:e] = True
    return a


def event_metrics(y, alarm, tolerance: int = 20):
    """
    event_recall     : fraction of true events with >= 1 alarm inside [start, end + tolerance)
    false_alarms     : alarm *episodes* (contiguous runs) that overlap no event (+tolerance)
    fa_per_10k       : false alarm episodes per 10,000 samples of normal time
    mean_delay       : samples from event start to first alarm, over detected events
    """
    y, a = np.asarray(y, bool), np.asarray(alarm, bool)
    events = segments(y)
    near = y.copy()
    for s, e in events:
        near[s:min(len(y), e + tolerance)] = True
    detected, delays = 0, []
    for s, e in events:
        hits = np.flatnonzero(a[s:min(len(y), e + tolerance)])
        if hits.size:
            detected += 1
            delays.append(int(hits[0]))
    fa = sum(1 for s, e in segments(a) if not near[s:e].any())
    normal_len = max(1, int((~near).sum()))
    return {
        "events": len(events),
        "event_recall": detected / len(events) if events else float("nan"),
        "false_alarms": fa,
        "fa_per_10k": 1e4 * fa / normal_len,
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
    }


def per_kind_recall(runs, alarms, tolerance: int = 20):
    """Which anomaly *types* does a detector catch? The table that tells you what to ensemble."""
    hit, tot = {}, {}
    for run, a in zip(runs, alarms):
        for ev in run.events:
            tot[ev.kind] = tot.get(ev.kind, 0) + 1
            if a[ev.start:min(len(a), ev.end + tolerance)].any():
                hit[ev.kind] = hit.get(ev.kind, 0) + 1
    return {k: hit.get(k, 0) / tot[k] for k in sorted(tot)}
