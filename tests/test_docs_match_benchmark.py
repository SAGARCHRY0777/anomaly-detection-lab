"""The lessons quote the benchmark's results. This checks that they still match.

Every results table in docs/ was written by hand once and then drifted: seven
lessons quoted figures the code no longer produced, and two of them stated a
conclusion the data contradicted. Prose is the part of a repository with no
compiler, so it rots silently.

This re-runs the benchmark and asserts that every figure quoted in a lesson is
one the run actually prints. It is slow -- it executes the benchmark twice, with
and without debounce, because the lessons quote both -- so it is marked ``slow``
and skippable with ``-m "not slow"``.

If it fails, the fix is almost never to edit an expected value here. It is to
regenerate the table from a run.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Labels the lessons use -> the name the benchmark prints.
ALIASES = {
    "z-score (global)": "zscore (global)",
    "zscore (global)": "zscore (global)",
    "robust z (rolling)": "robust z (rolling)",
    "stuck detector": "stuck detector",
    "mahalanobis": "mahalanobis",
    "pca t2": "pca T2",
    "pca t²": "pca T2",
    "pca spe": "pca SPE",
    "pca spe (best of lessons 4-6)": "pca SPE",
    "pca spe (best of lessons 4–6)": "pca SPE",
    "isolation forest": "isolation forest",
    "raw standardised sensors": "isolation forest",
    "residuals from lesson 7": "iforest on residuals",
    "knn distance": "kNN distance",
    "lof": "LOF",
    "residual (ridge)": "residual (ridge)",
    "residual (ridge, lags)": "residual (ridge)",
    "iforest on residuals": "iforest on residuals",
    "residual + cusum": "residual + CUSUM",
    "autoencoder": "autoencoder",
    "matrix profile": "matrix profile",
    "ensemble": "ENSEMBLE",
}

_MAIN_ROW = re.compile(
    r"^(.{22})\s*([\d.]+)\s+([\d.]+)\s+(-|[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$"
)


def _parse(stdout: str) -> dict[str, set[float]]:
    """Every number the run prints for each detector: main row and per-type row."""
    vals: dict[str, set[float]] = {}
    mode, kinds = None, []
    for line in stdout.splitlines():
        if line.startswith("detector") and "evt recall" in line:
            mode = "main"; continue
        if line.startswith("detector") and "spike" in line:
            kinds = line.split()[1:]; mode = "per"; continue
        if line.startswith("-") or not line.strip():
            continue
        if line.startswith("Read it"):
            mode = None; continue
        if mode == "main":
            m = _MAIN_ROW.match(line)
            if m:
                name = m.group(1).strip()
                got = {float(m.group(2)), float(m.group(3)),
                       float(m.group(5)), float(m.group(6))}
                if m.group(4) != "-":
                    got.add(float(m.group(4)))
                vals.setdefault(name, set()).update(got)
        elif mode == "per":
            m = re.match(r"^(.{22})\s*(.*)$", line)
            if m and m.group(2).strip():
                cells = m.group(2).split()
                if len(cells) == len(kinds):
                    vals.setdefault(m.group(1).strip(), set()).update(
                        float(c) for c in cells
                    )
    return vals


@pytest.fixture(scope="module")
def printed() -> dict[str, set[float]]:
    """Union of every figure both documented runs print, per detector."""
    merged: dict[str, set[float]] = {}
    for args in ([], ["--no-debounce"]):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "examples" / "benchmark.py"), *args],
            capture_output=True, text=True, timeout=1800, cwd=ROOT,
        )
        assert proc.returncode == 0, f"benchmark {args} failed:\n{proc.stderr[-2000:]}"
        for name, vals in _parse(proc.stdout).items():
            merged.setdefault(name, set()).update(vals)
    assert merged, "could not parse the benchmark output"
    return merged


def _doc_rows() -> list[tuple[str, str, list[float]]]:
    """Markdown rows in docs/ that name a detector and quote numbers."""
    rows = []
    for path in sorted(DOCS.glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|") or line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            key = ALIASES.get(re.sub(r"[*`]", "", cells[0]).strip().lower())
            if key is None:
                continue
            nums = [
                float(n)
                for n in re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w.])", " ".join(cells[1:]))
            ]
            if nums:
                rows.append((path.name, key, nums))
    return rows


@pytest.mark.slow
def test_lesson_tables_match_the_benchmark(printed):
    """Every figure a lesson quotes is one the benchmark actually prints."""
    rows = _doc_rows()
    assert len(rows) >= 15, f"expected the lessons' tables, found {len(rows)} rows"

    bad = []
    for fname, detector, nums in rows:
        assert detector in printed, f"{fname}: unknown detector {detector!r}"
        allowed = printed[detector]
        for n in nums:
            if not any(abs(n - v) < 0.005 for v in allowed):
                bad.append(f"{fname}: {detector} quotes {n}, which no run prints "
                           f"(it prints {sorted(allowed)})")

    assert not bad, (
        "Lesson tables disagree with examples/benchmark.py.\n"
        "Regenerate the table from a run rather than editing this test:\n  "
        + "\n  ".join(bad)
    )


@pytest.mark.slow
def test_isolation_forest_really_is_zero_on_raw_features(printed):
    """Lesson 5 tells the reader a row of zeros is the result, not a broken build.

    That claim is load-bearing -- it decides whether a reader trusts the lesson
    or files a bug -- so it is asserted rather than assumed.
    """
    assert 0.0 in printed["isolation forest"], "lesson 5's zero row no longer holds"
    assert any(v > 0.5 for v in printed["iforest on residuals"]), (
        "the representation-beats-algorithm contrast in lesson 5 no longer holds"
    )
