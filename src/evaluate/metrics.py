"""Forecast scoring metrics (plan §8). Outcomes are ordered [home, draw, away]."""
from __future__ import annotations

import numpy as np

OUTCOMES = ("home", "draw", "away")


def _onehot(actual: str) -> np.ndarray:
    return np.array([1.0 if o == actual else 0.0 for o in OUTCOMES])


def ranked_probability_score(probs: tuple[float, float, float], actual: str) -> float:
    """RPS — the standard 1X2 metric. Lower is better. Rewards being 'close'
    on an ordered outcome (home>draw>away)."""
    p = np.asarray(probs, float)
    o = _onehot(actual)
    cum_p = np.cumsum(p)
    cum_o = np.cumsum(o)
    # average of squared cumulative differences over the first r-1 categories
    return float(np.sum((cum_p[:-1] - cum_o[:-1]) ** 2) / (len(OUTCOMES) - 1))


def log_loss(probs: tuple[float, float, float], actual: str) -> float:
    p = dict(zip(OUTCOMES, probs))[actual]
    return float(-np.log(max(p, 1e-12)))


def brier_score(probs: tuple[float, float, float], actual: str) -> float:
    p = np.asarray(probs, float)
    return float(np.sum((p - _onehot(actual)) ** 2))


def calibration_bins(
    pred_probs: list[float], hits: list[int], n_bins: int = 10
) -> list[dict]:
    """For a reliability diagram: bucket predicted probs, compare to observed rate.
    `pred_probs` = predicted prob of an event; `hits` = 1 if it happened."""
    p = np.asarray(pred_probs); h = np.asarray(hits)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        out.append({
            "bin_mid": (edges[i] + edges[i + 1]) / 2,
            "predicted": float(p[m].mean()),
            "observed": float(h[m].mean()),
            "n": int(m.sum()),
        })
    return out


def actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals == away_goals:
        return "draw"
    return "away"
