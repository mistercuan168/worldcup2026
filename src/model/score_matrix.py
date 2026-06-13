"""Derive every market from one score matrix (plan §5/§6b).

P(home=x, away=y) lives in a single matrix; 1X2, top scores, BTTS and Over/Under
are all just sums over its cells. Deriving everything from one object guarantees
the outputs are mutually consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarketOutputs:
    p_home: float
    p_draw: float
    p_away: float
    exp_home_goals: float
    exp_away_goals: float
    top_scores: list[tuple[str, float]]  # [("2-1", 0.11), ...]
    p_btts: float
    p_over25: float


def outcome_probs(mat: np.ndarray) -> tuple[float, float, float]:
    p_home = float(np.tril(mat, -1).sum())  # x > y
    p_draw = float(np.trace(mat))           # x == y
    p_away = float(np.triu(mat, 1).sum())   # x < y
    return p_home, p_draw, p_away


def top_scores(mat: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    flat = [
        (f"{x}-{y}", float(mat[x, y]))
        for x in range(mat.shape[0])
        for y in range(mat.shape[1])
    ]
    flat.sort(key=lambda t: t[1], reverse=True)
    return flat[:k]


def expected_goals(mat: np.ndarray) -> tuple[float, float]:
    xs = np.arange(mat.shape[0])
    ys = np.arange(mat.shape[1])
    return float((mat.sum(axis=1) * xs).sum()), float((mat.sum(axis=0) * ys).sum())


def p_btts(mat: np.ndarray) -> float:
    return float(mat[1:, 1:].sum())  # both score >= 1


def p_over(mat: np.ndarray, line: float = 2.5) -> float:
    total = 0.0
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            if x + y > line:
                total += mat[x, y]
    return float(total)


def derive_all(mat: np.ndarray) -> MarketOutputs:
    ph, pd_, pa = outcome_probs(mat)
    eh, ea = expected_goals(mat)
    return MarketOutputs(
        p_home=ph, p_draw=pd_, p_away=pa,
        exp_home_goals=eh, exp_away_goals=ea,
        top_scores=top_scores(mat),
        p_btts=p_btts(mat),
        p_over25=p_over(mat, 2.5),
    )
