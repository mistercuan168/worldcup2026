"""Knockout resolution (plan §6e). Knockouts can't end drawn.

We take the 90-minute draw probability and redistribute it into extra-time +
penalties. The shootout is treated as a near coin-flip with a small tilt toward
the Elo-stronger side (kept modest on purpose).
"""
from __future__ import annotations


def shootout_edge(elo_home: float, elo_away: float) -> float:
    """Home win-share of a drawn game once it goes to ET/pens. ~0.5, gentle tilt."""
    diff = elo_home - elo_away
    # 200 Elo of gap -> ~0.55; capped so penalties stay close to a coin flip.
    tilt = max(-0.12, min(0.12, diff / 4000.0))
    return 0.5 + tilt


def resolve_knockout(
    p_home: float, p_draw: float, p_away: float,
    elo_home: float, elo_away: float,
) -> tuple[float, float]:
    """Return (p_advance_home, p_advance_away); they sum to 1."""
    edge = shootout_edge(elo_home, elo_away)
    adv_home = p_home + p_draw * edge
    adv_away = p_away + p_draw * (1.0 - edge)
    total = adv_home + adv_away
    return adv_home / total, adv_away / total
