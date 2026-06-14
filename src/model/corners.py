"""Expected corners — a TRANSPARENT ESTIMATE, not a trained model.

Important honesty note: the historical results dataset has only scores, not
corner counts, so corners cannot be modelled the way goals are (Elo / Dixon-Coles).
Instead we estimate them from each team's expected goals, using the well-known
real-world tendency that more attacking output produces more corners.

This is a heuristic, clearly labelled as such in the UI. The constants are set so
an even, average international match (~1.2 xG each) yields ~10 total corners, the
typical figure. If a corner dataset is added later, calibrate BASE / PER_XG here.
"""
from __future__ import annotations

BASE = 2.6       # corners a team tends to win even in a quiet match
PER_XG = 2.0     # extra corners per expected goal of attacking output


def expected_corners(xg_home: float, xg_away: float) -> tuple[float, float, float]:
    """Return (home_corners, away_corners, total) — all estimates."""
    home = BASE + PER_XG * xg_home
    away = BASE + PER_XG * xg_away
    return home, away, home + away
