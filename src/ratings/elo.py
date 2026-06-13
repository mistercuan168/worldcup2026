"""World Football Elo — rolling team-strength ratings over all of history.

Standard Elo tuned for national teams (plan §6a):
  - K (match weight) scales with how important the game is.
  - G (margin-of-victory multiplier) makes bigger wins move ratings more.
  - HFA (home-field advantage) applies only to non-neutral games.

Elo gives a clean win/draw/loss baseline and — crucially — a strong prior for
teams with very few recent matches (the cold-start problem in §2).
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

from src.db.models import get_connection

BASE_RATING = 1500.0
HFA = 100.0  # Elo points of home advantage; 0 on neutral ground.


def k_factor(competition: str) -> float:
    """Bigger games move ratings more (plan §6a)."""
    c = (competition or "").lower()
    if "world cup" in c and "qualif" not in c:
        return 60.0
    if any(x in c for x in ("euro", "copa", "nations", "gold cup", "asian cup",
                            "african cup", "confederations")) and "qualif" not in c:
        return 40.0
    if "qualif" in c:
        return 30.0
    return 20.0  # friendlies and everything else


def mov_multiplier(goal_diff: int) -> float:
    """World Football Elo margin-of-victory multiplier."""
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def expected_home(elo_home: float, elo_away: float, neutral: bool) -> float:
    """Elo expected score for the home team (1=win-equivalent, .5=draw share)."""
    hfa = 0.0 if neutral else HFA
    return 1.0 / (1.0 + 10 ** ((elo_away - elo_home - hfa) / 400.0))


@dataclass
class EloModel:
    ratings: dict[int, float]

    def get(self, team_id: int) -> float:
        return self.ratings.get(team_id, BASE_RATING)


def build_elo(
    conn: sqlite3.Connection | None = None,
    snapshot: bool = True,
    as_of: str | None = None,
) -> EloModel:
    """Process every played match in date order, updating ratings.

    `as_of` (ISO date) restricts to matches strictly BEFORE that date — essential
    for a leak-free backtest. If snapshot=True, writes each post-match rating into
    the elo_ratings table so a team's strength can be looked up as of any date.
    """
    own = conn is None
    if own:
        conn = get_connection()

    ratings: dict[int, float] = {}
    snapshots: list[tuple[int, str, float]] = []

    sql = """SELECT date, competition, home_id, away_id, home_goals, away_goals, neutral
             FROM matches
             WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL"""
    params: list = []
    if as_of:
        sql += " AND date < ?"
        params.append(as_of)
    sql += " ORDER BY date ASC, match_id ASC"
    rows = conn.execute(sql, params).fetchall()

    for r in rows:
        h, a = r["home_id"], r["away_id"]
        eh = ratings.get(h, BASE_RATING)
        ea = ratings.get(a, BASE_RATING)
        neutral = bool(r["neutral"])

        exp_h = expected_home(eh, ea, neutral)
        hg, ag = r["home_goals"], r["away_goals"]
        result_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        k = k_factor(r["competition"])
        g = mov_multiplier(hg - ag)
        delta = k * g * (result_h - exp_h)

        ratings[h] = eh + delta
        ratings[a] = ea - delta  # zero-sum

        if snapshot:
            snapshots.append((h, r["date"], ratings[h]))
            snapshots.append((a, r["date"], ratings[a]))

    if snapshot:
        conn.execute("DELETE FROM elo_ratings")
        conn.executemany(
            "INSERT OR REPLACE INTO elo_ratings(team_id, as_of_date, rating) VALUES (?,?,?)",
            snapshots,
        )
        conn.commit()

    if own:
        conn.close()
    return EloModel(ratings)


# ---- Baseline 1X2 from Elo ------------------------------------------------
# Split the Elo expectation into win/draw/loss. The draw is modelled as a bell
# curve in the rating gap: draws peak for evenly-matched teams and fade as the gap
# grows. DRAW_BASE / DRAW_WIDTH are sensible defaults; they can be tuned on the
# backtest later (plan §8). Expected points are preserved exactly.
DRAW_BASE = 0.29
DRAW_WIDTH = 230.0


def elo_1x2(
    elo_home: float, elo_away: float, neutral: bool
) -> tuple[float, float, float]:
    """Return (p_home, p_draw, p_away) from Elo alone."""
    hfa = 0.0 if neutral else HFA
    diff = elo_home + hfa - elo_away
    p_home_share = 1.0 / (1.0 + 10 ** (-diff / 400.0))  # expected points share

    p_draw = DRAW_BASE * math.exp(-((diff / DRAW_WIDTH) ** 2))
    p_home = p_home_share - p_draw / 2.0
    p_away = (1.0 - p_home_share) - p_draw / 2.0

    # Guard the tails where the draw term could push a side negative.
    p_home = max(p_home, 0.0)
    p_away = max(p_away, 0.0)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


if __name__ == "__main__":
    conn = get_connection()
    model = build_elo(conn)
    id_to_name = {r["team_id"]: r["name"] for r in conn.execute("SELECT team_id, name FROM teams")}
    conn.close()

    top = sorted(model.ratings.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print("Top 20 national teams by current World Football Elo:\n")
    print(f"  {'#':>2}  {'team':<22} {'Elo':>6}")
    print("  " + "-" * 32)
    for i, (tid, rating) in enumerate(top, 1):
        print(f"  {i:>2}  {id_to_name.get(tid, '?'):<22} {rating:>6.0f}")
