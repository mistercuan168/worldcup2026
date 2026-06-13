"""Recent-form layer (plan §6c).

Time decay already weights recent matches in the DC fit, but an explicit short-term
signal is useful both as a small nudge to expected goals and as display material
for the 'form breakdown' output. We summarise the last N matches: a weighted
results record (recent = heaviest) and a goal trend.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src.db.models import get_connection


@dataclass
class Form:
    record: str            # e.g. "4W-1D-1L"
    points_rate: float     # weighted points / 3, in [0,1]
    goals_for: float       # avg goals scored (recent window)
    goals_against: float   # avg goals conceded
    results: list[str]     # ['W','D','L',...] most-recent-first
    n: int

    def summary(self) -> str:
        return self.record


def get_form(
    team_id: int, n: int = 6, as_of: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> Form:
    own = conn is None
    if own:
        conn = get_connection()

    sql = """SELECT date, home_id, away_id, home_goals, away_goals
             FROM matches
             WHERE home_goals IS NOT NULL AND (home_id = ? OR away_id = ?)"""
    params = [team_id, team_id]
    if as_of:
        sql += " AND date < ?"
        params.append(as_of)
    sql += " ORDER BY date DESC LIMIT ?"
    params.append(n)
    rows = conn.execute(sql, params).fetchall()
    if own:
        conn.close()

    if not rows:
        return Form("no recent matches", 0.5, 0.0, 0.0, [], 0)

    results, gf, ga = [], [], []
    w_pts = 0.0
    w_sum = 0.0
    for i, r in enumerate(rows):  # i=0 is most recent
        is_home = r["home_id"] == team_id
        scored = r["home_goals"] if is_home else r["away_goals"]
        conceded = r["away_goals"] if is_home else r["home_goals"]
        gf.append(scored); ga.append(conceded)
        if scored > conceded:
            res, pts = "W", 3
        elif scored == conceded:
            res, pts = "D", 1
        else:
            res, pts = "L", 0
        results.append(res)
        weight = 0.85 ** i  # recent games weigh more
        w_pts += weight * pts
        w_sum += weight

    wins = results.count("W"); draws = results.count("D"); losses = results.count("L")
    record = f"{wins}W-{draws}D-{losses}L"
    points_rate = (w_pts / w_sum) / 3.0 if w_sum else 0.5
    return Form(
        record=record,
        points_rate=points_rate,
        goals_for=sum(gf) / len(gf),
        goals_against=sum(ga) / len(ga),
        results=results,
        n=len(rows),
    )
