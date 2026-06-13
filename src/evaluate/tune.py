"""Tune hyperparameters on a VALIDATION tournament, report on a separate TEST one.

Plan §8: pick `w` (blend) and `xi` (time decay) where validation RPS is lowest,
then quote final numbers on a tournament never used for tuning. This avoids
optimising on the same data we report.
"""
from __future__ import annotations

import itertools

from src.db.models import get_connection
from src.evaluate.metrics import actual_outcome, ranked_probability_score
from src.model.predictor import build_models, predict
from src.ratings.dixon_coles import DEFAULT_XI


def _matches(conn, start, end, competition):
    sql = """SELECT date, home_id, away_id, home_goals, away_goals, neutral
             FROM matches WHERE home_goals IS NOT NULL AND date>=? AND date<=?"""
    p = [start, end]
    if competition:
        sql += " AND competition=?"; p.append(competition)
    return conn.execute(sql + " ORDER BY date ASC", p).fetchall()


def mean_rps(conn, start, end, competition, w, xi):
    rows = _matches(conn, start, end, competition)
    bundle = None; cur = None; total = 0.0
    for m in rows:
        if m["date"] != cur:
            cur = m["date"]
            bundle = build_models(conn, as_of=cur, blend_w=w, xi=xi)
        pred = predict(bundle, m["home_id"], m["away_id"],
                       neutral=bool(m["neutral"]), conn=conn)
        actual = actual_outcome(m["home_goals"], m["away_goals"])
        total += ranked_probability_score((pred.p_home, pred.p_draw, pred.p_away), actual)
    return total / len(rows) if rows else float("nan")


def tune(
    val=("2024-06-14", "2024-07-14", "UEFA Euro"),
    ws=(0.4, 0.5, 0.6, 0.7, 0.8),
    xis=(DEFAULT_XI, DEFAULT_XI * 2),
):
    conn = get_connection()
    print(f"Tuning on validation tournament: {val[2]} {val[0]}→{val[1]}\n")
    results = []
    for w, xi in itertools.product(ws, xis):
        r = mean_rps(conn, val[0], val[1], val[2], w, xi)
        results.append((r, w, xi))
        print(f"  w={w:.2f} xi={xi:.4f}  ->  RPS={r:.4f}")
    conn.close()
    results.sort()
    best = results[0]
    print(f"\n  Best: w={best[1]:.2f}, xi={best[2]:.4f}  (val RPS={best[0]:.4f})")
    return best


if __name__ == "__main__":
    tune()
