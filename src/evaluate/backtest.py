"""Backtest with a strict time-split — no data leakage (plan §8).

For each test match we fit the models on data STRICTLY BEFORE that match's date,
predict, then score. Fits are cached per date (a tournament has few distinct dates),
so a whole tournament backtests in seconds.

We compare three forecasters:
  - blend     : our Elo + Dixon-Coles model
  - elo_only  : Elo baseline (internal strong-ish baseline)
  - base_rate : historical home/draw/away frequencies (the weak baseline we MUST beat)

Bookmaker-odds comparison (the strong external benchmark) needs an odds dataset,
which we don't ship; hook it in here when available.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.db.models import get_connection
from src.evaluate.metrics import (actual_outcome, brier_score, log_loss,
                                   ranked_probability_score)
from src.model.predictor import build_models, predict
from src.ratings.elo import elo_1x2

# Long-run international base rates (home/draw/away) — the naive baseline.
BASE_RATE = (0.49, 0.24, 0.27)


@dataclass
class Scores:
    n: int
    rps: float
    logloss: float
    brier: float
    top_pick_acc: float

    def __str__(self):
        return (f"n={self.n:4d}  RPS={self.rps:.4f}  logloss={self.logloss:.4f}  "
                f"brier={self.brier:.4f}  top-pick={self.top_pick_acc:.1%}")


def _aggregate(rows: list[tuple[tuple[float, float, float], str]]) -> Scores:
    if not rows:
        return Scores(0, float("nan"), float("nan"), float("nan"), float("nan"))
    rps = logl = brier = 0.0
    correct = 0
    order = ("home", "draw", "away")
    for probs, actual in rows:
        rps += ranked_probability_score(probs, actual)
        logl += log_loss(probs, actual)
        brier += brier_score(probs, actual)
        if order[int(max(range(3), key=lambda i: probs[i]))] == actual:
            correct += 1
    n = len(rows)
    return Scores(n, rps / n, logl / n, brier / n, correct / n)


def backtest(
    start: str, end: str, competition: str | None = None, verbose: bool = True
) -> dict[str, Scores]:
    """Backtest all played matches in [start, end], optionally one competition."""
    conn = get_connection()
    sql = """SELECT match_id, date, home_id, away_id, home_goals, away_goals, neutral
             FROM matches
             WHERE home_goals IS NOT NULL AND date >= ? AND date <= ?"""
    params = [start, end]
    if competition:
        sql += " AND competition = ?"
        params.append(competition)
    sql += " ORDER BY date ASC"
    matches = conn.execute(sql, params).fetchall()

    blend_rows, elo_rows, base_rows = [], [], []
    bundle = None
    cur_date = None

    for m in matches:
        # Refit only when the date advances (keeps the time-split strict + fast).
        if m["date"] != cur_date:
            cur_date = m["date"]
            bundle = build_models(conn, as_of=cur_date)

        actual = actual_outcome(m["home_goals"], m["away_goals"])
        neutral = bool(m["neutral"])
        pred = predict(bundle, m["home_id"], m["away_id"], neutral=neutral, conn=conn)
        blend_rows.append(((pred.p_home, pred.p_draw, pred.p_away), actual))

        e = elo_1x2(bundle.elo_of(m["home_id"]), bundle.elo_of(m["away_id"]), neutral)
        elo_rows.append((e, actual))
        base_rows.append((BASE_RATE, actual))

    conn.close()

    results = {
        "blend": _aggregate(blend_rows),
        "elo_only": _aggregate(elo_rows),
        "base_rate": _aggregate(base_rows),
    }
    if verbose:
        title = f"BACKTEST  {start} → {end}" + (f"  [{competition}]" if competition else "")
        print("=" * 72)
        print(f"  {title}")
        print("=" * 72)
        for name, s in results.items():
            print(f"  {name:10s} {s}")
        print("=" * 72)
        b, e, r = results["blend"], results["elo_only"], results["base_rate"]
        verdict = "✅ blend beats the base-rate baseline" if b.rps < r.rps else \
                  "❌ blend does NOT beat base-rate — investigate"
        print(f"  {verdict}  (lower RPS is better)")
        if b.rps < e.rps:
            print("  ✅ blend also edges Elo-only")
        else:
            print("  ⚠️ Elo-only matched/beat the blend here — consider tuning w/ξ")
        print("=" * 72)
    return results


if __name__ == "__main__":
    # WC 2022 (group stage onward) is a clean held-out tournament.
    backtest("2022-11-20", "2022-12-18", competition="FIFA World Cup")
