"""Load the historical international-results CSV (martj42 dataset) into SQLite.

Source: https://github.com/martj42/international_results (results.csv, 1872->present).
This is the single most important data source — Elo and Dixon-Coles train on it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.db.models import DATA_DIR, get_connection, init_db

RESULTS_CSV = DATA_DIR / "results.csv"


def load_results(csv_path: Path = RESULTS_CSV, conn: sqlite3.Connection | None = None) -> int:
    """Load historical results into the matches table. Returns rows inserted.

    Idempotent: the UNIQUE(date, home_id, away_id) constraint + INSERT OR IGNORE
    means re-running does not duplicate rows.
    """
    own_conn = conn is None
    if own_conn:
        init_db()
        conn = get_connection()

    df = pd.read_csv(csv_path)

    # Build the full set of team names first, insert them once.
    names = pd.unique(pd.concat([df["home_team"], df["away_team"]]).dropna())
    conn.executemany(
        "INSERT OR IGNORE INTO teams(name) VALUES (?)", [(n,) for n in names]
    )
    conn.commit()
    name_to_id = {
        row["name"]: row["team_id"]
        for row in conn.execute("SELECT team_id, name FROM teams")
    }

    rows = []
    for r in df.itertuples(index=False):
        home_id = name_to_id.get(r.home_team)
        away_id = name_to_id.get(r.away_team)
        if home_id is None or away_id is None:
            continue
        # Skip rows with missing scores (a handful of unplayed/unknown results).
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue
        neutral = 1 if str(r.neutral).strip().upper() == "TRUE" else 0
        rows.append(
            (
                str(r.date),
                str(r.tournament),
                home_id,
                away_id,
                int(r.home_score),
                int(r.away_score),
                neutral,
                "finished",
                "history_csv",
            )
        )

    before = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.executemany(
        """INSERT OR IGNORE INTO matches
           (date, competition, home_id, away_id, home_goals, away_goals,
            neutral, status, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    if own_conn:
        conn.close()
    return after - before


if __name__ == "__main__":
    n = load_results()
    print(f"Loaded {n} historical matches.")
