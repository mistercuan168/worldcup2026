"""Build the database from scratch: schema -> historical CSV -> WC26 fixtures.

This is the startup routine. On free hosts the .db file may be wiped on restart,
so we always rebuild from the committed CSV and re-pull fixtures (rate-limited).
"""
from __future__ import annotations

from src.db.models import get_connection, init_db
from src.ingest.csv_loader import load_results
from src.ingest.football_data_client import load_wc_fixtures


def build(verbose: bool = True) -> dict:
    init_db()
    conn = get_connection()

    hist = load_results(conn=conn)
    wc = load_wc_fixtures(conn=conn)

    stats = {
        "teams": conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0],
        "matches": conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
        "history_loaded": hist,
        "wc_fixtures_loaded": wc,
        "played": conn.execute(
            "SELECT COUNT(*) FROM matches WHERE home_goals IS NOT NULL"
        ).fetchone()[0],
        "date_min": conn.execute("SELECT MIN(date) FROM matches").fetchone()[0],
        "date_max": conn.execute("SELECT MAX(date) FROM matches").fetchone()[0],
    }
    conn.close()

    if verbose:
        print("=" * 52)
        print("  DATABASE BUILD COMPLETE")
        print("=" * 52)
        print(f"  Teams in DB ........... {stats['teams']:>8,}")
        print(f"  Matches in DB ......... {stats['matches']:>8,}")
        print(f"    of which played ..... {stats['played']:>8,}")
        print(f"  Date range ............ {stats['date_min']} -> {stats['date_max']}")
        print(f"  WC26 fixtures loaded .. {stats['wc_fixtures_loaded']:>8,}")
        if stats["wc_fixtures_loaded"] == 0:
            print("    (no FOOTBALL_DATA_API_KEY set — history-only, which is fine)")
        print("=" * 52)
    return stats


if __name__ == "__main__":
    build()
