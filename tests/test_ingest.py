"""Phase 0/1 acceptance tests: schema creates, CSV loads, counts are sane."""
from pathlib import Path

from src.db.models import get_connection, init_db
from src.ingest.csv_loader import load_results


def test_schema_creates(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    conn = get_connection(db)
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    conn.close()
    assert {"teams", "matches", "elo_ratings", "team_strengths", "predictions"} <= tables


def test_csv_loads_and_counts_sane():
    # Uses the committed data/results.csv against the real DB build.
    csv = Path(__file__).resolve().parents[1] / "data" / "results.csv"
    assert csv.exists(), "historical results.csv must be committed"

    init_db()
    conn = get_connection()
    # Wipe to a clean slate for a deterministic count check on a temp-like run.
    n = load_results(csv, conn=conn)
    total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    conn.close()

    assert total > 40000, "expected tens of thousands of historical matches"
    assert teams > 200, "expected hundreds of national teams"


def test_reload_is_idempotent():
    """Re-running the loader must not duplicate rows (cache/idempotency check)."""
    conn = get_connection()
    before = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    load_results(conn=conn)
    after = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.close()
    assert after == before
