"""football-data.org client — World Cup 2026 fixtures, results, standings.

Free tier: 10 requests/minute. Competition code for the World Cup is 'WC'.
Docs: https://docs.football-data.org/general/v4/index.html
"""
from __future__ import annotations

import sqlite3

from src.config import football_data_key
from src.db.models import get_connection, get_or_create_team, init_db
from src.ingest.http_cache import CachedClient

BASE_URL = "https://api.football-data.org/v4"
WC_COMPETITION = "WC"


def _client() -> CachedClient | None:
    key = football_data_key()
    if not key:
        return None
    # 10 req/min -> ~6s spacing. Fixtures are static -> long cache.
    return CachedClient(BASE_URL, headers={"X-Auth-Token": key}, min_interval=6.5)


def fetch_wc_matches() -> list[dict] | None:
    """Return raw WC 2026 match dicts, or None if no API key configured."""
    client = _client()
    if client is None:
        return None
    data = client.get(f"/competitions/{WC_COMPETITION}/matches")
    if not data:
        return None
    return data.get("matches", [])


_STAGE_MAP = {
    "GROUP_STAGE": "group",
    "LAST_32": "R32",
    "LAST_16": "R16",
    "QUARTER_FINALS": "QF",
    "SEMI_FINALS": "SF",
    "THIRD_PLACE": "3P",
    "FINAL": "F",
}


def load_wc_fixtures(conn: sqlite3.Connection | None = None) -> int:
    """Load WC 2026 fixtures/results into the matches table. Returns rows touched.

    Returns 0 if no API key is set — the model still works on history alone.
    """
    matches = fetch_wc_matches()
    if not matches:
        return 0

    own_conn = conn is None
    if own_conn:
        init_db()
        conn = get_connection()

    touched = 0
    for m in matches:
        home_name = (m.get("homeTeam") or {}).get("name")
        away_name = (m.get("awayTeam") or {}).get("name")
        if not home_name or not away_name:
            continue  # placeholder fixtures (e.g. "Winner Group A") — skip until set
        home_id = get_or_create_team(conn, home_name)
        away_id = get_or_create_team(conn, away_name)
        date = (m.get("utcDate") or "")[:10]
        score = m.get("score", {}).get("fullTime", {})
        hg, ag = score.get("home"), score.get("away")
        status_map = {"FINISHED": "finished", "IN_PLAY": "live", "PAUSED": "live"}
        status = status_map.get(m.get("status", ""), "scheduled")
        stage = _STAGE_MAP.get(m.get("stage", ""))

        conn.execute(
            """INSERT INTO matches
                 (date, competition, home_id, away_id, home_goals, away_goals,
                  neutral, stage, status, source)
               VALUES (?, 'FIFA World Cup', ?, ?, ?, ?, 1, ?, ?, 'football-data')
               ON CONFLICT(date, home_id, away_id) DO UPDATE SET
                 home_goals=excluded.home_goals,
                 away_goals=excluded.away_goals,
                 status=excluded.status,
                 stage=excluded.stage""",
            (date, home_id, away_id, hg, ag, stage, status),
        )
        touched += 1

    conn.commit()
    if own_conn:
        conn.close()
    return touched


if __name__ == "__main__":
    n = load_wc_fixtures()
    print(f"Loaded/updated {n} WC 2026 fixtures." if n else
          "No API key set (FOOTBALL_DATA_API_KEY) — skipped WC fixtures.")
