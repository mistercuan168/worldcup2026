"""API-Football (api-sports.io) client — lineups, team stats, and their own
/predictions endpoint.

Free tier: 100 requests/DAY — very tight. Cache aggressively; never loop blindly.
Their /predictions output is a black box: we store it only as a COMPARISON column,
never as training ground truth.
Docs: https://www.api-football.com/documentation-v3
"""
from __future__ import annotations

from src.config import api_football_key
from src.ingest.http_cache import CachedClient

BASE_URL = "https://v3.football.api-sports.io"


def _client() -> CachedClient | None:
    key = api_football_key()
    if not key:
        return None
    # 100/day is brutal: space calls and cache for a full day.
    return CachedClient(
        BASE_URL,
        headers={"x-apisports-key": key},
        min_interval=2.0,
        cache_ttl=24 * 3600,
    )


def fetch_prediction(fixture_id: int) -> dict | None:
    """API-Football's own prediction for a fixture — comparison only, not truth."""
    client = _client()
    if client is None:
        return None
    data = client.get("/predictions", params={"fixture": fixture_id})
    if not data:
        return None
    resp = data.get("response", [])
    return resp[0] if resp else None


def fetch_team_statistics(team_id: int, league: int, season: int) -> dict | None:
    """Team season stats (optional enrichment)."""
    client = _client()
    if client is None:
        return None
    data = client.get(
        "/teams/statistics",
        params={"team": team_id, "league": league, "season": season},
    )
    return data.get("response") if data else None


if __name__ == "__main__":
    print("API-Football client ready." if _client() else
          "No API key set (API_FOOTBALL_KEY) — client disabled.")
