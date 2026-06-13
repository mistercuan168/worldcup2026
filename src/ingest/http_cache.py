"""Shared HTTP layer: local cache -> rate limit -> request -> retry/backoff.

Every API client goes through CachedClient so we never burn free-tier quota on
data we already have. Static data (fixtures, history) is cached effectively forever;
live data is cached for a few seconds.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from src.db.models import DATA_DIR

CACHE_DIR = DATA_DIR / "cache"


class CachedClient:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        min_interval: float = 6.0,  # seconds between live calls (10/min default)
        cache_ttl: float = 7 * 24 * 3600,  # static data: 1 week
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.min_interval = min_interval
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self._last_call = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str, params: dict | None) -> Path:
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        return CACHE_DIR / f"{digest}.json"

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()

    def get(
        self,
        path: str,
        params: dict | None = None,
        cache_ttl: float | None = None,
    ) -> dict[str, Any] | None:
        """GET with caching. Returns parsed JSON, or None on persistent failure."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        ttl = self.cache_ttl if cache_ttl is None else cache_ttl
        cache_file = self._cache_path(url, params)

        # 1) Cache check
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < ttl:
            return json.loads(cache_file.read_text())

        # 2) Rate limit + 3) request with retry/backoff
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                resp = requests.get(url, headers=self.headers, params=params, timeout=20)
                if resp.status_code == 429:  # too many requests
                    time.sleep(2 ** attempt * 5)
                    continue
                resp.raise_for_status()
                data = resp.json()
                cache_file.write_text(json.dumps(data))
                return data
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    # Fall back to stale cache if we have any.
                    if cache_file.exists():
                        return json.loads(cache_file.read_text())
                    return None
                time.sleep(2 ** attempt)
        return None
