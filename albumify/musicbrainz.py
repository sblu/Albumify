"""MusicBrainz Web Service v2 client.

Polite rate limit: 1 req/sec (per their docs). Custom User-Agent required.
"""
from __future__ import annotations

import time
from urllib.parse import quote

import requests


class MBNotFound(Exception):
    pass


class MusicBrainzClient:
    BASE = "https://musicbrainz.org/ws/2"

    def __init__(self, user_agent: str, min_interval: float = 1.0):
        self.user_agent = user_agent
        self.min_interval = min_interval
        self._last_request_t = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request_t)
        if wait > 0:
            time.sleep(wait)
        self._last_request_t = time.monotonic()

    def search_release_group(self, artist: str, title: str) -> "str | None":
        """Return MBID of the canonical album release-group, or None."""
        query = f'artist:"{artist}" AND releasegroup:"{title}" AND primarytype:album'
        url = f"{self.BASE}/release-group?query={quote(query)}&fmt=json&limit=5"
        self._throttle()
        resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
        resp.raise_for_status()
        groups = resp.json().get("release-groups", [])

        # Prefer entries with no secondary types (skip Remix / Compilation / Live / etc.)
        for g in groups:
            if g.get("primary-type") == "Album" and not g.get("secondary-types"):
                return g["id"]
        return None
