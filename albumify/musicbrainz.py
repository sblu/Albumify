"""MusicBrainz Web Service v2 client.

Polite rate limit: 1 req/sec (per their docs). Custom User-Agent required.
"""
from __future__ import annotations

import time
from urllib.parse import quote

import requests


# Secondary types we ACCEPT alongside a primary "Album" type.
# Compilation + Soundtrack catch greatest-hits and OST best-sellers.
# Anything else (Remix, Live, Demo, DJ-mix, Mixtape/Street, Audiobook, etc.)
# is rejected because the cover art is not the canonical album cover.
ALLOWED_SECONDARY_TYPES = frozenset({"Compilation", "Soundtrack"})


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

    @staticmethod
    def _is_acceptable(group: dict) -> bool:
        if group.get("primary-type") != "Album":
            return False
        secondary = set(group.get("secondary-types") or [])
        # Empty set is fine (pure Album); otherwise every secondary must be on the allowlist.
        return secondary.issubset(ALLOWED_SECONDARY_TYPES)

    def search_release_group(self, artist: str, title: str) -> "str | None":
        """Return MBID of the canonical album release-group, or None.

        Prefers pure Albums (no secondary types) over Compilations/Soundtracks
        when both appear in the result set.
        """
        query = f'artist:"{artist}" AND releasegroup:"{title}" AND primarytype:album'
        url = f"{self.BASE}/release-group?query={quote(query)}&fmt=json&limit=5"
        self._throttle()
        resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
        resp.raise_for_status()
        groups = resp.json().get("release-groups", [])

        acceptable = [g for g in groups if self._is_acceptable(g)]
        if not acceptable:
            return None
        # Prefer pure Album (no secondary types) over Compilation/Soundtrack.
        pure = [g for g in acceptable if not g.get("secondary-types")]
        return (pure or acceptable)[0]["id"]
