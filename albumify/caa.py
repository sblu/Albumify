"""Cover Art Archive HTTP client.

Endpoints used:
    https://coverartarchive.org/release-group/{mbid}/front-1200
    https://coverartarchive.org/release-group/{mbid}/front       (fallback)

Behavior:
- Retry up to max_retries on 5xx with exponential backoff.
- Fall back to /front if /front-1200 keeps failing.
- Reject images smaller than min_dim on either axis (CAA returns native size if smaller than requested).
"""
from __future__ import annotations

import time
from io import BytesIO
from typing import Optional, Tuple

import requests
from PIL import Image


class CAATooSmall(Exception):
    pass


class CAAClient:
    BASE = "https://coverartarchive.org/release-group"

    def __init__(
        self,
        user_agent: str,
        min_interval: float = 0.5,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        min_dim: int = 256,
    ):
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.min_dim = min_dim
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, url: str) -> requests.Response:
        self._throttle()
        return requests.get(url, headers={"User-Agent": self.user_agent},
                            allow_redirects=True, timeout=60)

    def fetch_front(self, mbid: str) -> Optional[Tuple[bytes, Tuple[int, int]]]:
        """Fetch the front cover. Returns (bytes, (w, h)) or None on 404."""
        for url in (f"{self.BASE}/{mbid}/front-1200", f"{self.BASE}/{mbid}/front"):
            for attempt in range(self.max_retries):
                resp = self._get(url)
                if resp.status_code == 404:
                    return None  # no cover for this MBID at all
                if 500 <= resp.status_code < 600:
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                w, h = img.size
                if min(w, h) < self.min_dim:
                    raise CAATooSmall(f"{mbid}: native {w}x{h} below {self.min_dim}")
                return resp.content, (w, h)
            # exhausted retries at this URL; fall through to next URL
        return None
