import json
from pathlib import Path

REQUIRED_KEYS = {"rank", "slug", "artist", "title", "year"}
SLUG_RE = r"^[a-z0-9][a-z0-9-]*[a-z0-9]$"


def test_albums_json_loads_and_validates():
    path = Path(__file__).parent.parent / "data" / "albums.json"
    data = json.loads(path.read_text())
    assert isinstance(data, list) and len(data) >= 10, "expect at least 10 entries"
    seen_slugs = set()
    seen_ranks = set()
    import re
    for entry in data:
        missing = REQUIRED_KEYS - entry.keys()
        assert not missing, f"entry missing keys: {missing}: {entry}"
        assert re.match(SLUG_RE, entry["slug"]), f"bad slug: {entry['slug']}"
        assert entry["slug"] not in seen_slugs, f"duplicate slug: {entry['slug']}"
        assert entry["rank"] not in seen_ranks, f"duplicate rank: {entry['rank']}"
        assert 1900 <= entry["year"] <= 2030
        seen_slugs.add(entry["slug"])
        seen_ranks.add(entry["rank"])
