import json
from pathlib import Path
from unittest.mock import patch

import pytest

from albumify import fetch_covers


def test_fetch_covers_writes_files_and_report(tmp_path):
    albums = [
        {"rank": 1, "slug": "a", "artist": "Artist A", "title": "Album A", "year": 2000},
        {"rank": 2, "slug": "b", "artist": "Artist B", "title": "Album B", "year": 2001},
    ]
    albums_json = tmp_path / "albums.json"
    albums_json.write_text(json.dumps(albums))
    covers = tmp_path / "covers"
    report = tmp_path / "report.txt"

    # Mock: MB returns mbid for both; CAA returns bytes for both.
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (1200, 1200), (1, 2, 3)).save(buf, "JPEG")
    jpg = buf.getvalue()

    with patch("albumify.fetch_covers.MusicBrainzClient") as MB, \
         patch("albumify.fetch_covers.CAAClient") as CAA:
        MB.return_value.search_release_group.side_effect = ["mbid-a", "mbid-b"]
        CAA.return_value.fetch_front.side_effect = [(jpg, (1200, 1200)), (jpg, (1200, 1200))]
        n = fetch_covers.run(albums_json, covers, report, user_agent="test/0.1")

    assert n == 2
    assert (covers / "a.jpg").exists()
    assert (covers / "b.jpg").exists()
    text = report.read_text()
    assert "a OK" in text and "b OK" in text


def test_fetch_covers_logs_misses(tmp_path):
    albums = [{"rank": 1, "slug": "x", "artist": "X", "title": "Y", "year": 2000}]
    albums_json = tmp_path / "albums.json"
    albums_json.write_text(json.dumps(albums))
    covers = tmp_path / "covers"
    report = tmp_path / "report.txt"

    with patch("albumify.fetch_covers.MusicBrainzClient") as MB, \
         patch("albumify.fetch_covers.CAAClient") as CAA:
        MB.return_value.search_release_group.return_value = None
        n = fetch_covers.run(albums_json, covers, report, user_agent="test/0.1")
    assert n == 0
    assert "x no_mbid" in report.read_text()


def test_fetch_covers_skips_existing(tmp_path):
    albums = [{"rank": 1, "slug": "z", "artist": "Z", "title": "ZZ", "year": 2000}]
    albums_json = tmp_path / "albums.json"
    albums_json.write_text(json.dumps(albums))
    covers = tmp_path / "covers"
    covers.mkdir()
    (covers / "z.jpg").write_bytes(b"existing")
    report = tmp_path / "report.txt"

    with patch("albumify.fetch_covers.MusicBrainzClient") as MB, \
         patch("albumify.fetch_covers.CAAClient") as CAA:
        n = fetch_covers.run(albums_json, covers, report, user_agent="test/0.1")
        MB.return_value.search_release_group.assert_not_called()
        CAA.return_value.fetch_front.assert_not_called()
    assert n == 0
    assert "z skip_exists" in report.read_text()
