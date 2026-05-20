import time
from unittest.mock import MagicMock, patch
import pytest

from albumify.musicbrainz import MusicBrainzClient, MBNotFound


def make_response(json_payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_payload
    resp.raise_for_status = MagicMock()
    return resp


def test_search_returns_first_album_release_group_mbid():
    payload = {
        "release-groups": [
            {"id": "9162580e-5df4-32de-80cc-f45a8d8a9b1d",
             "title": "Abbey Road", "primary-type": "Album",
             "secondary-types": []},
        ]
    }
    client = MusicBrainzClient(user_agent="Albumify/0.1 (test)")
    with patch("albumify.musicbrainz.requests.get", return_value=make_response(payload)) as m:
        mbid = client.search_release_group("The Beatles", "Abbey Road")
    assert mbid == "9162580e-5df4-32de-80cc-f45a8d8a9b1d"
    args, kwargs = m.call_args
    assert "musicbrainz.org/ws/2/release-group" in args[0]
    assert kwargs["headers"]["User-Agent"] == "Albumify/0.1 (test)"


def test_search_skips_remix_live_demo_secondary_types():
    payload = {
        "release-groups": [
            {"id": "remix-id", "title": "Abbey Road Remix",
             "primary-type": "Album", "secondary-types": ["Remix"]},
            {"id": "live-id", "title": "Live at Abbey Road",
             "primary-type": "Album", "secondary-types": ["Live"]},
            {"id": "demo-id", "title": "Abbey Road Demos",
             "primary-type": "Album", "secondary-types": ["Demo"]},
            {"id": "canonical-id", "title": "Abbey Road",
             "primary-type": "Album", "secondary-types": []},
        ]
    }
    client = MusicBrainzClient(user_agent="Albumify/0.1 (test)")
    with patch("albumify.musicbrainz.requests.get", return_value=make_response(payload)):
        mbid = client.search_release_group("The Beatles", "Abbey Road")
    assert mbid == "canonical-id"


def test_search_accepts_compilation_and_soundtrack():
    """Greatest-hits / soundtrack best-sellers must not be filtered out."""
    payload = {
        "release-groups": [
            {"id": "comp-id", "title": "Their Greatest Hits",
             "primary-type": "Album", "secondary-types": ["Compilation"]},
        ]
    }
    client = MusicBrainzClient(user_agent="Albumify/0.1 (test)")
    with patch("albumify.musicbrainz.requests.get", return_value=make_response(payload)):
        assert client.search_release_group("Eagles", "Their Greatest Hits") == "comp-id"

    payload = {
        "release-groups": [
            {"id": "ost-id", "title": "The Bodyguard",
             "primary-type": "Album", "secondary-types": ["Soundtrack"]},
        ]
    }
    with patch("albumify.musicbrainz.requests.get", return_value=make_response(payload)):
        assert client.search_release_group("Whitney Houston", "The Bodyguard") == "ost-id"


def test_search_prefers_pure_album_over_compilation():
    """When both a pure Album and a Compilation exist, prefer the Album."""
    payload = {
        "release-groups": [
            {"id": "comp-id", "title": "X (Greatest Hits)",
             "primary-type": "Album", "secondary-types": ["Compilation"]},
            {"id": "album-id", "title": "X",
             "primary-type": "Album", "secondary-types": []},
        ]
    }
    client = MusicBrainzClient(user_agent="Albumify/0.1 (test)")
    with patch("albumify.musicbrainz.requests.get", return_value=make_response(payload)):
        assert client.search_release_group("Artist", "X") == "album-id"


def test_search_rejects_mixed_disallowed_secondary_types():
    """An entry like Compilation+Live should be rejected (Live is disallowed)."""
    payload = {
        "release-groups": [
            {"id": "mixed-id", "title": "Live Greatest Hits",
             "primary-type": "Album", "secondary-types": ["Compilation", "Live"]},
        ]
    }
    client = MusicBrainzClient(user_agent="Albumify/0.1 (test)")
    with patch("albumify.musicbrainz.requests.get", return_value=make_response(payload)):
        assert client.search_release_group("Artist", "X") is None


def test_search_returns_none_when_no_match():
    client = MusicBrainzClient(user_agent="Albumify/0.1 (test)")
    with patch("albumify.musicbrainz.requests.get", return_value=make_response({"release-groups": []})):
        assert client.search_release_group("Nobody", "Nothing") is None


def test_rate_limit_enforces_one_second_minimum():
    client = MusicBrainzClient(user_agent="Albumify/0.1 (test)", min_interval=0.5)
    with patch("albumify.musicbrainz.requests.get", return_value=make_response({"release-groups": []})):
        t0 = time.monotonic()
        client.search_release_group("A", "B")
        client.search_release_group("C", "D")
        elapsed = time.monotonic() - t0
    assert elapsed >= 0.5, f"rate limit not enforced (elapsed={elapsed:.3f}s)"
