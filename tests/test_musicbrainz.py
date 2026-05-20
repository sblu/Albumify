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


def test_search_skips_remix_compilation_secondary_types():
    payload = {
        "release-groups": [
            {"id": "remix-id", "title": "Abbey Road Remix",
             "primary-type": "Album", "secondary-types": ["Remix"]},
            {"id": "canonical-id", "title": "Abbey Road",
             "primary-type": "Album", "secondary-types": []},
        ]
    }
    client = MusicBrainzClient(user_agent="Albumify/0.1 (test)")
    with patch("albumify.musicbrainz.requests.get", return_value=make_response(payload)):
        mbid = client.search_release_group("The Beatles", "Abbey Road")
    assert mbid == "canonical-id"


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
