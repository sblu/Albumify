from unittest.mock import MagicMock, patch
import pytest

from albumify.caa import CAAClient, CAATooSmall


def make_resp(status, content=b"", content_length=None):
    r = MagicMock()
    r.status_code = status
    r.content = content
    r.headers = {"content-length": str(len(content) if content_length is None else content_length)}
    return r


def jpeg_bytes(w: int, h: int) -> bytes:
    # Build a real JPEG so PIL identifies dimensions correctly.
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (w, h), (128, 128, 128)).save(buf, "JPEG")
    return buf.getvalue()


def test_fetch_at_1200_success_first_try():
    client = CAAClient(user_agent="Albumify/0.1 (test)", min_interval=0)
    img1200 = jpeg_bytes(1200, 1200)
    with patch("albumify.caa.requests.get", return_value=make_resp(200, img1200)):
        data, dims = client.fetch_front("abbey-mbid")
    assert len(data) > 0
    assert dims == (1200, 1200)


def test_fetch_retries_on_5xx_then_succeeds():
    client = CAAClient(user_agent="Albumify/0.1 (test)", min_interval=0, max_retries=3, backoff_base=0.0)
    img = jpeg_bytes(1200, 1200)
    responses = [make_resp(500), make_resp(500), make_resp(200, img)]
    with patch("albumify.caa.requests.get", side_effect=responses) as m:
        data, dims = client.fetch_front("dsotm-mbid")
    assert m.call_count == 3
    assert dims == (1200, 1200)


def test_fetch_falls_back_to_original_after_persistent_1200_failure():
    client = CAAClient(user_agent="Albumify/0.1 (test)", min_interval=0, max_retries=2, backoff_base=0.0)
    img = jpeg_bytes(900, 900)
    # 2x 500 at /front-1200 then 200 at /front
    responses = [make_resp(500), make_resp(500), make_resp(200, img)]
    urls = []

    def side_effect(url, **kw):
        urls.append(url)
        return responses.pop(0)

    with patch("albumify.caa.requests.get", side_effect=side_effect):
        data, dims = client.fetch_front("falling-back-mbid")
    assert dims == (900, 900)
    assert urls[-1].endswith("/front"), f"final URL not /front: {urls}"


def test_fetch_raises_too_small_below_threshold():
    client = CAAClient(user_agent="Albumify/0.1 (test)", min_interval=0, min_dim=256)
    img = jpeg_bytes(200, 200)
    with patch("albumify.caa.requests.get", return_value=make_resp(200, img)):
        with pytest.raises(CAATooSmall):
            client.fetch_front("tiny-mbid")


def test_fetch_returns_none_on_404():
    client = CAAClient(user_agent="Albumify/0.1 (test)", min_interval=0)
    with patch("albumify.caa.requests.get", return_value=make_resp(404)):
        assert client.fetch_front("nonexistent-mbid") is None
