from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from albumify import gen_labels


def fake_image_response(width: int = 1024, height: int = 1024):
    """Mock structure of google-genai response with inline image data."""
    from io import BytesIO
    buf = BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, "PNG")
    inline_data = MagicMock()
    inline_data.data = buf.getvalue()
    inline_data.mime_type = "image/png"
    part = MagicMock()
    part.inline_data = inline_data
    part.text = None
    candidate = MagicMock()
    candidate.content.parts = [part]
    resp = MagicMock()
    resp.candidates = [candidate]
    return resp


def test_gen_labels_creates_png_for_each_cover(tmp_path):
    covers = tmp_path / "covers"
    covers.mkdir()
    labels = tmp_path / "labels"
    report = tmp_path / "labels_report.txt"
    Image.new("RGB", (1200, 1200)).save(covers / "a.jpg")
    Image.new("RGB", (1200, 1200)).save(covers / "b.jpg")

    with patch("albumify.gen_labels._make_client") as mc, \
         patch("albumify.gen_labels._genai_call", return_value=fake_image_response()) as call:
        n = gen_labels.run(covers, labels, report,
                           model="gemini-3.1-flash-image-preview",
                           api_key="test-key", sleep_s=0)
    assert n == 2
    assert call.call_count == 2
    assert (labels / "a.png").exists()
    assert (labels / "b.png").exists()
    assert " OK " in report.read_text()


def test_gen_labels_skips_existing(tmp_path):
    covers = tmp_path / "covers"
    covers.mkdir()
    labels = tmp_path / "labels"
    labels.mkdir()
    Image.new("RGB", (1200, 1200)).save(covers / "z.jpg")
    Image.new("RGB", (1024, 1024)).save(labels / "z.png")
    report = tmp_path / "labels_report.txt"

    with patch("albumify.gen_labels._make_client") as mc, \
         patch("albumify.gen_labels._genai_call") as call:
        n = gen_labels.run(covers, labels, report,
                           model="gemini-3.1-flash-image-preview",
                           api_key="test-key", sleep_s=0)
    assert n == 0
    call.assert_not_called()
    assert "skip_exists" in report.read_text()


def test_gen_labels_limit_applied(tmp_path):
    covers = tmp_path / "covers"
    covers.mkdir()
    labels = tmp_path / "labels"
    report = tmp_path / "labels_report.txt"
    for slug in ("a", "b", "c", "d", "e"):
        Image.new("RGB", (800, 800)).save(covers / f"{slug}.jpg")

    with patch("albumify.gen_labels._make_client"), \
         patch("albumify.gen_labels._genai_call", return_value=fake_image_response()) as call:
        n = gen_labels.run(covers, labels, report,
                           model="gemini-3.1-flash-image-preview",
                           api_key="test-key", limit=3, sleep_s=0)
    assert n == 3
    assert call.call_count == 3


def test_gen_labels_logs_api_errors(tmp_path):
    covers = tmp_path / "covers"
    covers.mkdir()
    labels = tmp_path / "labels"
    report = tmp_path / "labels_report.txt"
    Image.new("RGB", (800, 800)).save(covers / "x.jpg")

    def boom(*a, **k):
        raise RuntimeError("api boom")

    with patch("albumify.gen_labels._make_client"), \
         patch("albumify.gen_labels._genai_call", side_effect=boom):
        n = gen_labels.run(covers, labels, report,
                           model="gemini-3.1-flash-image-preview",
                           api_key="test-key", sleep_s=0)
    assert n == 0
    assert "api_error" in report.read_text()


def test_gen_labels_handles_response_with_no_image(tmp_path):
    covers = tmp_path / "covers"
    covers.mkdir()
    labels = tmp_path / "labels"
    report = tmp_path / "labels_report.txt"
    Image.new("RGB", (800, 800)).save(covers / "q.jpg")

    empty_resp = MagicMock()
    empty_resp.candidates = []  # no candidates, no image

    with patch("albumify.gen_labels._make_client"), \
         patch("albumify.gen_labels._genai_call", return_value=empty_resp):
        n = gen_labels.run(covers, labels, report,
                           model="gemini-3.1-flash-image-preview",
                           api_key="test-key", sleep_s=0)
    assert n == 0
    assert "no_image_in_response" in report.read_text()
