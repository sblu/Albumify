"""Tests for the review web app: DB ops, seeding, API behavior, regen path."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from albumify import review_app


def make_png(path: Path, color=(200, 200, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color).save(path, "PNG")


def make_jpg(path: Path, color=(150, 150, 150)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color).save(path, "JPEG")


def setup_dirs(tmp_path: Path, slugs=("a", "b", "c")) -> dict:
    covers = tmp_path / "covers"
    labels = tmp_path / "labels"
    history = tmp_path / "labels_history"
    db = tmp_path / "review.db"
    for s in slugs:
        make_jpg(covers / f"{s}.jpg")
        make_png(labels / f"{s}.png")
    return {"covers": covers, "labels": labels, "history": history, "db": db}


# --- Seeding ----------------------------------------------------------

def test_seed_db_creates_v1_for_each_label(tmp_path):
    d = setup_dirs(tmp_path, ("a", "b"))
    added = review_app.seed_db(d["db"], d["labels"], d["history"])
    assert added == 2
    items = review_app.fetch_items(d["db"])
    slugs = sorted(x["slug"] for x in items)
    assert slugs == ["a", "b"]
    for it in items:
        assert it["status"] == "pending"
        assert it["current_iter"] == 1
        assert len(it["iterations"]) == 1
        v1 = it["iterations"][0]
        assert v1["iter_n"] == 1
        assert v1["comment_addressed"] is None
        assert v1["png_path"] == f"labels_history/{it['slug']}/v1.png"
        assert (d["history"] / it["slug"] / "v1.png").exists()


def test_seed_db_is_idempotent(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])
    added2 = review_app.seed_db(d["db"], d["labels"], d["history"])
    assert added2 == 0
    items = review_app.fetch_items(d["db"])
    assert len(items) == 1


# --- DB helpers -------------------------------------------------------

def test_set_status_updates_row(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])
    review_app.set_status(d["db"], "a", "approved")
    item = review_app.fetch_items(d["db"])[0]
    assert item["status"] == "approved"


def test_append_iteration_bumps_current_iter(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])
    n = review_app.append_iteration(
        d["db"], "a",
        prompt="P2", model="m", png_path="labels_history/a/v2.png",
        comment_addressed="too dark",
    )
    assert n == 2
    item = next(x for x in review_app.fetch_items(d["db"]) if x["slug"] == "a")
    assert item["status"] == "rejected"
    assert item["current_iter"] == 2
    assert len(item["iterations"]) == 2
    assert item["iterations"][1]["comment_addressed"] == "too dark"


def test_build_refined_prompt_appends_comment():
    p = review_app.build_refined_prompt("BASE.", "fix the eyes")
    assert "BASE." in p
    assert "fix the eyes" in p
    assert "reviewer feedback" in p.lower()


# --- Flask API --------------------------------------------------------

def test_get_items_returns_seeded(tmp_path):
    d = setup_dirs(tmp_path, ("a", "b"))
    review_app.seed_db(d["db"], d["labels"], d["history"])
    app = review_app.create_app(
        db_path=d["db"], covers_dir=d["covers"], history_dir=d["history"], worker=None,
    )
    client = app.test_client()
    resp = client.get("/api/items")
    assert resp.status_code == 200
    data = resp.get_json()
    assert sorted(x["slug"] for x in data) == ["a", "b"]


def test_approve_marks_row_green(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])
    app = review_app.create_app(
        db_path=d["db"], covers_dir=d["covers"], history_dir=d["history"], worker=None,
    )
    client = app.test_client()
    resp = client.post("/api/items/a/approve")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "approved"


def test_reject_requires_comment(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])
    app = review_app.create_app(
        db_path=d["db"], covers_dir=d["covers"], history_dir=d["history"], worker=None,
    )
    client = app.test_client()
    resp = client.post("/api/items/a/reject", json={"comment": ""})
    assert resp.status_code == 400


def test_reject_with_worker_enqueues_and_marks_processing(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])
    worker = MagicMock()
    app = review_app.create_app(
        db_path=d["db"], covers_dir=d["covers"], history_dir=d["history"], worker=worker,
    )
    # Have enqueue actually flip status so the response reflects it.
    def fake_enqueue(slug, comment):
        review_app.set_status(d["db"], slug, "processing")
    worker.enqueue.side_effect = fake_enqueue
    client = app.test_client()
    resp = client.post("/api/items/a/reject", json={"comment": "too dark"})
    assert resp.status_code == 200
    worker.enqueue.assert_called_once_with("a", "too dark")
    assert resp.get_json()["status"] == "processing"


def test_serve_cover_and_history(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])
    app = review_app.create_app(
        db_path=d["db"], covers_dir=d["covers"], history_dir=d["history"], worker=None,
    )
    client = app.test_client()
    assert client.get("/covers/a.jpg").status_code == 200
    assert client.get("/labels_history/a/v1.png").status_code == 200


# --- Regen path ------------------------------------------------------

def test_regenerate_label_writes_v2_and_inserts_row(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])

    fake_resp = MagicMock()
    fake_inline = MagicMock()
    fake_inline.data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    part = MagicMock()
    part.inline_data = fake_inline
    cand = MagicMock()
    cand.content.parts = [part]
    fake_resp.candidates = [cand]

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_resp

    n = review_app.regenerate_label(
        slug="a", comment="brighter background",
        covers_dir=d["covers"], history_dir=d["history"], db_path=d["db"],
        api_key="k", make_client=lambda key: fake_client,
    )
    assert n == 2
    assert (d["history"] / "a" / "v2.png").exists()
    item = next(x for x in review_app.fetch_items(d["db"]) if x["slug"] == "a")
    assert item["current_iter"] == 2
    assert item["status"] == "rejected"
    assert item["iterations"][1]["comment_addressed"] == "brighter background"
    assert "brighter background" in item["iterations"][1]["prompt"]


def test_regenerate_label_handles_api_failure(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])
    review_app.set_status(d["db"], "a", "processing")

    def boom(api_key):
        c = MagicMock()
        c.models.generate_content.side_effect = RuntimeError("boom")
        return c

    n = review_app.regenerate_label(
        slug="a", comment="x",
        covers_dir=d["covers"], history_dir=d["history"], db_path=d["db"],
        api_key="k", make_client=boom,
    )
    assert n is None
    item = next(x for x in review_app.fetch_items(d["db"]) if x["slug"] == "a")
    assert item["status"] == "rejected"
    assert item["current_iter"] == 1


def test_regenerate_label_no_image_in_response(tmp_path):
    d = setup_dirs(tmp_path, ("a",))
    review_app.seed_db(d["db"], d["labels"], d["history"])

    empty_resp = MagicMock()
    empty_resp.candidates = []
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = empty_resp

    n = review_app.regenerate_label(
        slug="a", comment="x",
        covers_dir=d["covers"], history_dir=d["history"], db_path=d["db"],
        api_key="k", make_client=lambda k: fake_client,
    )
    assert n is None
    item = next(x for x in review_app.fetch_items(d["db"]) if x["slug"] == "a")
    assert item["status"] == "rejected"
