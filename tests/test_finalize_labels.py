from pathlib import Path

from PIL import Image

from albumify import review_app
from albumify.finalize_labels import finalize


def _seed(tmp_path: Path) -> dict:
    covers = tmp_path / "covers"; covers.mkdir()
    labels = tmp_path / "labels"; labels.mkdir()
    history = tmp_path / "labels_history"
    db = tmp_path / "review.db"
    for slug in ("a", "b", "c", "d"):
        Image.new("RGB", (32, 32)).save(covers / f"{slug}.jpg")
        Image.new("L", (32, 32), 200).save(labels / f"{slug}.png")  # v1 is gray
    review_app.seed_db(db, labels, history)
    return {"db": db, "labels": labels, "history": history}


def test_finalize_copies_only_approved_non_v1(tmp_path):
    d = _seed(tmp_path)
    # Promote 'a' through 2 iterations and approve at v2.
    Image.new("L", (32, 32), 50).save(d["history"] / "a" / "v2.png")  # dark
    review_app.append_iteration(
        d["db"], "a", prompt="P", model="m",
        png_path="labels_history/a/v2.png", comment_addressed="darker",
    )
    review_app.set_status(d["db"], "a", "approved")
    # 'b' approved at v1 (no change expected).
    review_app.set_status(d["db"], "b", "approved")
    # 'c' rejected — should be skipped.
    Image.new("L", (32, 32), 100).save(d["history"] / "c" / "v2.png")
    review_app.append_iteration(
        d["db"], "c", prompt="P", model="m",
        png_path="labels_history/c/v2.png", comment_addressed="meh",
    )  # leaves status 'rejected'
    # 'd' still pending.

    stats = finalize(db_path=d["db"], labels_dir=d["labels"], history_dir=d["history"])
    assert stats["copied"] == 1
    assert stats["skipped_v1"] == 1
    assert stats["skipped_not_approved"] == 2
    # 'a' should now be the dark image (value 50)
    arr_a = Image.open(d["labels"] / "a.png").getextrema()
    assert arr_a == (50, 50)
    # 'b' untouched (still gray 200)
    arr_b = Image.open(d["labels"] / "b.png").getextrema()
    assert arr_b == (200, 200)


def test_finalize_all_mode_includes_rejected(tmp_path):
    d = _seed(tmp_path)
    Image.new("L", (32, 32), 50).save(d["history"] / "a" / "v2.png")
    review_app.append_iteration(
        d["db"], "a", prompt="P", model="m",
        png_path="labels_history/a/v2.png", comment_addressed="x",
    )  # leaves status 'rejected'
    stats = finalize(
        db_path=d["db"], labels_dir=d["labels"], history_dir=d["history"],
        require_approved=False,
    )
    assert stats["copied"] == 1  # the rejected v2 of 'a' gets copied
