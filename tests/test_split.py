from pathlib import Path

from PIL import Image

from albumify import split


def touch_pair(tmp_path: Path, slug: str, *, has_cover: bool = True, has_label: bool = True) -> None:
    if has_cover:
        Image.new("RGB", (32, 32)).save(tmp_path / "covers" / f"{slug}.jpg")
    if has_label:
        Image.new("RGB", (32, 32)).save(tmp_path / "labels" / f"{slug}.png")


def setup_layout(tmp_path: Path) -> Path:
    (tmp_path / "covers").mkdir()
    (tmp_path / "labels").mkdir()
    return tmp_path


def test_discover_pair_slugs_only_returns_intersect(tmp_path):
    root = setup_layout(tmp_path)
    touch_pair(root, "a")
    touch_pair(root, "b")
    touch_pair(root, "only-cover", has_label=False)
    touch_pair(root, "only-label", has_cover=False)
    slugs = split.discover_pair_slugs(root / "covers", root / "labels")
    assert slugs == ["a", "b"]


def test_split_slugs_is_deterministic_for_same_seed():
    slugs = [f"s{i:03d}" for i in range(50)]
    t1, v1 = split.split_slugs(slugs, val_frac=0.2, seed=42)
    t2, v2 = split.split_slugs(slugs, val_frac=0.2, seed=42)
    assert t1 == t2 and v1 == v2
    # Insensitive to input ordering.
    t3, v3 = split.split_slugs(list(reversed(slugs)), val_frac=0.2, seed=42)
    assert t1 == t3 and v1 == v3


def test_split_slugs_sizes_match_fraction():
    slugs = [f"s{i:03d}" for i in range(100)]
    train, val = split.split_slugs(slugs, val_frac=0.1, seed=0)
    assert len(train) + len(val) == 100
    assert len(val) == 10
    # No overlap
    assert set(train).isdisjoint(set(val))


def test_split_slugs_handles_tiny_set():
    slugs = ["only"]
    train, val = split.split_slugs(slugs, val_frac=0.1, seed=0)
    # With one slug, val_frac=0.1 should still send it somewhere; we want at
    # least 1 in val to avoid an empty eval set, so train ends up empty.
    assert (len(train), len(val)) == (0, 1)


def test_write_and_read_splits_round_trip(tmp_path):
    train = ["a", "b", "c"]
    val = ["d"]
    split.write_splits(tmp_path, train, val)
    assert split.read_split(tmp_path / "train.txt") == train
    assert split.read_split(tmp_path / "val.txt") == val


def test_write_splits_handles_empty(tmp_path):
    split.write_splits(tmp_path, [], [])
    assert (tmp_path / "train.txt").read_text() == ""
    assert (tmp_path / "val.txt").read_text() == ""
