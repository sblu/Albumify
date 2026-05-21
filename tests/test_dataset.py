"""Tests for AlbumDataset. Skipped when torch isn't installed (local dev)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from pathlib import Path

from PIL import Image

from albumify import split
from albumify.dataset import AlbumDataset
from albumify.transforms import PairedTransformConfig


def _setup(tmp_path: Path, n: int = 4) -> tuple[Path, Path, Path]:
    covers = tmp_path / "covers"; covers.mkdir()
    labels = tmp_path / "labels"; labels.mkdir()
    splits = tmp_path / "splits"; splits.mkdir()
    for i in range(n):
        slug = f"s{i:02d}"
        Image.new("RGB", (400, 400), (i * 30, 50, 200)).save(covers / f"{slug}.jpg")
        Image.new("L", (400, 400), 255).save(labels / f"{slug}.png")
    train_slugs = [f"s{i:02d}" for i in range(n - 1)]
    val_slugs = [f"s{n - 1:02d}"]
    split.write_splits(splits, train_slugs, val_slugs)
    return covers, labels, splits


def test_album_dataset_shapes_and_ranges(tmp_path):
    covers, labels, splits = _setup(tmp_path)
    cfg = PairedTransformConfig(out_size=128, resize_short_to=160)
    ds = AlbumDataset(splits / "train.txt", covers, labels, train=True, cfg=cfg)
    cover, label, slug = ds[0]
    assert cover.shape == (3, 128, 128)
    assert label.shape == (1, 128, 128)
    assert cover.dtype == torch.float32 and label.dtype == torch.float32
    assert 0.0 <= float(cover.min()) <= float(cover.max()) <= 1.0
    assert 0.0 <= float(label.min()) <= float(label.max()) <= 1.0
    assert slug == "s00"


def test_album_dataset_eval_is_deterministic(tmp_path):
    covers, labels, splits = _setup(tmp_path)
    cfg = PairedTransformConfig(out_size=128, resize_short_to=160)
    ds = AlbumDataset(splits / "val.txt", covers, labels, train=False, cfg=cfg)
    c1, l1, _ = ds[0]
    c2, l2, _ = ds[0]
    assert torch.equal(c1, c2)
    assert torch.equal(l1, l2)


def test_album_dataset_len_matches_split(tmp_path):
    covers, labels, splits = _setup(tmp_path, n=5)
    ds = AlbumDataset(splits / "train.txt", covers, labels, train=True)
    assert len(ds) == 4
