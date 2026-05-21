"""End-to-end smoke test for the training loop. Requires torch + torchvision."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402
pytest.importorskip("torchvision")    # noqa: E402

from pathlib import Path

from PIL import Image

from albumify import split as split_mod
from albumify.train import TrainConfig, train


def test_train_smoke_runs_two_epochs_and_writes_ckpt(tmp_path: Path):
    covers = tmp_path / "covers"; covers.mkdir()
    labels = tmp_path / "labels"; labels.mkdir()
    splits = tmp_path / "splits"; splits.mkdir()
    runs = tmp_path / "runs"

    # 4 train + 2 val tiny imgs
    slugs = [f"s{i}" for i in range(6)]
    for s in slugs:
        Image.new("RGB", (64, 64), (40, 80, 160)).save(covers / f"{s}.jpg")
        # Pseudo-label: a horizontal line in the middle
        lbl = Image.new("L", (64, 64), 255)
        for x in range(64):
            lbl.putpixel((x, 32), 0)
        lbl.save(labels / f"{s}.png")
    split_mod.write_splits(splits, slugs[:4], slugs[4:])

    cfg = TrainConfig(
        splits_dir=str(splits),
        covers_dir=str(covers),
        labels_dir=str(labels),
        out_dir=str(runs / "smoke"),
        img_size=64,
        resize_short_to=72,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        num_workers=0,           # avoid worker spawn in CI
        lora_rank=2,
        lora_alpha=2.0,
        perceptual_weight=0.0,   # avoid VGG download in tests
        use_vgg_pretrained=False,
        n_residual_blocks=1,     # tiny model
        seed=0,
    )
    summary = train(cfg)
    assert (runs / "smoke" / "best.pt").exists() or (runs / "smoke" / "last.pt").exists()
    assert (runs / "smoke" / "config.json").exists()
    assert (runs / "smoke" / "summary.json").exists()
    assert "best_val_total" in summary
