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


def test_train_smoke_bce_loss_writes_apply_sigmoid_false(tmp_path: Path):
    """--loss bce should train Generator(sigmoid=False) and stamp ckpt metadata."""
    covers = tmp_path / "covers"; covers.mkdir()
    labels = tmp_path / "labels"; labels.mkdir()
    splits = tmp_path / "splits"; splits.mkdir()
    runs = tmp_path / "runs"

    slugs = [f"s{i}" for i in range(6)]
    for s in slugs:
        Image.new("RGB", (64, 64), (40, 80, 160)).save(covers / f"{s}.jpg")
        lbl = Image.new("L", (64, 64), 255)
        for x in range(64):
            lbl.putpixel((x, 32), 0)
        lbl.save(labels / f"{s}.png")
    split_mod.write_splits(splits, slugs[:4], slugs[4:])

    cfg = TrainConfig(
        splits_dir=str(splits),
        covers_dir=str(covers),
        labels_dir=str(labels),
        out_dir=str(runs / "smoke-bce"),
        img_size=64,
        resize_short_to=72,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        num_workers=0,
        use_lora=False,
        n_residual_blocks=1,
        perceptual_weight=0.0,    # avoid VGG download
        use_vgg_pretrained=False,
        edge_weight=19.0,
        seed=0,
        loss_type="bce",
        bce_weight=1.0,
    )
    summary = train(cfg)
    ckpt_path = runs / "smoke-bce" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = runs / "smoke-bce" / "last.pt"
    assert ckpt_path.exists()

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert ckpt.get("apply_sigmoid") is False
    assert ckpt.get("loss_type") == "bce"
    assert "best_val_total" in summary


def test_train_smoke_bce_no_lora_pretrained_warmstart(tmp_path: Path, capsys):
    """Plan E combo: --loss bce + --no-lora + --pretrained-ckpt.

    Builds a tiny Generator(sigmoid=True), saves only its model0.* weights
    as a partial pretrained ckpt (so model1..model4 are guaranteed missing
    on load), then trains BCE + no-LoRA from that warm-start. Verifies:
    1) the run completes and ckpt metadata is correct,
    2) the [pretrained] diagnostic includes key NAMES, not just counts,
       so we can see which layers warm-started vs random-inited.
    """
    from albumify.model import Generator

    covers = tmp_path / "covers"; covers.mkdir()
    labels = tmp_path / "labels"; labels.mkdir()
    splits = tmp_path / "splits"; splits.mkdir()
    runs = tmp_path / "runs"

    slugs = [f"s{i}" for i in range(6)]
    for s in slugs:
        Image.new("RGB", (64, 64), (40, 80, 160)).save(covers / f"{s}.jpg")
        lbl = Image.new("L", (64, 64), 255)
        for x in range(64):
            lbl.putpixel((x, 32), 0)
        lbl.save(labels / f"{s}.png")
    split_mod.write_splits(splits, slugs[:4], slugs[4:])

    # Build a tiny sigmoid=True generator and save only its model0.* keys
    # to simulate a partial upstream ckpt — guarantees missing-key set on load.
    src = Generator(n_residual_blocks=1, ngf=8, sigmoid=True)
    partial_state = {k: v for k, v in src.state_dict().items() if k.startswith("model0.")}
    pretrained_path = tmp_path / "partial-pretrained.pt"
    torch.save(partial_state, pretrained_path)

    cfg = TrainConfig(
        splits_dir=str(splits),
        covers_dir=str(covers),
        labels_dir=str(labels),
        pretrained_ckpt=str(pretrained_path),
        out_dir=str(runs / "smoke-bce-warm"),
        img_size=64,
        resize_short_to=72,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        num_workers=0,
        use_lora=False,
        n_residual_blocks=1,
        ngf=8,
        perceptual_weight=0.0,
        use_vgg_pretrained=False,
        edge_weight=19.0,
        seed=0,
        loss_type="bce",
        bce_weight=1.0,
    )
    summary = train(cfg)

    # 1) ckpt was saved and metadata correct
    ckpt_path = runs / "smoke-bce-warm" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = runs / "smoke-bce-warm" / "last.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert ckpt.get("apply_sigmoid") is False
    assert ckpt.get("loss_type") == "bce"
    assert "best_val_total" in summary

    # 2) the [pretrained] diagnostic includes at least one missing key NAME
    captured = capsys.readouterr().out
    assert "[pretrained] missing=" in captured, captured
    # When any keys are missing, the next line must include actual key names
    assert "[pretrained] missing keys" in captured, (
        "Expected enhanced diagnostic showing missing key names, got:\n" + captured
    )
    # At least one of the model{1,2,3,4} families must appear since we only saved model0.*
    assert any(f"model{i}." in captured for i in (1, 2, 3, 4)), (
        "Expected at least one model1./model2./model3./model4. key name in diagnostic, got:\n"
        + captured
    )
