"""Tests for the pure-numpy parts of eval.py + the torch run_eval path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from albumify.eval import edge_f1, render_grid, render_triplet_row, ssim_pair


def test_edge_f1_perfect_match_is_one():
    arr = np.ones((16, 16), dtype=np.float32)
    arr[4:12, 4:12] = 0.0  # a black square = "edge"
    m = edge_f1(arr, arr, edge_threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_edge_f1_total_mismatch_is_zero():
    pred = np.ones((16, 16), dtype=np.float32)
    pred[4:12, 4:12] = 0.0
    target = np.ones((16, 16), dtype=np.float32)
    target[0:4, 0:4] = 0.0
    m = edge_f1(pred, target, edge_threshold=0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_edge_f1_partial_overlap_between_zero_and_one():
    pred = np.ones((10, 10), dtype=np.float32)
    target = np.ones((10, 10), dtype=np.float32)
    pred[:, 0:6] = 0.0   # predicted left 6 cols as edges
    target[:, 0:4] = 0.0  # actually only left 4 cols are edges
    m = edge_f1(pred, target, edge_threshold=0.5)
    # tp = 40, fp = 20, fn = 0 → precision=2/3, recall=1, f1=4/5
    assert m["precision"] == 40 / 60
    assert m["recall"] == 1.0
    assert abs(m["f1"] - (2 * (40 / 60) * 1.0 / ((40 / 60) + 1.0))) < 1e-6


def test_ssim_pair_identity_is_high():
    arr = np.random.RandomState(0).rand(64, 64).astype(np.float32)
    val = ssim_pair(arr, arr)
    assert val >= 0.99


def test_ssim_pair_different_is_low():
    a = np.zeros((64, 64), dtype=np.float32)
    b = np.ones((64, 64), dtype=np.float32)
    val = ssim_pair(a, b)
    assert val < 0.5


def test_render_triplet_row_dims_match():
    cov = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)
    tgt = np.zeros((64, 64), dtype=np.float32)
    prd = np.zeros((64, 64), dtype=np.float32)
    img = render_triplet_row(cov, tgt, prd)
    assert img.size == (64 * 3, 64)
    assert img.mode == "RGB"


def test_render_triplet_row_resizes_cover_to_match():
    cov = np.random.RandomState(0).rand(128, 128, 3).astype(np.float32)
    tgt = np.zeros((64, 64), dtype=np.float32)
    prd = np.zeros((64, 64), dtype=np.float32)
    img = render_triplet_row(cov, tgt, prd)
    assert img.size == (64 * 3, 64)


def test_render_grid_stacks_rows():
    rows = [Image.new("RGB", (100, 50)) for _ in range(3)]
    g = render_grid(rows)
    assert g.size == (100, 50 * 3 + 2 * 2)


def test_render_grid_empty_returns_pixel():
    g = render_grid([])
    assert g.size == (1, 1)


def test_eval_loads_apply_sigmoid_false_ckpt_and_wraps_externally(tmp_path: Path):
    """Build a tiny no-sigmoid ckpt, run eval, assert metrics are finite + in range.

    If the sigmoid wrap weren't applied, the renderer would see raw logits and
    SSIM/edge-F1 would be NaN or out-of-range.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from albumify.eval import EvalConfig, run_eval
    from albumify.model import Generator

    covers = tmp_path / "covers"; covers.mkdir()
    labels = tmp_path / "labels"; labels.mkdir()
    splits = tmp_path / "splits"; splits.mkdir()
    out_dir = tmp_path / "eval-out"

    slugs = [f"v{i}" for i in range(2)]
    for s in slugs:
        Image.new("RGB", (64, 64), (40, 80, 160)).save(covers / f"{s}.jpg")
        lbl = Image.new("L", (64, 64), 255)
        for x in range(64):
            lbl.putpixel((x, 32), 0)
        lbl.save(labels / f"{s}.png")
    (splits / "val.txt").write_text("\n".join(slugs) + "\n")
    (splits / "train.txt").write_text("")

    g = Generator(n_residual_blocks=1, ngf=8, sigmoid=False)
    ckpt_path = tmp_path / "noprep.pt"
    torch.save({
        "model_state_dict": g.state_dict(),
        "epoch": 1,
        "val_total": 0.0,
        "apply_sigmoid": False,
        "loss_type": "bce",
    }, ckpt_path)

    cfg = EvalConfig(
        splits_dir=str(splits), covers_dir=str(covers), labels_dir=str(labels),
        ckpt_path=str(ckpt_path), out_dir=str(out_dir),
        img_size=64, n_residual_blocks=1, ngf=8,
        use_lora=False,
        use_lpips=False,
        grid_n=2,
    )
    summary = run_eval(cfg)
    assert np.isfinite(summary["ssim"])
    assert 0.0 <= summary["f1"] <= 1.0
    assert 0.0 <= summary["precision"] <= 1.0
    assert 0.0 <= summary["recall"] <= 1.0
