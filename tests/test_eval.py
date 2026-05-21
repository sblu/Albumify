"""Tests for the pure-numpy parts of eval.py. The torch run_eval path is
covered by manual + VM testing."""
from __future__ import annotations

import numpy as np
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
