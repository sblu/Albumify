"""Tests for GeomDepthLoss (Plan F3).

Same two-layer pattern as test_clip_loss.py: stub-injected tests always
run (exercise the cache lookup + tensor pipeline), while real-network
tests skip until the upstream feats2Geom checkpoint is on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch.nn as nn

from albumify.geom_loss import GeomDepthLoss


class _StubFeatExtractor(nn.Module):
    """Stand-in for InceptionV3.Mixed_6b feature extractor.

    Takes [B, 3, H, W], returns [B, 768, 17, 17] so downstream G_Geom
    sees the real input shape.
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return nn.functional.adaptive_avg_pool2d(h, (17, 17))


class _StubG_Geom(nn.Module):
    """Stand-in for the features-to-depth network.

    Takes [B, 768, 17, 17], returns [B, 1, 128, 128] Tanh — same
    "deeper-than-input" spatial output as the real G_Geom (which goes
    17 -> ~1088). The exact size doesn't matter: GeomDepthLoss resamples
    to match cached GT shape before computing L1.
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(768, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = nn.functional.interpolate(h, size=(128, 128), mode="bilinear", align_corners=False)
        return torch.tanh(h)


def _write_depth_cache(out_dir: Path, slugs: list[str], shape=(256, 256)) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in slugs:
        np.save(out_dir / f"{s}.npy", np.random.RandomState(hash(s) & 0xffffffff).rand(*shape).astype(np.float16))


def test_geom_loss_returns_scalar_with_grad(tmp_path):
    slugs = ["a", "b"]
    _write_depth_cache(tmp_path / "depth", slugs)
    loss_mod = GeomDepthLoss(
        feats2depth=_StubG_Geom(),
        feature_extractor=_StubFeatExtractor(),
        depth_cache_dir=tmp_path / "depth",
    )
    pred = torch.rand(2, 1, 256, 256, requires_grad=True)
    out = loss_mod(pred, slugs)
    assert out.ndim == 0, f"expected scalar, got shape {tuple(out.shape)}"
    assert out.requires_grad
    out.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


def test_geom_loss_keeps_feats2depth_and_features_frozen(tmp_path):
    slugs = ["a"]
    _write_depth_cache(tmp_path / "depth", slugs)
    g_geom = _StubG_Geom()
    feat = _StubFeatExtractor()
    loss_mod = GeomDepthLoss(
        feats2depth=g_geom, feature_extractor=feat, depth_cache_dir=tmp_path / "depth",
    )
    for n, p in loss_mod.feats2depth.named_parameters():
        assert not p.requires_grad, f"feats2depth/{n} should be frozen"
    for n, p in loss_mod.feature_extractor.named_parameters():
        assert not p.requires_grad, f"feature_extractor/{n} should be frozen"


def test_geom_loss_missing_slug_raises_informative_error(tmp_path):
    _write_depth_cache(tmp_path / "depth", ["a"])
    loss_mod = GeomDepthLoss(
        feats2depth=_StubG_Geom(),
        feature_extractor=_StubFeatExtractor(),
        depth_cache_dir=tmp_path / "depth",
    )
    pred = torch.rand(1, 1, 256, 256)
    with pytest.raises(KeyError) as exc:
        loss_mod(pred, ["does_not_exist"])
    assert "does_not_exist" in str(exc.value), f"error should mention missing slug, got {exc.value}"


def test_geom_loss_batch_lookup_matches_per_item(tmp_path):
    """L1 loss on a batch should equal mean of per-item L1 losses (the
    per-batch cache lookup must respect order).
    """
    slugs = ["alpha", "beta", "gamma"]
    _write_depth_cache(tmp_path / "depth", slugs)
    loss_mod = GeomDepthLoss(
        feats2depth=_StubG_Geom(),
        feature_extractor=_StubFeatExtractor(),
        depth_cache_dir=tmp_path / "depth",
    )
    torch.manual_seed(0)
    pred = torch.rand(3, 1, 256, 256)
    batch_loss = loss_mod(pred, slugs).item()
    per_item_losses = [loss_mod(pred[i:i+1], [slugs[i]]).item() for i in range(3)]
    assert pytest.approx(batch_loss, rel=1e-4) == sum(per_item_losses) / 3, (
        f"batch loss {batch_loss} != mean of per-item losses {per_item_losses}"
    )


def test_geom_loss_preloads_depth_cache_eagerly(tmp_path):
    """Constructor reads every .npy under depth_cache_dir into memory.

    Deleting the files after construction should not break forward — proving
    the cache is in RAM, not lazy-read from disk.
    """
    slugs = ["x", "y"]
    cache_dir = tmp_path / "depth"
    _write_depth_cache(cache_dir, slugs)
    loss_mod = GeomDepthLoss(
        feats2depth=_StubG_Geom(),
        feature_extractor=_StubFeatExtractor(),
        depth_cache_dir=cache_dir,
    )
    # Delete the cache files: if the loss is reading lazily, it will fail.
    for s in slugs:
        (cache_dir / f"{s}.npy").unlink()
    pred = torch.rand(2, 1, 256, 256)
    _ = loss_mod(pred, slugs)  # must not raise


def test_geom_loss_resamples_prediction_to_cached_gt_shape(tmp_path):
    """G_Geom returns 128x128 in the stub but cached GT is 256x256;
    the loss must resample one to match the other before L1.
    """
    slugs = ["a"]
    _write_depth_cache(tmp_path / "depth", slugs, shape=(256, 256))
    loss_mod = GeomDepthLoss(
        feats2depth=_StubG_Geom(),  # outputs 128x128
        feature_extractor=_StubFeatExtractor(),
        depth_cache_dir=tmp_path / "depth",
    )
    pred = torch.rand(1, 1, 256, 256)
    # Just verifying it does not raise from a shape mismatch.
    out = loss_mod(pred, slugs)
    assert torch.isfinite(out)


def test_geom_loss_empty_cache_dir_raises_at_construction(tmp_path):
    """Pre-load failing fast at construction is better than a cryptic KeyError
    on the first forward.
    """
    empty = tmp_path / "empty_depth"
    empty.mkdir()
    with pytest.raises((FileNotFoundError, ValueError)) as exc:
        GeomDepthLoss(
            feats2depth=_StubG_Geom(),
            feature_extractor=_StubFeatExtractor(),
            depth_cache_dir=empty,
        )
    msg = str(exc.value)
    assert "depth" in msg.lower() or "empty" in msg.lower() or str(empty) in msg


# ---- Real-checkpoint smoke (skipped unless feats2depth ckpt is present) ----


def test_real_load_feats2depth_strict_when_ckpt_present():
    from albumify.feats2depth import load_feats2depth
    ckpt = Path("artifacts/feats2Geom/feats2depth.pth")
    if not ckpt.exists():
        pytest.skip(f"upstream feats2Geom checkpoint not at {ckpt}")
    m = load_feats2depth(ckpt)
    assert sum(p.numel() for p in m.parameters()) > 1_000_000, "model suspiciously small"
