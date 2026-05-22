"""Tests for the BCE-with-logits + perceptual loss path (Plan C)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

import torch.nn.functional as F

from albumify.loss import (
    BCELogitsPerceptualLoss,
    VGGPerceptualLoss,
    edge_weighted_bce_logits,
)


# ---- edge_weighted_bce_logits ---------------------------------------------

def test_bce_logits_returns_finite_scalar_with_gradient():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 1, 8, 8)
    loss = edge_weighted_bce_logits(logits, target, edge_weight=0.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_bce_logits_with_edge_weight_0_matches_plain_bce():
    """edge_weight=0 should equal F.binary_cross_entropy_with_logits on the binarized target."""
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 16, 16)
    target = torch.rand(1, 1, 16, 16)
    target_bin = (target >= 0.5).float()

    ours = edge_weighted_bce_logits(logits, target, edge_threshold=0.5, edge_weight=0.0)
    ref = F.binary_cross_entropy_with_logits(logits, target_bin)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_bce_logits_edge_weight_increases_loss_when_edge_present():
    """Adding edge_weight on a target with edge pixels must produce a larger loss."""
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 16, 16)
    target = torch.zeros(1, 1, 16, 16)
    target[..., :2, :] = 0.0    # row of edges (dark)
    target[..., 2:, :] = 1.0    # background (bright)

    base = edge_weighted_bce_logits(logits, target, edge_weight=0.0)
    weighted = edge_weighted_bce_logits(logits, target, edge_weight=19.0)
    assert weighted.item() > base.item()


# ---- BCELogitsPerceptualLoss ---------------------------------------------

def test_bce_logits_perceptual_loss_no_vgg_returns_bce_only():
    loss_fn = BCELogitsPerceptualLoss(
        bce_weight=1.0, perceptual_weight=0.0, edge_weight=0.0,
    )
    logits = torch.randn(1, 1, 8, 8)
    target = torch.rand(1, 1, 8, 8)
    res = loss_fn(logits, target)
    assert set(res.keys()) == {"bce", "total"}
    assert torch.allclose(res["total"], res["bce"])


def test_bce_logits_perceptual_loss_requires_vgg_when_weight_positive():
    with pytest.raises(ValueError):
        BCELogitsPerceptualLoss(perceptual_weight=0.1, vgg=None)


def test_bce_logits_perceptual_loss_with_random_vgg_combines_terms():
    """Random-init VGG just for shape; check 'total' = bce + 0.1 * perc."""
    import torchvision.models as tv_models

    vgg_features = tv_models.vgg16(weights=None).features
    vgg = VGGPerceptualLoss(vgg_features)
    loss_fn = BCELogitsPerceptualLoss(
        bce_weight=1.0, perceptual_weight=0.1,
        edge_weight=0.0, vgg=vgg,
    )
    logits = torch.randn(1, 1, 32, 32)
    target = torch.rand(1, 1, 32, 32)
    res = loss_fn(logits, target)
    assert {"bce", "perc", "total"} <= set(res.keys())
    expected_total = 1.0 * res["bce"] + 0.1 * res["perc"]
    assert torch.allclose(res["total"], expected_total, atol=1e-6)
