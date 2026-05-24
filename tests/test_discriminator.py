"""Tests for PatchGAN70 + LSGAN helpers (Plan F4).

Stub-free: discriminator is tiny enough to build for real in every test.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from albumify.discriminator import PatchGAN70, lsgan_d_loss, lsgan_g_loss


def test_patchgan70_output_shape_for_256_input():
    """A 70x70 PatchGAN's feature map should be ~H/16 x W/16 for stride
    2,2,2,1 layers with kernel 4.
    """
    d = PatchGAN70(in_ch=1)
    x = torch.randn(2, 1, 256, 256)
    y = d(x)
    assert y.dim() == 4
    assert y.size(0) == 2
    assert y.size(1) == 1
    # 256 -> 128 -> 64 -> 32 -> 30 (stride 1, kernel 4) -> 29 (stride 1, kernel 4)
    # Exact size depends on padding choices; assert it's substantially smaller.
    assert 8 <= y.size(-1) <= 40, f"unexpected patch size {y.size(-1)}"


def test_patchgan70_propagates_gradient_to_input():
    d = PatchGAN70(in_ch=1)
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    y = d(x).mean()
    y.backward()
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x))


def test_lsgan_d_loss_is_zero_at_perfect_predictions():
    """D loss = 0 when D(real)=1 and D(fake)=0 everywhere."""
    d_real = torch.ones(1, 1, 30, 30)
    d_fake = torch.zeros(1, 1, 30, 30)
    loss = lsgan_d_loss(d_real, d_fake)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_lsgan_d_loss_is_one_at_swapped_predictions():
    """((0-1)^2 + 1^2) / 2 = 1.0 when D swaps real/fake calls."""
    d_real = torch.zeros(1, 1, 30, 30)
    d_fake = torch.ones(1, 1, 30, 30)
    loss = lsgan_d_loss(d_real, d_fake)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_lsgan_g_loss_is_zero_when_d_fooled():
    """G loss = 0 when D(fake) = 1 (G has perfectly fooled D)."""
    d_fake = torch.ones(1, 1, 30, 30)
    loss = lsgan_g_loss(d_fake)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_lsgan_g_loss_propagates_gradient():
    d_fake = torch.zeros(1, 1, 30, 30, requires_grad=True)
    loss = lsgan_g_loss(d_fake)
    loss.backward()
    assert d_fake.grad is not None
    # ∂/∂x (x - 1)^2 = 2(x - 1); at x=0 grad is -2/N for each element
    assert (d_fake.grad < 0).all(), "expected negative gradient pushing toward 1"


def test_patchgan70_3channel_input_also_works():
    """Accept 1ch (drawings) or 3ch (photos) via in_ch param."""
    d = PatchGAN70(in_ch=3)
    x = torch.randn(1, 3, 256, 256)
    y = d(x)
    assert y.size(0) == 1 and y.size(1) == 1
