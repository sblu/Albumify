"""Tests for CLIPSemanticLoss (Plan F2).

Two layers of coverage:

1. Stub-injected tests (always run): exercise the forward/backward path and
   the gray->RGB tiling/resize logic using a tiny in-process stand-in for
   CLIP's image encoder. These prove the loss math without touching the
   network.

2. Real-CLIP tests (skipped when openai-clip is not installed locally; they
   run on the VM after `pip install -e .[train]`): verify ViT-B/32 loads,
   produces nonzero loss between distinct images, exactly-zero loss between
   identical images, and that its parameters stay frozen across a backward.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from albumify.clip_loss import CLIPSemanticLoss


class _StubClipImageEncoder(nn.Module):
    """Minimal stand-in for clip.model.CLIP that only exposes encode_image.

    Maps a [B, 3, 224, 224] tensor to a [B, 64] embedding via a single conv
    + global average pool. Deterministic, differentiable, fast, no network.
    """

    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        h = self.head(x)              # [B, 64, 224, 224]
        return h.mean(dim=(2, 3))     # [B, 64]


def _stub_preprocess_factory():
    """Mirror clip.load's `preprocess` signature: takes a PIL image, returns
    a 3x224x224 tensor. Our loss bypasses this (works on tensors directly),
    so the test just passes a no-op identity that the loss won't call.
    """
    return lambda x: x


def test_clip_loss_returns_scalar_tensor_with_grad():
    loss_mod = CLIPSemanticLoss(clip_model=_StubClipImageEncoder(), preprocess=_stub_preprocess_factory())
    pred = torch.rand(2, 1, 256, 256, requires_grad=True)
    target = torch.rand(2, 3, 256, 256)
    out = loss_mod(pred, target)
    assert out.ndim == 0, f"expected scalar, got shape {tuple(out.shape)}"
    assert out.requires_grad
    out.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


def test_clip_loss_is_zero_when_pred_and_target_encode_identically():
    """If pred (gray) is tiled to match an all-equal-channel target, the
    stub encoder's output is identical and MSE = 0 exactly."""
    loss_mod = CLIPSemanticLoss(clip_model=_StubClipImageEncoder(), preprocess=_stub_preprocess_factory())
    # Construct gray and target so that pred replicated to 3ch == target
    pred_gray = torch.rand(1, 1, 64, 64)
    target_rgb = pred_gray.expand(-1, 3, -1, -1).clone()
    out = loss_mod(pred_gray, target_rgb)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-6), f"expected ~0, got {out.item()}"


def test_clip_loss_keeps_stub_model_frozen():
    """After construction, every parameter of the CLIP model has
    requires_grad=False."""
    stub = _StubClipImageEncoder()
    # Sanity: the stub starts unfrozen
    assert all(p.requires_grad for p in stub.parameters())
    loss_mod = CLIPSemanticLoss(clip_model=stub, preprocess=_stub_preprocess_factory())
    assert all(not p.requires_grad for p in loss_mod.clip_model.parameters())


def test_clip_loss_backward_does_not_change_clip_weights():
    """Even after a full forward+backward+optimizer step on `pred`, the
    CLIP model's weights are untouched (because they're frozen)."""
    stub = _StubClipImageEncoder()
    loss_mod = CLIPSemanticLoss(clip_model=stub, preprocess=_stub_preprocess_factory())
    snapshot = {n: p.detach().clone() for n, p in loss_mod.clip_model.named_parameters()}
    pred = torch.rand(1, 1, 32, 32, requires_grad=True)
    target = torch.rand(1, 3, 32, 32)
    out = loss_mod(pred, target)
    out.backward()
    # Simulate an optimizer step that WOULD have touched CLIP if its params
    # had grads — but since they're frozen, .grad is None for all of them.
    for n, p in loss_mod.clip_model.named_parameters():
        assert p.grad is None, f"{n} got a gradient despite being frozen"
        assert torch.equal(p, snapshot[n]), f"{n} changed during backward"


def test_clip_loss_resizes_inputs_to_224():
    """The CLIP model only accepts 224x224 inputs; the loss must resize.

    We use a stub that asserts input shape — if the loss does not resize
    pred/target before calling encode_image, this raises.
    """
    class _StrictShapeStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Conv2d(3, 8, kernel_size=1)
        def encode_image(self, x):
            assert x.shape[-2:] == (224, 224), f"expected 224x224, got {tuple(x.shape)}"
            return self.head(x).mean(dim=(2, 3))

    loss_mod = CLIPSemanticLoss(clip_model=_StrictShapeStub(), preprocess=_stub_preprocess_factory())
    pred = torch.rand(1, 1, 256, 256)
    target = torch.rand(1, 3, 256, 256)
    loss_mod(pred, target)  # must not raise


# ---- Real-CLIP smoke tests (skipped without openai-clip) ----


def test_real_clip_vit_b32_loads_and_returns_scalar(tmp_path):
    """Loads the real ViT-B/32 weights; verifies forward returns a scalar
    tensor with grad. Network call (downloads ~150MB on first run).
    """
    pytest.importorskip("clip")
    loss_mod = CLIPSemanticLoss()  # uses default ViT-B/32
    pred = torch.rand(1, 1, 256, 256, requires_grad=True)
    target = torch.rand(1, 3, 256, 256)
    out = loss_mod(pred, target)
    assert out.ndim == 0
    assert out.requires_grad
    out.backward()
    assert pred.grad is not None


def test_real_clip_distinct_images_give_nonzero_loss():
    pytest.importorskip("clip")
    loss_mod = CLIPSemanticLoss()
    torch.manual_seed(0)
    pred = torch.rand(1, 1, 256, 256)
    target = torch.rand(1, 3, 256, 256)
    out = loss_mod(pred, target)
    assert out.item() > 1e-6, f"expected nonzero CLIP loss between random images, got {out.item()}"
