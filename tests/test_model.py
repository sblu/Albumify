"""Tests for the Generator + LoRA-Conv plumbing. Requires torch."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

import torch.nn as nn

from albumify.lora import (
    LoRAConv2d,
    count_lora_params,
    freeze_non_lora,
    lora_parameters,
    merge_all_lora,
    wrap_conv2d_layers,
)
from albumify.model import Generator, ResidualBlock


# ---- Architecture ---------------------------------------------------------

def test_generator_output_shape_matches_input_size():
    g = Generator()
    g.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = g(x)
    assert y.shape == (1, 1, 64, 64)
    assert (y >= 0).all() and (y <= 1).all()  # sigmoid


def test_generator_with_sigmoid_false_produces_real_logits():
    """sigmoid=False is the Plan C training mode — output should not be clamped to [0,1]."""
    torch.manual_seed(42)
    g = Generator(sigmoid=False)
    g.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = g(x)
    assert y.shape == (1, 1, 64, 64)
    assert (y < 0).any() or (y > 1).any(), \
        f"sigmoid=False output should not be in [0,1]; got min={y.min()}, max={y.max()}"


def test_generator_param_count_in_expected_range():
    """Sanity check: ~4.4M params at defaults."""
    g = Generator()
    n = sum(p.numel() for p in g.parameters())
    assert 4_000_000 < n < 5_000_000, n


def test_residual_block_preserves_shape():
    rb = ResidualBlock(32)
    x = torch.randn(2, 32, 16, 16)
    y = rb(x)
    assert y.shape == x.shape


# ---- LoRA -----------------------------------------------------------------

def test_lora_conv2d_initially_acts_as_identity_over_base():
    """B is zero-init, so wrapped output must equal base output exactly."""
    base = nn.Conv2d(3, 8, 3, padding=1)
    lora = LoRAConv2d(base, rank=4, alpha=4.0)
    x = torch.randn(1, 3, 16, 16)
    base_out = base(x)
    lora_out = lora(x)
    assert torch.allclose(base_out, lora_out, atol=1e-6)


def test_lora_conv2d_freezes_base_params():
    base = nn.Conv2d(3, 8, 3, padding=1)
    lora = LoRAConv2d(base, rank=4, alpha=4.0)
    for p in lora.base.parameters():
        assert not p.requires_grad
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad


def test_lora_conv2d_changes_output_after_B_is_perturbed():
    base = nn.Conv2d(3, 8, 3, padding=1)
    lora = LoRAConv2d(base, rank=4, alpha=4.0)
    nn.init.normal_(lora.lora_B.weight, std=0.1)
    x = torch.randn(1, 3, 16, 16)
    assert not torch.allclose(base(x), lora(x))


def test_lora_conv2d_merge_matches_forward():
    """After merge, the plain Conv2d output equals the LoRA forward output."""
    base = nn.Conv2d(3, 8, 3, padding=1)
    lora = LoRAConv2d(base, rank=4, alpha=4.0)
    nn.init.normal_(lora.lora_B.weight, std=0.1)
    x = torch.randn(1, 3, 16, 16)
    expected = lora(x)
    merged = lora.merge_into_base()
    got = merged(x)
    assert torch.allclose(expected, got, atol=1e-5)


def test_wrap_conv2d_layers_replaces_all_conv2d_in_generator():
    g = Generator(n_residual_blocks=2)  # smaller for speed
    base_convs = sum(1 for m in g.modules() if isinstance(m, nn.Conv2d))
    n_wrapped = wrap_conv2d_layers(g, rank=4, alpha=4.0)
    after_lora = sum(1 for m in g.modules() if isinstance(m, LoRAConv2d))
    after_plain = sum(
        1 for m in g.modules()
        if isinstance(m, nn.Conv2d) and not isinstance(m, LoRAConv2d)
    )
    # Every original Conv2d is now wrapped; plain Conv2d count drops to the
    # inner-base count (each LoRAConv2d still owns a base Conv2d).
    assert n_wrapped == base_convs
    assert after_lora == base_convs
    # The "plain Conv2d" count is the inner base convs (one per wrapper) plus
    # the lora_A and lora_B convs (two per wrapper).
    assert after_plain == base_convs * 3


def test_wrap_skips_excluded_kernel_sizes():
    g = Generator(n_residual_blocks=1)
    # Count 7x7 convs (initial + output).
    n_7x7 = sum(
        1 for m in g.modules()
        if isinstance(m, nn.Conv2d) and (m.kernel_size == (7, 7))
    )
    assert n_7x7 == 2
    wrapped = wrap_conv2d_layers(g, rank=4, alpha=4.0, skip_kernel_sizes=(7,))
    # Wrapped count should exclude the 2 7x7 convs.
    total = sum(1 for m in g.modules() if isinstance(m, nn.Conv2d))
    # 'total' here counts inner-base + 2 unwrapped 7x7 + lora_A/B for each wrapped.
    assert wrapped == (sum(
        1 for m in g.modules() if isinstance(m, LoRAConv2d)
    ))


def test_freeze_non_lora_only_lora_trainable():
    g = Generator(n_residual_blocks=2)
    wrap_conv2d_layers(g, rank=4, alpha=4.0)
    freeze_non_lora(g)
    trainable = [p for p in g.parameters() if p.requires_grad]
    lora_ps = list(lora_parameters(g))
    assert len(trainable) == len(lora_ps)
    assert all(any(t is l for l in lora_ps) for t in trainable)


def test_merge_all_lora_preserves_forward():
    g = Generator(n_residual_blocks=2)
    g.eval()
    wrap_conv2d_layers(g, rank=4, alpha=4.0)
    # Perturb LoRA weights so merge has a non-trivial delta to bake in.
    for m in g.modules():
        if isinstance(m, LoRAConv2d):
            nn.init.normal_(m.lora_B.weight, std=0.05)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        before = g(x).clone()
    n_merged = merge_all_lora(g)
    assert n_merged > 0
    with torch.no_grad():
        after = g(x)
    assert torch.allclose(before, after, atol=1e-4)


def test_count_lora_params_reasonable():
    g = Generator(n_residual_blocks=9)
    wrap_conv2d_layers(g, rank=8, alpha=8.0)
    n = count_lora_params(g)
    # Should be much smaller than the full model's params (~4.4M).
    total = sum(p.numel() for p in g.parameters())
    assert n < total
    # And within a sensible LoRA budget (rank 8 over ~22 Conv2d layers
    # comes out around 400-500k).
    assert 100_000 < n < 1_000_000, n
