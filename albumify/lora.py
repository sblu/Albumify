"""LoRA-Conv adapters for nn.Conv2d.

Standard LoRA: `y = base(x) + (alpha/rank) * B(A(x))` where A is a 1x1 conv
into a rank-dim bottleneck and B is the same kernel/stride/padding as `base`.
A is kaiming-init, B is zero-init, so the adapter starts as a no-op and
the unadapted forward pass matches the pretrained baseline exactly.

When `wrap_conv2d_layers(model, rank, alpha)` is called, every nn.Conv2d in
the module tree is replaced in-place with a LoRAConv2d wrapper and the base
conv weights are frozen. nn.ConvTranspose2d is skipped (LoRA on transposed
convs is non-standard, and the generator has only 2 of them).
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(
            rank, base.out_channels, kernel_size=base.kernel_size,
            stride=base.stride, padding=base.padding, dilation=base.dilation,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))

    def merge_into_base(self) -> nn.Conv2d:
        """Bake LoRA update into a fresh nn.Conv2d so it can be exported standalone.

        For 1x1 LoRA-A and kxk LoRA-B, the merged delta weight is:
            ΔW[o, i, h, w] = scaling * sum_r B[o, r, h, w] * A[r, i, 0, 0]
        Composing kernel sizes is a tensor contraction; we do it in PyTorch.
        """
        base = self.base
        A = self.lora_A.weight  # (rank, in_ch, 1, 1)
        B = self.lora_B.weight  # (out_ch, rank, kH, kW)
        # squeeze A -> (rank, in_ch); contract: delta[o, i, h, w] = sum_r B[o, r, h, w] * A[r, i]
        A2 = A.view(A.shape[0], A.shape[1])  # (rank, in_ch)
        # einsum: o r h w, r i -> o i h w
        delta = torch.einsum("orhw,ri->oihw", B, A2) * self.scaling
        new_w = base.weight.detach().clone() + delta.to(base.weight.dtype)
        merged = nn.Conv2d(
            base.in_channels, base.out_channels, base.kernel_size,
            stride=base.stride, padding=base.padding, dilation=base.dilation,
            bias=(base.bias is not None),
        )
        with torch.no_grad():
            merged.weight.copy_(new_w)
            if base.bias is not None:
                merged.bias.copy_(base.bias.detach())
        return merged


def _replace_child(parent: nn.Module, name: str, new_child: nn.Module) -> None:
    setattr(parent, name, new_child)


def wrap_conv2d_layers(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    skip_kernel_sizes: tuple[int, ...] = (),
) -> int:
    """Replace every nn.Conv2d in `model` with LoRAConv2d. Returns count wrapped.

    `skip_kernel_sizes` lets the caller exclude specific kernel sizes (e.g. (7,)
    to leave the wide initial/output convs untouched).
    """
    wrapped = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d) and not isinstance(child, LoRAConv2d):
                k = child.kernel_size[0] if isinstance(child.kernel_size, tuple) else child.kernel_size
                if k in skip_kernel_sizes:
                    continue
                _replace_child(module, name, LoRAConv2d(child, rank=rank, alpha=alpha))
                wrapped += 1
    return wrapped


def merge_all_lora(model: nn.Module) -> int:
    """Replace every LoRAConv2d with a merged plain nn.Conv2d. Returns count merged."""
    merged = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRAConv2d):
                _replace_child(module, name, child.merge_into_base())
                merged += 1
    return merged


def lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            yield m.lora_A.weight
            yield m.lora_B.weight


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


def freeze_non_lora(model: nn.Module) -> None:
    """Set requires_grad=False on every parameter that isn't a LoRA A/B weight."""
    lora_ids = {id(p) for p in lora_parameters(model)}
    for p in model.parameters():
        p.requires_grad = id(p) in lora_ids
