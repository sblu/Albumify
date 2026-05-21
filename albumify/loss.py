"""Pixel + perceptual losses for line-drawing supervision.

L1 dominates training (it's the right loss for pixel-aligned line drawings).
VGG perceptual loss adds high-frequency structural pressure; it's optional
and weight 0 disables it cleanly. The label is grayscale (1ch); both pred
and target are tiled to 3ch before being fed to VGG.

VGG features are taken at relu1_2, relu2_2, relu3_3, relu4_3 — the same
selection used in Johnson et al. 2016 and most perceptual-loss work.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet stats for VGG input normalization
_VGG_MEAN = (0.485, 0.456, 0.406)
_VGG_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):
    """L1 between VGG16 feature maps of pred and target.

    `vgg_features` is a frozen `torchvision.models.vgg16().features` (or a
    compatible Sequential). Pass `pretrained=True` in main() to download
    ImageNet weights; tests can pass a random-init VGG for shape-only checks.
    """

    LAYER_OUT_INDICES = (3, 8, 15, 22)  # relu1_2, relu2_2, relu3_3, relu4_3 (post-ReLU)

    def __init__(self, vgg_features: nn.Module):
        super().__init__()
        self.vgg = vgg_features
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor(_VGG_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_VGG_STD).view(1, 3, 1, 1))

    def _normalize(self, x_3ch: torch.Tensor) -> torch.Tensor:
        return (x_3ch - self.mean) / self.std

    def _features(self, x_3ch: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        h = self._normalize(x_3ch)
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.LAYER_OUT_INDICES:
                feats.append(h)
            if i >= max(self.LAYER_OUT_INDICES):
                break
        return feats

    def forward(self, pred_gray: torch.Tensor, target_gray: torch.Tensor) -> torch.Tensor:
        # Replicate 1ch -> 3ch
        pred_rgb = pred_gray.expand(-1, 3, -1, -1) if pred_gray.size(1) == 1 else pred_gray
        target_rgb = target_gray.expand(-1, 3, -1, -1) if target_gray.size(1) == 1 else target_gray
        f_p = self._features(pred_rgb)
        f_t = self._features(target_rgb)
        loss = pred_rgb.new_tensor(0.0)
        for fp, ft in zip(f_p, f_t):
            loss = loss + F.l1_loss(fp, ft)
        return loss / len(f_p)


class L1PerceptualLoss(nn.Module):
    """Combined L1 + optional VGG perceptual loss."""

    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        vgg: Optional[VGGPerceptualLoss] = None,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.vgg = vgg
        if perceptual_weight > 0 and vgg is None:
            raise ValueError("perceptual_weight > 0 requires a VGGPerceptualLoss instance")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l1 = F.l1_loss(pred, target)
        total = self.l1_weight * l1
        result: dict[str, torch.Tensor] = {"l1": l1.detach()}
        if self.perceptual_weight > 0 and self.vgg is not None:
            perc = self.vgg(pred, target)
            total = total + self.perceptual_weight * perc
            result["perc"] = perc.detach()
        result["total"] = total
        return result
