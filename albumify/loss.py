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


def edge_weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    edge_threshold: float = 0.5,
    edge_weight: float = 0.0,
) -> torch.Tensor:
    """Per-pixel L1, with dark target pixels weighted (1 + edge_weight)x.

    Line-drawing labels are ~95% white background / ~5% dark edges, so plain
    L1 lets the model converge to "predict white everywhere" — gradient toward
    white dominates 19:1. Lifting edge_weight to 9 brings the effective
    contribution of edge pixels up by 10x, giving roughly balanced gradients
    and pushing the model to commit dark where dark belongs.

    Edge_weight=0 is identical to plain L1.
    """
    abs_err = (pred - target).abs()
    if edge_weight == 0:
        return abs_err.mean()
    # Mark target pixels that are "edge" (dark) and dilate 1px so the model
    # also gets credit on near-edge pixels.
    edges = (target < edge_threshold).float()
    edges = F.max_pool2d(edges, kernel_size=3, stride=1, padding=1)
    weights = 1.0 + edge_weight * edges
    return (abs_err * weights).mean()


class L1PerceptualLoss(nn.Module):
    """Combined L1 + optional VGG perceptual loss with optional edge weighting."""

    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        edge_weight: float = 0.0,
        edge_threshold: float = 0.5,
        vgg: Optional["VGGPerceptualLoss"] = None,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.edge_threshold = edge_threshold
        self.vgg = vgg
        if perceptual_weight > 0 and vgg is None:
            raise ValueError("perceptual_weight > 0 requires a VGGPerceptualLoss instance")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l1 = edge_weighted_l1(
            pred, target,
            edge_threshold=self.edge_threshold, edge_weight=self.edge_weight,
        )
        total = self.l1_weight * l1
        result: dict[str, torch.Tensor] = {"l1": l1.detach()}
        if self.perceptual_weight > 0 and self.vgg is not None:
            perc = self.vgg(pred, target)
            total = total + self.perceptual_weight * perc
            result["perc"] = perc.detach()
        result["total"] = total
        return result


def edge_weighted_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    edge_threshold: float = 0.5,
    edge_weight: float = 19.0,
) -> torch.Tensor:
    """BCE-with-logits on hard-thresholded targets, with edge-pixel weighting.

    Mirrors `edge_weighted_l1`: target is grayscale in [0, 1], thresholded to
    {0, 1} at `edge_threshold`. A dilated edge mask (max_pool2d k=3) up-weights
    edge pixels (1 + edge_weight)x so the loss is not dominated by the ~95%
    background pixels.

    `edge_weight=0` reduces to plain `F.binary_cross_entropy_with_logits` on
    the binarized target.
    """
    target_bin = (target >= edge_threshold).float()
    if edge_weight == 0:
        return F.binary_cross_entropy_with_logits(logits, target_bin)
    edges = (target < edge_threshold).float()
    edges = F.max_pool2d(edges, kernel_size=3, stride=1, padding=1)
    weights = 1.0 + edge_weight * edges
    return F.binary_cross_entropy_with_logits(logits, target_bin, weight=weights)


class BCELogitsPerceptualLoss(nn.Module):
    """BCE-with-logits + optional VGG perceptual on sigmoid(logits)."""

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        edge_weight: float = 19.0,
        edge_threshold: float = 0.5,
        vgg: Optional["VGGPerceptualLoss"] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.edge_threshold = edge_threshold
        self.vgg = vgg
        if perceptual_weight > 0 and vgg is None:
            raise ValueError("perceptual_weight > 0 requires a VGGPerceptualLoss instance")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        bce = edge_weighted_bce_logits(
            logits, target,
            edge_threshold=self.edge_threshold, edge_weight=self.edge_weight,
        )
        total = self.bce_weight * bce
        result: dict[str, torch.Tensor] = {"bce": bce.detach()}
        if self.perceptual_weight > 0 and self.vgg is not None:
            pred_prob = torch.sigmoid(logits)
            perc = self.vgg(pred_prob, target)
            total = total + self.perceptual_weight * perc
            result["perc"] = perc.detach()
        result["total"] = total
        return result
