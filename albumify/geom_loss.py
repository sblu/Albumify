"""Depth/geometry loss (Informative-Drawings paper Eq. 2, Plan F3).

Implements L_geom = || G_Geom(I(G_A(a))) − F(a) ||_1  where:
  G_A(a)     — our generated line drawing (`pred_gray` to forward)
  I(·)       — frozen InceptionV3 features at Mixed_6b
  G_Geom(·)  — frozen pretrained features-to-depth network from upstream
  F(a)       — MiDaS DPT-Large depth of the original cover, **precomputed
              and cached** to disk (see precompute_depth.py)

Why pre-compute F(a) rather than run MiDaS in the train loop:
  DPT-Large is ~340M params; running it per step alongside our 4.4M
  generator would dominate compute. Caching once costs ~3 min total and
  the cached tensors fit comfortably in RAM (~55 MB for 424 covers at
  256² float16).

Per the paper's ablation (Table 2), this loss is the **single most
important loss for the contour-drawing style** — 92.2% user preference
for "with depth" vs "without depth" on Contour vs only 48.3% on Anime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeomDepthLoss(nn.Module):
    def __init__(
        self,
        *,
        feats2depth: Optional[nn.Module] = None,
        feature_extractor: Optional[nn.Module] = None,
        depth_cache_dir: Path | str,
        feats2depth_ckpt: Optional[Path | str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        feats2depth        Pre-constructed G_Geom (frozen). If None,
                           `feats2depth_ckpt` must be provided.
        feature_extractor  Pre-constructed Mixed_6b feature extractor
                           (frozen). If None, a default InceptionV3-based
                           one is built (torchvision pretrained).
        depth_cache_dir    Directory of {slug}.npy float16 depth maps
                           produced by precompute_depth.py.
        feats2depth_ckpt   Path to the upstream feats2depth.pth.
        device             Optional placement target.
        """
        super().__init__()

        # Build/freeze G_Geom.
        if feats2depth is None:
            if feats2depth_ckpt is None:
                raise ValueError("Provide either `feats2depth` or `feats2depth_ckpt`.")
            from albumify.feats2depth import load_feats2depth
            feats2depth = load_feats2depth(feats2depth_ckpt, map_location="cpu")
        for p in feats2depth.parameters():
            p.requires_grad = False
        feats2depth.eval()
        self.feats2depth = feats2depth

        # Build/freeze feature extractor.
        if feature_extractor is None:
            from albumify.feats2depth import InceptionMixed6bExtractor
            feature_extractor = InceptionMixed6bExtractor(pretrained=True)
        for p in feature_extractor.parameters():
            p.requires_grad = False
        feature_extractor.eval()
        self.feature_extractor = feature_extractor

        # Pre-load depth cache. Fail fast on empty/missing.
        depth_cache_dir = Path(depth_cache_dir)
        if not depth_cache_dir.exists():
            raise FileNotFoundError(f"depth_cache_dir does not exist: {depth_cache_dir}")
        cache_files = sorted(depth_cache_dir.glob("*.npy"))
        if not cache_files:
            raise ValueError(
                f"depth_cache_dir is empty: {depth_cache_dir}. "
                "Run `python -m albumify.precompute_depth` first."
            )
        self._depth_cache: dict[str, torch.Tensor] = {}
        for fp in cache_files:
            arr = np.load(fp).astype(np.float32)
            t = torch.from_numpy(arr)
            if t.ndim == 2:
                t = t.unsqueeze(0)  # [1, H, W]
            self._depth_cache[fp.stem] = t
        if device is not None:
            self.to(device)

    def _gather_targets(self, slugs: Sequence[str], device: torch.device) -> torch.Tensor:
        """Stack cached depth tensors in the order matching `slugs`."""
        missing = [s for s in slugs if s not in self._depth_cache]
        if missing:
            raise KeyError(
                f"No cached depth for slugs: {missing[:5]}"
                f"{' (+ more)' if len(missing) > 5 else ''}. "
                f"Re-run precompute_depth.py for the full split."
            )
        return torch.stack([self._depth_cache[s] for s in slugs], dim=0).to(device)

    def forward(self, pred_gray: torch.Tensor, slugs: Sequence[str]) -> torch.Tensor:
        # Replicate 1ch -> 3ch so InceptionV3 accepts it.
        pred_3ch = pred_gray.expand(-1, 3, -1, -1) if pred_gray.size(1) == 1 else pred_gray
        feats = self.feature_extractor(pred_3ch)            # [B, 768, 17, 17]
        depth_pred = self.feats2depth(feats)                # [B, Cd, H', W'], Tanh in [-1, 1]
        # Released feats2depth.pth outputs 3 channels (a depth-encoding) —
        # collapse to 1ch by mean before L1 against the 1ch cached GT.
        if depth_pred.size(1) > 1:
            depth_pred = depth_pred.mean(dim=1, keepdim=True)
        depth_pred = (depth_pred + 1.0) * 0.5               # rescale to [0, 1] to match cached GT
        depth_gt = self._gather_targets(slugs, pred_gray.device)  # [B, 1, H, W]
        if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
            depth_pred = F.interpolate(
                depth_pred, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False,
            )
        return F.l1_loss(depth_pred, depth_gt)
