"""CLIP semantic loss (Informative-Drawings paper Eq. 3, Plan F2).

`CLIPSemanticLoss` matches the CLIP image embedding of the generated line
drawing to the CLIP embedding of the original photograph. Plays the role of
a learned perceptual loss but with the rich visual-text-trained features
of OpenAI's CLIP ViT-B/32, which (per the paper) "adds the most lines."

Forward signature:
    pred_gray   [B, 1, H, W]  — sigmoid output of our Generator, in [0, 1]
    target_rgb  [B, 3, H, W]  — original cover, in [0, 1]
    returns     scalar Tensor with grad — MSE between CLIP embeddings.

Implementation notes:
- pred (1ch) is replicated to 3ch by tile before CLIP. CLIP was never
  trained on grayscale; the paper applies the same trick.
- Both pred and target are bilinearly resized to 224x224 (CLIP's native
  input size) before encoding.
- CLIP-canonical channel normalization is applied so the cover and the
  prediction live in the same distribution CLIP was trained on.
- CLIP parameters are frozen at construction. Their .requires_grad is
  False, and they are not in `self.parameters()` for any optimizer.
- For testing, a stub `clip_model` exposing `.encode_image(x)` can be
  injected, avoiding the ~150MB model download.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# CLIP's canonical image-channel normalization (from openai/CLIP preprocess).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CLIP_RES = 224


class CLIPSemanticLoss(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        clip_model: Optional[nn.Module] = None,
        preprocess=None,  # accepted for API symmetry with clip.load; not used internally
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if clip_model is None:
            import clip  # type: ignore
            dev = device if device is not None else torch.device("cpu")
            clip_model, _ = clip.load(model_name, device=str(dev), jit=False)
        self.clip_model = clip_model
        # CLIP is sometimes loaded in fp16; force fp32 for stable training-loop math.
        self.clip_model.float()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.register_buffer("_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1))

    def _prepare(self, x_3ch: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x_3ch, size=(_CLIP_RES, _CLIP_RES), mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    def forward(self, pred_gray: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        pred_3ch = pred_gray.expand(-1, 3, -1, -1) if pred_gray.size(1) == 1 else pred_gray
        target_3ch = target_rgb if target_rgb.size(1) == 3 else target_rgb.expand(-1, 3, -1, -1)
        pred_emb = self.clip_model.encode_image(self._prepare(pred_3ch))
        target_emb = self.clip_model.encode_image(self._prepare(target_3ch))
        # CLIP's encode_image can return fp16 in some configs; cast to match.
        pred_emb = pred_emb.float()
        target_emb = target_emb.float()
        return F.mse_loss(pred_emb, target_emb)
