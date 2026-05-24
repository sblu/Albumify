"""Features-to-depth network + InceptionV3 feature extractor (Plan F3).

Vendored from carolineec/informative-drawings (model.py). The pretrained
checkpoint `feats2depth.pth` for `GlobalGenerator2` was released by the
paper authors via Google Drive
(1Ov1BNue74Yu-57X2rpdjqZy0o-fnFoly); architecture must match exactly for
strict-load to succeed.

Pipeline at training time (see geom_loss.py):

  cover (3x256x256)
    → resize 299 + ImageNet normalize
    → InceptionV3.Mixed_6b features (768 x 17 x 17)
    → GlobalGenerator2(input_nc=768, output_nc=1, n_downsampling=3,
                       n_blocks=9, use_sig=False)
    → predicted depth ~ (1 x 1088 x 1088) with Tanh in [-1, 1]

The same pipeline runs on a generated line drawing (1ch replicated to 3ch)
to produce a "depth-from-drawing" prediction that is then compared (L1)
to the cached DPT-Large depth of the original cover.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---- Vendored GlobalGenerator2 (pix2pixHD-style) ----

class _ResnetBlock(nn.Module):
    def __init__(self, dim: int, padding_type: str, norm_layer, activation=nn.ReLU(True), use_dropout: bool = False):
        super().__init__()
        self.conv_block = self._build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    @staticmethod
    def _build_conv_block(dim: int, padding_type: str, norm_layer, activation, use_dropout: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), activation]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        p = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class GlobalGenerator2(nn.Module):
    """Vendored from upstream informative-drawings/model.py.

    Default constructor arguments match the **actual released
    `feats2depth.pth`** layout (verified by strict state_dict load):

        input_nc=768, output_nc=3, ngf=64,
        n_downsampling=1, n_blocks=9, n_UPsampling=3,
        BatchNorm2d, Tanh output.

    The paper supplemental Table 7 prints these as "n_downsampling=3"
    but the released checkpoint was actually trained with n_downsampling=1
    (one 512→256 transposed conv) and n_UPsampling=3 (three transposed
    convs back down to 64 channels). The ResNet blocks live at 256ch.

    Note "downsample" is a misnomer — the loop uses ConvTranspose2d, so
    it doubles spatial size. From Mixed_6b's 17×17 input, output spatial
    is roughly 17 → 18 (pad+conv) → 36 (1 transpose) → 36 (resnets) →
    72→144→288 (3 transposes) = 288×288 with 3 channels.
    """

    def __init__(
        self,
        input_nc: int = 768,
        output_nc: int = 3,
        ngf: int = 64,
        n_downsampling: int = 1,
        n_blocks: int = 9,
        norm_layer=nn.BatchNorm2d,
        padding_type: str = "reflect",
        use_sig: bool = False,
        n_UPsampling: int = 3,
    ):
        assert n_blocks >= 0
        super().__init__()
        activation = nn.ReLU(True)
        mult = 8
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(4),
            nn.Conv2d(input_nc, ngf * mult, kernel_size=7, padding=0),
            norm_layer(ngf * mult),
            activation,
        ]
        for _ in range(n_downsampling):
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=4, stride=2, padding=1),
                norm_layer(ngf * mult // 2),
                activation,
            ]
            mult = mult // 2
        if n_UPsampling <= 0:
            n_UPsampling = n_downsampling
        for _ in range(n_blocks):
            layers.append(_ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer))
        for _ in range(n_UPsampling):
            next_mult = mult // 2
            if next_mult == 0:
                next_mult = 1
                mult = 1
            layers += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * next_mult), kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(int(ngf * next_mult)),
                activation,
            ]
            mult = next_mult
        if use_sig:
            layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        else:
            layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_feats2depth(ckpt_path: Path | str, *, map_location: str = "cpu") -> GlobalGenerator2:
    """Construct G_Geom at the released checkpoint's exact shape and load weights.

    The upstream ckpt is typically wrapped as either a plain state_dict or
    a dict with key 'model' / 'state_dict'. We try a few common shapes.
    """
    sd = torch.load(str(ckpt_path), map_location=map_location)
    for key in ("model", "state_dict", "netGeom", "net_Geom"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    model = GlobalGenerator2()  # defaults match upstream training
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Surface for debugging — caller can also inspect.
        print(f"[feats2depth] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[feats2depth] missing[:5]: {missing[:5]}")
        if unexpected:
            print(f"[feats2depth] unexpected[:5]: {unexpected[:5]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ---- InceptionV3 Mixed_6b feature extractor ----

# ImageNet stats used by torchvision's pretrained InceptionV3.
_INCEPTION_MEAN = (0.485, 0.456, 0.406)
_INCEPTION_STD = (0.229, 0.224, 0.225)
_INCEPTION_SIZE = 299


class InceptionMixed6bExtractor(nn.Module):
    """Frozen InceptionV3 (ImageNet pretrained) returning Mixed_6b features.

    Input:  [B, 3, H, W] in [0, 1] (will be resized to 299x299 + normalized).
    Output: [B, 768, 17, 17] — the Mixed_6b activation.

    Why Mixed_6b: the upstream Informative-Drawings paper supplemental
    Sec 6.3 says "we extract features from the Mixed 6b node" specifically,
    not the penultimate layer. Mid-network features transfer better across
    the photo / line-drawing domain gap.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        # aux_logits=True is required to load weights but we never call AuxLogits in the forward.
        m = models.inception_v3(weights=weights, aux_logits=True, init_weights=not pretrained)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        self.model = m
        self.register_buffer("_mean", torch.tensor(_INCEPTION_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_INCEPTION_STD).view(1, 3, 1, 1))

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(_INCEPTION_SIZE, _INCEPTION_SIZE), mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = self._prepare(x)
        # Mirror torchvision's InceptionV3.forward up to Mixed_6b. We can't
        # call the public forward — it would run all the way to logits and
        # we'd lose intermediate features.
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)   # [B, 768, 17, 17]
        return x
