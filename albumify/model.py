"""Informative-Drawings generator (Chan et al., SIGGRAPH 2022), vendored.

Architecture follows the official `informative-drawings` repo. ResNet-style
generator with 64 base channels, 2 downsamples, N residual blocks (default 9),
2 upsamples, sigmoid output. ~4.4 M params at the defaults.

Reference: https://github.com/carolineec/informative-drawings/blob/main/model.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class Generator(nn.Module):
    """ResNet-style generator, vendored to match the official informative-drawings
    state_dict layout exactly: five Sequential children named model0..model4
    so a checkpoint produced by the upstream repo loads strict=True.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 1,
        n_residual_blocks: int = 9,
        ngf: int = 64,
        sigmoid: bool = True,
    ):
        super().__init__()
        # ---- model0: initial 7x7 reflect-pad conv ----
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # ---- model1: 2x downsampling ----
        self.model1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )

        # ---- model2: N residual blocks at ngf*4 channels ----
        self.model2 = nn.Sequential(
            *[ResidualBlock(ngf * 4) for _ in range(n_residual_blocks)]
        )

        # ---- model3: 2x upsampling ----
        self.model3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # ---- model4: final 7x7 reflect-pad conv (+ optional sigmoid) ----
        tail: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
        ]
        if sigmoid:
            tail.append(nn.Sigmoid())
        self.model4 = nn.Sequential(*tail)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model0(x)
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        return x


def load_pretrained(
    model: Generator,
    ckpt_path: Path | str,
    *,
    map_location: str = "cpu",
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load a pretrained checkpoint into a Generator.

    The official Informative-Drawings checkpoint stores keys at the top level
    (i.e. `model.0.weight`, `model.1.weight`, ...) matching `Generator.model`'s
    Sequential children. If the checkpoint is wrapped in a dict, common keys
    like 'netG_A', 'state_dict', or 'generator' are unwrapped.

    Returns (missing_keys, unexpected_keys) from load_state_dict.
    """
    sd = torch.load(str(ckpt_path), map_location=map_location)
    for wrapper_key in ("netG_A", "netG", "state_dict", "generator", "model_state_dict"):
        if isinstance(sd, dict) and wrapper_key in sd and isinstance(sd[wrapper_key], dict):
            sd = sd[wrapper_key]
            break
    # Strip a leading 'module.' if the checkpoint was saved from DataParallel.
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    out = model.load_state_dict(sd, strict=strict)
    return list(out.missing_keys), list(out.unexpected_keys)
