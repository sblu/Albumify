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
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 1,
        n_residual_blocks: int = 9,
        ngf: int = 64,
        sigmoid: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # Downsampling x2
        in_ch = ngf
        out_ch = ngf * 2
        for _ in range(2):
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            out_ch = in_ch * 2
        # Residual blocks
        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(in_ch))
        # Upsampling x2
        out_ch = in_ch // 2
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    in_ch, out_ch, 3, stride=2, padding=1, output_padding=1,
                ),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            out_ch = in_ch // 2
        # Final 1-channel conv + sigmoid
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, output_nc, 7),
        ]
        if sigmoid:
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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
