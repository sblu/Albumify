"""70x70 PatchGAN discriminator + LSGAN loss helpers (Plan F4).

Vanilla pix2pix/CycleGAN PatchGAN: 4 strided conv blocks (Conv-InstanceNorm-
LeakyReLU) at strides 2,2,2,1 with kernel 4, then a final 1x1 conv head
producing one logit per ~70x70 patch.

Use:
  D = PatchGAN70(in_ch=1).to(device)
  opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
  # per train step:
  pred = G(cover)
  d_real = D(label)
  d_fake = D(pred.detach())
  loss_d = lsgan_d_loss(d_real, d_fake)
  loss_d.backward(); opt_D.step()
  # then G step:
  d_fake_for_g = D(pred)
  g_gan = lsgan_g_loss(d_fake_for_g)
  total_g = ... + cfg.gan_weight * g_gan

Per the paper's Eq. 1, LSGAN is used (not vanilla BCE GAN). LSGAN gives
smoother gradients early in training when D is overconfident.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchGAN70(nn.Module):
    """N-layer PatchGAN matching pix2pix's 70x70 receptive field default.

    Per-batch input: [B, in_ch, H, W]. Output: [B, 1, ~H/16, ~W/16] logits,
    one value per receptive-field patch. No sigmoid — LSGAN loss expects
    raw scores.
    """

    def __init__(self, in_ch: int = 1, ngf: int = 64, n_layers: int = 3, use_instance_norm: bool = True):
        super().__init__()
        norm = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d
        # First layer has no normalization (standard for PatchGAN).
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, ngf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Strided conv layers — each halves spatial.
        mult = 1
        for n in range(1, n_layers):
            prev_mult = mult
            mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ngf * prev_mult, ngf * mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm(ngf * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        # One more conv at stride 1 to push the receptive field up to ~70.
        prev_mult = mult
        mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ngf * prev_mult, ngf * mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm(ngf * mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Head: 1x1 conv to a single logit per patch.
        layers.append(nn.Conv2d(ngf * mult, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def lsgan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN discriminator loss (paper Eq. 1, D term).

    L_D = 0.5 * ( E[(D(real) - 1)^2] + E[D(fake)^2] )
    Returns scalar with grad.
    """
    real_term = ((d_real - 1.0) ** 2).mean()
    fake_term = (d_fake ** 2).mean()
    return 0.5 * (real_term + fake_term)


def lsgan_g_loss(d_fake_for_g: torch.Tensor) -> torch.Tensor:
    """LSGAN generator loss (paper Eq. 1, G term).

    L_G = E[(D(G(a)) - 1)^2]
    G wants D to think its outputs are real.
    """
    return ((d_fake_for_g - 1.0) ** 2).mean()
