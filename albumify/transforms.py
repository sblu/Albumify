"""PIL-level paired augmentation for (cover, label) image pairs.

The geometric transforms (resize, random crop, horizontal flip) MUST be
applied to both images with identical parameters so the line drawing stays
aligned with its source. Photometric jitter (brightness/contrast/saturation)
is applied only to the cover — the label is a binary-ish line drawing and
must not be re-coloured.

Pure-Python + PIL only so it can be unit-tested without torch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageEnhance


@dataclass
class PairedTransformConfig:
    out_size: int = 256
    resize_short_to: int = 288  # resize-then-crop adds a small zoom margin
    hflip_prob: float = 0.5
    jitter_brightness: float = 0.15
    jitter_contrast: float = 0.15
    jitter_saturation: float = 0.15
    enable_jitter: bool = True


def _resize_short_side(img: Image.Image, target_short: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    if w <= h:
        new_w = target_short
        new_h = int(round(h * target_short / w))
    else:
        new_h = target_short
        new_w = int(round(w * target_short / h))
    return img.resize((new_w, new_h), Image.BICUBIC)


def _resize_short_side_nearest(img: Image.Image, target_short: int) -> Image.Image:
    """Same as _resize_short_side but with NEAREST resampling for binary labels."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    if w <= h:
        new_w = target_short
        new_h = int(round(h * target_short / w))
    else:
        new_h = target_short
        new_w = int(round(w * target_short / h))
    return img.resize((new_w, new_h), Image.NEAREST)


def paired_transform(
    cover: Image.Image,
    label: Image.Image,
    *,
    cfg: Optional[PairedTransformConfig] = None,
    rng: Optional[random.Random] = None,
    train: bool = True,
) -> tuple[Image.Image, Image.Image]:
    """Apply synchronized geometric augs + cover-only photometric jitter.

    Cover and label are forced to identical (square) dimensions up front:
    a direct PIL resize to `resize_short_to` x `resize_short_to`. Real-world
    covers from CAA aren't always square (e.g. Hotel California is 300x298)
    and the Gemini labels are always 1024x1024, so aspect-preserving resize
    would yield different shapes per side and break paired cropping. The
    cover absorbs a small aspect distortion; the label (already square) is
    not affected.

    When `train=False`, applies a deterministic resize to (out_size, out_size)
    with no jitter.
    """
    cfg = cfg or PairedTransformConfig()
    rng = rng or random.Random()

    cover_rgb = cover.convert("RGB")
    label_l = label.convert("L")  # line drawings are grayscale; collapse to 1 channel

    if train:
        # Direct square resize so cover + label always agree on dimensions.
        side = cfg.resize_short_to
        cover_r = cover_rgb.resize((side, side), Image.BICUBIC)
        label_r = label_l.resize((side, side), Image.NEAREST)

        max_xy = max(0, side - cfg.out_size)
        x = rng.randint(0, max_xy)
        y = rng.randint(0, max_xy)
        box = (x, y, x + cfg.out_size, y + cfg.out_size)
        cover_c = cover_r.crop(box)
        label_c = label_r.crop(box)

        if rng.random() < cfg.hflip_prob:
            cover_c = cover_c.transpose(Image.FLIP_LEFT_RIGHT)
            label_c = label_c.transpose(Image.FLIP_LEFT_RIGHT)

        if cfg.enable_jitter:
            b = 1.0 + rng.uniform(-cfg.jitter_brightness, cfg.jitter_brightness)
            c = 1.0 + rng.uniform(-cfg.jitter_contrast, cfg.jitter_contrast)
            s = 1.0 + rng.uniform(-cfg.jitter_saturation, cfg.jitter_saturation)
            cover_c = ImageEnhance.Brightness(cover_c).enhance(b)
            cover_c = ImageEnhance.Contrast(cover_c).enhance(c)
            cover_c = ImageEnhance.Color(cover_c).enhance(s)

        return cover_c, label_c

    # Eval: deterministic square resize at out_size.
    return (
        cover_rgb.resize((cfg.out_size, cfg.out_size), Image.BICUBIC),
        label_l.resize((cfg.out_size, cfg.out_size), Image.NEAREST),
    )
