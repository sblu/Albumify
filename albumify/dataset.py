"""AlbumDataset: PyTorch Dataset of (cover, label) pairs from a split file.

Imported only when torch is available (i.e. on the training VM). The
PIL-level paired transform lives in `albumify.transforms` and is testable
without torch.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from albumify.split import read_split
from albumify.transforms import PairedTransformConfig, paired_transform


def _pil_rgb_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert RGB PIL → CHW float tensor in [0, 1].

    Uses np.array (copy) rather than np.asarray (view) because PIL's buffer
    is non-writable and torch.from_numpy emits a warning otherwise.
    """
    arr = np.array(img, dtype=np.uint8)  # HxWx3
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def _pil_gray_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert L-mode PIL → 1xHxW float tensor in [0, 1]."""
    arr = np.array(img, dtype=np.uint8)  # HxW
    t = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
    return t


class AlbumDataset(Dataset):
    """Loads (cover.jpg, label.png) pairs whose slugs are listed in a split file.

    Each item returns (cover: 3xHxW float in [0,1], label: 1xHxW float in [0,1], slug).
    """

    def __init__(
        self,
        split_file: Path | str,
        covers_dir: Path | str,
        labels_dir: Path | str,
        *,
        train: bool,
        cfg: Optional[PairedTransformConfig] = None,
        seed: int = 0,
    ):
        self.split_file = Path(split_file)
        self.covers_dir = Path(covers_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train
        self.cfg = cfg or PairedTransformConfig()
        self.seed = seed
        self.slugs = read_split(self.split_file)

    def __len__(self) -> int:
        return len(self.slugs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        slug = self.slugs[idx]
        cover_path = self.covers_dir / f"{slug}.jpg"
        label_path = self.labels_dir / f"{slug}.png"
        with Image.open(cover_path) as c, Image.open(label_path) as l:
            c.load(); l.load()
            cover_img, label_img = c, l
        # Per-worker, per-sample RNG so DataLoader workers don't all see the same
        # augmentation sequence but the same (seed, idx) always reproduces.
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random((self.seed, worker_id, idx, self.train))
        cover_t, label_t = paired_transform(
            cover_img, label_img, cfg=self.cfg, rng=rng, train=self.train,
        )
        return _pil_rgb_to_tensor(cover_t), _pil_gray_to_tensor(label_t), slug
