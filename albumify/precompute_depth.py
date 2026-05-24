"""One-time MiDaS DPT-Large pseudo-GT depth precompute for Plan F3.

For each cover in --covers-dir, runs DPT-Large to produce a depth map,
normalizes per-image to [0, 1], saves as float16 NumPy to
--out-dir/{slug}.npy. Skips slugs that are already cached.

DPT-Large is the highest-quality single-shot MiDaS variant (per user
decision 2026-05-23: "do it right"). On L4 it takes ~0.4 s per cover;
424 covers ~3 min total.

Output spatial size matches --resize (default 256, matching our training
crop). The geometry loss compares G_Geom's predicted depth to these
cached pseudo-GT maps via L1 (after resampling).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _load_midas(device: torch.device, variant: str = "DPT_Large"):
    midas = torch.hub.load("intel-isl/MiDaS", variant)
    midas.eval().to(device)
    for p in midas.parameters():
        p.requires_grad = False
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if variant in ("DPT_Large", "DPT_Hybrid") else transforms.small_transform
    return midas, transform


def precompute(
    covers_dir: Path,
    out_dir: Path,
    *,
    resize: int = 256,
    device: str = "cuda",
    variant: str = "DPT_Large",
    overwrite: bool = False,
    glob: str = "*.jpg",
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    print(f"[depth] loading MiDaS {variant} on {dev} ...")
    midas, transform = _load_midas(dev, variant=variant)
    cover_paths = sorted(covers_dir.glob(glob))
    done = skipped = failed = 0
    for i, cp in enumerate(cover_paths):
        slug = cp.stem
        out_path = out_dir / f"{slug}.npy"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            img = np.array(Image.open(cp).convert("RGB"))
        except Exception as exc:
            print(f"[depth] skip {slug}: read failed ({exc})")
            failed += 1
            continue
        # MiDaS' own transform handles resize/normalize and returns a CHW tensor.
        batch = transform(img).to(dev)
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        with torch.no_grad():
            depth = midas(batch)
            # Up- or down-sample to the training crop so the cached GT
            # spatially aligns with G_Geom's prediction (after final resize).
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=(resize, resize), mode="bicubic", align_corners=False,
            ).squeeze(1)[0]
        d = depth.cpu().float().numpy()
        # Per-image min-max normalize to [0, 1] so the L1 loss against
        # G_Geom(Tanh in [-1,1] rescaled to [0,1]) is on the same scale
        # for every cover.
        dmin, dmax = float(d.min()), float(d.max())
        if dmax - dmin < 1e-6:
            d_n = np.zeros_like(d, dtype=np.float16)
        else:
            d_n = ((d - dmin) / (dmax - dmin)).astype(np.float16)
        np.save(out_path, d_n)
        done += 1
        if done % 25 == 0:
            print(f"[depth] {done + skipped}/{len(cover_paths)} cached ({skipped} pre-existing)")
    print(f"[depth] done: wrote={done} skipped={skipped} failed={failed} total={len(cover_paths)}")
    return {"wrote": done, "skipped": skipped, "failed": failed, "total": len(cover_paths)}


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-cache DPT-Large depth maps for every cover.")
    p.add_argument("--covers-dir", default="data/covers", type=Path)
    p.add_argument("--out-dir",    default="data/depth",  type=Path)
    p.add_argument("--resize",     type=int, default=256, help="spatial resolution of cached depth (must match train img-size)")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--variant",    choices=("DPT_Large", "DPT_Hybrid", "MiDaS_small"), default="DPT_Large")
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--glob",       default="*.jpg")
    args = p.parse_args()
    precompute(
        args.covers_dir, args.out_dir,
        resize=args.resize, device=args.device, variant=args.variant,
        overwrite=args.overwrite, glob=args.glob,
    )


if __name__ == "__main__":
    main()
