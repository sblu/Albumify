"""Validation metrics + visual grid renderer.

Metrics:
- SSIM between pred and target (single-channel)
- Edge-F1: per-image F1 over "edge" pixels (label < edge_threshold).
  Precision/recall computed against the binarized target; reports the F1.
- LPIPS (optional, perceptual): requires the lpips package.

The visual grid stitches (cover | target | pred) triples into one wide PNG
for quick human spot-checking on the val set.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ---- Pure-numpy metrics (testable without torch) -------------------------


def edge_f1(
    pred_gray: np.ndarray,
    target_gray: np.ndarray,
    *,
    edge_threshold: float = 0.5,
) -> dict[str, float]:
    """Binarize via threshold (dark = edge), compute precision/recall/F1.

    Inputs are HxW float arrays in [0, 1]. A pixel is an "edge" if its value
    is < edge_threshold (black lines on white background).
    """
    pred_e = pred_gray < edge_threshold
    tgt_e = target_gray < edge_threshold
    tp = int(np.logical_and(pred_e, tgt_e).sum())
    fp = int(np.logical_and(pred_e, ~tgt_e).sum())
    fn = int(np.logical_and(~pred_e, tgt_e).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def ssim_pair(pred_gray: np.ndarray, target_gray: np.ndarray) -> float:
    """SSIM between two HxW float arrays in [0, 1]. Falls back to MSE-based
    estimate if scikit-image isn't installed."""
    try:
        from skimage.metrics import structural_similarity as _ssim
    except ImportError:
        # Degraded fallback so unit tests don't require skimage at import time.
        mse = float(np.mean((pred_gray - target_gray) ** 2))
        return float(max(0.0, 1.0 - mse))
    return float(_ssim(pred_gray, target_gray, data_range=1.0))


# ---- Visual grid ---------------------------------------------------------


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render_triplet_row(
    cover_rgb: np.ndarray,
    target_gray: np.ndarray,
    pred_gray: np.ndarray,
) -> Image.Image:
    """Stitch (cover | target | pred) horizontally as a single RGB PIL Image."""
    h, w = target_gray.shape
    # Tile grayscale targets/preds to RGB for stitching
    tgt_rgb = np.stack([_to_uint8(target_gray)] * 3, axis=-1)
    pred_rgb = np.stack([_to_uint8(pred_gray)] * 3, axis=-1)
    cov = _to_uint8(cover_rgb) if cover_rgb.dtype != np.uint8 else cover_rgb
    if cov.shape[:2] != (h, w):
        cov_img = Image.fromarray(cov).resize((w, h), Image.BICUBIC)
        cov = np.asarray(cov_img)
    canvas = np.concatenate([cov, tgt_rgb, pred_rgb], axis=1)
    return Image.fromarray(canvas)


def render_grid(rows: list[Image.Image]) -> Image.Image:
    if not rows:
        return Image.new("RGB", (1, 1), (0, 0, 0))
    w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows) + 2 * (len(rows) - 1)
    canvas = Image.new("RGB", (w, total_h), (32, 32, 32))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 2
    return canvas


# ---- Full evaluation loop (torch path) -----------------------------------


@dataclass
class EvalConfig:
    splits_dir: str = "data/splits"
    covers_dir: str = "data/covers"
    labels_dir: str = "data/labels"
    ckpt_path: str = "runs/lora-default/best.pt"
    out_dir: str = "runs/lora-default/eval"
    img_size: int = 256
    n_residual_blocks: int = 9
    lora_rank: int = 8
    lora_alpha: float = 8.0
    skip_kernel_sizes_for_lora: tuple = ()
    use_lpips: bool = False
    grid_n: int = 16
    edge_threshold: float = 0.5


def run_eval(cfg: EvalConfig) -> dict[str, float]:
    """Score the val split end-to-end and write a visual grid."""
    import torch  # local import so this module is importable without torch
    from torch.utils.data import DataLoader

    from albumify.dataset import AlbumDataset
    from albumify.lora import freeze_non_lora, wrap_conv2d_layers
    from albumify.model import Generator
    from albumify.transforms import PairedTransformConfig

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build on CPU, wrap LoRA (creates new CPU convs), load state, then move
    # once to `device` so every parameter ends up co-located. Same ordering
    # as albumify/train.py — moving to device before wrap leaves the new
    # lora_A/lora_B convs on CPU and the first forward crashes.
    model = Generator(n_residual_blocks=cfg.n_residual_blocks)
    wrap_conv2d_layers(
        model, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
        skip_kernel_sizes=tuple(cfg.skip_kernel_sizes_for_lora),
    )
    freeze_non_lora(model)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    tf_cfg = PairedTransformConfig(out_size=cfg.img_size, resize_short_to=cfg.img_size)
    val_ds = AlbumDataset(
        Path(cfg.splits_dir) / "val.txt", cfg.covers_dir, cfg.labels_dir,
        train=False, cfg=tf_cfg,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    lpips_fn = None
    if cfg.use_lpips:
        import lpips  # type: ignore
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    rows: list[Image.Image] = []
    per_item: list[dict] = []
    sums = {"ssim": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "lpips": 0.0}
    n = 0
    with torch.no_grad():
        for cover, label, slug in val_loader:
            cover = cover.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pred = model(cover)

            cov_np = cover[0].cpu().permute(1, 2, 0).numpy()
            tgt_np = label[0, 0].cpu().numpy()
            prd_np = pred[0, 0].cpu().numpy()

            ssim = ssim_pair(prd_np, tgt_np)
            ef1 = edge_f1(prd_np, tgt_np, edge_threshold=cfg.edge_threshold)
            entry = {
                "slug": slug[0], "ssim": ssim, **ef1,
            }
            if lpips_fn is not None:
                pred_rgb = pred.expand(-1, 3, -1, -1) * 2 - 1
                tgt_rgb = label.expand(-1, 3, -1, -1) * 2 - 1
                lp = float(lpips_fn(pred_rgb, tgt_rgb).item())
                entry["lpips"] = lp
                sums["lpips"] += lp
            sums["ssim"] += ssim
            sums["f1"] += ef1["f1"]
            sums["precision"] += ef1["precision"]
            sums["recall"] += ef1["recall"]
            n += 1
            per_item.append(entry)
            if len(rows) < cfg.grid_n:
                rows.append(render_triplet_row(cov_np, tgt_np, prd_np))

    grid = render_grid(rows)
    grid.save(out_dir / "val_grid.png")
    summary = {k: (v / n if n else 0.0) for k, v in sums.items()}
    summary["n"] = n
    (out_dir / "per_item.json").write_text(json.dumps(per_item, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained LoRA checkpoint.")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--covers-dir", default="data/covers")
    p.add_argument("--labels-dir", default="data/labels")
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out-dir",   default="runs/lora-default/eval")
    p.add_argument("--img-size",  type=int, default=256)
    p.add_argument("--n-residual-blocks", type=int, default=9)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument("--use-lpips", action="store_true")
    p.add_argument("--grid-n",    type=int, default=16)
    args = p.parse_args()
    cfg = EvalConfig(
        splits_dir=args.splits_dir, covers_dir=args.covers_dir, labels_dir=args.labels_dir,
        ckpt_path=args.ckpt_path, out_dir=args.out_dir, img_size=args.img_size,
        n_residual_blocks=args.n_residual_blocks, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, use_lpips=args.use_lpips, grid_n=args.grid_n,
    )
    summary = run_eval(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
