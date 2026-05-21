"""Training entry point for the LoRA fine-tune.

Loads a pretrained Informative-Drawings generator, wraps Conv2d layers with
LoRA-Conv adapters, freezes everything else, and trains the adapters with
L1 (+ optional VGG perceptual) loss on (cover, Gemini-label) pairs.

Designed to be runnable on a single GCP T4 (16 GB) at batch 8 / 256x256.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from albumify.dataset import AlbumDataset
from albumify.lora import (
    count_lora_params,
    freeze_non_lora,
    lora_parameters,
    wrap_conv2d_layers,
)
from albumify.loss import L1PerceptualLoss, VGGPerceptualLoss
from albumify.model import Generator, load_pretrained
from albumify.transforms import PairedTransformConfig


@dataclass
class TrainConfig:
    splits_dir: str = "data/splits"
    covers_dir: str = "data/covers"
    labels_dir: str = "data/labels"
    pretrained_ckpt: Optional[str] = None  # Informative-Drawings checkpoint
    out_dir: str = "runs/lora-default"
    img_size: int = 256
    resize_short_to: int = 288
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    lora_rank: int = 8
    lora_alpha: float = 8.0
    l1_weight: float = 1.0
    perceptual_weight: float = 0.1
    use_vgg_pretrained: bool = True
    n_residual_blocks: int = 9
    seed: int = 0
    log_every: int = 20
    eval_every: int = 1  # epochs
    skip_kernel_sizes_for_lora: tuple = ()  # e.g. (7,) to skip the 7x7 head/tail


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_vgg(use_pretrained: bool, device: torch.device) -> Optional[VGGPerceptualLoss]:
    try:
        from torchvision import models
    except ImportError:
        return None
    weights = models.VGG16_Weights.IMAGENET1K_V1 if use_pretrained else None
    vgg = models.vgg16(weights=weights).features.eval()
    return VGGPerceptualLoss(vgg).to(device)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: L1PerceptualLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    l1_acc = 0.0
    perc_acc = 0.0
    n_batches = 0
    with torch.no_grad():
        for cover, label, _ in loader:
            cover = cover.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pred = model(cover)
            res = loss_fn(pred, label)
            total += float(res["total"])
            l1_acc += float(res["l1"])
            if "perc" in res:
                perc_acc += float(res["perc"])
            n_batches += 1
    model.train()
    if n_batches == 0:
        return {"val_total": 0.0, "val_l1": 0.0, "val_perc": 0.0}
    return {
        "val_total": total / n_batches,
        "val_l1": l1_acc / n_batches,
        "val_perc": perc_acc / n_batches,
    }


def _try_tb_writer(log_dir: Path):
    """Create a TensorBoard SummaryWriter if torch.utils.tensorboard is installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(log_dir=str(log_dir))


def train(cfg: TrainConfig) -> dict[str, float]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    _seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb = _try_tb_writer(out_dir / "tb")
    metrics_jsonl = open(out_dir / "metrics.jsonl", "a", buffering=1)

    # ---- Model + LoRA ----
    # Build on CPU so the pretrained checkpoint (saved on CPU) loads without
    # device-conversion surprises, wrap with LoRA (creates new CPU conv layers),
    # then move the whole thing to `device` in one shot so every parameter
    # ends up on the same device.
    model = Generator(n_residual_blocks=cfg.n_residual_blocks)
    if cfg.pretrained_ckpt:
        missing, unexpected = load_pretrained(model, cfg.pretrained_ckpt, map_location="cpu")
        print(f"[pretrained] missing={len(missing)} unexpected={len(unexpected)}")
    n_wrapped = wrap_conv2d_layers(
        model, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
        skip_kernel_sizes=tuple(cfg.skip_kernel_sizes_for_lora),
    )
    freeze_non_lora(model)
    model = model.to(device)
    n_lora = count_lora_params(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] wrapped={n_wrapped} lora_params={n_lora:,} / total={n_total:,}")

    # ---- Data ----
    tf_cfg = PairedTransformConfig(out_size=cfg.img_size, resize_short_to=cfg.resize_short_to)
    train_ds = AlbumDataset(
        Path(cfg.splits_dir) / "train.txt", cfg.covers_dir, cfg.labels_dir,
        train=True, cfg=tf_cfg, seed=cfg.seed,
    )
    val_ds = AlbumDataset(
        Path(cfg.splits_dir) / "val.txt", cfg.covers_dir, cfg.labels_dir,
        train=False, cfg=tf_cfg, seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Loss ----
    vgg = _make_vgg(cfg.use_vgg_pretrained, device) if cfg.perceptual_weight > 0 else None
    loss_fn = L1PerceptualLoss(
        l1_weight=cfg.l1_weight, perceptual_weight=cfg.perceptual_weight, vgg=vgg,
    ).to(device)

    # ---- Optimizer ----
    opt = torch.optim.AdamW(
        list(lora_parameters(model)), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ---- Training loop ----
    best_val = math.inf
    global_step = 0
    last_log: dict[str, float] = {}
    for epoch in range(cfg.epochs):
        t_epoch = time.time()
        model.train()
        running_total = 0.0
        n_batches = 0
        for cover, label, _ in train_loader:
            cover = cover.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pred = model(cover)
            res = loss_fn(pred, label)
            opt.zero_grad(set_to_none=True)
            res["total"].backward()
            opt.step()
            running_total += float(res["total"].detach())
            n_batches += 1
            global_step += 1
            if tb is not None:
                tb.add_scalar("loss/train_step_total", float(res["total"]), global_step)
                tb.add_scalar("loss/train_step_l1", float(res["l1"]), global_step)
                if "perc" in res:
                    tb.add_scalar("loss/train_step_perc", float(res["perc"]), global_step)
            if global_step % cfg.log_every == 0:
                avg = running_total / max(1, n_batches)
                print(f"epoch={epoch} step={global_step} loss={avg:.4f}")
        avg_train = running_total / max(1, n_batches)

        if (epoch + 1) % cfg.eval_every == 0:
            metrics = _evaluate(model, val_loader, loss_fn, device)
        else:
            metrics = {"val_total": float("nan"), "val_l1": float("nan"), "val_perc": float("nan")}
        last_log = {
            "epoch": epoch + 1,
            "train_total": avg_train,
            **metrics,
            "epoch_s": time.time() - t_epoch,
        }
        print("[epoch]", json.dumps(last_log, default=lambda x: round(x, 4) if isinstance(x, float) else x))
        metrics_jsonl.write(json.dumps(last_log) + "\n")
        if tb is not None:
            tb.add_scalar("loss/train_epoch_total", avg_train, epoch + 1)
            if not math.isnan(metrics.get("val_total", float("nan"))):
                tb.add_scalar("loss/val_total", metrics["val_total"], epoch + 1)
                tb.add_scalar("loss/val_l1", metrics["val_l1"], epoch + 1)
                if metrics.get("val_perc", 0) > 0:
                    tb.add_scalar("loss/val_perc", metrics["val_perc"], epoch + 1)
            tb.flush()

        # Save state on best val
        val_total = metrics.get("val_total", float("nan"))
        if not math.isnan(val_total) and val_total < best_val:
            best_val = val_total
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_total": val_total},
                out_dir / "best.pt",
            )
            print(f"[ckpt] saved best.pt (val_total={val_total:.4f})")
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch + 1},
            out_dir / "last.pt",
        )

    summary = {"best_val_total": float(best_val), **last_log}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    metrics_jsonl.close()
    if tb is not None:
        tb.close()
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train LoRA fine-tune of Informative-Drawings.")
    p.add_argument("--splits-dir",        default="data/splits")
    p.add_argument("--covers-dir",        default="data/covers")
    p.add_argument("--labels-dir",        default="data/labels")
    p.add_argument("--pretrained-ckpt",   default=None)
    p.add_argument("--out-dir",           default="runs/lora-default")
    p.add_argument("--img-size",          type=int, default=256)
    p.add_argument("--epochs",            type=int, default=30)
    p.add_argument("--batch-size",        type=int, default=8)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--num-workers",       type=int, default=4)
    p.add_argument("--lora-rank",         type=int, default=8)
    p.add_argument("--lora-alpha",        type=float, default=8.0)
    p.add_argument("--l1-weight",         type=float, default=1.0)
    p.add_argument("--perceptual-weight", type=float, default=0.1)
    p.add_argument("--no-vgg-pretrained", action="store_true")
    p.add_argument("--n-residual-blocks", type=int, default=9)
    p.add_argument("--seed",              type=int, default=0)
    args = p.parse_args()
    cfg = TrainConfig(
        splits_dir=args.splits_dir, covers_dir=args.covers_dir, labels_dir=args.labels_dir,
        pretrained_ckpt=args.pretrained_ckpt, out_dir=args.out_dir,
        img_size=args.img_size, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, num_workers=args.num_workers,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        l1_weight=args.l1_weight, perceptual_weight=args.perceptual_weight,
        use_vgg_pretrained=not args.no_vgg_pretrained,
        n_residual_blocks=args.n_residual_blocks, seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()
