"""Local + Pi inference CLI: cover JPG -> line-drawing PNG via ONNX Runtime.

Uses CPU execution; no torch, no GPU. The exported model has dynamic
spatial dims, so you can pass --size to run at any resolution.

Usage:
    albumify --model artifacts/model.int8.onnx --in cover.jpg --out line.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def preprocess(img_path: Path | str, size: int = 256) -> np.ndarray:
    """Read JPG/PNG, resize-to-square + center-crop, return (1,3,H,W) float32 in [0,1]."""
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        # short-side resize
        short = min(w, h)
        scale = size / short
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        im = im.resize((new_w, new_h), Image.BICUBIC)
        # center crop
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        im = im.crop((left, top, left + size, top + size))
    arr = np.asarray(im, dtype=np.float32) / 255.0   # HxWx3
    arr = arr.transpose(2, 0, 1)[None, ...]          # 1x3xHxW
    return arr


def postprocess(out_arr: np.ndarray, threshold: float | None = None) -> Image.Image:
    """1x1xHxW float in [0,1] -> L-mode PIL Image.

    If `threshold` is set (a value in [0,1]), the output is binarized:
    pixels < threshold become 0 (pure black), others become 255 (pure white).
    Useful when the model produces faint "ghost" line drawings that need
    contrast boosted at inference time.
    """
    if threshold is not None:
        cutoff = float(threshold)
        mask = out_arr[0, 0] < cutoff
        a = np.where(mask, 0, 255).astype(np.uint8)
    else:
        a = np.clip(out_arr[0, 0] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="L")


def _build_session(model_path: Path | str, threads: int):
    import onnxruntime as ort  # local import so module is importable headless

    so = ort.SessionOptions()
    if threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path), sess_options=so, providers=["CPUExecutionProvider"],
    )


def infer_one(
    *,
    session,
    in_path: Path | str,
    out_path: Path | str,
    size: int = 256,
    threshold: float | None = None,
) -> Tuple[float, Path]:
    """Run inference on one file, save PNG, return (latency_s, out_path)."""
    x = preprocess(in_path, size=size)
    t0 = time.time()
    y = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: x})[0]
    dt = time.time() - t0
    img = postprocess(y, threshold=threshold)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return dt, out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a line-drawing PNG from an album cover.")
    p.add_argument("--model", required=True, type=Path,
                   help="Path to an ONNX file (FP32 or INT8).")
    p.add_argument("--in",  dest="in_path", required=True, type=Path)
    p.add_argument("--out", dest="out_path", required=True, type=Path)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--threads", type=int, default=0,
                   help="onnxruntime intra-op threads (0 = library default).")
    p.add_argument("--threshold", type=float, default=None,
                   help="If set (0..1), binarize the output: pixels darker than "
                        "this become pure black, lighter pixels pure white. "
                        "Useful when the model produces faint ghost lines.")
    args = p.parse_args()
    session = _build_session(args.model, args.threads)
    dt, out = infer_one(
        session=session, in_path=args.in_path, out_path=args.out_path,
        size=args.size, threshold=args.threshold,
    )
    print(f"wrote {out} in {dt * 1000:.1f} ms (size={args.size}, model={args.model.name})")


if __name__ == "__main__":
    main()
