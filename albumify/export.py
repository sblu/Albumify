"""Export a trained LoRA checkpoint to ONNX and INT8-quantize.

Steps:
1. Build Generator with the same LoRA config used during training.
2. Load the trained state_dict.
3. Merge LoRA into base Conv2d weights (so the exported graph has no LoRA structure).
4. Export to FP32 ONNX with dynamic spatial dims so we can run at any resolution.
5. (Optional) Dynamic INT8 quantize via onnxruntime.quantization.

INT8 dynamic quantization is the simplest path and works well for conv-heavy
models on CPU. For ARM (Pi 5), static quantization with a calibration dataset
would be slightly better, but dynamic is good enough for the on-device target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def export_onnx(
    *,
    ckpt_path: Path | str,
    out_fp32_path: Path | str,
    n_residual_blocks: int = 9,
    lora_rank: int = 8,
    lora_alpha: float = 8.0,
    skip_kernel_sizes_for_lora: tuple = (),
    example_size: int = 256,
    opset: int = 17,
) -> dict[str, str]:
    """Merge LoRA + export to FP32 ONNX. Returns a small report dict."""
    import torch
    from albumify.lora import freeze_non_lora, merge_all_lora, wrap_conv2d_layers
    from albumify.model import Generator

    out_fp32_path = Path(out_fp32_path)
    out_fp32_path.parent.mkdir(parents=True, exist_ok=True)

    model = Generator(n_residual_blocks=n_residual_blocks)
    wrap_conv2d_layers(
        model, rank=lora_rank, alpha=lora_alpha,
        skip_kernel_sizes=tuple(skip_kernel_sizes_for_lora),
    )
    freeze_non_lora(model)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    n_merged = merge_all_lora(model)
    model.eval()

    dummy = torch.randn(1, 3, example_size, example_size)
    torch.onnx.export(
        model, dummy, str(out_fp32_path),
        input_names=["cover"],
        output_names=["line"],
        dynamic_axes={
            "cover": {0: "batch", 2: "H", 3: "W"},
            "line":  {0: "batch", 2: "H", 3: "W"},
        },
        opset_version=opset,
    )
    return {"fp32_path": str(out_fp32_path), "merged_lora_layers": str(n_merged)}


def quantize_int8(
    *,
    fp32_path: Path | str,
    int8_path: Path | str,
) -> dict[str, str]:
    """Dynamic INT8 quantization (per-channel weights, dynamic activations)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return {"int8_path": str(int8_path)}


def smoke_compare(
    *,
    torch_ckpt_path: Path | str,
    fp32_onnx_path: Path | str,
    n_residual_blocks: int = 9,
    lora_rank: int = 8,
    lora_alpha: float = 8.0,
    skip_kernel_sizes_for_lora: tuple = (),
    img_size: int = 64,
    atol: float = 5e-4,
) -> dict[str, float]:
    """Run the same input through PyTorch (merged) and ONNX; report max abs diff."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from albumify.lora import freeze_non_lora, merge_all_lora, wrap_conv2d_layers
    from albumify.model import Generator

    model = Generator(n_residual_blocks=n_residual_blocks)
    wrap_conv2d_layers(
        model, rank=lora_rank, alpha=lora_alpha,
        skip_kernel_sizes=tuple(skip_kernel_sizes_for_lora),
    )
    freeze_non_lora(model)
    ckpt = torch.load(str(torch_ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    merge_all_lora(model)
    model.eval()

    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        y_torch = model(x).numpy()

    sess = ort.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(["line"], {"cover": x.numpy()})[0]

    max_diff = float(np.abs(y_torch - y_onnx).max())
    return {"max_abs_diff": max_diff, "passes_tolerance": float(max_diff <= atol)}


def main() -> None:
    p = argparse.ArgumentParser(description="Export LoRA checkpoint to ONNX + INT8.")
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out-dir",   default="artifacts")
    p.add_argument("--n-residual-blocks", type=int, default=9)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument("--example-size", type=int, default=256)
    p.add_argument("--opset",     type=int, default=17)
    p.add_argument("--int8",      action="store_true",
                   help="Also produce model.int8.onnx via dynamic quantization.")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    fp32 = out_dir / "model.fp32.onnx"
    report = export_onnx(
        ckpt_path=args.ckpt_path, out_fp32_path=fp32,
        n_residual_blocks=args.n_residual_blocks,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        example_size=args.example_size, opset=args.opset,
    )
    if args.int8:
        report.update(quantize_int8(fp32_path=fp32, int8_path=out_dir / "model.int8.onnx"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
