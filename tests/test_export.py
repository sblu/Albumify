"""End-to-end ONNX export smoke test. Requires torch + onnx + onnxruntime."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")            # noqa: E402
pytest.importorskip("onnx")                     # noqa: E402
ort = pytest.importorskip("onnxruntime")        # noqa: E402

from pathlib import Path

import numpy as np

from albumify.lora import wrap_conv2d_layers
from albumify.model import Generator
from albumify.export import export_onnx, quantize_int8, smoke_compare


def test_export_onnx_round_trip_matches_torch(tmp_path: Path):
    # Build a tiny model and save a checkpoint
    g = Generator(n_residual_blocks=1)
    wrap_conv2d_layers(g, rank=2, alpha=2.0)
    # Perturb LoRA B so merge actually does something non-trivial.
    for m in g.modules():
        if hasattr(m, "lora_B"):
            torch.nn.init.normal_(m.lora_B.weight, std=0.05)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": g.state_dict()}, ckpt_path)

    fp32 = tmp_path / "model.fp32.onnx"
    export_onnx(
        ckpt_path=ckpt_path, out_fp32_path=fp32,
        n_residual_blocks=1, lora_rank=2, lora_alpha=2.0, example_size=64,
    )
    assert fp32.exists() and fp32.stat().st_size > 0

    result = smoke_compare(
        torch_ckpt_path=ckpt_path, fp32_onnx_path=fp32,
        n_residual_blocks=1, lora_rank=2, lora_alpha=2.0, img_size=64,
        atol=2e-3,
    )
    assert result["passes_tolerance"] == 1.0, result


def test_int8_quantization_produces_runnable_model(tmp_path: Path):
    g = Generator(n_residual_blocks=1)
    wrap_conv2d_layers(g, rank=2, alpha=2.0)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": g.state_dict()}, ckpt_path)

    fp32 = tmp_path / "model.fp32.onnx"
    int8 = tmp_path / "model.int8.onnx"
    export_onnx(
        ckpt_path=ckpt_path, out_fp32_path=fp32,
        n_residual_blocks=1, lora_rank=2, lora_alpha=2.0, example_size=64,
    )
    quantize_int8(fp32_path=fp32, int8_path=int8)
    assert int8.exists() and int8.stat().st_size > 0
    # And the INT8 model should run (we don't assert numerical fidelity here —
    # INT8 loses precision; checking it doesn't crash is the smoke).
    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).randn(1, 3, 64, 64).astype(np.float32)
    y = sess.run(["line"], {"cover": x})[0]
    assert y.shape == (1, 1, 64, 64)


def test_export_wraps_sigmoid_for_apply_sigmoid_false_ckpt(tmp_path: Path):
    """Plan C ckpts (apply_sigmoid=False) must produce ONNX in [0,1]."""
    g = Generator(n_residual_blocks=1, ngf=8, sigmoid=False)
    ckpt_path = tmp_path / "ckpt-noprep.pt"
    torch.save({
        "model_state_dict": g.state_dict(),
        "apply_sigmoid": False,
        "loss_type": "bce",
    }, ckpt_path)

    fp32 = tmp_path / "plan-c.fp32.onnx"
    export_onnx(
        ckpt_path=ckpt_path, out_fp32_path=fp32,
        n_residual_blocks=1, ngf=8,
        use_lora=False,
        example_size=64,
    )
    assert fp32.exists() and fp32.stat().st_size > 0

    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    cover = np.random.RandomState(0).rand(1, 3, 64, 64).astype(np.float32)
    out = sess.run(["line"], {"cover": cover})[0]
    assert out.shape == (1, 1, 64, 64)
    assert (out >= 0).all() and (out <= 1).all(), \
        f"ONNX output out of [0,1]: min={out.min()}, max={out.max()}"
