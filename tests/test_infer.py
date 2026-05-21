"""Tests for the infer.py CLI pipeline. Pre/post pure-numpy; full path
requires onnxruntime + a tiny ONNX file."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from albumify.infer import postprocess, preprocess


def test_preprocess_returns_expected_shape_and_range(tmp_path: Path):
    Image.new("RGB", (640, 320), (128, 128, 128)).save(tmp_path / "c.jpg")
    x = preprocess(tmp_path / "c.jpg", size=128)
    assert x.shape == (1, 3, 128, 128)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() <= x.max() <= 1.0


def test_preprocess_resizes_short_side_then_center_crops(tmp_path: Path):
    # 400x100 wide rectangle -> short side 100 -> scale to 64 -> width 256, crop center 64
    img = Image.new("RGB", (400, 100), (200, 0, 0))
    img.save(tmp_path / "c.jpg")
    x = preprocess(tmp_path / "c.jpg", size=64)
    assert x.shape == (1, 3, 64, 64)


def test_postprocess_clips_and_returns_L_mode():
    arr = np.array([[[[0.5, 1.2], [-0.3, 0.8]]]], dtype=np.float32)
    img = postprocess(arr)
    assert img.mode == "L"
    assert img.size == (2, 2)
    # 0.5*255=127, 1.2 clipped to 255, -0.3 clipped to 0, 0.8*255=204
    arr_back = np.asarray(img)
    assert int(arr_back[0, 0]) == 128
    assert int(arr_back[0, 1]) == 255
    assert int(arr_back[1, 0]) == 0
    assert int(arr_back[1, 1]) == 204


def test_full_inference_via_tiny_onnx_model(tmp_path: Path):
    """Build a 1-conv identity-ish ONNX manually and verify the CLI flow runs."""
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")

    import torch.nn as nn
    from albumify.infer import _build_session, infer_one

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 1, 1)
            nn.init.zeros_(self.c.weight)
            nn.init.zeros_(self.c.bias)
        def forward(self, x):
            return torch.sigmoid(self.c(x))

    onnx_path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        Tiny().eval(), torch.randn(1, 3, 64, 64), str(onnx_path),
        input_names=["cover"], output_names=["line"],
        dynamic_axes={"cover": {0: "batch", 2: "H", 3: "W"},
                       "line":  {0: "batch", 2: "H", 3: "W"}},
        opset_version=17,
    )

    Image.new("RGB", (320, 240), (100, 50, 200)).save(tmp_path / "c.jpg")
    session = _build_session(onnx_path, threads=1)
    dt, out = infer_one(
        session=session, in_path=tmp_path / "c.jpg",
        out_path=tmp_path / "out.png", size=64,
    )
    assert out.exists()
    assert dt >= 0
    img = Image.open(out)
    assert img.size == (64, 64)
    assert img.mode == "L"
