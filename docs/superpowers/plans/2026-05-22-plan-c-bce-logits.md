# Plan C — BCE-with-logits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--loss bce` training mode that drops the terminal sigmoid and trains with `binary_cross_entropy_with_logits` on hard-thresholded targets. Sigmoid is re-applied at eval/export/infer time based on checkpoint metadata so existing v0.1.0 / v0.2.0 artifacts and the L1 path are unaffected.

**Architecture:** Parallel structures to today's L1 path — `edge_weighted_bce_logits` alongside `edge_weighted_l1`, `BCELogitsPerceptualLoss` alongside `L1PerceptualLoss`. `TrainConfig` gains `loss_type` and writes `apply_sigmoid` + `loss_type` into the checkpoint. `eval.py` and `export.py` read that metadata, build `Generator(sigmoid=...)` accordingly, and wrap `torch.sigmoid()` externally when the checkpoint was trained sigmoid-free.

**Tech Stack:** Python 3.10+, PyTorch, torchvision (VGG perceptual), onnx + onnxruntime, pytest.

**Spec:** `docs/superpowers/specs/2026-05-21-plan-c-bce-logits-design.md`

---

## File Map

- Modify `albumify/loss.py` — add `edge_weighted_bce_logits` + `BCELogitsPerceptualLoss`.
- Modify `albumify/train.py` — new `loss_type` config field, `--loss` CLI flag, sigmoid toggle on Generator, branch on loss class, write `apply_sigmoid` + `loss_type` into ckpt, make `_evaluate` loss-agnostic.
- Modify `albumify/eval.py` — read `apply_sigmoid` from ckpt, build `Generator` with matching flag, wrap forward with `torch.sigmoid` when needed.
- Modify `albumify/export.py` — same read; wrap loaded model with a `_SigmoidWrap` `nn.Module` before `torch.onnx.export` so the ONNX always emits values in [0, 1].
- Create `tests/test_loss.py` — unit tests for the two new loss units.
- Modify `tests/test_model.py` — assert `Generator(sigmoid=False)` produces real logits.
- Modify `tests/test_train.py` — smoke test for `--loss bce`.
- Modify `tests/test_eval.py` — eval-time sigmoid wrap test.
- Modify `tests/test_export.py` — exported ONNX in [0, 1] for sigmoid-less ckpt.

---

### Task 1: Create feature branch

**Files:** none (git state only)

- [ ] **Step 1: Confirm working tree is clean and on `main`**

Run: `git status -sb`
Expected: `## main...origin/main` with no `M` / `A` / `D` entries. Untracked exploration files (`AbbeyRoad.jpg`, etc.) are fine.

- [ ] **Step 2: Pull latest**

Run: `git pull --ff-only`
Expected: `Already up to date.`

- [ ] **Step 3: Create + switch to feature branch**

Run: `git switch -c feat/plan-c-bce-logits`
Expected: `Switched to a new branch 'feat/plan-c-bce-logits'`

- [ ] **Step 4: Confirm venv ready**

Run: `. .venv/bin/activate && python -c "import torch, pytest; print(torch.__version__)"`
Expected: a version like `2.x.y+cpu` prints, no `ModuleNotFoundError`.

---

### Task 2: Failing tests for `edge_weighted_bce_logits`

**Files:**
- Create: `tests/test_loss.py`
- Test: `pytest tests/test_loss.py -v`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loss.py` with:

```python
"""Tests for the BCE-with-logits + perceptual loss path (Plan C)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

import torch.nn.functional as F

from albumify.loss import (
    BCELogitsPerceptualLoss,
    VGGPerceptualLoss,
    edge_weighted_bce_logits,
)


# ---- edge_weighted_bce_logits ---------------------------------------------

def test_bce_logits_returns_finite_scalar_with_gradient():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 1, 8, 8)
    loss = edge_weighted_bce_logits(logits, target, edge_weight=0.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_bce_logits_with_edge_weight_0_matches_plain_bce():
    """edge_weight=0 should equal F.binary_cross_entropy_with_logits on the binarized target."""
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 16, 16)
    # Target with both edges (<0.5) and background (>=0.5).
    target = torch.rand(1, 1, 16, 16)
    target_bin = (target >= 0.5).float()

    ours = edge_weighted_bce_logits(logits, target, edge_threshold=0.5, edge_weight=0.0)
    ref = F.binary_cross_entropy_with_logits(logits, target_bin)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_bce_logits_edge_weight_increases_loss_when_edge_present():
    """Adding edge_weight on a target with edge pixels must produce a larger loss."""
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 16, 16)
    target = torch.zeros(1, 1, 16, 16)
    target[..., :2, :] = 0.0    # row of edges (dark)
    target[..., 2:, :] = 1.0    # background (bright)

    base = edge_weighted_bce_logits(logits, target, edge_weight=0.0)
    weighted = edge_weighted_bce_logits(logits, target, edge_weight=19.0)
    assert weighted.item() > base.item()


# ---- BCELogitsPerceptualLoss ---------------------------------------------

def test_bce_logits_perceptual_loss_no_vgg_returns_bce_only():
    loss_fn = BCELogitsPerceptualLoss(
        bce_weight=1.0, perceptual_weight=0.0, edge_weight=0.0,
    )
    logits = torch.randn(1, 1, 8, 8)
    target = torch.rand(1, 1, 8, 8)
    res = loss_fn(logits, target)
    assert set(res.keys()) == {"bce", "total"}
    assert torch.allclose(res["total"], res["bce"])


def test_bce_logits_perceptual_loss_requires_vgg_when_weight_positive():
    with pytest.raises(ValueError):
        BCELogitsPerceptualLoss(perceptual_weight=0.1, vgg=None)


def test_bce_logits_perceptual_loss_with_random_vgg_combines_terms():
    """Random-init VGG just for shape; check 'total' = bce + 0.1 * perc."""
    import torchvision.models as tv_models

    vgg_features = tv_models.vgg16(weights=None).features
    vgg = VGGPerceptualLoss(vgg_features)
    loss_fn = BCELogitsPerceptualLoss(
        bce_weight=1.0, perceptual_weight=0.1,
        edge_weight=0.0, vgg=vgg,
    )
    logits = torch.randn(1, 1, 32, 32)
    target = torch.rand(1, 1, 32, 32)
    res = loss_fn(logits, target)
    assert {"bce", "perc", "total"} <= set(res.keys())
    expected_total = 1.0 * res["bce"] + 0.1 * res["perc"]
    assert torch.allclose(res["total"], expected_total, atol=1e-6)
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `. .venv/bin/activate && pytest tests/test_loss.py -v`
Expected: collection error or `ImportError: cannot import name 'edge_weighted_bce_logits' from 'albumify.loss'` (red — function doesn't exist yet).

- [ ] **Step 3: Commit**

Run:
```bash
git add tests/test_loss.py
git commit -m "test(loss): failing tests for edge_weighted_bce_logits + BCELogitsPerceptualLoss"
```

---

### Task 3: Implement `edge_weighted_bce_logits` + `BCELogitsPerceptualLoss`

**Files:**
- Modify: `albumify/loss.py`
- Test: `pytest tests/test_loss.py -v`

- [ ] **Step 1: Append the new function + class to `albumify/loss.py`**

After the existing `class L1PerceptualLoss` block (end of file), append:

```python
def edge_weighted_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    edge_threshold: float = 0.5,
    edge_weight: float = 19.0,
) -> torch.Tensor:
    """BCE-with-logits on hard-thresholded targets, with edge-pixel weighting.

    Mirrors `edge_weighted_l1`: target is grayscale in [0, 1], thresholded to
    {0, 1} at `edge_threshold`. A dilated edge mask (max_pool2d k=3) up-weights
    edge pixels (1 + edge_weight)x so the loss is not dominated by the ~95%
    background pixels.

    `edge_weight=0` reduces to plain `F.binary_cross_entropy_with_logits` on
    the binarized target.
    """
    target_bin = (target >= edge_threshold).float()
    if edge_weight == 0:
        return F.binary_cross_entropy_with_logits(logits, target_bin)
    edges = (target < edge_threshold).float()
    edges = F.max_pool2d(edges, kernel_size=3, stride=1, padding=1)
    weights = 1.0 + edge_weight * edges
    return F.binary_cross_entropy_with_logits(logits, target_bin, weight=weights)


class BCELogitsPerceptualLoss(nn.Module):
    """BCE-with-logits + optional VGG perceptual on sigmoid(logits)."""

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        edge_weight: float = 19.0,
        edge_threshold: float = 0.5,
        vgg: Optional["VGGPerceptualLoss"] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.edge_threshold = edge_threshold
        self.vgg = vgg
        if perceptual_weight > 0 and vgg is None:
            raise ValueError("perceptual_weight > 0 requires a VGGPerceptualLoss instance")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        bce = edge_weighted_bce_logits(
            logits, target,
            edge_threshold=self.edge_threshold, edge_weight=self.edge_weight,
        )
        total = self.bce_weight * bce
        result: dict[str, torch.Tensor] = {"bce": bce.detach()}
        if self.perceptual_weight > 0 and self.vgg is not None:
            pred_prob = torch.sigmoid(logits)
            perc = self.vgg(pred_prob, target)
            total = total + self.perceptual_weight * perc
            result["perc"] = perc.detach()
        result["total"] = total
        return result
```

- [ ] **Step 2: Run loss tests, expect pass**

Run: `. .venv/bin/activate && pytest tests/test_loss.py -v`
Expected: all 6 tests pass. (`-v` shows them by name.)

- [ ] **Step 3: Run full test file, sanity check nothing else regressed**

Run: `pytest tests/test_loss.py tests/test_model.py -v --no-header`
Expected: previously-passing model tests still pass; new loss tests pass.

- [ ] **Step 4: Commit**

```bash
git add albumify/loss.py
git commit -m "feat(loss): add edge_weighted_bce_logits + BCELogitsPerceptualLoss"
```

---

### Task 4: Test that `Generator(sigmoid=False)` produces real logits

**Files:**
- Modify: `tests/test_model.py`
- Test: `pytest tests/test_model.py -v -k sigmoid`

- [ ] **Step 1: Append the failing test**

After the existing `test_generator_output_shape_matches_input_size` test (around line 28), append:

```python
def test_generator_with_sigmoid_false_produces_real_logits():
    """sigmoid=False is the Plan C training mode — output should not be clamped to [0,1]."""
    g = Generator(sigmoid=False)
    g.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = g(x)
    assert y.shape == (1, 1, 64, 64)
    # We can't guarantee both signs on every random input, but at least one
    # element must lie outside [0, 1] when sigmoid is off and weights are
    # default-init non-trivial — otherwise the toggle is silently broken.
    assert (y < 0).any() or (y > 1).any(), \
        f"sigmoid=False output should not be in [0,1]; got min={y.min()}, max={y.max()}"
```

- [ ] **Step 2: Run the new test, expect pass (model already supports the flag)**

Run: `. .venv/bin/activate && pytest tests/test_model.py::test_generator_with_sigmoid_false_produces_real_logits -v`
Expected: PASS. (The `sigmoid: bool = True` constructor parameter already plumbs to `model4` — see `albumify/model.py:47-90`.)

If it FAILS (output happens to land in [0,1] for the random init): rerun with `torch.manual_seed(42)` prepended or pick a slightly larger image so logit variance is bigger. Don't change the production code.

- [ ] **Step 3: Run full test_model.py**

Run: `pytest tests/test_model.py -v --no-header`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_model.py
git commit -m "test(model): cover Generator(sigmoid=False) logits path"
```

---

### Task 5: Plumb `--loss` flag through `TrainConfig` + `train()` + CLI

**Files:**
- Modify: `albumify/train.py`
- Test: deferred to Task 6 (smoke test).

- [ ] **Step 1: Extend the import block**

At the top of `albumify/train.py`, change:

```python
from albumify.loss import L1PerceptualLoss, VGGPerceptualLoss
```

to:

```python
from albumify.loss import BCELogitsPerceptualLoss, L1PerceptualLoss, VGGPerceptualLoss
```

- [ ] **Step 2: Extend `TrainConfig`**

In the `@dataclass class TrainConfig:` block (around line 36–60), append two fields immediately before the closing of the dataclass (after `seed: int = 0`):

```python
    loss_type: str = "l1"                  # "l1" (default) or "bce"
    bce_weight: float = 1.0                # weight on the BCE term when loss_type == "bce"
```

- [ ] **Step 3: Toggle Generator's sigmoid in `train()`**

In `train()` find:

```python
    model = Generator(n_residual_blocks=cfg.n_residual_blocks, ngf=cfg.ngf)
```

and replace with:

```python
    apply_sigmoid_in_model = cfg.loss_type != "bce"
    model = Generator(
        n_residual_blocks=cfg.n_residual_blocks,
        ngf=cfg.ngf,
        sigmoid=apply_sigmoid_in_model,
    )
```

- [ ] **Step 4: Branch on loss class**

Find the existing loss construction in `train()`:

```python
    loss_fn = L1PerceptualLoss(
        l1_weight=cfg.l1_weight,
        perceptual_weight=cfg.perceptual_weight,
        edge_weight=cfg.edge_weight,
        edge_threshold=cfg.edge_threshold,
        vgg=vgg,
    )
```

Replace with:

```python
    if cfg.loss_type == "bce":
        loss_fn = BCELogitsPerceptualLoss(
            bce_weight=cfg.bce_weight,
            perceptual_weight=cfg.perceptual_weight,
            edge_weight=cfg.edge_weight,
            edge_threshold=cfg.edge_threshold,
            vgg=vgg,
        )
    else:
        loss_fn = L1PerceptualLoss(
            l1_weight=cfg.l1_weight,
            perceptual_weight=cfg.perceptual_weight,
            edge_weight=cfg.edge_weight,
            edge_threshold=cfg.edge_threshold,
            vgg=vgg,
        )
```

- [ ] **Step 5: Make `_evaluate` loss-agnostic**

Find the `_evaluate` function (around line 87–115) and replace its body with:

```python
def _evaluate(
    model,
    loader,
    loss_fn,
    device,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    primary_acc = 0.0           # tracks "l1" or "bce" depending on loss_fn
    perc_acc = 0.0
    n_batches = 0
    primary_key: Optional[str] = None
    with torch.no_grad():
        for cover, label, _ in loader:
            cover = cover.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            pred = model(cover)
            res = loss_fn(pred, label)
            if primary_key is None:
                primary_key = "bce" if "bce" in res else "l1"
            total += float(res["total"])
            primary_acc += float(res[primary_key])
            if "perc" in res:
                perc_acc += float(res["perc"])
            n_batches += 1
    model.train()
    if n_batches == 0:
        return {"val_total": 0.0, "val_l1": 0.0, "val_bce": 0.0, "val_perc": 0.0}
    out = {
        "val_total": total / n_batches,
        "val_perc": perc_acc / n_batches,
    }
    out[f"val_{primary_key}"] = primary_acc / n_batches
    return out
```

- [ ] **Step 6: Save sigmoid + loss type into checkpoint**

Find the two `torch.save({...}, ...)` calls (around lines 250–258) and update each dict to include the new keys.

Replace the `best.pt` save:

```python
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_total": val_total},
                out_dir / "best.pt",
            )
```

with:

```python
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_total": val_total,
                    "apply_sigmoid": apply_sigmoid_in_model,
                    "loss_type": cfg.loss_type,
                },
                out_dir / "best.pt",
            )
```

Same edit for the `last.pt` save: add the two extra keys.

- [ ] **Step 7: Train-step logging**

The train loop's per-step `print(f"epoch={epoch} step={global_step} loss={avg:.4f}")` line and TB scalar writes use `res["l1"]` and `res["total"]`. Find:

```python
                tb.add_scalar("loss/train_step_total", float(res["total"].detach()), global_step)
                tb.add_scalar("loss/train_step_l1", float(res["l1"]), global_step)
```

and replace with:

```python
                tb.add_scalar("loss/train_step_total", float(res["total"].detach()), global_step)
                primary_key = "bce" if "bce" in res else "l1"
                tb.add_scalar(f"loss/train_step_{primary_key}", float(res[primary_key]), global_step)
```

(The per-epoch TB scalars in the same block use `metrics["val_l1"]` — these need parallel treatment.) Find:

```python
                if metrics is not None:
                    tb.add_scalar("loss/val_total", metrics["val_total"], epoch + 1)
                    tb.add_scalar("loss/val_l1", metrics["val_l1"], epoch + 1)
                    if "val_perc" in metrics and metrics["val_perc"] > 0:
                        tb.add_scalar("loss/val_perc", metrics["val_perc"], epoch + 1)
```

and replace with:

```python
                if metrics is not None:
                    tb.add_scalar("loss/val_total", metrics["val_total"], epoch + 1)
                    val_primary = "val_bce" if "val_bce" in metrics else "val_l1"
                    tb.add_scalar(f"loss/{val_primary}", metrics[val_primary], epoch + 1)
                    if "val_perc" in metrics and metrics["val_perc"] > 0:
                        tb.add_scalar("loss/val_perc", metrics["val_perc"], epoch + 1)
```

- [ ] **Step 8: Add `--loss` CLI flag + plumbing**

In the `def main()` argparse block, after `--edge-threshold` add:

```python
    p.add_argument("--loss", choices=("l1", "bce"), default="l1",
                   help="Training loss. 'l1' = today's L1 (+ perceptual). "
                        "'bce' = BCE-with-logits on hard-thresholded targets "
                        "(Plan C: drops sigmoid from Generator).")
    p.add_argument("--bce-weight", type=float, default=1.0,
                   help="Weight on the BCE term when --loss bce.")
```

And in the `cfg = TrainConfig(...)` block, append the two new fields:

```python
        loss_type=args.loss,
        bce_weight=args.bce_weight,
```

Also: when `--loss bce` and the user did not pass `--edge-weight`, default to `19.0`. After parsing args:

```python
    if args.loss == "bce" and args.edge_weight == 0.0:
        args.edge_weight = 19.0
```

(insert immediately after `args = p.parse_args()`).

- [ ] **Step 9: Log empirical edge fraction at startup (sanity check)**

Inside `train()`, after the train dataset is built and before the training loop starts, add:

```python
    # Sanity-check the edge fraction assumption (BCE pos-weight derivation).
    if cfg.loss_type == "bce":
        from albumify.dataset import _read_split_slugs  # already used by AlbumDataset
        import numpy as _np
        from PIL import Image as _Image
        sample = list((Path(cfg.labels_dir)).glob("*.png"))[:20]
        if sample:
            fracs = []
            for p in sample:
                arr = _np.asarray(_Image.open(p).convert("L"), dtype=_np.float32) / 255.0
                fracs.append(float((arr < cfg.edge_threshold).mean()))
            mean_frac = float(_np.mean(fracs))
            print(f"[bce] empirical edge fraction over {len(fracs)} labels: {mean_frac:.3f} "
                  f"(default edge_weight=19 assumes ~0.05)")
```

Note: the `_read_split_slugs` import is harmless if it doesn't exist on disk — only the `Path.glob` is used. If linting flags the unused name, drop that import line; it's a defensive guard against future refactors.

- [ ] **Step 10: Quick syntax check**

Run: `. .venv/bin/activate && python -c "from albumify import train as _; print('train.py imports OK')"`
Expected: `train.py imports OK`. If `SyntaxError`, fix locally and re-run.

- [ ] **Step 11: Existing train smoke test still passes**

Run: `pytest tests/test_train.py -v --no-header`
Expected: the existing `test_train_smoke_runs_two_epochs_and_writes_ckpt` still passes (default `loss_type="l1"` keeps behavior identical).

- [ ] **Step 12: Commit**

```bash
git add albumify/train.py
git commit -m "feat(train): --loss bce path with sigmoid toggle and ckpt metadata"
```

---

### Task 6: Smoke test for `--loss bce`

**Files:**
- Modify: `tests/test_train.py`
- Test: `pytest tests/test_train.py -v`

- [ ] **Step 1: Append the new smoke test**

After the existing test in `tests/test_train.py`, append:

```python
def test_train_smoke_bce_loss_writes_apply_sigmoid_false(tmp_path: Path):
    """--loss bce should train Generator(sigmoid=False) and stamp ckpt metadata."""
    covers = tmp_path / "covers"; covers.mkdir()
    labels = tmp_path / "labels"; labels.mkdir()
    splits = tmp_path / "splits"; splits.mkdir()
    runs = tmp_path / "runs"

    slugs = [f"s{i}" for i in range(6)]
    for s in slugs:
        Image.new("RGB", (64, 64), (40, 80, 160)).save(covers / f"{s}.jpg")
        lbl = Image.new("L", (64, 64), 255)
        for x in range(64):
            lbl.putpixel((x, 32), 0)
        lbl.save(labels / f"{s}.png")
    split_mod.write_splits(splits, slugs[:4], slugs[4:])

    cfg = TrainConfig(
        splits_dir=str(splits),
        covers_dir=str(covers),
        labels_dir=str(labels),
        out_dir=str(runs / "smoke-bce"),
        img_size=64,
        resize_short_to=72,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        num_workers=0,
        use_lora=False,
        n_residual_blocks=1,
        perceptual_weight=0.0,    # avoid VGG download
        use_vgg_pretrained=False,
        edge_weight=19.0,
        seed=0,
        loss_type="bce",
        bce_weight=1.0,
    )
    summary = train(cfg)
    ckpt_path = runs / "smoke-bce" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = runs / "smoke-bce" / "last.pt"
    assert ckpt_path.exists()

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert ckpt.get("apply_sigmoid") is False
    assert ckpt.get("loss_type") == "bce"
    assert "best_val_total" in summary
```

- [ ] **Step 2: Run the BCE smoke test**

Run: `. .venv/bin/activate && pytest tests/test_train.py::test_train_smoke_bce_loss_writes_apply_sigmoid_false -v`
Expected: PASS. (2 epochs on 4 tiny images on CPU should complete in well under 30 s.)

- [ ] **Step 3: Run full test_train.py**

Run: `pytest tests/test_train.py -v --no-header`
Expected: both tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_train.py
git commit -m "test(train): smoke test --loss bce writes apply_sigmoid=False ckpt"
```

---

### Task 7: `eval.py` reads metadata + wraps sigmoid externally

**Files:**
- Modify: `albumify/eval.py`
- Test: deferred to Task 8.

- [ ] **Step 1: Locate the model load block**

Around lines 142–152 of `albumify/eval.py`, find:

```python
    model = Generator(n_residual_blocks=cfg.n_residual_blocks, ngf=cfg.ngf)
    if cfg.use_lora:
        wrap_conv2d_layers(...)
        freeze_non_lora(model)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
```

Replace with:

```python
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    apply_sigmoid_in_model = bool(ckpt.get("apply_sigmoid", True))
    model = Generator(
        n_residual_blocks=cfg.n_residual_blocks,
        ngf=cfg.ngf,
        sigmoid=apply_sigmoid_in_model,
    )
    if cfg.use_lora:
        wrap_conv2d_layers(
            model, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
            skip_kernel_sizes=tuple(cfg.skip_kernel_sizes_for_lora),
        )
        freeze_non_lora(model)
    model.load_state_dict(ckpt["model_state_dict"])
    if not apply_sigmoid_in_model:
        # Wrap so SSIM/edge-F1/val_grid all see values in [0, 1].
        import torch.nn as _nn
        class _SigmoidWrap(_nn.Module):
            def __init__(self, inner: _nn.Module):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                return torch.sigmoid(self.inner(x))
        model = _SigmoidWrap(model)
    model = model.to(device)
    model.eval()
```

- [ ] **Step 2: Quick syntax + import check**

Run: `. .venv/bin/activate && python -c "from albumify import eval as _; print('eval.py imports OK')"`
Expected: `eval.py imports OK`.

- [ ] **Step 3: Existing eval test still passes**

Run: `pytest tests/test_eval.py -v --no-header`
Expected: existing tests pass. Defaults still wire to sigmoid-in-model.

- [ ] **Step 4: Commit**

```bash
git add albumify/eval.py
git commit -m "feat(eval): read apply_sigmoid from ckpt + wrap externally when false"
```

---

### Task 8: Eval test — `apply_sigmoid=False` ckpt routes through wrap

**Files:**
- Modify: `tests/test_eval.py`
- Test: `pytest tests/test_eval.py -v -k sigmoid_false`

- [ ] **Step 1: Inspect existing test_eval.py to match style**

Run: `head -40 tests/test_eval.py`
You should see the standard `pytest.importorskip("torch")` header and existing fixture patterns. Mirror them.

- [ ] **Step 2: Append the new test**

Append to `tests/test_eval.py`:

```python
def test_eval_loads_apply_sigmoid_false_ckpt_and_wraps_externally(tmp_path: Path):
    """Build a tiny no-sigmoid ckpt, run eval, assert pred is bounded in [0,1]."""
    import torch as _torch
    from albumify.eval import EvalConfig, run_eval
    from albumify.model import Generator

    covers = tmp_path / "covers"; covers.mkdir()
    labels = tmp_path / "labels"; labels.mkdir()
    splits = tmp_path / "splits"; splits.mkdir()
    out_dir = tmp_path / "eval-out"

    slugs = [f"v{i}" for i in range(2)]
    for s in slugs:
        Image.new("RGB", (64, 64), (40, 80, 160)).save(covers / f"{s}.jpg")
        Image.new("L", (64, 64), 255).save(labels / f"{s}.png")
    (splits / "val.txt").write_text("\n".join(slugs) + "\n")
    (splits / "train.txt").write_text("")  # required by AlbumDataset constructor

    # Build a tiny sigmoid-less Generator + save a Plan C ckpt.
    g = Generator(n_residual_blocks=1, ngf=8, sigmoid=False)
    ckpt_path = tmp_path / "noprep.pt"
    _torch.save({
        "model_state_dict": g.state_dict(),
        "epoch": 1,
        "val_total": 0.0,
        "apply_sigmoid": False,
        "loss_type": "bce",
    }, ckpt_path)

    cfg = EvalConfig(
        splits_dir=str(splits), covers_dir=str(covers), labels_dir=str(labels),
        ckpt_path=str(ckpt_path), out_dir=str(out_dir),
        img_size=64, n_residual_blocks=1, ngf=8,
        use_lora=False,
        use_lpips=False,
        grid_n=2,
    )
    summary = run_eval(cfg)
    # The eval grid renderer expects pred in [0, 1]; we don't read pixels
    # here, but if the sigmoid wrap weren't applied the SSIM/edge-F1 numbers
    # below would be NaN / out of range because pred would contain raw logits.
    assert _np.isfinite(summary["ssim"])
    assert 0.0 <= summary["f1"] <= 1.0
```

You may need to add `import numpy as _np` near the top of the test file (check whether existing tests already do this — if so, no change needed).

- [ ] **Step 3: Run the new test**

Run: `. .venv/bin/activate && pytest tests/test_eval.py::test_eval_loads_apply_sigmoid_false_ckpt_and_wraps_externally -v`
Expected: PASS.

- [ ] **Step 4: Run full test_eval.py**

Run: `pytest tests/test_eval.py -v --no-header`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_eval.py
git commit -m "test(eval): apply_sigmoid=False ckpt routes through external sigmoid"
```

---

### Task 9: `export.py` reads metadata + wraps before ONNX trace

**Files:**
- Modify: `albumify/export.py`
- Test: deferred to Task 10.

- [ ] **Step 1: Locate the export model construction**

Around lines 40–60 of `albumify/export.py`, find:

```python
    model = Generator(n_residual_blocks=n_residual_blocks, ngf=ngf)
    n_merged = 0
    if use_lora:
        wrap_conv2d_layers(...)
        freeze_non_lora(model)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    if use_lora:
        n_merged = merge_all_lora(model)
    model.eval()
```

Replace with:

```python
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    apply_sigmoid_in_model = bool(ckpt.get("apply_sigmoid", True))
    model = Generator(
        n_residual_blocks=n_residual_blocks, ngf=ngf,
        sigmoid=apply_sigmoid_in_model,
    )
    n_merged = 0
    if use_lora:
        wrap_conv2d_layers(
            model, rank=lora_rank, alpha=lora_alpha,
            skip_kernel_sizes=tuple(skip_kernel_sizes_for_lora),
        )
        freeze_non_lora(model)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    if use_lora:
        n_merged = merge_all_lora(model)
    if not apply_sigmoid_in_model:
        import torch.nn as _nn
        class _SigmoidWrap(_nn.Module):
            def __init__(self, inner: _nn.Module):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                return torch.sigmoid(self.inner(x))
        model = _SigmoidWrap(model)
    model.eval()
```

- [ ] **Step 2: Syntax + import check**

Run: `. .venv/bin/activate && python -c "from albumify import export as _; print('export.py imports OK')"`
Expected: `export.py imports OK`.

- [ ] **Step 3: Existing export tests still pass**

Run: `pytest tests/test_export.py -v --no-header`
Expected: existing tests pass. Defaults still wire to sigmoid-in-model.

- [ ] **Step 4: Commit**

```bash
git add albumify/export.py
git commit -m "feat(export): wrap exported model with sigmoid when ckpt has apply_sigmoid=False"
```

---

### Task 10: Export test — sigmoid-less ckpt produces ONNX in [0, 1]

**Files:**
- Modify: `tests/test_export.py`
- Test: `pytest tests/test_export.py -v -k sigmoid_false`

- [ ] **Step 1: Append the new test**

Append to `tests/test_export.py`:

```python
def test_export_wraps_sigmoid_for_apply_sigmoid_false_ckpt(tmp_path: Path):
    """Plan C ckpts (apply_sigmoid=False) must produce ONNX in [0,1]."""
    # Build a tiny no-sigmoid Generator and save a Plan C ckpt.
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
```

- [ ] **Step 2: Run the new test**

Run: `. .venv/bin/activate && pytest tests/test_export.py::test_export_wraps_sigmoid_for_apply_sigmoid_false_ckpt -v`
Expected: PASS.

- [ ] **Step 3: Run full test_export.py**

Run: `pytest tests/test_export.py -v --no-header`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_export.py
git commit -m "test(export): apply_sigmoid=False ckpt yields ONNX in [0,1]"
```

---

### Task 11: Full test sweep

**Files:** none

- [ ] **Step 1: Run the entire suite**

Run: `. .venv/bin/activate && pytest -q`
Expected: all green. Note any skipped tests (e.g., LPIPS-dependent eval pieces); these are pre-existing skips.

- [ ] **Step 2: If any failures, investigate before pushing**

Each failure should map to one of the modified files. Re-read the relevant Task above to ensure all steps were applied. No new code beyond the plan tasks should be needed.

---

### Task 12: Push the branch

**Files:** none

- [ ] **Step 1: Confirm clean working tree**

Run: `git status -sb`
Expected: branch `feat/plan-c-bce-logits`, no `M`/`A`/`D` lines (apart from the persistent untracked exploration files).

- [ ] **Step 2: Push to origin with upstream tracking**

Run: `git push -u origin feat/plan-c-bce-logits`
Expected: `Branch 'feat/plan-c-bce-logits' set up to track 'origin/feat/plan-c-bce-logits'`.

- [ ] **Step 3: Note the training command for tomorrow**

The branch is now ready to be checked out on a VM. Tomorrow's launch command on the VM is the same as Plan B's, with the additional `--loss bce` flag:

```bash
python -m albumify.train \
  --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
  --no-lora --ngf 96 \
  --loss bce \
  --out-dir runs/ngf96-bce --epochs 60 --batch-size 8 --lr 2e-4 \
  --perceptual-weight 0.1
```

(`--edge-weight` auto-defaults to 19 when `--loss bce`; override only if you want a different ratio.)

---

## Self-Review Notes

**Spec coverage check:**
- Loss layer (spec §loss.py) → Tasks 2–3. ✓
- Model unchanged (spec §model.py) → Task 4 covers the existing flag works. ✓
- Train.py: `--loss` flag, ckpt metadata, edge-weight default, edge-fraction log → Task 5 covers all four. ✓
- Eval.py: metadata read + external wrap → Task 7. ✓
- Export.py: metadata read + wrap before ONNX → Task 9. ✓
- Infer.py: spec says "no change". Not in tasks. ✓
- Tests for all five new behaviors → Tasks 2, 4, 6, 8, 10. ✓
- Run-time validation criteria (spec §Risks) → captured at Task 12 step 3 as the training command; verification happens during tomorrow's run, not in this plan.

**Placeholder scan:** none. Every step has concrete code or commands.

**Type/name consistency:**
- `loss_type` (lowercase string) used consistently across `TrainConfig`, ckpt dict, CLI flag.
- `apply_sigmoid` boolean — same in ckpt dict, eval, export.
- `_SigmoidWrap` defined twice (eval, export) — intentional inline definition so each module is self-contained; small enough that DRYing it out via a shared helper would add more import surface than it saves.
- Function name `edge_weighted_bce_logits` — used in `loss.py` and `tests/test_loss.py` only.
