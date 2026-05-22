# Plan C — BCE-with-logits line drawing model

Date: 2026-05-21
Status: Design approved, awaiting implementation plan

## Summary

Train the `Generator` without its terminal sigmoid, using
`binary_cross_entropy_with_logits` on hard-thresholded {0, 1} targets,
weighted per-pixel by an edge mask (same dilation as today's
`edge_weighted_l1`). Add the sigmoid back at eval/export/infer time so the
exported ONNX continues to produce values in [0, 1] — identical CLI contract
for users.

The motivation: Plan B (v0.2.0-ngf96) still produces faint "ghost" pixels that
need `--threshold 0.95` at inference to look like clean line drawings. The
sigmoid saturates near 1.0, the L1 gradient is flat at saturation, and the
model never commits to binary outputs. BCE-with-logits removes the
saturation by computing in logit space.

## Goals

- Trained model whose post-sigmoid output is meaningfully binary — clean
  black-on-white drawings without `--threshold 0.95`.
- Symmetric code structure to Plan B's L1 path so future A/B comparisons
  remain easy.
- Full backwards compatibility: existing v0.1.0 and v0.2.0 checkpoints
  continue to load and produce the same outputs they do today.

## Non-goals

- No architecture changes beyond toggling the terminal sigmoid.
- No warm-start from v0.2.0 ngf-96 weights — train from scratch for a clean
  A/B vs Plan B.
- No deprecation of the L1 path — `--loss l1` stays the default.
- No changes to the existing inference CLI or ONNX contract.

## Architecture changes

### `albumify/model.py`

No real change. `Generator.__init__(sigmoid: bool = True)` already exists.
Training in Plan C mode constructs `Generator(..., sigmoid=False)`.

### `albumify/loss.py`

Add two new units alongside the existing `edge_weighted_l1` /
`L1PerceptualLoss`:

```python
def edge_weighted_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    edge_threshold: float = 0.5,
    edge_weight: float = 19.0,
) -> torch.Tensor:
    """BCE-with-logits on hard-thresholded targets, with edge-pixel weighting.

    target is grayscale in [0, 1]; hard-thresholded to {0, 1} at edge_threshold.
    A dilated edge mask weights edge pixels (1 + edge_weight)x — mirrors
    edge_weighted_l1 so the two paths can be compared apples-to-apples.
    """
```

```python
class BCELogitsPerceptualLoss(nn.Module):
    """BCE-with-logits (+ optional VGG perceptual on sigmoid(logits))."""

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        edge_weight: float = 19.0,
        edge_threshold: float = 0.5,
        vgg: Optional[VGGPerceptualLoss] = None,
    ): ...

    def forward(self, logits, target) -> dict[str, torch.Tensor]:
        """Returns {"bce", "total", optional "perc"}."""
```

`BCELogitsPerceptualLoss` applies perceptual to `sigmoid(logits)` because
VGG requires inputs in [0, 1]. BCE consumes raw logits directly.

Default `edge_weight=19` matches the measured 5% edge fraction
((1 − 0.05) / 0.05 ≈ 19) so per-pixel gradients are balanced.

### `albumify/train.py`

New CLI surface:

- `--loss {l1,bce}` — default `l1` (today's behavior preserved).
- `--edge-weight` already exists; default remains `0` for `l1`, but when
  `--loss bce` is selected and the user did not pass `--edge-weight`, it
  defaults to `19`.

When `--loss bce`:

- Construct `Generator(..., sigmoid=False)`.
- Construct `BCELogitsPerceptualLoss(...)` instead of `L1PerceptualLoss(...)`.
- Same `--epochs 60`, `--batch-size 8`, `--lr 2e-4`, `--perceptual-weight 0.1`
  as Plan B — only the loss changes.
- At dataset load, log the empirical edge fraction across the training set
  and warn if it is more than ±2 percentage points from the assumed 5%.

Extended checkpoint format:

```python
{
    "model_state_dict": ...,
    "epoch": ...,
    "val_total": ...,
    "apply_sigmoid": <bool>,   # NEW — false for Plan C, true otherwise
    "loss_type": "<l1|bce>",   # NEW — what loss produced this ckpt
}
```

`_evaluate()` returns `{val_total, val_bce or val_l1, val_perc, ...}` so
the per-loss component is always logged.

## Backwards compatibility — eval, export, infer

Principle: **the checkpoint owns the truth** about whether sigmoid is
internal. Loaders read `apply_sigmoid` from the .pt metadata; CLI flags do
not override.

### `albumify/eval.py`

- Read `apply_sigmoid` (default `True` when key missing → matches v0.1/v0.2).
- Build `Generator(..., sigmoid=apply_sigmoid)`.
- If `apply_sigmoid=False`, wrap `model(x)` with `torch.sigmoid(...)` inside
  `_evaluate()` and `render_grid()` so SSIM, edge-F1, val_grid all see
  values in [0, 1].
- Report both `val_total` and the per-loss component
  (`val_bce` or `val_l1`) so numbers from different runs do not collide
  silently.

### `albumify/export.py`

- Same metadata read.
- If `apply_sigmoid=False`, wrap the loaded Generator with a thin
  `nn.Module` that does `sigmoid(model(x))` *before* tracing for ONNX.
  The exported ONNX always produces values in [0, 1].
- INT8 quantization path unchanged.

### `albumify/infer.py`

No change. The ONNX file always produces [0, 1] post-sigmoid; the existing
`--threshold` post-process still works. Plan C just makes the flag
optional rather than required.

## Testing

### `tests/test_loss.py` (new)

- `edge_weighted_bce_logits` returns a finite scalar; gradient flows.
- With `edge_weight=0` it equals
  `F.binary_cross_entropy_with_logits(logits, target_bin)`.
- With `edge_weight=19` and a synthetic single-edge target, the result
  differs from `edge_weight=0` by approximately
  `19 * loss_at_edge / N` (wiring check on the weight map).
- `BCELogitsPerceptualLoss.forward` returns dict with `bce`, `total`, and
  optional `perc`. `total` reconstructs from `bce_weight * bce` (+ perc).

### `tests/test_model.py` (extend)

- `Generator(sigmoid=False).forward(x)` produces output with both negative
  and positive values — real logits, not [0, 1].

### `tests/test_train.py` (extend)

- Smoke test: invoke `train.py` with `--loss bce`, 1 epoch, tiny dataset.
- Assert saved `best.pt` contains `apply_sigmoid=False` and
  `loss_type="bce"`.

### `tests/test_eval.py` (extend)

- Build a fake ckpt with `apply_sigmoid=False`, feed it to eval's loader.
- Assert the rendered pred column is in [0, 1] — sigmoid was applied
  externally.

### `tests/test_export.py` (extend)

- Same fake ckpt → export ONNX → load with onnxruntime → outputs in [0, 1].

### Run-time validation (not unit-tested)

The actual "does Plan C work" verdict is the training run. Success criteria:

- val_loss decreases steadily; no NaN; comparable or better
  `val_l1(sigmoid(logits))` vs Plan B's val_l1.
- `val_grid` pred column with **no** thresholding looks visually binary:
  clean black lines on white, not gray ghosts.
- Re-shot Nevermind/Wall/Thriller at 256/512/1024 with **no** `--threshold`
  match the visual quality of the v0.2.0-ngf96 versions with
  `--threshold 0.95`.

## Risks

- **Training instability.** Without sigmoid, raw logits can blow up early.
  Same conservative `--lr 2e-4` as Plan B. Add `clip_grad_norm_` only if
  NaNs appear — not preemptively.
- **Class imbalance miscalibrated.** `edge_weight=19` derives from a
  5-image histogram. Mitigation: log empirical fraction at startup; warn
  if more than ±2 pp from 5%.
- **Perceptual+BCE conflict.** Perceptual is grayscale-sensitive; BCE
  pushes for binary commitment. Mitigation: keep
  `--perceptual-weight 0.1`. If outputs look too gray, drop to 0 in a
  follow-up run.
- **Eval/export forgetting to wrap sigmoid.** Mitigation: the sigmoid
  wrap is conditional on a shared metadata read used by `eval.py`,
  `export.py`, `infer.py`. Round-trip tests cover the wiring.

## Out of scope

- Architecture changes (residual block count, InstanceNorm → BatchNorm,
  ReflectionPad → ZeroPad).
- Warm-start from v0.2.0 ngf-96 weights.
- Dataset expansion or relabeling.
- Static (calibration-based) INT8 quantization.

All deferred to future plans.
