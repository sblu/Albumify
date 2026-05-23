# Plan D — BCE-with-logits + pretrained warm-start

Date: 2026-05-23
Status: Design approved, ready to run

## Summary

Combine Plan C's BCE-with-logits loss (which solved the L1+sigmoid saturation
problem) with Plan B v0.1.0's warm-start from `informative_drawings.pth` and
LoRA fine-tuning (which already had the line-drawing inductive bias baked in).
Drop perceptual loss to 0 to remove the perceptual-vs-BCE tug-of-war that
caused Plan C's outputs to come out grey/diffuse.

## Motivation — what Plan C taught us

Plan C trained Generator from scratch at ngf=96 with BCE + perceptual=0.1.
Result (see `runs/ngf96-bce/`):

- val_bce 2.22 at best epoch (19), then overfit through epoch 60.
- Raw predictions look grey/diffuse, but with `--threshold 0.60` they resolve
  into recognizable line drawings (lettering legible, subjects visible).
- Lines are dotted/sketchy — the model knows where edges roughly are but not
  exactly. Edge localization is the remaining problem, not commitment.
- Spec's predicted risk materialized: perceptual_weight=0.1 fought BCE for
  grayscale realism and won. The spec's documented fallback was "drop
  perceptual to 0 in a follow-up run."

The from-scratch ngf=96 approach asked 25M params to learn "what's a line
drawing" from 424 covers, which is too little data for crisp localization.
Plan B v0.1.0 (LoRA on the pretrained Informative-Drawings generator)
produced recognizable ghosts because it inherited line-drawing structure
from pretraining.

## Hypothesis

BCE-with-logits *plus* the pretrained line-drawing prior *plus* no perceptual
interference will produce crisp single-pixel lines without thresholding:

- Pretrained `informative_drawings.pth` brings the edge-localization prior.
- BCE forces binary commitment at those localized edges (Plan C's contribution).
- `--perceptual-weight 0` removes the grayscale-realism pull.
- LoRA rank 16 gives more adaptation headroom than v0.1.0's r=8 (which was
  too constrained to commit to the BCE objective even if BCE were active).

## Architecture

No code changes. All configuration is via existing CLI flags:

```bash
python -m albumify.train \
  --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
  --pretrained-ckpt artifacts/informative_drawings.pth \
  --out-dir runs/lora-r16-bce \
  --ngf 64 \
  --lora-rank 16 --lora-alpha 16 \
  --loss bce \
  --perceptual-weight 0 \
  --epochs 25 \
  --batch-size 8 \
  --lr 1e-3 \
  --weight-decay 1e-4
```

Why these specific values:

- `--ngf 64`: must match the pretrained ckpt's shape (no resize/projection).
- `--lora-rank 16`: 2× v0.1.0's r=8. The L1 model at r=8 already produced
  ghosts; we want more adaptation capacity to commit to the BCE objective.
- `--lora-alpha 16`: convention is to match alpha to rank.
- `--epochs 25`: Plan C overfit at epoch 19 → 25 is a generous ceiling, the
  `best.pt` saver will capture the actual best epoch.
- `--lr 1e-3`: matches v0.1.0. LoRA tolerates a higher LR than full
  fine-tunes (which used 2e-4) because only the adapter weights move.
- `--edge-weight` auto-defaults to 19 when `--loss bce` (per Plan C's wiring),
  matching the 0.051 measured edge fraction.

Sigmoid is dropped from the model (BCE on logits), then re-applied externally
at eval/export/infer time via the existing Plan C apply_sigmoid metadata
plumbing.

## Backwards compatibility

Same checkpoint format as Plan C (`apply_sigmoid=False`, `loss_type="bce"`).
eval/export/infer code already reads these and wraps `torch.sigmoid()`
externally. No new flags in any user-facing CLI.

## Testing

The training-code surface is unchanged from Plan C. Tests covering the
BCE path and the apply_sigmoid metadata plumbing (Plan C's Tasks 2, 4, 6,
8, 10) still apply. No new unit tests needed for Plan D — it is a
configuration variation.

The actual "did Plan D work" verdict is the run itself. Success criteria:

- Predictions look crisp (single-pixel-ish lines, not dotted/sketchy)
  at **no threshold** — the original Plan C visual goal.
- val_grid pred column: clean black lines on white, no grey haze.
- Re-shot Nevermind/Wall/Thriller at 256/512/1024 with **no** `--threshold`
  match or beat v0.2.0+threshold 0.95 and Plan C+threshold 0.60.

Fallback (not auto-coded): if outputs are crisp but slightly over- or
under-committed, tune the threshold at infer time as today.

## Risks

- **Pretrained weights are sigmoid-trained.** The pretrained model was
  trained with sigmoid in-graph, so its final-layer pre-sigmoid logits are
  biased toward a sigmoid-friendly range. Loading these into a
  `Generator(sigmoid=False)` and continuing with BCE means the first few
  epochs may have larger gradients as the model adapts to the new output
  convention. Mitigation: LR is conservative (1e-3 on a LoRA-only opt
  space); the apply_sigmoid metadata plumbing means eval still sees [0,1]
  outputs even if logits are large.
- **LoRA r=16 still might be too constrained.** If the BCE objective
  requires substantially different weights than the pretrained model has,
  even r=16 may not have enough rank. Mitigation: `last.pt` is saved every
  epoch, so a follow-up run could continue from last.pt with higher rank
  or unfreeze the base. Not coded here.
- **Edge-weight=19 with no perceptual could push too hard for dark.**
  Without perceptual smoothing, edge-weighted BCE could over-predict edges.
  Mitigation: review val_grid; if too many false edges, reduce edge_weight
  to ~10–15 in a follow-up.

## Out of scope

- New CLI flags or loss surfaces (everything via existing Plan C wiring).
- INT8 export (toolchain regression still blocks Pi deployment; see
  `[[reference-toolchain-int8-regression]]`).
- ngf=96 + warm-start (would require porting pretrained weights to a wider
  Generator — explicitly deferred).
