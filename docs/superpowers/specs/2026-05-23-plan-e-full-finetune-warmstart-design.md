# Plan E — Full fine-tune from Informative-Drawings warm-start with BCE

Date: 2026-05-23
Status: Design approved; code prepared, training run pending.

## Summary

Train every parameter of the `Generator` (no LoRA freeze) using
`--loss bce`, warm-starting weights from `artifacts/informative_drawings.pth`
at the upstream architecture (`ngf=64`, 9 residual blocks). This is the
combination Plans B/C/D each had two of three pieces of:

- Plan B v0.2.0 (L1, ngf=96, from scratch): full tunability + BCE-free; clean
  thresholded output but no inherited prior. *Current winner with thr=0.95.*
- Plan C (BCE, ngf=96, from scratch): BCE pushed commitment but model
  spent capacity learning line-drawing structure from 424 covers. *Sketchy.*
- Plan D (BCE, ngf=64, LoRA r=16, warm-start): had the prior but
  847K trainable params couldn't override the pretrained soft-pencil bias.
  *Soft, sparse.*

Plan E gives the model both the **inductive bias** (warm-start) **and the
capacity** (no-LoRA, all 11.7M params tunable) **and the commitment driver**
(BCE) at the same time.

## Hypothesis

With every weight free to move and an initialization in the line-drawing
manifold, BCE-with-logits can push outputs toward crisp binary lines without
either: (a) the from-scratch data-poverty of Plan C, or (b) the LoRA-rank
bottleneck of Plan D.

Falsifiable: holdout 512px renders of Nevermind/Wall/Thriller without any
threshold should be at least as crisp as v0.2.0 + threshold 0.95, and
visibly cleaner than Plan C + threshold 0.60.

## Architecture

No model code changes. CLI invocation:

```bash
python -m albumify.train \
  --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
  --pretrained-ckpt artifacts/informative_drawings.pth \
  --out-dir runs/full-bce-warm-ngf64 \
  --no-lora --ngf 64 \
  --loss bce \
  --perceptual-weight 0 \
  --epochs 25 --batch-size 8 --lr 2e-4 \
  --weight-decay 1e-4
```

Param choices:

- `--no-lora`: trains all 11.7M params (vs Plan D's 847K).
- `--ngf 64`: matches the pretrained ckpt's shape so warm-start works.
- `--loss bce`: drops sigmoid from the model, uses
  `binary_cross_entropy_with_logits` on hard-thresholded targets
  (default `--edge-weight 19` auto-applied per Plan C wiring).
- `--perceptual-weight 0`: the spec-documented fallback for the Plan C
  "perceptual+BCE conflict" risk.
- `--lr 2e-4`: full fine-tunes need lower LR than LoRA. Matches Plan B
  v0.2.0's 60-epoch run. The LR Plan D used (`1e-3`) is appropriate when
  only LoRA-adapter weights move; here we're touching every weight.
- `--epochs 25`: Plan C overfit at epoch 19 from scratch. With warm-start
  the model converges faster; expect best.pt to land before epoch 20.
  Generous ceiling.
- `--weight-decay 1e-4`: train.py default. Light regularization helps with
  the 424-cover dataset size.

## Backwards compatibility

Same checkpoint format as Plans C and D (`apply_sigmoid=False`,
`loss_type="bce"`). eval.py/export.py/infer.py already handle the
external-sigmoid wrap via the Plan C metadata plumbing.

## Diagnostic improvement (the only code change)

Plan D's startup log printed `[pretrained] missing=24 unexpected=0` —
unactionable. For Plan E we want to *see* which 24 keys didn't load so we
know how much of the warm-start was actually effective. Tiny train.py
patch: extend the existing `print(f"[pretrained] missing=...")` to include
the first 8 missing and unexpected key names.

Why this matters: for the LoRA path (Plans B v0.1, D), partial warm-start
is fine because adapters compensate. For Plan E's full fine-tune, the
missing-key set IS the part that starts from random init. If a critical
layer (e.g., the final tail conv `model4.1`) is in the missing set,
warm-start is less meaningful than expected, and we should look at fixing
`load_pretrained`'s key remapping rather than accepting a half-warm start.

## Testing

- Add a smoke test for the BCE + `--no-lora` + `--pretrained-ckpt` combo
  (the three flags Plan E combines for the first time).
- Add a unit test that the pretrained-load diagnostic includes key names,
  not just counts.
- All existing Plan C tests (loss path, ckpt metadata, eval/export sigmoid
  wrap) still apply unchanged.

## Risks

- **The 24 missing keys turn out to be the critical layers.** If so,
  warm-start has no benefit and Plan E degrades to Plan C from scratch
  but with a worse-than-random init in those layers. Mitigation: the
  diagnostic print will surface this in the first 2 seconds of training.
  If catastrophic, abort the run and fix `load_pretrained`'s key remapping
  (likely a "model.0.*" vs "model0.0.*" naming mismatch — see
  `albumify/model.py:108-113`).
- **Catastrophic forgetting.** Full fine-tune can overwrite the pretrained
  prior in early epochs if the LR is too high. Mitigation: conservative
  2e-4 LR + watch the first 3 epochs in TB.
- **Overfit faster than Plan C.** Plan C overfit at epoch 19 from random
  init; warm-start moves the optimum closer, so overfit may come at
  epoch 10-15. Mitigation: best.pt saver captures the true best regardless.

## Out of scope

- INT8 export verification (toolchain regression still open — see
  `[[reference-toolchain-int8-regression]]`).
- ngf=96 + warm-start (would require porting pretrained weights to a wider
  Generator — explicitly deferred).
- A/B comparison code or a public release; this run produces an artifact
  and a verdict, nothing more.

## Cost & time estimate

- L4 g2-standard-4 on-demand: ~$0.85/hr.
- Train (~16-18 s/epoch × 25 epochs on L4): ~8 min.
- Eval + export + holdouts + scp: ~5 min.
- Total VM time: ~15 min → ~$0.21.
