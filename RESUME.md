# Resume notes — pick up training tomorrow

Last touched: 2026-05-23. Three BCE experiments (Plans C, D, E) attempted and
failed to beat **v0.2.0** (the L1+ngf=96+threshold-0.95 model already shipped
as a GitHub Release). v0.2.0 remains the recommended model.

## TL;DR

- v0.2.0-ngf96 is still the official line-drawing model. Don't run more BCE
  + warm-start experiments — three different configurations all failed in the
  same direction. The bottleneck is **data quantity**, not algorithm choice.
- All experiment artifacts (ckpts, ONNX, val_grids, holdout renders, 6-way
  comparison montages) live under `runs/`. The relevant branches are pushed
  to origin and waiting on a merge-strategy decision.
- The smaller, important improvement that came out of today: the `[pretrained]`
  diagnostic in `train.py` now prints the first 8 missing/unexpected key names,
  which surfaced the upstream-vs-ours residual-block mismatch (3 vs 9). Worth
  cherry-picking back to main on its own.

## What we've trained (running scoreboard)

| run | recipe | best val | F1 | visual verdict |
|---|---|---|---|---|
| `runs/lora-rank8` (v0.1.0) | LoRA r=8 on contour-style, L1, edge_weight=0 | — | — | Faint ghosts. Recognizable only at `--threshold 0.95`. |
| `runs/ngf96` (v0.2.0, **shipped**) | Full fine-tune, L1, ngf=96 from scratch, edge-weighted | — | — | **Winner.** Clean bold lines at `--threshold 0.95`. |
| `runs/ngf96-bce` (Plan C) | Full fine-tune, BCE, ngf=96 from scratch, perc=0.1 | 2.22 @ ep19 | 0.258 | Sketchy/dotted but recognizable at thr 0.60. |
| `runs/lora-r16-bce` (Plan D) | LoRA r=16 + warm-start, BCE, perc=0 | 2.22 @ ep25 | 0.152 | Pretrained soft-pencil dominates. Sparse at thr 0.67. |
| `runs/full-bce-warm-ngf64` (Plan E) | Full fine-tune + warm-start, BCE, ngf=64, perc=0 | 2.25 @ ep25 | 0.160 | Same as Plan D — pretrained character sticky. |

## What today actually taught us

1. **The pretrained Informative-Drawings checkpoint is "sticky."** Plans D and
   E both warm-started from it; one with LoRA r=16 (847K trainable), one with
   full fine-tune (11.4M trainable). Both landed in the same visual basin:
   soft photographic pencil sketches that, when thresholded, lose most
   non-title structure. Capacity wasn't the bottleneck.

2. **Architectural mismatch we didn't know about:** upstream Informative
   Drawings has 3 residual blocks; our `Generator` instantiates 9. The new
   diagnostic print revealed all 24 missing keys are from `model2.3` through
   `model2.8` — i.e., residual blocks 3–8 are random-init in every
   warm-started run we've ever done (including Plan B v0.1.0 and v0.2.0
   when the pretrained ckpt is loaded).

3. **BCE-with-logits works as designed**, but only "modestly." Plan C
   (no warm-start) got to a more committed regime (need `--threshold 0.60`
   instead of v0.2.0's `0.95`) but at the cost of dotted/sketchy lines.

4. **L1 + threshold 0.95 is still the cleanest path.** Three BCE variants
   couldn't beat it on visual quality.

## Branches still in play

- `feat/plan-c-bce-logits` (pushed): the BCE wiring — `--loss bce` CLI,
  `apply_sigmoid` ckpt metadata, eval/export external-sigmoid wrap. **Worth
  merging to main eventually** — clean code, all tests pass.
- `feat/plan-d-bce-warm-start` (pushed): docs only on top of Plan C. Merge
  upstream from Plan C.
- `feat/plan-e-full-bce-warmstart` (pushed): docs + the diagnostic-print
  improvement on top of Plan C. **The diagnostic-print commit is independently
  useful** — cherry-pick it to main regardless of whether Plan E merges.

Merge order to consider: cherry-pick `da875d9` (diagnostic print) onto main,
then merge `feat/plan-c-bce-logits` onto main, then optionally tag Plans D
and E's docs by merging their branches as historical record.

## What to try next (in EV order)

1. **More data.** ~500 more cover/label pairs through the existing review-app
   pipeline. This is the single highest-EV next move. Three different
   loss/init experiments couldn't compensate for 424 covers; another 500
   should give the model the localization signal it currently lacks.
2. **Cheap post-processing of v0.2.0 outputs.** Morphological close +
   skeletonize on the thresholded result. Free, no retraining. Could close
   the "scattered dots vs single lines" gap.
3. **Match the upstream architecture more carefully.** Try `--n-residual-blocks 3`
   for warm-start runs so the pretrained ckpt loads strict — eliminates the
   24 random-init residual block params. This is a one-flag-change retest,
   ~$0.15, worth doing before declaring BCE+warmstart dead-end if you want
   to be thorough. But evidence so far says don't bother.

## What NOT to try next

- Another BCE variant on the current dataset. Three runs converged to the
  same place; one more won't move the needle.
- Bigger model (ngf=128, more blocks). No data to back that up.
- LoRA at higher rank. Plan D already proved LoRA-on-warm-start has the
  pretrained-bias problem, and rank wasn't the dominant factor.

## Pre-existing issues still open (none blocking)

- `tests/test_model.py::test_generator_param_count_in_expected_range` —
  stale 4–5M assertion; actual is 11.4M at ngf=64. One-line fix.
- `tests/test_export.py::test_int8_quantization_produces_runnable_model` —
  toolchain regression in the DLVM's newer onnxruntime: int8 export
  produces a `ConvInteger(10)` op the runtime can't load. **Blocks Pi
  deployment** of any newly-exported INT8 ONNX from a fresh VM. Workarounds:
  use FP32 ONNX (works), or fix the export pre-processing per the warning
  message. v0.2.0's existing INT8 release artifact is unaffected.

## Infra notes for next VM run

- **L4 on `g2-standard-4` is the reliable option.** T4 capacity has been
  exhausted across every US zone on at least two consecutive nights. L4 is
  faster anyway (~2.5×) and costs about the same total per run.
- **Use `--tunnel-through-iap` for all `gcloud compute ssh/scp` calls.**
  Direct port 22 over the public IP times out in the `albumartifier` project's
  VPC. IAP works fine.
- Spin-up: `PROJECT=albumartifier SPOT=0 MACHINE_TYPE=g2-standard-4 GPU_TYPE=nvidia-l4 ZONE=us-east1-d ./infra/create_vm.sh`
- Pricing: ~$0.85/hr on-demand × ~12-15 min per Plan-C-style run = ~$0.20.
