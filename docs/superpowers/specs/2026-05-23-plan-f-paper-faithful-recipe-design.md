# Plan F — Paper-faithful Informative-Drawings recipe (CLIP + depth + GAN)

Date: 2026-05-23
Status: Design draft pending review. No code written yet.

## Summary

Stop training with paired-L1 alone and rebuild the loss function the
Informative-Drawings paper (Chan et al., CVPR 2022) actually used. Add three
new supervisions on top of (or, ultimately, in place of) our pixel-L1:

1. **CLIP semantic loss** — MSE between CLIP(G(cover)) and CLIP(cover).
2. **Depth/geometry loss** — L1 between predicted depth-from-line-drawing
   and pseudo-GT depth of the photo from MiDaS.
3. **PatchGAN adversarial loss** — 70×70 PatchGAN discriminator trained on
   our 424 Gemini labels as an unpaired drawing-style corpus.

Plus three structural fixes the previous Plans glossed over:

4. Generator at 3 residual blocks (matches the upstream pretrained ckpt
   shape; eliminates the 24 random-init middle-block params that have
   silently corrupted every warm-start since v0.1.0).
5. Adam (not AdamW), lr=2e-4, β=(0.5, 0.999), no weight decay — matches
   upstream training hyperparameters.
6. Full fine-tune (no LoRA freeze) on the now-strictly-loaded warm start.

The experiment ramps up in five stages (F1–F5) so each new loss / data
change is attributable.

## Background & motivation

[[plan-e-result]] concluded "data is the bottleneck, not algorithm — stop
iterating on loss/init." That conclusion is correct *within the paired-L1
family*: Plans C/D/E all converged to roughly the same place because they
varied only init/freeze/loss-shape, not the supervisions feeding the model.

But the Informative-Drawings paper trains with **four** loss terms (Eq. 5):

> L = 10·L_CLIP + 10·L_geom + 1·L_GAN + 0.1·L_cycle

Where:

- **L_GAN** (Eq. 1, LSGAN) — adversarial pressure from D_A and D_B on both
  translation directions.
- **L_geom** (Eq. 2) — `‖G_Geom(I(G_A(a))) − F(a)‖` with `F`=MiDaS pseudo-GT
  depth, `I`=InceptionV3 features at Mixed 6b node, `G_Geom`=pix2pixHD
  GlobalGenerator2 mapping features→depth.
- **L_CLIP** (Eq. 3) — `‖CLIP(G_A(a)) − CLIP(a)‖` (MSE) using CLIP ViT-B/32.
- **L_cycle** (Eq. 4) — L1 reconstruction both directions, weight 0.1.

Critical ablation from Table 2:

| Loss removed | Contour style (user preference for full method) | Anime style | Total |
|---|---|---|---|
| Without depth | **92.2%** | 48.3% | 70.3% |
| Without CLIP | 98.9% | 84.9% | 92.0% |
| Without cycle | 87.0% | 64.9% | 76.0% |

Paper text: *"The depth loss is most useful for **sparse styles such as the
Contour Drawings style**, where it adds occluding contours and textures."*
*"The CLIP loss adds the most lines."*

We deliberately picked `contour_style.pth` because it matched our reference
samples — i.e. we picked the style most dependent on the depth loss being
present during training. Then we stripped both depth and CLIP. The failure
modes we've seen ("faint ghosts," "soft pencil dominates," "lines not in
meaningful places") map directly onto the paper's "without depth" and
"without CLIP" ablation visuals (Fig. 5).

## Hypothesis

Adding CLIP + depth supervision to the existing paired-L1 setup will give
the model semantic and geometric anchors that 424 paired examples alone
cannot provide. The PatchGAN on unpaired labels will sharpen the output
style. Net effect: a single-stage F4 run produces visibly cleaner,
better-localized line drawings than v0.2.0 at threshold 0.95.

Falsifiable: on the same Nevermind / The Wall / Thriller holdout covers,
side-by-side rendering of F4 vs v0.2.0 at 512px shows F4 wins on at least
2/3 covers per blind review.

Stronger hypothesis (F5, two-stage): pretraining on 10k COCO photos with
our 424 labels as an unpaired style corpus, then short paired fine-tune,
produces a model that handles non-album photographs gracefully (a property
v0.2.0 does not have, and which we currently can't even evaluate).

## Architecture changes

### New module: `albumify/clip_loss.py`

```python
class CLIPSemanticLoss(nn.Module):
    """MSE between CLIP image embeddings of pred and target.

    Pred is 1-channel; replicated to 3ch before CLIP. CLIP model is frozen.
    Uses ViT-B/32 from openai/clip per the paper.
    """
    def __init__(self, device): ...
    def forward(self, pred_gray, target_rgb) -> Tensor: ...
```

- Backbone: `openai/clip` ViT-B/32 (`pip install git+https://github.com/openai/CLIP.git`).
- Input: pred is `[B,1,H,W]` ∈ [0,1] (post-sigmoid); target is the original
  RGB cover `[B,3,H,W]` ∈ [0,1].
- Resize both to 224 before CLIP (CLIP's native resolution).
- Loss: `((pred_emb - target_emb)**2).mean()` per Eq. 3.

### New module: `albumify/geom_loss.py`

```python
class GeomDepthLoss(nn.Module):
    """L1 between depth-from-line-drawing and MiDaS depth of original cover.

    Uses pretrained InceptionV3 (Mixed 6b features) + pretrained G_Geom
    (pix2pixHD GlobalGenerator2) from upstream's feats2Geom checkpoint.
    Pseudo-GT depth from MiDaS is precomputed once per cover and cached.
    """
    def __init__(self, g_geom_ckpt, device, midas_cache_dir): ...
    def forward(self, pred_drawing, cover_id) -> Tensor: ...
```

- `G_Geom`: download `feats2Geom.zip` from the upstream repo's pretrained
  checkpoints, unpack to `artifacts/feats2Geom/`. Architecture is the
  pix2pixHD GlobalGenerator2 (the paper supplemental Table 7 shows the
  exact layer spec — 4 conv + 9 ResNet + 3 deconv).
- `InceptionV3`: `torchvision.models.inception_v3(weights=IMAGENET1K_V1)`,
  features extracted at the Mixed_6b node (paper supplemental Sec 6.3).
- **MiDaS depth pre-computation:** rather than running MiDaS in the
  training loop, precompute `F(cover)` once per cover into
  `data/depth/{cover_id}.npy`. Saves ~50ms/sample/step on the GPU.
  Use `torch.hub.load("intel-isl/MiDaS", "DPT_Large")` — the high-quality
  variant. (Decision 2026-05-23: "do it right." The paper uses Miangoleh
  et al. boost-merge, which wraps MiDaS in a multi-resolution merging
  pipeline; DPT-Large is the closest single-shot substitute without
  pulling in their separate repo. If F3 results show depth noise is
  bottlenecking quality, escalate to a full Miangoleh integration.)
- Loss: L1 between `G_Geom(I(pred_drawing))` and the cached `F(cover)`.

### New module: `albumify/discriminator.py`

```python
class PatchGAN70(nn.Module):
    """70×70 PatchGAN per pix2pix/CycleGAN."""
    # 4 conv-instance_norm-LeakyReLU blocks, then 1x1 head, single-ch out
    def __init__(self, in_ch=1, ngf=64): ...
```

- LSGAN loss (Eq. 1): D outputs raw logits, loss is
  `((D(real) - 1)**2).mean() + (D(fake)**2).mean()` for D, and
  `((D(fake) - 1)**2).mean()` for G.
- Single discriminator D_B on the drawing side (we skip D_A — domain A is
  album covers, which we never generate, so D_A has nothing to do unless
  we add Stage-5 two-direction cycle training).
- Trained with its own Adam optimizer, lr=2e-4, β=(0.5, 0.999).

### Changed: `albumify/model.py`

No code change. We just invoke `Generator(n_residual_blocks=3)`. The pretrained
`contour_style.pth` ckpt loads strict=True at 3 blocks (eliminates the 24
random-init `model2.3` through `model2.8` keys flagged by the Plan E
diagnostic).

### Changed: `albumify/train.py`

- Add `--optimizer {adam,adamw}` (default adam, was implicitly adamw).
- Add `--clip-weight`, `--geom-weight`, `--gan-weight` flags.
- Add `--n-residual-blocks` (already exists, just used now).
- Discriminator: separate optimizer, alternating G/D steps (1:1).
- Loss assembly:

  ```
  total = l1_weight * L_l1
        + clip_weight * L_clip
        + geom_weight * L_geom
        + gan_weight * L_gan_G
  ```

- The L1 paired term **stays** in F1–F4 with weight 1.0; we still have
  pixel-aligned targets and it costs nothing to keep them informing the
  optimizer. (Departs from the paper's pure unpaired setup but is the
  honest use of our data.) For F5 Stage 1, L1 weight = 0 (no paired
  signal between COCO photos and our labels).

## New dependencies & artifacts

Add to `pyproject.toml` `[train]` extra:

- `clip @ git+https://github.com/openai/CLIP.git`
- `timm` (MiDaS via torch.hub already pulls this, but pin explicitly)

Add to `artifacts/` via `infra/setup_vm.sh`:

- `artifacts/feats2Geom/feats2depth_state.pth` — from upstream repo
  (Google Drive link in their README, "Features-to-Depth network")
- `artifacts/dpt_large.pt` — auto-downloaded by torch.hub on first use,
  but cache to artifacts/ for reproducible VM setup (~1.3GB)
- `artifacts/clip_vit_b32.pt` — auto-downloaded on first CLIP load

Precomputed per-cover depth: stored under `data/depth/`, gitignored,
regenerated by a new `python -m albumify.precompute_depth` script that
walks `data/covers/` and writes `data/depth/{stem}.npy` (float16, ~50KB each
at 256² = ~22MB for 424 covers).

## Data strategy

### F1–F4: paired-only on existing 424 covers

Same dataset as Plans C/D/E. No data changes. Each loss adds a new
supervision *over the same examples*, so EV per dollar is very high.

### F5: two-stage with COCO + unpaired labels

This is the most ambitious bet and the one that addresses the
"data quantity is the bottleneck" finding head-on by repurposing our
labels as unpaired style supervision.

Stage 1 (broad pretraining):
- Domain A: 10,000-image subset of COCO train2017 (matches paper).
  License: CC4.0. Download via `wget` of the public split.
- Domain B (unpaired): our 424 Gemini labels, no paired correspondence to
  COCO photos.
- Losses: CLIP (w=10) + depth (w=10) + GAN (w=1) + cycle (w=0.1).
  No L1 paired term — there's no correspondence.
- Train ~30 epochs at batch 6, lr=2e-4 (matches paper).
- Output: `runs/plan-f-stage1/best.pt` — a "Gemini-styled
  Informative-Drawings" model that generalizes to arbitrary photos.

Stage 2 (cover-specific fine-tune):
- Domain A: our 424 covers.
- Losses: L1 (w=1) + CLIP (w=1, downweighted) + depth (w=1, downweighted).
  Drop GAN to avoid destabilizing the fine-tune. Drop cycle for the
  same reason.
- Warm-start from Stage 1's best.pt.
- Train ~10 epochs at batch 8, lr=5e-5 (low — we're polishing).
- Output: `runs/plan-f-stage2/best.pt` — the actual shipped model.

## Experiment matrix

Each row is one training run. Stop conditions are checks to apply before
authorizing the next run.

| Run | Config (deltas from F1) | Goal | Stop condition before next |
|---|---|---|---|
| F1 | 3 blocks + full fine-tune + Adam(2e-4) + paired L1 only | Isolate "did we ever properly warm-start with all params loaded" | F1 holdouts visibly cleaner than v0.1.0 with same threshold; if not, the architecture mismatch was never the issue |
| F2 | F1 + CLIP loss (w=10) | Isolate CLIP's contribution. Paper's "biggest single loss." | F2 holdouts add lines vs F1 (per paper ablation Fig. 5 showing CLIP "adds the most lines") |
| F3 | F2 + depth loss (w=10) | Add the contour-style-critical loss | F3 holdouts add occluding contours / texture vs F2 |
| F4 | F3 + PatchGAN (w=1) | Full single-stage paper recipe on paired data | F4 ≥ v0.2.0 on the holdout blind review |
| F5a | Two-stage, Stage 1 only (COCO + unpaired labels) | Validate broad pretraining works on Gemini style | Stage 1 produces recognizable Gemini-aesthetic line drawings on COCO val images |
| F5b | F5a → Stage 2 paired fine-tune | Combine broad pretraining + cover-specific polish | F5b > F4 on holdouts |

Authorize each next run only after the prior stop condition is met. Don't
run F2 if F1 doesn't beat v0.1.0 — that signals the architecture fix isn't
delivering and we need to debug load_pretrained before adding losses.

## Eval methodology

**Quantitative (per run, automatic):**
- val_total, val_l1, val_clip, val_geom, val_gan_D — written to
  metrics.jsonl per epoch.
- F1 score @ thr=0.5, 0.7, 0.9 on val set (already wired in eval.py).

**Qualitative (per run, after training):**
- Render Nevermind / The Wall / Thriller at 512px with each of: no
  threshold, threshold 0.5, 0.7, 0.9, 0.95.
- Render the same 3 covers through every model trained so far (v0.1.0,
  v0.2.0, Plans C/D/E, F1..F5) into a single 8×5 montage PNG.
- Blind self-review: file names anonymized; rank by visual quality.

**Ship criterion:** F-something must beat v0.2.0 in the blind review on
at least 2/3 holdout covers. Otherwise we keep v0.2.0 as the shipped model
and document Plan F as a research result.

## Testing strategy

Per-component unit tests, all CPU-runnable so they fit in the existing
fast test suite:

- `tests/test_clip_loss.py`
  - CLIP loss returns a scalar Tensor with grad.
  - CLIP loss is 0 when pred and target are pixel-identical (sanity, not
    perfectly true due to gray→RGB tile but close to 0).
  - Frozen CLIP weights: forward+backward leaves CLIP params unchanged.
- `tests/test_geom_loss.py`
  - With a stub G_Geom + InceptionV3, loss returns scalar with grad.
  - Pre-computed depth path: cache hit returns same tensor as cache miss
    + write.
- `tests/test_discriminator.py`
  - PatchGAN output shape is [B, 1, H/16, W/16] (standard 70px receptive
    field).
  - LSGAN loss: D loss is 0 when D(real)=1 and D(fake)=0; G loss is 0
    when D(fake)=1.
- `tests/test_train_plan_f.py`
  - Smoke test: 2 covers, 1 epoch, all four losses enabled, asserts
    metrics.jsonl has clip/geom/gan columns.
  - Smoke test: `--no-clip --no-geom --no-gan` reduces to current
    paired-L1 path (backwards compat).

## Backwards compatibility

- F1–F5 produce `apply_sigmoid=True` checkpoints (we're going back to
  sigmoid-in-model since we drop BCE-with-logits). eval/export/infer already
  handle this — it's the v0.2.0 layout, not the Plan C layout.
- Existing `--loss l1` / `--loss bce` flags untouched. F runs use `--loss l1`
  with the new `--clip-weight` / `--geom-weight` / `--gan-weight` flags
  layered on top.
- Tests for Plan C BCE path still pass (no shared code path with the new
  losses).

## Risks & open questions

- **CLIP on grayscale line drawings.** CLIP was trained on natural RGB
  photos; embedding a 1-channel sigmoid output (replicated to 3ch) is
  out-of-distribution. Paper does this and it works, but worth verifying
  early: on F2 day-one, compute `CLIP(label) − CLIP(cover)` distance for
  10 pairs and sanity-check the gradient signal isn't dominated by
  domain-gap noise.
- **DPT-Large vs the paper's boost-merge depth.** Per 2026-05-23 decision
  we use DPT-Large (the highest-quality single-shot MiDaS variant). The
  paper's Miangoleh boost-merge is still strictly higher resolution but
  requires integrating a separate repo. Stay with DPT-Large unless F3
  results visibly suffer from depth quality, then escalate.
- **PatchGAN instability on 424 examples.** GAN training with N=424 is
  thin. Mitigations: spectral norm on D, low GAN weight (1.0 vs L1's 1.0
  is reasonable), early stopping per-G val loss not GAN val loss.
- **Memory budget on L4.** Adding CLIP forward (ViT-B/32, ~150M params)
  + InceptionV3 (~25M, only for L_geom) + G_Geom (~30M) + D (~3M) onto
  the 11.7M G means total fwd memory roughly triples. L4 has 24GB; this
  should fit at batch 6, but if not, drop to batch 4.
- **F5 COCO download licensing.** COCO is CC4.0, fine to use for
  research. We don't redistribute. Mention in repo NOTICE.
- **Stage-1 risk: model overfits to general line-drawing style and
  loses the Gemini-specific aesthetic.** Stage 2 paired fine-tune is the
  corrective; if even after Stage 2 the outputs don't look like our
  Gemini labels, the Stage-1 → Stage-2 transition needs more epochs at
  higher L1 weight.

## Out of scope

- Cycle loss (the paper's 4th loss). Weight 0.1 in the paper; per Table 2
  ablation, removing it has the smallest impact (76% total). Add later
  if F4/F5 land below v0.2.0 and we need every last drop.
- D_A (the photo-side discriminator). Only matters in two-direction
  cycle training, and we're keeping the cycle pruned.
- New evaluation protocols (depth-prediction or caption-similarity
  metrics from the paper). Our blind-review-on-holdouts is sufficient
  for the ship decision; the paper's evals are for academic comparison.
- INT8 export — unaffected by Plan F (same Generator architecture for
  inference). [[reference-toolchain-int8-regression]] still open
  separately.
- Anime-style starting weights. Documented in the Plan F prior research
  but explicitly deferred: we should change one variable at a time, and
  contour_style is what the existing evidence is built on.

## Cost & time estimate

L4 g2-standard-4 on-demand at ~$0.85/hr.

| Run | Compute time | Notes | Cost |
|---|---|---|---|
| F1 | ~10 min | Same as Plan E + 1 epoch buffer | ~$0.15 |
| F2 | ~25 min | CLIP fwd adds ~1.5s/step | ~$0.40 |
| F3 | ~35 min | + InceptionV3 + G_Geom fwd | ~$0.55 |
| F4 | ~50 min | + PatchGAN G+D steps | ~$0.75 |
| F5a (Stage 1) | ~6 hours | 10k COCO × 30 epochs × all losses | ~$5.10 |
| F5b (Stage 2) | ~10 min | Small fine-tune | ~$0.15 |
| Depth precompute (one-time, 424 covers) | ~5 min on L4 GPU or 30 min CPU | 424 × DPT-Large | ~$0.10 |
| Depth precompute (one-time, 10k COCO for F5) | ~2 hours on L4 | 10k × DPT-Large | ~$1.70 |
| **Total (F1–F5b + depth precomputes)** | **~10 hours VM time** | | **~$8.90** |

Bracket runs (one F1 retry if architecture fix needs debugging, one F4
retry to dial GAN weight, one F5b retry): add ~$2. **Budget envelope: $11.**

Far inside the "multiple training runs is fine and well within budget"
the user authorized.

## Sequencing & next steps

Per 2026-05-23 decision: ship F1 early in parallel with planning the
bigger work, since F1 needs zero new dependencies and produces an
immediate gate signal.

1. This spec — user reviewed, three decisions made (DPT-Large for depth,
   10k COCO for F5, F1-in-parallel). ✓
2. **F1 implementation in parallel:** smallest possible diff — add
   `--optimizer` flag (adam | adamw) to `train.py`, default adam with
   β=(0.5, 0.999), no weight decay. Run F1 on L4 immediately, ~10 min.
   F1 = `--n-residual-blocks 3 --no-lora --optimizer adam --lr 2e-4
         --weight-decay 0 --pretrained-ckpt artifacts/...`
3. **Plan doc in parallel:** while F1 trains, write
   `docs/superpowers/plans/2026-05-23-plan-f-paper-faithful-recipe.md`
   with TDD task breakdown for F2–F5 modules.
4. After F1 returns: blind review against v0.1.0. If F1 ≥ v0.1.0 → proceed
   to F2 implementation. If F1 < v0.1.0 → debug `load_pretrained` key
   remapping before any new module work.
5. Implement CLIP loss → run F2 → review.
6. Implement depth loss (CPU MiDaS precompute first, then training-time
   InceptionV3 + G_Geom) → run F3 → review.
7. Implement PatchGAN → run F4 → review.
8. F5 only after F4 ships or after F4 plateaus and we explicitly choose
   to escalate to the two-stage bet.
