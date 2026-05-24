# Plan F — Paper-faithful recipe implementation/run plan

**Spec:** `docs/superpowers/specs/2026-05-23-plan-f-paper-faithful-recipe-design.md`

**Strategy:** ship F1 first (smallest possible diff, gates everything
downstream), review, then write follow-up plan docs for F2–F5b as each is
authorized.

---

## Stage F1 — Architecture & optimizer fix only

**Goal:** Test the hypothesis "we've never actually had a clean warm-start"
in isolation, before adding any new loss machinery.

**Code surface:** `albumify/train.py` gets `--optimizer {adam,adamw}` flag.
That's it. Everything else is a CLI configuration change against existing
code (`--n-residual-blocks 3`, `--no-lora`, `--pretrained-ckpt`).

### Task 1 — Branch (done)

- [x] `git switch -c feat/plan-f-paper-faithful` (off main or off the
      diagnostic-print commit `da875d9`, whichever is current HEAD)

### Task 2 — Failing test: `make_optimizer` produces Adam with paper β (done)

**File:** `tests/test_train.py` (extend)

- [x] Append `test_make_optimizer_adam_uses_torch_adam_with_paper_betas`:
      construct `TrainConfig(optimizer="adam", lr=2e-4, weight_decay=0)`,
      call `make_optimizer(cfg, params)`, assert exact type is
      `torch.optim.Adam`, `betas == (0.5, 0.999)`, `lr == 2e-4`,
      `weight_decay == 0`.
- [x] Append `test_make_optimizer_adamw_preserves_legacy_behavior`: same
      but `optimizer="adamw"`, assert exact type is `torch.optim.AdamW`
      with the lr/wd that were passed.

### Task 3 — Implement `make_optimizer` + CLI flag (done)

**File:** `albumify/train.py`

- [x] Add `optimizer: str = "adam"` field to `TrainConfig`. Default "adam"
      because F1+ is paper-faithful by default; old invocations that don't
      pass the flag get the new default (intended).
- [x] Define module-level `make_optimizer(cfg, params)` per the spec's
      Architecture section. Switch on `cfg.optimizer`, raise on unknown.
- [x] Replace the inline `torch.optim.AdamW(...)` in `train()` with a
      single call to `make_optimizer(cfg, opt_params)`.
- [x] Add `--optimizer` argparse flag with `choices=("adam", "adamw")`,
      default `"adam"`, and a help string referencing Plan F.
- [x] Thread `args.optimizer` into the `TrainConfig(...)` construction.

### Task 4 — Verify ckpt loads strict at 3 blocks (done)

This is a local sanity check, not a committed test (artifact files are
not in the repo). Already executed:

- `Generator(n_residual_blocks=3, ngf=64)` + `load_pretrained` on
  `artifacts/informative_drawings.pth` returns **missing=0, unexpected=0**.
  4,290,945 params (matches paper's "~4.4M" claim).

### Task 5 — Commit, push, set up F1 invocation

- [ ] `git add docs/superpowers/specs/2026-05-23-plan-f-paper-faithful-recipe-design.md \
      docs/superpowers/plans/2026-05-24-plan-f-paper-faithful-recipe.md \
      albumify/train.py tests/test_train.py`
- [ ] Three commits: one for the spec doc, one for the plan doc, one for
      the train.py + tests change.
- [ ] `git push -u origin feat/plan-f-paper-faithful`

### Task 6 — Run F1 on GCP

1. Spin up VM (per [[reference-gcp-vm-quirks]]):
   ```bash
   PROJECT=albumartifier SPOT=0 MACHINE_TYPE=g2-standard-4 \
     GPU_TYPE=nvidia-l4 ZONE=us-east1-d ./infra/create_vm.sh
   ```
2. Bootstrap:
   ```bash
   gcloud compute ssh albumify-train --zone us-east1-d \
     --project albumartifier --tunnel-through-iap
   git clone https://github.com/sblu/Albumify.git && cd Albumify
   git switch feat/plan-f-paper-faithful
   ./infra/setup_vm.sh
   ```
3. Upload data + pretrained ckpt:
   ```bash
   gcloud compute scp --tunnel-through-iap --recurse \
     data/covers data/labels data/splits data/albums.json \
     albumify-train:~/Albumify/data/ \
     --zone us-east1-d --project albumartifier
   gcloud compute scp --tunnel-through-iap \
     artifacts/informative_drawings.pth \
     albumify-train:~/Albumify/artifacts/ \
     --zone us-east1-d --project albumartifier
   ```
4. Run F1 in tmux:
   ```bash
   tmux new -s train
   mkdir -p runs/plan-f1-arch-fix
   python -m albumify.train \
     --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
     --pretrained-ckpt artifacts/informative_drawings.pth \
     --out-dir runs/plan-f1-arch-fix \
     --no-lora \
     --n-residual-blocks 3 \
     --ngf 64 \
     --optimizer adam \
     --lr 2e-4 --weight-decay 0 \
     --loss l1 \
     --perceptual-weight 0 \
     --epochs 30 --batch-size 8 \
     2>&1 | tee runs/plan-f1-arch-fix/train.log
   ```
5. **First-30-seconds check:** the `[pretrained] missing=` line must read
   **`missing=0 unexpected=0`**. If anything else, abort and debug — the
   whole point of F1 is testing a strict-loaded warm-start. (Local check
   already passed; remote should match.)
6. Eval + export + holdout renders:
   ```bash
   python -m albumify.eval --ckpt runs/plan-f1-arch-fix/best.pt \
     --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
     --ngf 64 --n-residual-blocks 3
   python -m albumify.export --ckpt runs/plan-f1-arch-fix/best.pt \
     --out runs/plan-f1-arch-fix/model.onnx --ngf 64 --n-residual-blocks 3
   # Holdout renders at 256/512/1024 for Nevermind / The Wall / Thriller
   ```
7. `scp` ckpt + renders back, delete VM.

### Stop condition (gate to F2)

**F1 must beat v0.1.0 (the original LoRA r=8 warm-start) on the same
holdout covers**, at the same threshold (0.95) or better. v0.1.0 produced
"faint ghosts requiring threshold 0.95 to be recognizable" (RESUME.md L24).

| F1 outcome | next step |
|---|---|
| crisp clean lines, no threshold needed | huge surprise; possibly ship as v0.3.0 directly |
| visibly stronger than v0.1.0 at thr=0.95, still soft | expected; proceed to F2 (add CLIP loss) |
| ≈ v0.1.0 or worse | the architecture mismatch wasn't the bottleneck; rethink before adding losses |
| `missing>0` in pretrained log | bug in `load_pretrained` or wrong ckpt; abort, debug |

### Decision after F1

- F1 ≥ v0.1.0 → write Plan F2 plan doc, implement `albumify/clip_loss.py`,
  run F2.
- F1 < v0.1.0 → re-examine assumptions, possibly fall back to v0.2.0 and
  reconsider the entire spec. (Unlikely given the 0/0 strict-load.)

---

## Stage F2 — Add CLIP semantic loss (NOT YET PLANNED)

Triggers after F1 review.

Sketch:
- New file `albumify/clip_loss.py` with `CLIPSemanticLoss` per spec.
- New CLI flag `--clip-weight` (default 0.0; F2 sets 10.0).
- Add `clip` package to `[train]` extra in pyproject.toml.
- Tests (per spec testing section): frozen-weights, scalar+grad output.
- Run: same F1 command + `--clip-weight 10`.

Estimated effort: ~2 hours implementation, ~25 min run, ~$0.40.

---

## Stage F3 — Add depth/geometry loss (NOT YET PLANNED)

Triggers after F2 review.

Sketch:
- New file `albumify/geom_loss.py` (G_Geom + InceptionV3 + cached MiDaS).
- New file `albumify/precompute_depth.py` (one-time DPT-Large pass over
  `data/covers/` → `data/depth/`).
- New CLI flags `--geom-weight`, `--depth-cache-dir`.
- Download `feats2Geom` ckpt in `infra/setup_vm.sh`.
- Tests per spec.
- Run: F2 command + depth precompute + `--geom-weight 10`.

Estimated effort: ~3 hours implementation, ~35 min run, ~$0.65 (incl.
~$0.10 one-time depth precompute on 424 covers).

---

## Stage F4 — Add PatchGAN discriminator (NOT YET PLANNED)

Triggers after F3 review.

Sketch:
- New file `albumify/discriminator.py` with `PatchGAN70` per spec.
- LSGAN loss, separate Adam optimizer, alternating G/D steps in `train()`.
- New CLI flag `--gan-weight`.
- Tests per spec.
- Run: F3 command + `--gan-weight 1`.

Estimated effort: ~3 hours implementation, ~50 min run, ~$0.75.

---

## Stage F5a/F5b — Two-stage with COCO + paired fine-tune (NOT YET PLANNED)

Triggers only if F4 plateaus or we explicitly choose to escalate.

Sketch:
- New file `albumify/coco_dataset.py` (unpaired dataset for COCO photos).
- New file `albumify/precompute_depth.py` extended for COCO (~10k images,
  ~$1.70 one-time).
- Refactor `train()` to support `--paired none` (Stage 1: photos + unpaired
  labels) and `--paired only` (Stage 2: cover→label).
- Sets dropdown of pretrained-style files (`stage1/best.pt`) for warm-start
  into Stage 2.

Estimated effort: ~1 day implementation, ~6 hours Stage 1 + 10 min Stage 2,
~$5.25 + bracket.

---

## Outcome — TBD

(Populated after F1 runs.)
