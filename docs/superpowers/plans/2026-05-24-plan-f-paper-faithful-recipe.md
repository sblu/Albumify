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

## Stage F3 — Add depth/geometry loss (CODE READY, RUN PENDING)

**Status: code prepared 2026-05-24. Branch `feat/plan-f-paper-faithful`
has every F3 change committed and pushed; just spin up a VM and run.**

### What's already done

- `albumify/feats2depth.py` — vendored `GlobalGenerator2` (G_Geom from
  upstream model.py, matches released ckpt shape) + frozen
  `InceptionMixed6bExtractor` (torchvision Inception_V3 IMAGENET1K_V1).
- `albumify/precompute_depth.py` — CLI that runs MiDaS DPT-Large over
  `data/covers/` and caches `data/depth/{slug}.npy` (float16, min-max
  normalized per image, default 256×256).
- `albumify/geom_loss.py` — `GeomDepthLoss(pred_gray, slugs)` pre-loads
  the depth cache eagerly at construction, fails fast on empty/missing
  cache, raises informative `KeyError` on missing slugs.
- `albumify/train.py` — `--geom-weight`, `--depth-cache-dir`,
  `--feats2depth-ckpt` flags; slug propagation through train + eval
  loops; tensorboard logs `train_step_geom` and `val_geom`.
- `tests/test_geom_loss.py` — 7 stub-injected tests (run locally) + 1
  real-ckpt smoke (skips without `artifacts/feats2Geom/feats2depth.pth`).
- `pyproject.toml` — adds `timm` (MiDaS dep) and `gdown` (Google Drive).
- `infra/setup_vm.sh` — pulls `feats2depth.zip` from upstream Google
  Drive (Drive ID `1Ov1BNue74Yu-57X2rpdjqZy0o-fnFoly`) and unzips it
  to `artifacts/feats2Geom/`. Idempotent.

### Pickup procedure (the "tomorrow" GCP run)

1. **Spin up VM** (per [[reference-gcp-vm-quirks]]):
   ```bash
   PROJECT=albumartifier SPOT=0 MACHINE_TYPE=g2-standard-4 \
     GPU_TYPE=nvidia-l4 ZONE=us-east1-d ./infra/create_vm.sh
   ```

2. **Bootstrap** (the new setup_vm.sh handles feats2depth download):
   ```bash
   gcloud compute ssh albumify-train --zone us-east1-d \
     --project albumartifier --tunnel-through-iap
   git clone https://github.com/sblu/Albumify.git && cd Albumify
   git switch feat/plan-f-paper-faithful
   ./infra/setup_vm.sh   # installs timm/gdown, downloads feats2depth.zip
   ```

   **Verify after setup:** `ls artifacts/feats2Geom/` should show at
   least one `.pth` file. If gdown failed (rate-limited), the script
   prints fallback instructions; download manually from
   <https://drive.google.com/file/d/1Ov1BNue74Yu-57X2rpdjqZy0o-fnFoly/view>
   and unzip to `artifacts/feats2Geom/`.

3. **Upload data + base ckpt** (use the tarball trick from F2's lessons
   to avoid the sftp file-by-file IAP slowness):
   ```bash
   # On local machine
   tar czf /tmp/albumify-data.tar.gz data/covers data/labels data/splits
   gcloud compute scp --tunnel-through-iap /tmp/albumify-data.tar.gz \
     albumify-train:~/Albumify/ --zone us-east1-d --project albumartifier
   gcloud compute scp --tunnel-through-iap artifacts/informative_drawings.pth \
     albumify-train:~/Albumify/artifacts/ --zone us-east1-d --project albumartifier
   # On VM
   cd ~/Albumify && tar xzf albumify-data.tar.gz && rm albumify-data.tar.gz
   ```

4. **Precompute depth** (one-time, ~3 min on L4):
   ```bash
   .venv/bin/python -m albumify.precompute_depth \
     --covers-dir data/covers --out-dir data/depth --resize 256
   # Expect: "[depth] done: wrote=424 skipped=0 failed=0 total=424"
   ```

   The cache is ~55 MB; subsequent runs reuse it.

5. **Auto-detect the feats2depth ckpt filename:** the zip's contents
   aren't documented. After step 2 finishes, identify the .pth file:
   ```bash
   ls artifacts/feats2Geom/
   F2D_CKPT=$(ls artifacts/feats2Geom/*.pth | head -1)
   echo "Will use: $F2D_CKPT"
   ```

6. **Run F3 training** in tmux:
   ```bash
   tmux new -s train
   mkdir -p runs/plan-f3-arch-fix-plus-clip-plus-geom
   nohup .venv/bin/python -m albumify.train \
     --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
     --pretrained-ckpt artifacts/informative_drawings.pth \
     --out-dir runs/plan-f3-arch-fix-plus-clip-plus-geom \
     --no-lora --n-residual-blocks 3 --ngf 64 \
     --optimizer adam --lr 2e-4 --weight-decay 0 \
     --loss l1 --perceptual-weight 0 \
     --clip-weight 10 \
     --geom-weight 10 \
     --depth-cache-dir data/depth \
     --feats2depth-ckpt "$F2D_CKPT" \
     --epochs 30 --batch-size 8 \
     > runs/plan-f3-arch-fix-plus-clip-plus-geom/train.log 2>&1 &
   ```

7. **First-30-seconds check** — head of `train.log` must show:
   ```
   [pretrained] missing=0 unexpected=0
   [clip] enabled weight=10.0 model=ViT-B/32 ...
   [feats2depth] missing=0 unexpected=0   <-- if non-zero, abort and inspect ckpt layout
   [geom] enabled weight=10.0 cache=data/depth ...
   ```

8. **Eval, holdouts, download** (same flags as F2, just point at the
   new run dir):
   ```bash
   .venv/bin/python -m albumify.eval \
     --ckpt-path runs/plan-f3-arch-fix-plus-clip-plus-geom/best.pt \
     --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
     --ngf 64 --n-residual-blocks 3 --no-lora \
     --out-dir runs/plan-f3-arch-fix-plus-clip-plus-geom/eval
   tar czf /tmp/plan-f3-results.tar.gz runs/plan-f3-arch-fix-plus-clip-plus-geom/
   # download via scp + delete VM
   ```

9. **Render holdouts locally** with the same one-liner used for F1/F2
   (saved in conversation history; basically a snippet that loads
   best.pt and runs each `data/holdout/*.png` through with thr=0.5,
   0.7, 0.85, 0.95).

### Expected per-step compute

Per F2's actual numbers (~10.7 sec/epoch with CLIP added), F3 adds:
- InceptionV3 + G_Geom forward per step: ~50 ms (much smaller models
  than CLIP's ViT)
- Depth cache GPU upload per batch: negligible (pre-loaded into RAM)

Estimated F3 epoch: ~12-13 sec → ~7 min total training time. Plus
~3 min depth precompute + ~3 min VM setup. **Total expected cost
~$1.10-1.30 for the F3 run.**

### Decision tree after F3

| F3 outcome | next step |
|---|---|
| Clean line drawings on occluding contours (paper-like) | tag v0.3.0-paper-faithful, ship |
| Less "halftone fill" than F2 but still too dense | proceed to F4 (PatchGAN on labels) for style pressure |
| Same density as F2 with structure shifted | depth loss did its job; F4 needed for stylization |
| Worse than F2 | something off in feats2depth load; inspect missing keys, may need to re-derive ckpt arch |

### Open risks for tomorrow

- **feats2depth.zip filename inside the archive is unknown.** Step 5
  handles this with auto-detection but if the zip contains nested
  directories the path may differ. Adjust if needed.
- **gdown rate limits** sometimes refuse anonymous Google Drive
  downloads. Manual fallback documented in setup_vm.sh + step 2.
- **DPT-Large hub download is ~340 MB.** First run of
  `precompute_depth.py` waits on this; subsequent runs use the
  torch.hub cache.

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
