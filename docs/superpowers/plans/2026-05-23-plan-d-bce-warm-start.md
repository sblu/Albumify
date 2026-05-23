# Plan D — BCE + warm-start implementation/run plan

> No code changes. This is a training-configuration variation that uses
> existing CLI flags from Plan B (`--pretrained-ckpt`, `--lora-rank`) and
> Plan C (`--loss bce`, `--perceptual-weight 0`).
>
> The "plan" is mostly: create branch, run training on VM, eval, render
> holdouts, A/B against Plan C@thr0.6 and Plan B v0.2.0@thr0.95, decide.

**Goal:** Produce a Plan-C-style BCE-trained model that, with the pretrained
line-drawing prior in place, generates *crisp* line drawings without
post-process thresholding.

**Spec:** `docs/superpowers/specs/2026-05-23-plan-d-bce-warm-start-design.md`

---

### Task 1 — Create branch

- [ ] `git status -sb` → clean (untracked exploration files OK).
- [ ] `git switch -c feat/plan-d-bce-warm-start`
- [ ] Commit the spec + plan: `git add docs/superpowers/{specs,plans}/2026-05-23-plan-d-*.md && git commit -m "docs(plan-d): BCE + warm-start design + run plan"`
- [ ] `git push -u origin feat/plan-d-bce-warm-start`

### Task 2 — Spin up VM

- [ ] L4 on g2-standard-4 (T4 capacity is unreliable; per [[reference-gcp-vm-quirks]]):
  ```bash
  PROJECT=albumartifier SPOT=0 MACHINE_TYPE=g2-standard-4 GPU_TYPE=nvidia-l4 ZONE=us-east1-d ./infra/create_vm.sh
  ```
  Fall back to other L4 zones (us-east1-b, us-east1-c, us-central1-b, etc.) if exhausted.
- [ ] All gcloud ssh/scp must use `--tunnel-through-iap` (port 22 unreachable on direct IP in this VPC).

### Task 3 — Bootstrap

- [ ] `git clone https://github.com/sblu/Albumify.git && cd Albumify && git switch feat/plan-d-bce-warm-start`
- [ ] `./infra/setup_vm.sh`
- [ ] scp data: covers/ + labels/ + splits/ + albums.json + **`artifacts/informative_drawings.pth`** (Plan D needs this — unlike Plan C which trained from scratch).

### Task 4 — Train

- [ ] In tmux:
  ```bash
  cd ~/Albumify && . .venv/bin/activate
  mkdir -p runs/lora-r16-bce
  python -m albumify.train \
    --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
    --pretrained-ckpt artifacts/informative_drawings.pth \
    --out-dir runs/lora-r16-bce \
    --ngf 64 \
    --lora-rank 16 --lora-alpha 16 \
    --loss bce \
    --perceptual-weight 0 \
    --epochs 25 --batch-size 8 --lr 1e-3 \
    2>&1 | tee runs/lora-r16-bce/train.log
  ```
- [ ] Watch startup: `[pretrained] missing=N unexpected=M` (small N+M, ideally 0 if architecture matches), `[lora] wrapped=K lora_params=...`, `[bce] empirical edge fraction ...`.
- [ ] If val_total stops improving by epoch ~15-18, expect best.pt to settle there.

### Task 5 — Eval + export

- [ ] On VM:
  ```bash
  python -m albumify.eval \
    --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
    --ckpt-path runs/lora-r16-bce/best.pt \
    --lora-rank 16 --lora-alpha 16 \
    --out-dir runs/lora-r16-bce/eval --grid-n 32

  python -m albumify.export \
    --ckpt-path runs/lora-r16-bce/best.pt \
    --lora-rank 16 --lora-alpha 16 \
    --out-dir runs/lora-r16-bce \
    --int8   # may fail per the toolchain regression; fp32 will succeed
  ```

### Task 6 — Holdout renders, NO threshold

- [ ] On VM, 3 albums × 3 sizes:
  ```bash
  mkdir -p runs/lora-r16-bce/holdout
  for cover in nevermind-nirvana the-wall-pink-floyd thriller-michael-jackson; do
    for size in 256 512 1024; do
      python -m albumify.infer \
        --model runs/lora-r16-bce/model.fp32.onnx \
        --in "data/covers/${cover}.jpg" \
        --out "runs/lora-r16-bce/holdout/${cover}-${size}.png" \
        --size $size --threads 4
    done
  done
  ```

### Task 7 — Pull artifacts to laptop

- [ ] `scp --recurse` everything under `runs/lora-r16-bce/` to local `runs/lora-r16-bce/`.

### Task 8 — A/B verdict

Compare the holdout 512-px images across three models:

| run | threshold used | expected crispness |
|---|---|---|
| v0.2.0 ngf=96 from-scratch L1 | 0.95 | clean, sometimes sparse |
| Plan C ngf=96 from-scratch BCE | 0.60 | recognizable but sketchy/dotted |
| **Plan D LoRA-r16-BCE + warm-start** | **none, ideally** | **goal: crisp, no threshold** |

Decision tree:

- Plan D crisp w/o threshold → tag `v0.3.0-bce-lora-r16`, update README.
- Plan D similar to Plan C@0.6 → marginal win, document, decide on next.
- Plan D worse than Plan C → unwind, return to Plan C@0.6 as v0.3.0.

### Task 9 — Delete VM

- [ ] `gcloud compute instances delete albumify-train --zone <ZONE> --project albumartifier --quiet`
- [ ] Confirm `gcloud compute instances list --project=albumartifier` shows zero.

---

## Cost & time estimate

- L4 g2-standard-4 on-demand: ~$0.85/hr.
- Train: ~12 min (LoRA on ngf=64 is much faster than the 30 min Plan C ngf=96 full-finetune).
- Eval + export + holdout renders + scp: ~5 min.
- Total VM time: ~20 min → ~$0.30.
