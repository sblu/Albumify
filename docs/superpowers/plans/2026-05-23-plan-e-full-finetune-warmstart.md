# Plan E — Full fine-tune + BCE + warm-start implementation/run plan

> The only code changes are diagnostic (richer pretrained-load print + a
> smoke test for the new flag combination). The actual training is a
> CLI-only configuration variation.

**Goal:** Land code so a single `gcloud compute ssh ...` + training command
can run Plan E with full warm-start visibility.

**Spec:** `docs/superpowers/specs/2026-05-23-plan-e-full-finetune-warmstart-design.md`

---

### Task 1 — Branch (done)

Branched off `feat/plan-c-bce-logits` (the BCE wiring branch). Plan D's
branch is independent and merges separately.

- [x] `git switch feat/plan-c-bce-logits`
- [x] `git switch -c feat/plan-e-full-bce-warmstart`

### Task 2 — Failing test: pretrained-load diagnostic shows key names

**File:** `tests/test_train.py` (extend)

- [ ] Append a test that loads a Generator + saves a partial state_dict
      (e.g., only `model0.*` keys), feeds it to a `train()` call with
      `pretrained_ckpt=<that partial ckpt>`, captures stdout, and asserts
      the printed line includes the names of missing keys, not just a
      count.

The point: keep the diagnostic surface stable so future refactors don't
regress to count-only logging.

### Task 3 — Implement diagnostic print

**File:** `albumify/train.py`

- [ ] Replace the single-line `[pretrained] missing=N unexpected=M` print
      (around L154) with:
      - count line as today
      - if `missing`: print `[pretrained] missing keys (first N): k1, k2, ...`
      - if `unexpected`: print `[pretrained] unexpected keys (first N): k1, k2, ...`

Make the cap small (8) so the log stays scannable.

### Task 4 — Failing test: Plan E flag combo runs

**File:** `tests/test_train.py` (extend)

- [ ] Smoke test: `--no-lora` + `--loss bce` + `--pretrained-ckpt` on a
      tiny dataset (2 epochs, batch 2, tiny generator). Build a partial
      pretrained state_dict from a `Generator(sigmoid=True)`, save it to
      tmp, call `train()` with that as `pretrained_ckpt`. Assert:
      - the run completes without exception,
      - `best.pt` exists with `apply_sigmoid=False` and `loss_type="bce"`,
      - the pretrained load was at least partially successful (not 100%
        missing — sanity check that the path is real).

This catches the LoRA-vs-no-LoRA × pretrained-load interaction.

### Task 5 — Commit + push

- [ ] `git add docs/superpowers/specs/2026-05-23-plan-e-* docs/superpowers/plans/2026-05-23-plan-e-* albumify/train.py tests/test_train.py`
- [ ] Two commits: one for docs, one for the train.py + test changes.
- [ ] `git push -u origin feat/plan-e-full-bce-warmstart`

---

## When ready to actually train

1. Spin up the VM (per [[reference-gcp-vm-quirks]]):
   ```bash
   PROJECT=albumartifier SPOT=0 MACHINE_TYPE=g2-standard-4 GPU_TYPE=nvidia-l4 ZONE=us-east1-d ./infra/create_vm.sh
   ```
2. Bootstrap:
   ```bash
   gcloud compute ssh albumify-train --zone us-east1-d --project albumartifier --tunnel-through-iap
   git clone https://github.com/sblu/Albumify.git && cd Albumify
   git switch feat/plan-e-full-bce-warmstart
   ./infra/setup_vm.sh
   ```
3. Upload data **and** `informative_drawings.pth` (Plan E needs it):
   ```bash
   gcloud compute scp --tunnel-through-iap --recurse data/{covers,labels,splits} data/albums.json artifacts/informative_drawings.pth ... :~/Albumify/data/   # and artifacts/
   ```
4. Run training in tmux:
   ```bash
   python -m albumify.train \
     --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
     --pretrained-ckpt artifacts/informative_drawings.pth \
     --out-dir runs/full-bce-warm-ngf64 \
     --no-lora --ngf 64 \
     --loss bce \
     --perceptual-weight 0 \
     --epochs 25 --batch-size 8 --lr 2e-4 \
     --weight-decay 1e-4 \
     2>&1 | tee runs/full-bce-warm-ngf64/train.log
   ```
5. Watch the first 30 seconds: the `[pretrained] missing keys` line shows
   what the warm-start actually loaded. If `model4.1.weight` (the final
   tail conv) is missing, abort and fix `load_pretrained` first.
6. Standard post-training: eval → export → holdout renders (no threshold)
   → scp → delete VM.

## Decision tree after the run

| outcome | next step |
|---|---|
| crisp clean lines, no threshold | tag `v0.3.0-bce-warmstart-full`, update README, ship |
| crisper than Plan C but needs thr ~0.5–0.7 | minor improvement; consider as v0.3.0 with default threshold |
| ≈ Plan C or worse | unwind, stick with v0.2.0; investigate dataset expansion (option 2) |
| `model4.1` was missing → catastrophic | fix `load_pretrained` key remapping, re-run |
