# Albumify — Design Spec

**Date:** 2026-05-20
**Repo:** https://github.com/sblu/Albumify.git
**Status:** Approved design, awaiting implementation plan

## 1. Goal

Produce a tiny on-device model that turns a color album cover into a black-and-white line drawing in the style of the two reference samples (`AbbeyRoad-SingleLine.png`, `PinkFloydTheWall-SingleLine.png`), runnable on a Raspberry Pi 5 with 1 GB of available RAM, in under 5 seconds per image.

## 2. Why not Gemma (the original proposal)

The original `Overview.md` proposed fine-tuning Gemma. Gemma 3 has multimodal *input* (it can read images via its vision encoder) but it emits **text tokens only** — there is no image decoder in the architecture. Fine-tuning cannot add one. "Gemma 4" does not exist as of 2026-05.

This task is image-to-image translation, for which the established tool family is small CNN generators (pix2pix, CycleGAN, U-Nets). We picked **Informative-Drawings** (Chan et al., SIGGRAPH 2022) as the base: it is purpose-built for photo→line-drawing, ships pretrained checkpoints, and is ~17 MB FP32. LoRA-style adapters (the user's original request) are applied to its Conv2d layers.

## 3. End-to-end architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          (one-time, on laptop)                       │
│  1. DATA: data/albums.json → MusicBrainz → Cover Art Archive        │
│           → data/covers/{slug}.jpg  (500 covers, ~250 MB)           │
│  2. LABELS: Gemini 3.1 Flash Image Preview API                      │
│             → data/labels/{slug}.png (500 line drawings, ~250 MB)   │
│             ← CHECKPOINT: review 10-image sample before full run    │
└─────────────────────────────────────────────────────────────────────┘
                                     │ gsutil cp → GCS bucket
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       (GCP VM with T4 GPU, spot)                     │
│  3. TRAIN: PyTorch, Informative-Drawings + LoRA-Conv (rank 8)       │
│            450 train / 50 eval, ~30 min/run, L1 + VGG perceptual    │
│  4. EVAL: SSIM, LPIPS, side-by-side grid on held-out covers         │
└─────────────────────────────────────────────────────────────────────┘
                                     │ gsutil cp ← GCS bucket
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. EXPORT: merge LoRA → ONNX → INT8 quantize → ~5 MB file          │
└─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6a. Local: Python CLI, onnxruntime (CPU), <1 s per 256×256 image   │
│  6b. Pi 5:  Python CLI, onnxruntime ARM64, 2–5 s, ~200 MB RAM       │
└─────────────────────────────────────────────────────────────────────┘
```

**Targets:**
- Input/output resolution: 256×256 RGB→grayscale
- Deployed model size: ~5 MB (INT8 ONNX)
- Peak Pi RAM: ~200–300 MB
- Pi inference: 2–5 s/image
- End-to-end project cost: ~$40 (dominated by Gemini API)

## 4. Repo layout

```
Albumify/
├── README.md
├── pyproject.toml                  # one install set for laptop, one for Pi
├── .gitignore                      # data/covers, data/labels, spot_check
├── .env.example                    # GEMINI_API_KEY placeholder
│
├── data/
│   ├── albums.json                 # committed: curated top-500 list
│   ├── splits/                     # committed: train.txt / eval.txt (slugs only)
│   ├── covers/                     # gitignored: downloaded RGB covers
│   └── labels/                     # gitignored: Gemini-generated line drawings
│
├── albumify/                       # python package
│   ├── __init__.py
│   ├── dataset.py                  # CAA client + torch Dataset + paired augmentation
│   ├── gen_labels.py               # Gemini 2.5 Flash Image API client
│   ├── model.py                    # Informative-Drawings arch + LoRA-Conv adapters
│   ├── train.py                    # LoRA fine-tune loop
│   ├── eval.py                     # SSIM / LPIPS + visual grid
│   ├── export.py                   # merge LoRA → ONNX → INT8
│   └── infer.py                    # CLI: image_in → image_out (laptop & Pi)
│
├── deploy/
│   ├── local.md                    # laptop install (CPU onnxruntime)
│   └── pi.md                       # Pi 5 install (ARM64 onnxruntime)
│
├── infra/
│   ├── gcp_setup.md                # step-by-step VM provisioning + cost
│   ├── create_vm.sh                # gcloud command (T4 spot)
│   ├── vm_startup.sh               # CUDA + repo clone + deps install
│   └── delete_vm.sh                # cleanup script
│
├── docs/
│   └── superpowers/specs/2026-05-20-albumify-design.md  # this file
│
└── tests/
    ├── test_dataset.py
    └── test_infer.py               # smoke test using AbbeyRoad.jpg
```

**Module contracts:**

| Module | Purpose | Public interface | Dependencies |
|---|---|---|---|
| `dataset.py` | Fetch + load (cover, line) pairs | `class AlbumDataset(split: str)` → `(rgb_tensor, line_tensor)` | CAA HTTP API, PIL, torch |
| `gen_labels.py` | One-time label generation | `python -m albumify.gen_labels [--limit N] --in data/covers --out data/labels` | `google-genai`, `GEMINI_API_KEY` |
| `model.py` | Model + LoRA wrapper | `InformativeDrawings()`, `apply_lora(model, rank=8)`, `merge_lora(model)` | torch |
| `train.py` | Fine-tuning loop | `python -m albumify.train --epochs 50 --rank 8 --out runs/...` | dataset, model, torchvision (VGG) |
| `eval.py` | Quantitative + visual eval | `python -m albumify.eval --ckpt runs/.../best.pt --out eval/` | lpips, scikit-image |
| `export.py` | Deployment artifact | `python -m albumify.export --ckpt runs/.../best.pt --out artifacts/` | onnx, onnxruntime.quantization |
| `infer.py` | The deployed CLI | `albumify --model artifacts/albumify_int8.onnx --in cover.jpg --out line.png` | onnxruntime only |

**Isolation property:** the Pi installs only `onnxruntime`, `Pillow`, `numpy` (~30 MB total). Torch, CUDA, and training-only deps stay on the GCP VM.

## 5. Data pipeline

### 5.1 `data/albums.json`

Committed, hand-curated, ~500 entries seeded from Wikipedia's *"List of best-selling albums"* and RIAA. The originally-requested top 100 is the first 100 entries by rank.

```json
[
  {"rank": 1, "slug": "thriller-michael-jackson",
   "artist": "Michael Jackson", "title": "Thriller", "year": 1982},
  ...
]
```

`slug` is the only field used downstream. Files on disk are `{slug}.jpg` and `{slug}.png`.

### 5.2 Cover fetching (`fetch_covers.py` inside `dataset.py`)

Algorithm:
```
for entry in albums.json:
  if covers/{slug}.jpg exists: skip
  mbid = mb_search(entry.artist, entry.title)   # MusicBrainz, 1 req/sec, custom UA
  if mbid is None: log "no MBID" → skip
  img = caa_fetch(mbid, size=1200)              # 3 retries, exp backoff
  if img.width < 256: log "too small" → skip    # quality floor
  save covers/{slug}.jpg
```

Verified live during design:
- MusicBrainz search endpoint: `/ws/2/release-group?query=artist:"X" AND releasegroup:"Y" AND primarytype:album&fmt=json`
- CAA endpoint: `https://coverartarchive.org/release-group/{MBID}/front-1200`
- 6/6 sample best-sellers resolved end-to-end. Two real-world quirks observed and handled:
  - CAA returns native res if smaller than the requested size (Hotel California native = 300×298). Anything ≥ 256 is acceptable.
  - CAA occasionally returns transient 500s on the resize endpoint. Retry policy: 3 attempts with 1s/2s/4s backoff, falling back to `/front` (original).
- Required `User-Agent`: `Albumify/0.1 (scottbluman@gmail.com)`.

Expected runtime: ~10–15 min for 500 entries (rate-limited).

### 5.3 Label generation (`gen_labels.py`)

Uses **Gemini 3.1 Flash Image Preview**, Standard tier, 1024×1024 output. The newer 3.1 model is expected to match or improve on the Gemini 2.5 Flash Image ("Nano Banana") outputs the user's `Overview.md` references for the reference samples.

```
PROMPT = "Create a single line black-and-white line drawing of this image. " \
         "White background, black lines only, no shading or fills, " \
         "preserving the main subjects and composition of the album cover."

for cover in data/covers/*.jpg:
  if labels/{slug}.png exists: skip
  resp = genai.generate_content(
      model="gemini-3.1-flash-image-preview",
      contents=[PROMPT, PIL.Image.open(cover)],
      config={"image_config": {"image_size": "1K"}})
  save labels/{slug}.png
  sleep 1.0
```

- Output: 1024×1024 PNG (downsampled to 256 for training, so 1K is sufficient and saves ~33% vs 2K).
- Cost (Standard tier, 1K): ~$0.067/image → **~$33.50 for 500**. Verified against `ai.google.dev/gemini-api/docs/pricing` on 2026-05-20.
- Auth: `GEMINI_API_KEY` from `.env`.
- "Preview" model status caveat: this model can change or be deprecated with short notice. Pin the exact model name and document the fallback (drop to `gemini-2.5-flash-image` at $0.039/image, ~$19.50 total) in `gen_labels.py`.

**CHECKPOINT (mandatory):** First run with `--limit 10`, generating each cover with **both** `gemini-3.1-flash-image-preview` **and** `gemini-2.5-flash-image` (20 outputs total, ~$1 sunk cost). User reviews and chooses the model for the full 500-image run. If 3.1 isn't visibly better, we save ~$14 by sticking with 2.5. This caps the blast radius of any quality issue *and* validates the model upgrade.

### 5.4 Splits

After labels exist, a one-shot script writes:
- `data/splits/train.txt` — ~450 slugs
- `data/splits/eval.txt` — ~50 slugs, stratified by rank (every 10th)

The two existing user-provided samples (Abbey Road, Pink Floyd) are *excluded* from training and used as out-of-distribution sanity images in eval.

### 5.5 Augmentation (training time only)

| Transform | Cover (input) | Label (target) |
|---|---|---|
| Resize 1024, random crop → 256×256 | same params | same params |
| Horizontal flip (p=0.5) | same | same |
| Rotation ±5° | same | same |
| Color jitter (brightness/sat) | applied | never |
| Normalize to [-1, 1] | applied | applied |

Implemented as a single `paired_transform()` so geometric transforms can't drift between input and label.

### 5.6 What is committed vs ignored

| Committed | Gitignored |
|---|---|
| `data/albums.json` (metadata only) | `data/covers/` (copyrighted) |
| `data/splits/*.txt` (slug lists only) | `data/labels/` (derivative of copyrighted) |
| | `spot_check/` |

Repo stays under ~2 MB; all derivative copyrighted material lives only on local disk and GCS.

## 6. Model and LoRA design

### 6.1 Base architecture

Informative-Drawings generator `G: RGB → line drawing`:
```
input 256×256×3
  → ReflectionPad + Conv7×7 + InstanceNorm + ReLU       (64 ch)
  → 2× downsample (Conv4×4 stride 2)                    (128, 256 ch)
  → 9× residual blocks (2× Conv3×3 each)                (256 ch)
  → 2× upsample (ConvTranspose4×4 stride 2)             (128, 64 ch)
  → ReflectionPad + Conv7×7 + Tanh                      (1 ch)
output 256×256×1
```
Total: ~4.4 M params, ~17 MB FP32.

Starting checkpoint: the published `contour_style.pth` (closest match to the reference samples).

### 6.2 LoRA-Conv adapter

For each Conv2d layer in the generator:
```
y = (W₀ * x) + (B * (A * x))
   └─frozen─┘  └──trainable──┘
A: Conv2d(in_ch → r,   kernel=1×1)              # rank-r projection
B: Conv2d(r    → out_ch, kernel=kxk, same pad)   # restores receptive field
```

- **Attach points:** every Conv2d in encoder + residual blocks + decoder.
- **Skip:** InstanceNorm, biases, the Tanh output.
- **Default rank:** 8 (sweepable: {4, 8, 16}).
- **Default alpha:** 16 (scale factor `alpha/rank = 2`).
- **Trainable param count at rank 8:** ~75 K (~1.7% of base 4.4 M).
- **At export time:** `merge_lora()` folds `B·A` into `W₀`, producing an architecturally vanilla generator. Zero LoRA overhead at inference.

## 7. Training

### 7.1 Loss

```
L_total = 1.0 · L1(G(x), y)  +  0.1 · Perceptual_VGG(G(x), y)
```
- L1 (not L2): sharper for image-to-image translation.
- Perceptual: pretrained VGG16, features at `relu_2_2` and `relu_3_3`.
- **No adversarial loss / discriminator.** Justification: the base generator already produces good line drawings; we're only shifting style toward album-cover aesthetics. Discriminators are unstable with 450-sample fine-tunes. If quality is weak after the first run, a small PatchGAN can be added in a follow-up.

### 7.2 Hyperparameters

| | Value | Rationale |
|---|---|---|
| Image size | 256×256 | Matches Pi inference target |
| Batch size | 16 | Fits T4 16 GB with VGG loaded |
| Epochs | 50 | ~1400 steps; early stop on eval L1 |
| Optimizer | AdamW, lr=1e-4, wd=1e-2 | Standard for LoRA |
| Scheduler | Cosine annealing | Stable for short runs |
| LoRA rank | 8 (default) | Sweep {4, 8, 16} in a follow-up |
| LoRA alpha | 16 | `alpha/rank = 2` |
| Precision | bf16 mixed | Supported on T4/L4, ~1.5× faster |
| Seed | 42 | Determinism for comparable runs |

### 7.3 Run output (`runs/2026-05-20-rank8/`)

```
config.json              # full hyperparams + git SHA + dep versions
train_log.jsonl          # per-step loss
eval_log.jsonl           # per-epoch SSIM / LPIPS / L1
ckpt_epoch_{N}.pt        # LoRA-only weights (~300 KB), last 3 + best
best.pt                  # symlink to lowest-eval-L1 checkpoint
samples/epoch_{N}.png    # visualization grid on eval set
events.out.tfevents.*    # TensorBoard
```

- Eval every 5 epochs.
- 5-image smoke test runs at the very start of every job to catch broken plumbing before any real compute is spent.
- Checkpoints uploaded to GCS bucket after each eval to survive spot preemption.

### 7.4 Expected training cost

| GPU | Time/epoch | Total (50 ep) | $ spot |
|---|---|---|---|
| T4 | ~30 s | ~25 min | ~$0.09 |
| L4 | ~15 s | ~13 min | ~$0.05 |

Budget for 3 runs (rank sweep): **~$0.30 of GPU compute.**

## 8. Evaluation

### 8.1 Quantitative (on 50-cover held-out split)

| Metric | Target |
|---|---|
| L1 (mean abs diff) | < 0.10 on [0,1] |
| SSIM | > 0.75 |
| LPIPS (AlexNet backbone) | < 0.25 |

These metrics are "distance from Gemini's output" — we are measuring how well we mimic the teacher.

### 8.2 Visual

- 3-column grid `[input | Gemini label | our output]` for all 50 eval slugs, written to `eval_out/grid.png`.
- The two user-provided samples (Abbey Road, Pink Floyd) are run as out-of-distribution sanity tests and included in the grid.
- `eval_out/report.md` written with numeric table + thumbnails.

## 9. Export & INT8 quantization

```
python -m albumify.export --ckpt runs/.../best.pt --out artifacts/
```

Produces two artifacts:

| File | Format | Size | Purpose |
|---|---|---|---|
| `albumify_fp32.onnx` | ONNX | ~17 MB | Reference, laptop default |
| `albumify_int8.onnx` | ONNX INT8 | ~5 MB | Pi default |

Pipeline:
1. **Merge LoRA** (`merge_lora()`) into base weights.
2. **Export ONNX** with `torch.onnx.export(opset=17, dynamic_axes={"input": {0: "batch"}})`.
3. **INT8 quantize** via `onnxruntime.quantization.quantize_static`, MinMax calibration, QOperator format, calibration set = 100 images sampled from train (never eval).
4. **Numerical check:** assert max-abs diff between FP32 and INT8 outputs on the 50 eval images is below ~0.05 on [-1, 1] outputs. If it fails, fall back to `quantize_dynamic` (slower but more robust for unusual ops).

## 10. Deployment

### 10.1 CLI

```
albumify --model artifacts/albumify_int8.onnx --in cover.jpg --out line.png
         [--size 256]    # input size; default 256
         [--threads 4]   # CPU threads; default = onnxruntime's choice
```

`infer.py` is small (~50 lines): PIL load → resize+normalize → ORT run → denormalize → PIL save. **No torch at deploy time.**

### 10.2 Local (laptop)

```
pip install onnxruntime pillow numpy   # ~30 MB
```
Runs on CPU, ~250 ms per 256×256 image on a modern laptop. macOS/Linux/Windows via manylinux wheels.

### 10.3 Pi 5 (Raspberry Pi OS Bookworm 64-bit)

```
sudo apt install python3-pip python3-pil
pip install --break-system-packages onnxruntime numpy
scp laptop:artifacts/albumify_int8.onnx ~/
albumify --in cover.jpg --out line.png
```

**RAM budget (estimate, to be measured during deploy phase):**

| Component | Estimated RAM |
|---|---|
| INT8 model weights | ~5 MB |
| ORT session + arena | ~30–50 MB |
| Peak activations (256×256) | ~80–120 MB |
| Python + libs | ~40 MB |
| **Total** | **~155–215 MB** |

Headroom budget on 1 GB Pi: ~700 MB. **Fallbacks if measurement exceeds budget**, in order:
1. Drop input size 256 → 192 (cuts activations ~45%).
2. Set ORT `arena_extend_strategy=kSameAsRequested` (disable memory pooling).
3. Per-tensor INT8 instead of per-channel.
4. Rebuild ORT with `--minimal_build`.

**Latency target:** 2–5 s/image on Pi 5 quad-core ARM Cortex-A76 at 256×256 INT8. If > 5 s, drop to 192×192.

The 1 GB Pi is the primary target. The user's 16 GB Pi 5 is the fallback for quality A/B if the 1 GB budget forces compromises.

## 11. GCP VM infra

### 11.1 Machine

| Item | Value |
|---|---|
| Machine type | `n1-standard-4` (4 vCPU, 15 GB RAM) |
| GPU | 1× `nvidia-tesla-t4` (16 GB) |
| Provisioning | **SPOT** with `instance-termination-action=DELETE` |
| Image | `pytorch-2-3-cu121-debian-12` (Deep Learning VM family) |
| Boot disk | 100 GB `pd-ssd` |
| Region/zone | `us-central1-a` (cheapest+best T4 availability; verify at create time) |

L4 (`g2-standard-4`) is documented as an alternative for faster iteration; T4 is the default.

### 11.2 Prerequisites (one-time)

1. Install gcloud CLI, run `gcloud init`.
2. Create or select a GCP project; enable billing and the Compute Engine API.
3. **Request GPU quota** — new projects have a default of **0**. Console → IAM & Admin → Quotas → `nvidia_t4_gpus` in chosen region → request 1. Usually granted in <30 min.
4. Pick a region; default `us-central1-a`.

### 11.3 Provisioning — `infra/create_vm.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:?set to your GCP project ID}"
ZONE="us-central1-a"
NAME="albumify-train"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-3-cu121-debian-12 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --metadata-from-file=startup-script=infra/vm_startup.sh \
  --scopes=cloud-platform
```

### 11.4 Startup — `infra/vm_startup.sh`

```bash
#!/usr/bin/env bash
set -euxo pipefail
cd /opt
git clone https://github.com/sblu/Albumify.git
cd Albumify
python3 -m pip install --upgrade pip
pip install -e ".[train]"
nvidia-smi > /tmp/gpu_check.txt
```

### 11.5 Workflow

```
LOCAL                                            VM
─────                                            ──
1. python -m albumify.gen_labels --limit 10
   USER REVIEWS 10 SAMPLES; only on approval:
2. python -m albumify.gen_labels   (full 500)
3. gsutil cp -r data/{covers,labels} \
       gs://YOUR-BUCKET/albumify-data/  ──────►
                                                 ──── ./infra/create_vm.sh
                                                 ──── gcloud compute ssh albumify-train
                                                 ──── gsutil cp -r gs://.../albumify-data ./data
                                                 ──── python -m albumify.train ...
                                                 ──── python -m albumify.eval  ...
                                                 ──── python -m albumify.export ...
                                                 ──── gsutil cp artifacts/*.onnx \
                                                 ────       gs://YOUR-BUCKET/artifacts/
4. gsutil cp gs://.../artifacts/*.onnx \
       ./artifacts/         ◄───────
5. ./infra/delete_vm.sh                   # MANDATORY: prevent zombie instance
```

Data transfer via GCS bucket (not `gcloud compute scp`) is faster for the ~650 MB payload. Bucket storage cost: <$0.02/month for the few GB we use.

### 11.6 Cost estimate

| Line item | Calculation | Cost |
|---|---|---|
| Gemini 3.1 Flash Image Preview (500 labels, 1K standard) | 500 × ~$0.067 | ~$33.50 |
| 10-sample A/B preview (2.5 + 3.1, ~20 calls) | 10 × $0.039 + 10 × $0.067 | ~$1.06 |
| GCP T4 spot (3 runs, ~30 min each) | 1.5 hr × $0.21 | ~$0.32 |
| 100 GB pd-ssd (~4 hours of VM life) | 100 × $0.17/mo × 4/720 | ~$0.09 |
| GCS storage (1 GB, 1 month) | 1 × $0.02/mo | ~$0.02 |
| Egress (~5 MB artifact pull) | within free tier | $0 |
| **Subtotal (happy path)** | | **~$34.99** |
| Buffer for re-runs / mistakes | +15% | +$5 |
| **Realistic total** | | **~$40** |

If the 10-sample checkpoint shows 2.5 is sufficient, total drops to ~$25 (the 2.5 dataset cost is $19.50 vs $33.50).

**Forgotten-VM risk:**
- T4 spot left 24 hrs: ~$5/day
- T4 on-demand left 24 hrs: ~$13/day

Mitigations: SPOT provisioning auto-deletes on preemption; `infra/delete_vm.sh` shipped as a one-liner; `infra/gcp_setup.md` ends with a "did you delete the VM?" reminder; recommend a $50 GCP budget alert.

## 12. Risks and open verification items

Items to confirm at implementation time, not blockers for the design:

1. **Gemini 3.1 Flash Image Preview availability and pricing.** $0.067/image at 1K standard verified 2026-05-20 against `ai.google.dev/gemini-api/docs/pricing`. The "Preview" designation means the model can change or be deprecated; `gen_labels.py` ships with a fallback path to `gemini-2.5-flash-image` ($0.039/image). The 10-sample A/B in §5.3 also lets us decide whether 3.1's quality justifies the price bump.
2. **`pytorch-2-3-cu121-debian-12` image tag.** The Deep Learning VM family name is stable; the specific image version moves. Verify at create time, swap in the latest in the family if needed.
3. **CAA coverage at 500 entries.** Spot-check on 6 best-sellers was 6/6. Expect ≥95% hit rate at 500, but the `fetch_report.txt` output will show actual misses, and we may need to substitute alternate releases (e.g., remastered editions) for the few that fail.
4. **Pi RAM measurement.** The ~200 MB estimate is computed, not measured. The first Pi deployment step is to measure peak RSS and verify against budget. Fallbacks ordered in §10.3 if measurement is over.
5. **GPU quota lead time.** New GCP projects can have a longer-than-30-min wait for the first quota grant. Request quota before the rest of the workflow.

## 13. Decisions summary

| # | Decision |
|---|---|
| Approach | Fine-tune Informative-Drawings with LoRA-Conv adapters |
| Target hardware | Pi 5 / 1 GB primary, Pi 5 / 16 GB fallback for quality A/B |
| Output style | Detailed line drawing matching reference samples |
| Dataset | 500 covers (top 100 best-sellers as ranked subset) |
| Cover source | MusicBrainz → Cover Art Archive `front-1200` |
| Label source | Gemini 3.1 Flash Image Preview at 1K standard, with 10-sample A/B vs 2.5 (fallback to 2.5 if 3.1 isn't visibly better) |
| Model size | ~17 MB FP32 → ~5 MB INT8 ONNX |
| Inference resolution | 256×256 |
| Latency target | 2–5 s/image on Pi 5 |
| Training | 50 epochs, LoRA rank 8, AdamW lr=1e-4, L1 + VGG perceptual, no discriminator |
| Compute | GCP T4 spot in `us-central1-a`, Deep Learning VM image |
| Cost (realistic total) | ~$40 (drops to ~$25 if checkpoint chooses 2.5) |
| Repo | https://github.com/sblu/Albumify.git |
