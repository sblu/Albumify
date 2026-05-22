# Albumify

Turn an album cover into a single-line black-and-white drawing with a tiny
LoRA-fine-tuned model that fits on a Raspberry Pi 5 in ~5 MB.

| stage | what it does | input | output |
| --- | --- | --- | --- |
| 1. cover fetch | MusicBrainz → Cover Art Archive | `data/albums.json` | `data/covers/<slug>.jpg` |
| 2. label gen   | Gemini 3.1 Flash Image Preview | `data/covers/` | `data/labels/<slug>.png` |
| 3. review      | thumbs-up/down web UI; async regen for thumbs-down | `data/labels/` | curated `data/labels/` |
| 4. train       | LoRA-Conv fine-tune of Informative-Drawings on GCP T4 spot | `data/{covers,labels,splits}` | `runs/.../best.pt` |
| 5. export      | merge LoRA → ONNX FP32 → INT8 quantize | `best.pt` | `artifacts/model.int8.onnx` |
| 6. infer       | onnxruntime CLI, runs on laptop or Pi 5 | `cover.jpg` | `line.png` |

## Try the pretrained model (rank-8 LoRA preview)

A rank-8 LoRA model trained on ~470 album covers is attached to the
[`v0.1.0-rank8-preview`](https://github.com/sblu/Albumify/releases/tag/v0.1.0-rank8-preview)
release. It produces faint *ghost-line* drawings — recognizable, but not yet
shippable. A larger from-scratch model is in training; this preview lets you
exercise the inference pipeline without retraining.

```bash
git clone https://github.com/sblu/Albumify.git
cd Albumify
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install -e .                    # pillow + numpy + onnxruntime + CLI

mkdir -p artifacts
curl -L -o artifacts/model.int8.onnx \
  https://github.com/sblu/Albumify/releases/download/v0.1.0-rank8-preview/model.int8.onnx

albumify --model artifacts/model.int8.onnx \
  --in path/to/cover.jpg \
  --out line.png \
  --size 256 \
  --threshold 0.95            # lifts the faint ghost lines to clean black
```

Raspberry Pi 5 deployment: see [`deploy/pi.md`](deploy/pi.md).

## Quick map

- `albumify/musicbrainz.py`, `albumify/caa.py`, `albumify/fetch_covers.py` — fetch pipeline
- `albumify/gen_labels.py` — Gemini label generation
- `albumify/review_app.py` — Flask review UI (table view, async regen, sync magnifier)
- `albumify/split.py` — deterministic train/val split
- `albumify/transforms.py` — paired (cover, label) augmentation
- `albumify/dataset.py` — `AlbumDataset(torch.utils.data.Dataset)`
- `albumify/model.py` — Informative-Drawings generator (vendored)
- `albumify/lora.py` — LoRA-Conv adapters + merge
- `albumify/loss.py` — L1 + VGG perceptual
- `albumify/train.py` — training loop
- `albumify/eval.py` — SSIM, edge F1, LPIPS + visual grid
- `albumify/export.py` — ONNX export + INT8 quantization
- `albumify/infer.py` — `albumify` CLI entry point
- `deploy/pi.md` — Raspberry Pi 5 walkthrough
- `infra/gcp_setup.md`, `infra/create_vm.sh`, `infra/setup_vm.sh` — GCP training
- `docs/superpowers/specs/2026-05-20-albumify-design.md` — design doc
- `docs/superpowers/plans/2026-05-20-albumify.md` — implementation plan
- `data/holdout/README.md` — held-out test set (10 covers from Afterglow)

## Setup

```bash
git clone https://github.com/sblu/Albumify.git
cd Albumify
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install -e ".[data]"          # fetch + label gen + review app
# pip install -e ".[train]"       # also installs torch + lpips + tensorboard (use on the GCP VM)
cp .env.example .env              # then edit and set GEMINI_API_KEY
```

## End-to-end

```bash
# 1. Pull covers (MusicBrainz + Cover Art Archive)
python -m albumify.fetch_covers

# 2. Generate Gemini line-drawing labels
python -m albumify.gen_labels

# 3. Review + iteratively refine (open http://127.0.0.1:5005/)
python -m albumify.review_app

# 4. Write train/val split
python -m albumify.split

# 5. Train on a GCP T4 spot VM — see infra/gcp_setup.md
#    Pull best.pt back to your laptop.

# 6. Export ONNX + INT8
python -m albumify.export --ckpt-path runs/lora-rank8/best.pt --out-dir artifacts --int8

# 7. Run inference on your laptop or Pi 5
albumify --model artifacts/model.int8.onnx --in cover.jpg --out line.png --size 256
```

## Cost & timing

- Cover fetch: free (MusicBrainz, CAA)
- Gemini labels (500 covers @ 3.1 Flash Image Preview): ~$31
- GCP T4 spot training (30 epochs): ~$0.50–$1.00 per run
- Inference: ~free, runs in milliseconds on Pi 5

## Tests

```bash
pytest -q
```

Tests that need `torch` are marked with `pytest.importorskip("torch")` and
skip cleanly on a laptop install. They run on the GCP VM.

## License

This repo's source code is MIT. The Informative-Drawings architecture and
pretrained weights are © Chan et al. and used under their license; see
https://github.com/carolineec/informative-drawings. Gemini-generated label
images derive from copyrighted album covers and are kept out of the public
repo (gitignored).
