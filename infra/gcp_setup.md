# GCP fine-tune walkthrough

End-to-end: create a T4 spot VM in `us-central1`, copy the dataset onto it,
run the LoRA fine-tune, copy the trained checkpoint + ONNX artifacts back
to your laptop, then **delete the VM**.

Total billable time per run: ~30–40 min (T4 spot ≈ $0.11/hr GPU + ~$0.06/hr
n1-standard-4 + disk). Budget ~$0.50–$1.00 per fine-tune.

## 0. Prereqs

- GCP account with billing enabled.
- The following quotas (one row each) at ≥ 1 in `us-central1`:
  - `GPUS_ALL_REGIONS` (global)
  - `NVIDIA_T4_GPUS` (us-central1)
  - `PREEMPTIBLE_NVIDIA_T4_GPUS` (us-central1)
- `gcloud` CLI authenticated: `gcloud auth login && gcloud config set project YOUR_PROJECT_ID`
- This repo cloned locally with a populated `data/covers/`, `data/labels/`,
  `data/splits/` (the cover JPGs and label PNGs are gitignored; you need to
  carry them up to the VM yourself in step 3).

## 1. Spin up the VM

```bash
export PROJECT=YOUR_PROJECT_ID
./infra/create_vm.sh
# wait ~30s, then:
gcloud compute ssh albumify-train --zone us-central1-a --project "$PROJECT"
```

Spot instances can be reclaimed at any time, but the `--instance-termination-action=DELETE`
flag ensures you never pay for an idle stopped VM if it does get preempted.
If preempted mid-train, just re-run `create_vm.sh` and re-upload the data.

## 2. On the VM: clone repo + install

```bash
git clone https://github.com/sblu/Albumify.git
cd Albumify
./infra/setup_vm.sh
```

This creates a `.venv`, installs `albumify[train]` (torch + torchvision +
lpips + tensorboard), and prints a torch/CUDA sanity check.

## 3. Upload the dataset

From your laptop:

```bash
PROJECT=YOUR_PROJECT_ID
gcloud compute scp --zone us-central1-a --project "$PROJECT" --recurse \
  data/covers data/labels data/splits data/albums.json \
  albumify-train:~/Albumify/data/
```

Don't upload `data/holdout/` — that's the held-out test set; we evaluate on
the val split (a slice of `data/splits/val.txt`) during training and only
touch the holdout after final model selection.

You also need the Informative-Drawings pretrained checkpoint. Download the
official `model.pth` from https://github.com/carolineec/informative-drawings
(README links the Google Drive) and:

```bash
gcloud compute scp --zone us-central1-a --project "$PROJECT" \
  ~/Downloads/informative_drawings.pth \
  albumify-train:~/Albumify/artifacts/informative_drawings.pth
```

## 4. Train

On the VM:

```bash
. .venv/bin/activate
mkdir -p artifacts runs
python -m albumify.train \
  --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
  --pretrained-ckpt artifacts/informative_drawings.pth \
  --out-dir runs/lora-rank8 \
  --epochs 30 --batch-size 8 --lr 1e-3 \
  --lora-rank 8 --lora-alpha 8.0 \
  --perceptual-weight 0.1
```

Expected wall-clock on T4: roughly 30–60s per epoch at batch 8 / 256 px, so
30 epochs in ~25–30 minutes.

## 5. Evaluate

```bash
python -m albumify.eval \
  --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
  --ckpt-path runs/lora-rank8/best.pt \
  --out-dir runs/lora-rank8/eval \
  --grid-n 32
```

Inspect `runs/lora-rank8/eval/val_grid.png` and `summary.json`. If SSIM is
above ~0.85 and edge-F1 above ~0.55, ship it. If not, iterate on
`--perceptual-weight`, `--lora-rank`, or `--lr`.

## 6. Export ONNX + INT8

```bash
python -m albumify.export \
  --ckpt-path runs/lora-rank8/best.pt \
  --out-dir artifacts \
  --int8
```

Produces `artifacts/model.fp32.onnx` and `artifacts/model.int8.onnx`. The
INT8 model should be ~5 MB.

## 7. Pull artifacts back to your laptop

```bash
gcloud compute scp --zone us-central1-a --project "$PROJECT" --recurse \
  albumify-train:~/Albumify/runs/lora-rank8 \
  albumify-train:~/Albumify/artifacts \
  ./
```

## 8. DELETE the VM

```bash
gcloud compute instances delete albumify-train --zone us-central1-a --project "$PROJECT"
```

This is the most important step — a forgotten T4 VM is the most expensive
thing in this project by a wide margin.

## 9. Test against the holdout set on your laptop

```bash
for f in data/holdout/*.png; do
  out="data/holdout_predictions/$(basename "$f")"
  albumify --model artifacts/model.int8.onnx --in "$f" --out "$out" --size 256
done
```

Eye-check the 10 results. These 10 PNGs (Abbey Road, Dark Side, etc.) were
explicitly held out — they did not appear in training — so this is your
true generalization read.
