#!/usr/bin/env bash
# Run ON the GCP VM (Deep Learning VM image) to install Albumify deps.
# Idempotent — safe to re-run after a preemption replacement.

set -euo pipefail

cd "$(dirname "$0")/.."  # repo root assumed two levels up from infra/

# The current pytorch-2-9-cu129-ubuntu-2204-nvidia-580 DLVM image is a 'stage'
# variant: NVIDIA driver + CUDA, but no Python ML stack and no python3-venv.
# Refresh apt and install just enough to create a venv. Idempotent.
echo ">>> Refreshing apt + installing python3-venv + pip + unzip"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-venv python3-pip unzip

# Build the venv from scratch if it doesn't already have an activate script
# (a half-built .venv from a failed prior run will trip up `source` otherwise).
if [ ! -f ".venv/bin/activate" ]; then
  rm -rf .venv
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip

# Install with the [train] extra (pulls torch + torchvision + lpips + tensorboard).
pip install -e ".[train]"

# Optional: install lpips here too if eval needs it.
pip install lpips || true

# Plan F3: download the upstream feats2depth checkpoint (G_Geom) from
# Google Drive on first setup. Idempotent — skips if already extracted.
# Source: README of carolineec/informative-drawings
#   "Place the pre-trained features to depth network in ./checkpoints/feats2Geom"
F2D_DIR="artifacts/feats2Geom"
F2D_ZIP="artifacts/feats2depth.zip"
F2D_DRIVE_ID="1Ov1BNue74Yu-57X2rpdjqZy0o-fnFoly"
if [ -d "$F2D_DIR" ] && [ -n "$(ls -A "$F2D_DIR" 2>/dev/null)" ]; then
  echo ">>> feats2depth ckpt already present at $F2D_DIR (skip download)"
else
  echo ">>> Downloading feats2depth.zip from upstream Google Drive"
  mkdir -p artifacts
  python -m gdown "$F2D_DRIVE_ID" -O "$F2D_ZIP" || {
    echo "[WARN] gdown failed — F3 will fall back to skip if --geom-weight 0."
    echo "[WARN] Manual fix: download https://drive.google.com/file/d/${F2D_DRIVE_ID}/view"
    echo "[WARN]              unzip to ${F2D_DIR}/"
  }
  if [ -f "$F2D_ZIP" ]; then
    mkdir -p "$F2D_DIR"
    (cd artifacts && unzip -o "$(basename "$F2D_ZIP")" -d feats2Geom)
    rm -f "$F2D_ZIP"
    echo ">>> Extracted feats2depth into $F2D_DIR:"
    ls "$F2D_DIR"
  fi
fi

echo ">>> Verifying torch + CUDA"
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo
echo ">>> Ready. Plan F3 run sequence:"
echo "    . .venv/bin/activate"
echo "    # 1. Precompute DPT-Large depth (one-time, ~3 min on L4)"
echo "    python -m albumify.precompute_depth \\"
echo "      --covers-dir data/covers --out-dir data/depth --resize 256"
echo "    # 2. Train with all three losses (L1 + CLIP + geom)"
echo "    python -m albumify.train \\"
echo "      --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \\"
echo "      --pretrained-ckpt artifacts/informative_drawings.pth \\"
echo "      --out-dir runs/plan-f3 --no-lora --n-residual-blocks 3 --ngf 64 \\"
echo "      --optimizer adam --lr 2e-4 --weight-decay 0 --epochs 30 --batch-size 8 \\"
echo "      --clip-weight 10 --geom-weight 10 \\"
echo "      --depth-cache-dir data/depth \\"
echo "      --feats2depth-ckpt artifacts/feats2Geom/feats2depth.pth"
