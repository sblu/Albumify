#!/usr/bin/env bash
# Run ON the GCP VM (Deep Learning VM image) to install Albumify deps.
# Idempotent — safe to re-run after a preemption replacement.

set -euo pipefail

cd "$(dirname "$0")/.."  # repo root assumed two levels up from infra/

# The current pytorch-2-9-cu129-ubuntu-2204-nvidia-580 DLVM image is a 'stage'
# variant: NVIDIA driver + CUDA, but no Python ML stack and no python3-venv.
# Refresh apt and install just enough to create a venv. Idempotent.
echo ">>> Refreshing apt + installing python3-venv + pip"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-venv python3-pip

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

echo ">>> Verifying torch + CUDA"
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo
echo ">>> Ready. Typical run:"
echo "    . .venv/bin/activate"
echo "    python -m albumify.train \\"
echo "      --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \\"
echo "      --pretrained-ckpt artifacts/informative_drawings.pth \\"
echo "      --out-dir runs/lora-rank8 --epochs 30 --batch-size 8 --lr 1e-3"
