# Resume notes — pick up training tomorrow

Last touched: 2026-05-21 early AM.
Working tree = `origin/main` at `af390ae` (commit listing below).

## TL;DR

End-to-end scaffolding is done and tested. The first trained model is
visible-but-not-shippable. We diagnosed why (sigmoid saturation +
class imbalance), shipped fixes, started the bigger-model retrain,
hit GCP T4 capacity exhaustion across us-central1 zones a/b/c/f.
Stopped to wait for capacity.

## What works right now

- 471 cover-label pairs cleaned, split 424/47 train/val. ✓
- Review web app produced final approved labels. ✓
- LoRA rank-8 model trained successfully on GCP T4 spot (~25 min, ~$0.25). ✓
- Exported INT8 ONNX (`artifacts/model.int8.onnx`, 12.6 MB). ✓
- Pi 5 inference pipeline works at 256/512/1024 sizes. ✓
- `--threshold` post-process in `albumify` CLI lets you pull faint
  ghost predictions into binary lines. ✓
- 10-image holdout set at `data/holdout/`. ✓

## What we've trained (and what they produced)

| run | params | result | location |
|---|---|---|---|
| `runs/lora-rank8` (epoch 29) | LoRA r=8 on contour-style, edge_weight=0 | Faint ghost line drawings. Recognizable at `--threshold 0.95`. Not shippable. | `artifacts/model.int8.onnx` on laptop + Pi |
| `runs/lora-rank16-edge` (preempted at epoch 13) | LoRA r=16, edge_weight=15, lr=3e-3 | Was finally improving (val_l1 0.708 → 0.685) when GCP preempted the VM. Lost. | gone |

## Root cause we identified

The model output saturates near sigmoid(+∞)=1 ("predict all white").
Plain L1 loss has a flat gradient at saturation, so the model can't
escape "all white" no matter how high we crank LR or edge_weight.

The first signs were:
- Train converges quickly, val barely moves.
- val_l1 sits at ~0.70 with edge_weight=15 — which is exactly the value
  you'd get if the model predicts pred=1 everywhere.

## Plan B (already coded, ready to run)

Bigger model from scratch (no LoRA), with edge-weighted L1, longer epochs.

Code changes are all in `main`:
- `--ngf 96` flag on train/eval/export
- `--no-lora` flag on train/eval/export
- `SPOT=0` env var on `infra/create_vm.sh` for on-demand VM

Training command for next time, on the VM:

```bash
mkdir -p runs/ngf96-scratch
python -m albumify.train \
  --splits-dir data/splits \
  --covers-dir data/covers \
  --labels-dir data/labels \
  --no-lora --ngf 96 \
  --out-dir runs/ngf96-scratch \
  --epochs 60 --batch-size 8 --lr 2e-4 \
  --perceptual-weight 0.1 \
  --edge-weight 10.0 \
  2>&1 | tee runs/ngf96-scratch/train.log
```

Notes for that run:
- ~25.5M params at ngf=96 vs ~11.7M at ngf=64.
- ~75 min on T4 (vs ~25 min for the LoRA runs).
- INT8 export will be ~25-30 MB (vs 12.6 MB). Pi 5 1 GB RAM handles it fine
  at 256 and 512; 1024 may OOM.
- No pretrained checkpoint needed (training from scratch). Skipping the
  `informative_drawings.pth` upload saves a minute.
- On-demand cost: ~$0.55 if it runs without preemption.

## Plan C (if Plan B's bigger-model run still looks like ghost lines)

Architectural fix: remove sigmoid from Generator, train with
`F.binary_cross_entropy_with_logits` (no saturation in the gradient).
Then add sigmoid back at export time. Not yet coded. ~30 min of work
before running.

## How to pick up tomorrow

### 0. (skip if obvious) Verify state

```
cd /home/scott/Desktop/AlbumArtModelClaude
git status                     # should be clean
git pull                       # nothing to pull; double-check
ls artifacts/                  # informative_drawings.pth + model.int8.onnx
ls data/holdout/ | head        # 10 PNGs + README.md
```

### 1. Spin up a VM

T4 capacity in us-central1 was tight last night. Sequence to try
(stop at first success):

```bash
# Try on-demand zones in order
PROJECT=albumartifier SPOT=0 ZONE=us-central1-a ./infra/create_vm.sh
PROJECT=albumartifier SPOT=0 ZONE=us-central1-b ./infra/create_vm.sh
PROJECT=albumartifier SPOT=0 ZONE=us-central1-c ./infra/create_vm.sh
PROJECT=albumartifier SPOT=0 ZONE=us-central1-f ./infra/create_vm.sh
# Fall back to spot if all on-demand fails
PROJECT=albumartifier ZONE=us-central1-a ./infra/create_vm.sh
```

If a create succeeds in a zone other than `us-central1-a`, remember to
pass that `--zone` to every subsequent `gcloud compute ssh|scp` call.

### 2. SSH in + install deps

```
gcloud compute ssh albumify-train --zone <ZONE> --project albumartifier

# On the VM:
sudo apt update
sudo apt install -y python3-venv python3-pip
git clone https://github.com/sblu/Albumify.git
cd Albumify
./infra/setup_vm.sh
```

### 3. Upload data + pretrained ckpt (from laptop)

```
cd /home/scott/Desktop/AlbumArtModelClaude
gcloud compute ssh albumify-train --zone <ZONE> --project albumartifier \
  --command 'mkdir -p ~/Albumify/artifacts'
gcloud compute scp --zone <ZONE> --project albumartifier --recurse \
  data/covers data/labels data/splits data/albums.json \
  albumify-train:~/Albumify/data/
gcloud compute scp --zone <ZONE> --project albumartifier \
  artifacts/informative_drawings.pth \
  albumify-train:~/Albumify/artifacts/informative_drawings.pth
```

### 4. Start the Plan-B training

```
cd ~/Albumify
. .venv/bin/activate
mkdir -p runs/ngf96-scratch
tmux new -s train
# ... run the python command above ...
# Ctrl-b d to detach
```

### 5. Monitor

Set up the SSH tunnel from laptop:
```
gcloud compute ssh albumify-train --zone <ZONE> --project albumartifier -- \
  -L 6006:localhost:6006 -N
```
Then on a 2nd VM session, start `tensorboard --logdir runs/ngf96-scratch/tb --port 6006 --bind_all`.
Open http://localhost:6006/ in your browser.

### 6. After training (~75 min):

```
# Eval (specify --ngf 96 --no-lora to match training)
python -m albumify.eval \
  --splits-dir data/splits --covers-dir data/covers --labels-dir data/labels \
  --ckpt-path runs/ngf96-scratch/best.pt \
  --no-lora --ngf 96 \
  --out-dir runs/ngf96-scratch/eval --grid-n 32

# Export
python -m albumify.export \
  --ckpt-path runs/ngf96-scratch/best.pt \
  --no-lora --ngf 96 \
  --out-dir artifacts --int8

# Pull val_grid + ONNX to laptop, push ONNX to Pi
# (commands identical to the rank-8 run; see the chat history if needed)
```

### 7. DELETE THE VM

```
gcloud compute instances delete albumify-train --zone <ZONE> --project albumartifier
```

## Key flags / numbers to remember

- Project: `albumartifier`. Active config: `albumify`.
- Pi IP on LAN: `192.168.86.84` (user `scott`).
- Best threshold for the current rank-8 model on the Pi: `0.95`.
- **Pi 5 (1 GB) inference budget:** stick to `--size 512` max. At 1024,
  intermediate feature maps total ~600 MB working set → swap → 141 sec
  per image (measured). 256 is ~300 ms, 512 is ~1.5 s. For 1024
  you'd want a 4 GB Pi.
- Edge fraction in our labels: ~5%. With edge_weight=N, "predict white
  everywhere" gives val_l1 ≈ 0.05*N — sanity check when reading numbers.
- VGG16 perceptual weights (~530 MB) download once per fresh VM; expect
  a 30s pause early in the first epoch of any run with --perceptual-weight > 0.

## Recent commits (for context if returning here)

```
af390ae feat(infra): SPOT=0 env var picks on-demand VM (no preemption risk)
ebf815d feat(train/eval/export): --ngf and --no-lora CLI flags for bigger models
6c08297 chore(deps): add onnxscript to [train] (torch>=2.5 ONNX exporter imports it)
7edee00 feat(loss+infer): edge-weighted L1 + --threshold post-process for line drawings
5af841a fix(eval): same .to(device) ordering as train.py to avoid cpu/cuda mix
1aa410a feat(eval+export+infer): metrics, ONNX export, INT8, and inference CLI
1879313 feat(train): Informative-Drawings + LoRA-Conv + training loop
628713f feat(data): train/val split + paired transforms + AlbumDataset
```
