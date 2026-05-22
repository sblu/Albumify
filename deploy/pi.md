# Deploy to Raspberry Pi 5 (1 GB)

Target: load `model.int8.onnx` (~5 MB on disk) and run `albumify --in cover.jpg
--out line.png` at 256×256 in well under a second on a stock Pi 5 with 1 GB RAM.

## 0. Hardware + OS

- Raspberry Pi 5 (1 GB or 4 GB), 64-bit aarch64
- Raspberry Pi OS Bookworm (64-bit) or Ubuntu 24.04 aarch64
- An SD card with `sudo apt update && sudo apt upgrade -y` recently run

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libopenblas0 libjpeg62-turbo libpng16-16
```

## 2. Project + virtualenv

```bash
git clone https://github.com/sblu/Albumify.git
cd Albumify
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .                  # installs pillow + numpy + onnxruntime + CLI entry point
```

`pip install -e .` registers the `albumify` console script via the
`[project.scripts]` entry in `pyproject.toml`.

## 3. Drop the trained model on the device

Two published models are available — pick by size / quality. Both expect
`--threshold 0.95` at inference.

| Release | Size | Notes |
|:--|:-:|:--|
| `v0.2.0-ngf96` (recommended) | 27 MB | Clearer, bolder lines |
| `v0.1.0-rank8-preview` | 12.6 MB | Faint "ghost" lines, fastest |

Pull whichever (or both) directly on the Pi:

```bash
mkdir -p ~/Albumify/artifacts

# v0.2.0-ngf96 (recommended)
curl -L -o ~/Albumify/artifacts/model.int8.ngf96.onnx \
  https://github.com/sblu/Albumify/releases/download/v0.2.0-ngf96/model.int8.onnx

# v0.1.0-rank8-preview
curl -L -o ~/Albumify/artifacts/model.int8.rank8.onnx \
  https://github.com/sblu/Albumify/releases/download/v0.1.0-rank8-preview/model.int8.onnx
```

Or scp a locally-trained model from the laptop:

```bash
scp artifacts/model.int8.onnx pi@pi5.local:/home/pi/Albumify/artifacts/
```

## 4. Run

```bash
# ngf-96 (recommended)
albumify --model artifacts/model.int8.ngf96.onnx \
  --in some-cover.jpg \
  --out some-line.png \
  --size 256 \
  --threshold 0.95

# rank-8 (smaller, fainter)
albumify --model artifacts/model.int8.rank8.onnx \
  --in some-cover.jpg \
  --out some-line.png \
  --size 256 \
  --threshold 0.95
```

Measured Pi 5 wall times at 256×256 (INT8, CPU): rank-8 ≈ 2.65 s, ngf-96 ≈
TBD (see deployment results in the release notes).

## 5. Tuning

- **Threads**: `--threads 4` pins onnxruntime to all 4 Pi 5 cores. Leave 0
  (default) to let ORT pick.
- **Size**: drop to `--size 192` for ~2× speedup; `--size 320` if you want
  more detail and have the headroom.
- **Memory budget**: at 1 GB RAM the largest pressure is ORT's model + the
  decoded input PNG. Stay at 256 and avoid concurrent CPU-heavy jobs.

## 6. Sanity check

```bash
albumify --model artifacts/model.int8.onnx \
  --in AbbeyRoad.jpg \
  --out abbey_line.png
```

The output should be a recognisable black-on-white line drawing of the cover.
If it's solid black or solid white, you probably loaded the wrong ONNX or
mis-pasted the model. The FP32 model (`model.fp32.onnx`) works too but is
~4× larger and ~3× slower on aarch64 CPU.
