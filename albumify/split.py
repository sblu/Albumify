"""Deterministic train/val split over the slugs present in data/labels/.

The split is determined by (1) sorting slugs lexicographically so the input
ordering doesn't matter and (2) shuffling with a fixed seed. We only emit
slugs whose paired cover JPG also exists.

Splits are written as one slug per line to data/splits/{train,val}.txt.
The dataset class resolves slug -> (cover_path, label_path) at load time.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable


def discover_pair_slugs(covers_dir: Path, labels_dir: Path) -> list[str]:
    """Return slugs that have both a <slug>.jpg cover and a <slug>.png label."""
    covers = {p.stem for p in Path(covers_dir).glob("*.jpg")}
    labels = {p.stem for p in Path(labels_dir).glob("*.png")}
    return sorted(covers & labels)


def split_slugs(slugs: Iterable[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic train/val split.

    Sorts first so the input collection order does not affect output.
    """
    ordered = sorted(set(slugs))
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac)) if shuffled else 0
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def write_splits(out_dir: Path, train: list[str], val: list[str]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train) + ("\n" if train else ""))
    (out_dir / "val.txt").write_text("\n".join(val) + ("\n" if val else ""))


def read_split(path: Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Write deterministic train/val splits.")
    p.add_argument("--covers", default="data/covers", type=Path)
    p.add_argument("--labels", default="data/labels", type=Path)
    p.add_argument("--out",    default="data/splits", type=Path)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed",   type=int, default=0)
    args = p.parse_args()

    slugs = discover_pair_slugs(args.covers, args.labels)
    train, val = split_slugs(slugs, args.val_frac, args.seed)
    write_splits(args.out, train, val)
    print(f"Discovered {len(slugs)} paired slugs.")
    print(f"Wrote {len(train)} train + {len(val)} val to {args.out}/")


if __name__ == "__main__":
    main()
