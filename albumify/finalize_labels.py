"""Copy each approved slug's current iteration from labels_history/ into labels/.

After review, the canonical training label for slug X is the v<current_iter> PNG
in data/labels_history/X/. The unaltered v1 still lives in data/labels/X.png
from the initial Gemini run. This script overwrites data/labels/<slug>.png
with the latest iteration so the training pipeline picks up reviewer-approved
images.

Slugs with current_iter == 1 are no-ops (v1 in labels_history/ is identical to
the file in data/labels/, since seed_db copied it there).
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


def finalize(
    *,
    db_path: Path,
    labels_dir: Path,
    history_dir: Path,
    require_approved: bool = True,
) -> dict[str, int]:
    """Copy v<current_iter>.png from history_dir into labels_dir for each row.

    Returns a stat dict. When require_approved=True, only rows with status
    'approved' are copied; others are skipped and counted.
    """
    labels_dir = Path(labels_dir)
    history_dir = Path(history_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stats = {"copied": 0, "skipped_v1": 0, "skipped_not_approved": 0, "missing_src": 0}
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT slug, status, current_iter FROM labels ORDER BY slug"
        ):
            if require_approved and row["status"] != "approved":
                stats["skipped_not_approved"] += 1
                continue
            if row["current_iter"] == 1:
                stats["skipped_v1"] += 1
                continue
            src = history_dir / row["slug"] / f"v{row['current_iter']}.png"
            dst = labels_dir / f"{row['slug']}.png"
            if not src.exists():
                stats["missing_src"] += 1
                continue
            shutil.copyfile(src, dst)
            stats["copied"] += 1
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Copy approved iterations into data/labels/.")
    p.add_argument("--db",      default="data/review.db",         type=Path)
    p.add_argument("--labels",  default="data/labels",            type=Path)
    p.add_argument("--history", default="data/labels_history",    type=Path)
    p.add_argument("--all", action="store_true",
                   help="Include non-approved rows (default: only 'approved').")
    args = p.parse_args()
    stats = finalize(
        db_path=args.db, labels_dir=args.labels, history_dir=args.history,
        require_approved=not args.all,
    )
    print(
        f"copied={stats['copied']} "
        f"skipped_v1={stats['skipped_v1']} "
        f"skipped_not_approved={stats['skipped_not_approved']} "
        f"missing_src={stats['missing_src']}"
    )


if __name__ == "__main__":
    main()
