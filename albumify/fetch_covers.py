"""Orchestrator: read albums.json -> resolve MBIDs -> download covers.

Idempotent: rerunning skips already-downloaded files. Writes a report with
per-slug status so misses can be triaged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from albumify.musicbrainz import MusicBrainzClient
from albumify.caa import CAAClient, CAATooSmall


def run(albums_json: Path, covers_dir: Path, report_path: Path, user_agent: str) -> int:
    """Run fetch. Returns count of NEW covers downloaded this run."""
    albums = json.loads(Path(albums_json).read_text())
    covers_dir = Path(covers_dir)
    covers_dir.mkdir(parents=True, exist_ok=True)
    mb = MusicBrainzClient(user_agent=user_agent)
    caa = CAAClient(user_agent=user_agent)

    new_count = 0
    lines = []
    for entry in albums:
        slug = entry["slug"]
        target = covers_dir / f"{slug}.jpg"
        if target.exists():
            lines.append(f"{slug} skip_exists")
            continue
        try:
            mbid = mb.search_release_group(entry["artist"], entry["title"])
        except Exception as exc:
            lines.append(f"{slug} mb_error {exc}")
            continue
        if mbid is None:
            lines.append(f"{slug} no_mbid")
            continue
        try:
            result = caa.fetch_front(mbid)
        except CAATooSmall as exc:
            lines.append(f"{slug} too_small {exc}")
            continue
        except Exception as exc:
            lines.append(f"{slug} caa_error {exc}")
            continue
        if result is None:
            lines.append(f"{slug} no_cover")
            continue
        data, (w, h) = result
        target.write_bytes(data)
        new_count += 1
        lines.append(f"{slug} OK {w}x{h}")

    Path(report_path).write_text("\n".join(lines) + "\n")
    return new_count


def main() -> None:
    p = argparse.ArgumentParser(description="Download album covers from CAA.")
    p.add_argument("--albums", default="data/albums.json", type=Path)
    p.add_argument("--out", default="data/covers", type=Path)
    p.add_argument("--report", default="data/fetch_report.txt", type=Path)
    p.add_argument("--user-agent", default="Albumify/0.1 (scottbluman@gmail.com)")
    args = p.parse_args()
    n = run(args.albums, args.out, args.report, args.user_agent)
    print(f"Downloaded {n} new covers. Report: {args.report}")


if __name__ == "__main__":
    main()
