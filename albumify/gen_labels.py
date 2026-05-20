"""Generate line-drawing labels for downloaded covers via Gemini.

Default model: gemini-3.1-flash-image-preview at 1024x1024 (1K) output.
Fallback: gemini-2.5-flash-image (Nano Banana, $0.039/image at 1K-equivalent).

Idempotent (skips already-generated labels). API errors are logged to the
report file, not raised, so a long run never aborts mid-stream.

The Gemini SDK is imported lazily inside `_make_client` and `_genai_call`
so this module can be imported and unit-tested without google-genai installed.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional


PROMPT = (
    "Create a simple single line black-and-white line drawing of this image. "
    "White background, thin black outline strokes only. "
    "Critical constraint: every visible element must be drawn as an outline. "
    "Do NOT fill any shape with solid black. "
    "Render all text, letters, and logos as outline strokes only -- never as "
    "filled-in characters. Hair, clothing, and dark objects from the source "
    "must also be rendered as outline contours, not solid black areas. "
    "Preserve the main subjects and composition of the album cover."
)


def _make_client(api_key: str):
    """Construct a google-genai client. Lazy import for testability."""
    from google import genai
    return genai.Client(api_key=api_key)


def _genai_call(client, model: str, image_bytes: bytes, mime_type: str = "image/jpeg"):
    """Single call wrapper so tests can patch this function.

    Note: google-genai 1.47 only accepts `aspect_ratio` on ImageConfig (not
    `image_size`). We omit ImageConfig entirely and let the model return its
    default output size, which empirically is 1024x1024 PNG for both
    gemini-3.1-flash-image-preview and gemini-2.5-flash-image as of 2026-05-20.
    """
    from google.genai import types as gtypes
    image_part = gtypes.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    config = gtypes.GenerateContentConfig(
        response_modalities=["IMAGE"],
    )
    return client.models.generate_content(
        model=model,
        contents=[PROMPT, image_part],
        config=config,
    )


def _extract_png(response) -> Optional[bytes]:
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    return None


def run(
    covers_dir,
    labels_dir,
    report_path,
    *,
    model: str,
    api_key: str,
    limit: Optional[int] = None,
    sleep_s: float = 1.0,
) -> int:
    """Generate labels for every cover in covers_dir not already in labels_dir.

    Returns the count of newly-generated labels.
    """
    covers_dir = Path(covers_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    client = _make_client(api_key)

    new_count = 0
    lines: list = []
    covers = sorted(covers_dir.glob("*.jpg"))
    if limit is not None:
        covers = covers[:limit]

    for cover in covers:
        slug = cover.stem
        target = labels_dir / f"{slug}.png"
        if target.exists():
            lines.append(f"{slug} skip_exists")
            continue
        try:
            resp = _genai_call(client, model, cover.read_bytes(), "image/jpeg")
        except Exception as exc:
            lines.append(f"{slug} api_error {exc}")
            continue
        png = _extract_png(resp)
        if png is None:
            lines.append(f"{slug} no_image_in_response")
            continue
        target.write_bytes(png)
        new_count += 1
        lines.append(f"{slug} OK {len(png)}B")
        if sleep_s:
            time.sleep(sleep_s)

    Path(report_path).write_text("\n".join(lines) + "\n")
    return new_count


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description="Generate line-drawing labels via Gemini.")
    p.add_argument("--covers", default="data/covers", type=Path)
    p.add_argument("--out", default="data/labels", type=Path)
    p.add_argument("--report", default="data/labels_report.txt", type=Path)
    p.add_argument("--model", default="gemini-3.1-flash-image-preview",
                   choices=["gemini-3.1-flash-image-preview", "gemini-2.5-flash-image"])
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N covers (for preview checkpoints).")
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated slug list; if set, only those covers are processed.")
    args = p.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing. Copy .env.example to .env and fill it in.")

    # --only filter: copy matching covers into a temp dir, run on that, then merge back.
    if args.only:
        import tempfile, shutil
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        with tempfile.TemporaryDirectory() as td:
            tmp_covers = Path(td) / "covers"
            tmp_covers.mkdir()
            for slug in wanted:
                src = args.covers / f"{slug}.jpg"
                if src.exists():
                    shutil.copy(src, tmp_covers / f"{slug}.jpg")
            n = run(tmp_covers, args.out, args.report,
                    model=args.model, api_key=api_key, sleep_s=args.sleep)
    else:
        n = run(args.covers, args.out, args.report,
                model=args.model, api_key=api_key, limit=args.limit, sleep_s=args.sleep)
    print(f"Generated {n} new labels with {args.model}. Report: {args.report}")


if __name__ == "__main__":
    main()
