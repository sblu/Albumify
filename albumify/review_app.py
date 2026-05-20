"""Flask web app to review and refine Gemini-generated labels.

Single-page table of all cover/label pairs. User marks 👍/👎; thumbs-down
with a comment triggers async re-generation through a background worker
that addresses the comment by appending it to the base prompt.

All iterations (prompt + model + png + comment) persist in SQLite at
data/review.db. Generated images are copied to data/labels_history/<slug>/v<n>.png
so the original data/labels/ tree stays untouched until the user approves.

The genai SDK is imported lazily inside the regen path so this module
remains importable without google-genai (and unit tests don't need it).
"""
from __future__ import annotations

import json
import queue
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, request, send_from_directory


BASE_PROMPT = (
    "Create a simple single line black-and-white line drawing of this image. "
    "White background, thin black outline strokes only. "
    "Critical constraint: every visible element must be drawn as an outline. "
    "Do NOT fill any shape with solid black. "
    "Render all text, letters, and logos as outline strokes only -- never as "
    "filled-in characters. Hair, clothing, and dark objects from the source "
    "must also be rendered as outline contours, not solid black areas. "
    "Preserve the main subjects and composition of the album cover."
)

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


# --- DB helpers ---------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
  slug TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'pending',
  current_iter INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS iterations (
  slug TEXT NOT NULL,
  iter_n INTEGER NOT NULL,
  prompt TEXT NOT NULL,
  model TEXT NOT NULL,
  png_path TEXT NOT NULL,
  comment_addressed TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (slug, iter_n)
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def seed_db(
    db_path: Path,
    labels_dir: Path,
    history_dir: Path,
    *,
    base_prompt: str = BASE_PROMPT,
    model: str = DEFAULT_MODEL,
) -> int:
    """Seed v1 iteration rows from existing label PNGs. Idempotent."""
    init_db(db_path)
    history_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    with _connect(db_path) as conn:
        for png in sorted(labels_dir.glob("*.png")):
            slug = png.stem
            row = conn.execute("SELECT 1 FROM labels WHERE slug = ?", (slug,)).fetchone()
            if row:
                continue
            slug_dir = history_dir / slug
            slug_dir.mkdir(parents=True, exist_ok=True)
            v1_path = slug_dir / "v1.png"
            if not v1_path.exists():
                shutil.copyfile(png, v1_path)
            rel = f"labels_history/{slug}/v1.png"
            conn.execute(
                "INSERT INTO labels (slug, status, current_iter) VALUES (?, 'pending', 1)",
                (slug,),
            )
            conn.execute(
                "INSERT INTO iterations (slug, iter_n, prompt, model, png_path, comment_addressed) "
                "VALUES (?, 1, ?, ?, ?, NULL)",
                (slug, base_prompt, model, rel),
            )
            added += 1
        conn.commit()
    return added


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def fetch_items(db_path: Path) -> list[dict]:
    with _connect(db_path) as conn:
        labels = [_row_to_dict(r) for r in conn.execute(
            "SELECT slug, status, current_iter, updated_at FROM labels ORDER BY slug"
        )]
        iters = [_row_to_dict(r) for r in conn.execute(
            "SELECT slug, iter_n, prompt, model, png_path, comment_addressed, created_at "
            "FROM iterations ORDER BY slug, iter_n"
        )]
    by_slug: dict[str, list[dict]] = {}
    for it in iters:
        by_slug.setdefault(it["slug"], []).append(it)
    for lab in labels:
        lab["iterations"] = by_slug.get(lab["slug"], [])
    return labels


def set_status(db_path: Path, slug: str, status: str) -> None:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE labels SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE slug = ?",
            (status, slug),
        )
        if cur.rowcount == 0:
            raise KeyError(slug)
        conn.commit()


def append_iteration(
    db_path: Path,
    slug: str,
    *,
    prompt: str,
    model: str,
    png_path: str,
    comment_addressed: str,
) -> int:
    """Insert a new iteration row and bump labels.current_iter. Returns iter_n."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT current_iter FROM labels WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            raise KeyError(slug)
        next_n = row["current_iter"] + 1
        conn.execute(
            "INSERT INTO iterations (slug, iter_n, prompt, model, png_path, comment_addressed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (slug, next_n, prompt, model, png_path, comment_addressed),
        )
        conn.execute(
            "UPDATE labels SET current_iter = ?, status = 'rejected', "
            "updated_at = CURRENT_TIMESTAMP WHERE slug = ?",
            (next_n, slug),
        )
        conn.commit()
        return next_n


# --- Regen ---------------------------------------------------------------

def build_refined_prompt(base: str, comment: str) -> str:
    return (
        f"{base}\n\n"
        f"Additional refinement based on reviewer feedback: {comment.strip()}"
    )


def regenerate_label(
    *,
    slug: str,
    comment: str,
    covers_dir: Path,
    history_dir: Path,
    db_path: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_prompt: str = BASE_PROMPT,
    genai_call=None,
    make_client=None,
) -> Optional[int]:
    """Run one regen: call Gemini, write PNG, append iteration row.

    Returns the new iter_n on success, None on failure (status set to
    'rejected' so the row remains actionable rather than stuck in gray).
    """
    if make_client is None:
        from albumify.gen_labels import _make_client as make_client
    if genai_call is None:
        from albumify.gen_labels import _genai_call as genai_call
    from albumify.gen_labels import _extract_png

    cover = covers_dir / f"{slug}.jpg"
    if not cover.exists():
        set_status(db_path, slug, "rejected")
        return None

    prompt = build_refined_prompt(base_prompt, comment)
    client = make_client(api_key)
    try:
        from google.genai import types as gtypes  # type: ignore
        image_part = gtypes.Part.from_bytes(data=cover.read_bytes(), mime_type="image/jpeg")
        config = gtypes.GenerateContentConfig(response_modalities=["IMAGE"])
        # Use a custom prompt instead of the module-level PROMPT.
        resp = client.models.generate_content(
            model=model,
            contents=[prompt, image_part],
            config=config,
        )
    except Exception:
        set_status(db_path, slug, "rejected")
        return None

    png = _extract_png(resp)
    if png is None:
        set_status(db_path, slug, "rejected")
        return None

    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT current_iter FROM labels WHERE slug = ?", (slug,)
        ).fetchone()
        next_n = (row["current_iter"] if row else 0) + 1

    slug_dir = history_dir / slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    out_path = slug_dir / f"v{next_n}.png"
    out_path.write_bytes(png)
    rel = f"labels_history/{slug}/v{next_n}.png"
    append_iteration(
        db_path, slug,
        prompt=prompt, model=model, png_path=rel, comment_addressed=comment,
    )
    return next_n


# --- Worker --------------------------------------------------------------

class RegenWorker:
    """Single background thread that pops slugs+comments and regenerates."""

    def __init__(
        self,
        *,
        db_path: Path,
        covers_dir: Path,
        history_dir: Path,
        api_key: str,
        model: str = DEFAULT_MODEL,
        sleep_s: float = 1.0,
    ):
        self.db_path = db_path
        self.covers_dir = covers_dir
        self.history_dir = history_dir
        self.api_key = api_key
        self.model = model
        self.sleep_s = sleep_s
        self.q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def enqueue(self, slug: str, comment: str) -> None:
        set_status(self.db_path, slug, "processing")
        self.q.put((slug, comment))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                slug, comment = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                regenerate_label(
                    slug=slug, comment=comment,
                    covers_dir=self.covers_dir, history_dir=self.history_dir,
                    db_path=self.db_path, api_key=self.api_key, model=self.model,
                )
            except Exception:
                try:
                    set_status(self.db_path, slug, "rejected")
                except Exception:
                    pass
            time.sleep(self.sleep_s)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# --- Flask app -----------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Albumify Label Review</title>
<style>
  :root { --zoom: 1; --cover-base: 200px; --current-base: 220px; --prev-base: 110px; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 0; background: #fafafa; }
  header { position: sticky; top: 0; background: #222; color: #fff; padding: 10px 16px; z-index: 10; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; }
  header .stat { font-size: 13px; opacity: 0.9; }
  header .stat b { color: #ffeb3b; }
  header .zoom { display: flex; align-items: center; gap: 8px; margin-left: auto; font-size: 12px; }
  header .zoom input[type=range] { width: 160px; }
  header .zoom .pct { font-variant-numeric: tabular-nums; min-width: 42px; text-align: right; color: #ffeb3b; }
  header .zoom button { background: #444; color: #fff; border: 1px solid #666; padding: 2px 8px; font-size: 11px; cursor: pointer; border-radius: 3px; }
  header .zoom button:hover { background: #555; }
  table { width: 100%; border-collapse: collapse; }
  thead th { position: sticky; top: 42px; background: #eee; padding: 8px; text-align: left; font-size: 12px; border-bottom: 1px solid #ccc; }
  tbody tr { border-bottom: 1px solid #e0e0e0; vertical-align: top; }
  tbody tr.pending  { background: #ffffff; }
  tbody tr.approved { background: #d4f5d4; }
  tbody tr.rejected { background: #fde0e0; }
  tbody tr.processing { background: #d0d0d0; }
  td { padding: 8px; font-size: 13px; }
  td.slug { width: 180px; font-family: ui-monospace, Menlo, monospace; font-size: 11px; word-break: break-all; }
  td.cover img { width: calc(var(--cover-base) * var(--zoom)); height: calc(var(--cover-base) * var(--zoom)); object-fit: cover; border: 1px solid #bbb; display: block; }
  td.iters .iter { margin-bottom: 8px; }
  td.iters .iter img { display: block; border: 1px solid #bbb; }
  td.iters .iter.current img { width: calc(var(--current-base) * var(--zoom)); height: calc(var(--current-base) * var(--zoom)); }
  td.iters .iter.prev img { width: calc(var(--prev-base) * var(--zoom)); height: calc(var(--prev-base) * var(--zoom)); opacity: 0.7; }
  td.iters .prompt { font-size: 10px; color: #555; margin-top: 3px; max-width: calc(var(--current-base) * var(--zoom)); white-space: pre-wrap; }
  td.iters .iter.prev .prompt { max-width: calc(var(--prev-base) * var(--zoom)); }
  td.iters .comment-shown { font-size: 11px; color: #b00; font-style: italic; margin-top: 2px; }
  td.actions { width: 110px; }
  td.actions label { display: block; cursor: pointer; padding: 4px 0; }
  td.comment textarea { width: 260px; height: 70px; font-family: inherit; font-size: 12px; padding: 4px; box-sizing: border-box; }
  td.submit { width: 90px; }
  td.submit button { padding: 6px 12px; font-size: 13px; cursor: pointer; background: #1976d2; color: #fff; border: none; border-radius: 3px; }
  td.submit button:disabled { background: #888; cursor: not-allowed; }
  td.status { width: 80px; font-size: 11px; text-align: center; text-transform: uppercase; color: #555; }
</style>
</head>
<body>
<header>
  <h1>Albumify label review</h1>
  <span class="stat"><b id="cnt-approved">0</b> approved</span>
  <span class="stat"><b id="cnt-pending">0</b> pending</span>
  <span class="stat"><b id="cnt-rejected">0</b> need re-review</span>
  <span class="stat"><b id="cnt-processing">0</b> processing</span>
  <span class="stat">/ <b id="cnt-total">0</b> total</span>
  <span class="zoom">
    <button id="zoom-reset" title="Reset to 100%">reset</button>
    <label for="zoom-range">zoom</label>
    <input id="zoom-range" type="range" min="0.5" max="3" step="0.1" value="1">
    <span class="pct" id="zoom-pct">100%</span>
  </span>
</header>
<table>
  <thead>
    <tr>
      <th>Slug</th>
      <th>Cover</th>
      <th>Iterations (newest top)</th>
      <th>Rating</th>
      <th>Comment</th>
      <th></th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody id="rows"></tbody>
</table>

<script>
let items = [];

function renderRow(it) {
  const tr = document.createElement('tr');
  tr.id = 'row-' + it.slug;
  tr.className = it.status;

  const tdSlug = document.createElement('td');
  tdSlug.className = 'slug';
  tdSlug.textContent = it.slug;
  tr.appendChild(tdSlug);

  const tdCover = document.createElement('td');
  tdCover.className = 'cover';
  const cimg = document.createElement('img');
  cimg.src = '/covers/' + it.slug + '.jpg';
  cimg.loading = 'lazy';
  tdCover.appendChild(cimg);
  tr.appendChild(tdCover);

  // Iterations: when approved, show only current. Else newest-first stack.
  const tdIters = document.createElement('td');
  tdIters.className = 'iters';
  const showAll = (it.status !== 'approved');
  const itersToShow = showAll ? [...it.iterations].reverse() : [it.iterations[it.iterations.length - 1]];
  itersToShow.forEach((iter, idx) => {
    if (!iter) return;
    const div = document.createElement('div');
    div.className = 'iter ' + (idx === 0 ? 'current' : 'prev');
    const img = document.createElement('img');
    img.src = '/' + iter.png_path;
    img.loading = 'lazy';
    div.appendChild(img);
    if (iter.comment_addressed) {
      const c = document.createElement('div');
      c.className = 'comment-shown';
      c.textContent = '↳ ' + iter.comment_addressed;
      div.appendChild(c);
    }
    const p = document.createElement('div');
    p.className = 'prompt';
    p.textContent = 'v' + iter.iter_n + ' [' + iter.model + ']: ' + iter.prompt;
    div.appendChild(p);
    tdIters.appendChild(div);
  });
  tr.appendChild(tdIters);

  const tdActions = document.createElement('td');
  tdActions.className = 'actions';
  const rUp = document.createElement('label');
  rUp.innerHTML = '<input type="radio" name="r-' + it.slug + '" value="up"> 👍 approve';
  const rDown = document.createElement('label');
  rDown.innerHTML = '<input type="radio" name="r-' + it.slug + '" value="down"> 👎 reject';
  tdActions.appendChild(rUp);
  tdActions.appendChild(rDown);
  tr.appendChild(tdActions);

  const tdComment = document.createElement('td');
  tdComment.className = 'comment';
  const ta = document.createElement('textarea');
  ta.placeholder = 'Required for 👎';
  ta.id = 'c-' + it.slug;
  tdComment.appendChild(ta);
  tr.appendChild(tdComment);

  const tdSubmit = document.createElement('td');
  tdSubmit.className = 'submit';
  const btn = document.createElement('button');
  btn.textContent = 'Submit';
  btn.disabled = (it.status === 'processing');
  btn.onclick = () => submitRow(it.slug, btn);
  tdSubmit.appendChild(btn);
  tr.appendChild(tdSubmit);

  const tdStatus = document.createElement('td');
  tdStatus.className = 'status';
  tdStatus.textContent = it.status;
  tr.appendChild(tdStatus);

  return tr;
}

function rerender() {
  const tbody = document.getElementById('rows');
  tbody.innerHTML = '';
  let cnt = {approved: 0, pending: 0, rejected: 0, processing: 0};
  items.forEach(it => {
    cnt[it.status] = (cnt[it.status] || 0) + 1;
    tbody.appendChild(renderRow(it));
  });
  document.getElementById('cnt-approved').textContent = cnt.approved;
  document.getElementById('cnt-pending').textContent = cnt.pending;
  document.getElementById('cnt-rejected').textContent = cnt.rejected;
  document.getElementById('cnt-processing').textContent = cnt.processing;
  document.getElementById('cnt-total').textContent = items.length;
}

function patchRow(updated) {
  const i = items.findIndex(x => x.slug === updated.slug);
  if (i >= 0) items[i] = updated;
  const old = document.getElementById('row-' + updated.slug);
  if (old) old.replaceWith(renderRow(updated));
  rerenderCounters();
}

function rerenderCounters() {
  let cnt = {approved: 0, pending: 0, rejected: 0, processing: 0};
  items.forEach(it => { cnt[it.status] = (cnt[it.status] || 0) + 1; });
  document.getElementById('cnt-approved').textContent = cnt.approved;
  document.getElementById('cnt-pending').textContent = cnt.pending;
  document.getElementById('cnt-rejected').textContent = cnt.rejected;
  document.getElementById('cnt-processing').textContent = cnt.processing;
  document.getElementById('cnt-total').textContent = items.length;
}

async function submitRow(slug, btn) {
  const choice = document.querySelector('input[name="r-' + slug + '"]:checked');
  if (!choice) { alert('Pick 👍 or 👎 first.'); return; }
  if (choice.value === 'up') {
    btn.disabled = true;
    const resp = await fetch('/api/items/' + slug + '/approve', {method: 'POST'});
    const data = await resp.json();
    patchRow(data);
  } else {
    const comment = document.getElementById('c-' + slug).value.trim();
    if (!comment) { alert('Comment required for 👎.'); return; }
    btn.disabled = true;
    const resp = await fetch('/api/items/' + slug + '/reject', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({comment}),
    });
    const data = await resp.json();
    patchRow(data);
  }
}

async function load() {
  const resp = await fetch('/api/items');
  items = await resp.json();
  rerender();
}

async function poll() {
  // Only poll when at least one row is processing
  if (!items.some(x => x.status === 'processing')) return;
  const resp = await fetch('/api/items');
  const fresh = await resp.json();
  fresh.forEach(f => {
    const cur = items.find(x => x.slug === f.slug);
    if (!cur || cur.status !== f.status || cur.current_iter !== f.current_iter) {
      patchRow(f);
    }
  });
}

function applyZoom(z) {
  document.documentElement.style.setProperty('--zoom', z);
  document.getElementById('zoom-pct').textContent = Math.round(z * 100) + '%';
  document.getElementById('zoom-range').value = z;
  localStorage.setItem('albumify-zoom', String(z));
}

(function initZoom() {
  const saved = parseFloat(localStorage.getItem('albumify-zoom') || '1');
  applyZoom(isFinite(saved) ? saved : 1);
  document.getElementById('zoom-range').addEventListener('input', e => {
    applyZoom(parseFloat(e.target.value));
  });
  document.getElementById('zoom-reset').addEventListener('click', () => applyZoom(1));
})();

load();
setInterval(poll, 4000);
</script>
</body>
</html>
"""


def create_app(
    *,
    db_path: Path,
    covers_dir: Path,
    history_dir: Path,
    worker: Optional[RegenWorker] = None,
) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.get("/api/items")
    def api_items():
        return jsonify(fetch_items(db_path))

    @app.post("/api/items/<slug>/approve")
    def api_approve(slug: str):
        try:
            set_status(db_path, slug, "approved")
        except KeyError:
            abort(404)
        item = next((x for x in fetch_items(db_path) if x["slug"] == slug), None)
        return jsonify(item)

    @app.post("/api/items/<slug>/reject")
    def api_reject(slug: str):
        body = request.get_json(silent=True) or {}
        comment = (body.get("comment") or "").strip()
        if not comment:
            return jsonify({"error": "comment required"}), 400
        if worker is None:
            # Tests path: just mark rejected without queueing.
            try:
                set_status(db_path, slug, "rejected")
            except KeyError:
                abort(404)
        else:
            try:
                worker.enqueue(slug, comment)
            except KeyError:
                abort(404)
        item = next((x for x in fetch_items(db_path) if x["slug"] == slug), None)
        return jsonify(item)

    covers_abs = Path(covers_dir).resolve()
    history_abs = Path(history_dir).resolve()

    @app.get("/covers/<path:fname>")
    def serve_cover(fname: str):
        return send_from_directory(covers_abs, fname)

    @app.get("/labels_history/<path:relpath>")
    def serve_history(relpath: str):
        return send_from_directory(history_abs, relpath)

    return app


def main() -> None:
    import argparse
    import os
    from dotenv import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser(description="Run the label review web app.")
    p.add_argument("--data", default="data", type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    data_dir = args.data.resolve()
    covers_dir = data_dir / "covers"
    labels_dir = data_dir / "labels"
    history_dir = data_dir / "labels_history"
    db_path = data_dir / "review.db"

    added = seed_db(db_path, labels_dir, history_dir, model=args.model)
    if added:
        print(f"Seeded {added} new labels into {db_path}.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY missing. Copy .env.example to .env and fill it in.")

    worker = RegenWorker(
        db_path=db_path, covers_dir=covers_dir, history_dir=history_dir,
        api_key=api_key, model=args.model,
    )
    worker.start()
    app = create_app(
        db_path=db_path, covers_dir=covers_dir, history_dir=history_dir, worker=worker,
    )
    print(f"Review app: http://{args.host}:{args.port}/")
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
