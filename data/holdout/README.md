# Holdout test set (10 difficult covers)

These 10 PNGs are reserved as a **held-out evaluation set**. They MUST NOT
appear in training data. The training/eval split code (Phase 5) treats this
directory as its own partition.

Source: `jelsas/Afterglow` (private), `backend/tests/fixtures/images/`.
Fetched 2026-05-20.

| file | notes |
| --- | --- |
| `abbey_road.png` | The Beatles — Abbey Road. Removed from `albums.json` to prevent train leakage. |
| `b52s.png` | The B-52's (debut). |
| `circles.png` | Abstract test pattern. |
| `dark_side_moon.png` | Pink Floyd — The Dark Side of the Moon. Removed from `albums.json`. |
| `face.png` | Stylized portrait test pattern. |
| `geometric.png` | Abstract geometric test pattern. |
| `kind_of_blue.png` | Miles Davis — Kind of Blue. |
| `loveless.png` | My Bloody Valentine — Loveless. |
| `sgt_peppers.png` | The Beatles — Sgt. Pepper's. Removed from `albums.json`. |
| `synthwave.png` | Stylized illustration test pattern. |

The PNGs themselves are gitignored (binary, large). Only this README is
tracked so the holdout set's provenance + train-leakage decisions are
auditable in git history.
