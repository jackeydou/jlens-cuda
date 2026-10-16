"""Export saved sessions as a static, read-only presentation site.

Produces the layout web/index.html reads in presentation mode, so the
result deploys to any static host (GitHub Pages, Cloudflare Pages) with no
server and no model:

    <out>/index.html                  the UI, JLENS_MODE="presentation", with
                                      the meta + session manifest baked in
    <out>/meta.json                   model/lens metadata (fallback copy)
    <out>/sessions/index.json         session manifest (fallback copy)
    <out>/sessions/<id>.<k>.gz        gzipped session payload chunks
    <out>/.nojekyll                   GitHub Pages: serve files verbatim

Chunk 0 is the conversation envelope plus the first snapshots (enough to
render); chunks 1.. carry the remaining snapshots and stream in behind it.
Each conversation gets a shareable link: <site>/#session=<id>.

Layers below --min-layer (default 19: the sensory/surface band, L0-L18)
are dropped from every grid, cutting the payload by roughly a third; the
UI shows them as one "not shown in this export" placeholder column. Pass
--min-layer 0 to keep every layer. Markups on dropped layers are dropped.

Metadata (model id, layer count, functional bands) comes from a running
server when --server is given; otherwise it is inferred from the sessions'
recorded layers plus --model-id / --bands.

Run:
    python scripts/export_static.py --out site
    python scripts/export_static.py --out site data/sessions/2026*.json \
        --server http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "web" / "index.html"
SESSIONS_DIR = REPO / "data" / "sessions"
BANDS_DIR = REPO / "data" / "bands"
AUTOSAVE_ID = "autosave-latest.json"
MODE_LINE = 'window.JLENS_MODE = "active";'

# GitHub Pages rejects files over 100 MB (git) and Cloudflare Pages over
# 25 MiB; budgets are on RAW JSON bytes, so gzipped chunks land far below.
# Chunk 0 is kept small so the first render arrives quickly.
FIRST_CHUNK_BYTES = 4 << 20
CHUNK_BYTES = 32 << 20
DEFAULT_MIN_LAYER = 19


def session_preview(data: dict) -> str:
    """Mirror of serve._session_preview (serve.py imports torch)."""
    preview = str(data.get("preview") or "").strip()
    if preview:
        return preview[:160]
    for msg in data.get("messages") or []:
        if msg.get("role") == "user":
            content = str(msg.get("content") or "").strip()
            if content:
                return re.sub(r"\s+", " ", content)[:160]
    return "Untitled session"


def drop_layers(data: dict, min_layer: int) -> tuple[dict, int]:
    """Copy of a session without grid columns below min_layer.

    Returns (session, number of markups dropped with their layers)."""
    dropped = sorted(int(l) for l in data.get("layers") or [] if int(l) < min_layer)
    layers = [l for l in data.get("layers") or [] if int(l) >= min_layer]
    if data.get("layers") and not layers:
        raise SystemExit(f"--min-layer {min_layer} drops every recorded layer")
    snapshots = []
    for snap in data.get("snapshots") or []:
        grid = snap.get("grid")
        if grid:
            grid = {**grid,
                    "layers": [l for l in grid.get("layers") or [] if int(l) >= min_layer],
                    "cells": {l: c for l, c in (grid.get("cells") or {}).items()
                              if int(l) >= min_layer}}
            snap = {**snap, "grid": grid}
        snapshots.append(snap)
    markups = data.get("markups") or []
    kept = [m for m in markups if not isinstance(m.get("layer"), (int, float))
            or m["layer"] >= min_layer]
    out = {**data, "layers": layers, "snapshots": snapshots, "markups": kept}
    if dropped:
        # The UI draws this range as a placeholder column (state.omittedLayers).
        out["omittedLayers"] = {"start": dropped[0], "end": dropped[-1]}
    return out, len(markups) - len(kept)


def chunk_snapshots(snapshots: list[dict]) -> list[list[dict]]:
    chunks: list[list[dict]] = [[]]
    used = 0
    for snap in snapshots:
        size = len(json.dumps(snap, ensure_ascii=False, separators=(",", ":")))
        budget = FIRST_CHUNK_BYTES if len(chunks) == 1 else CHUNK_BYTES
        if chunks[-1] and used + size > budget:
            chunks.append([])
            used = 0
        chunks[-1].append(snap)
        used += size
    return chunks


def write_gz_json(path: Path, obj) -> int:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
    # mtime=0 keeps re-exports byte-identical, so unchanged sessions do not
    # show up as diffs in the Pages repo.
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return path.stat().st_size


def fetch_server_meta(server: str) -> dict:
    def get(endpoint: str) -> dict:
        with urllib.request.urlopen(server.rstrip("/") + endpoint, timeout=30) as resp:
            return json.loads(resp.read())
    return {"model": get("/api/model"), "lens": get("/api/lens")}


def resolve_bands(arg: str | None) -> list[dict] | None:
    if arg == "none":
        return None
    if arg:
        path = Path(arg)
    else:
        candidates = sorted(BANDS_DIR.glob("*.json"))
        if len(candidates) != 1:
            if candidates:
                print(f"BANDS: {len(candidates)} files in {BANDS_DIR}; pass --bands "
                      "to choose one. Using the UI's fallback split.", file=sys.stderr)
            return None
        path = candidates[0]
        print(f"BANDS: using the only bands file, {path.name}", file=sys.stderr)
    bands = json.loads(path.read_text()).get("bands")
    if not isinstance(bands, list) or not bands:
        raise SystemExit(f"{path}: no 'bands' list")
    return bands


def infer_meta(sessions: list[dict], model_id: str, bands: list[dict] | None) -> dict:
    # Sessions record the lens source layers plus the final layer
    # (serve._record_layers), so the deepest recorded layer is n_layers - 1.
    # (After --min-layer, source_layers lists only the exported ones.)
    layers = sorted({int(l) for s in sessions for l in (s.get("layers") or [])})
    if not layers:
        raise SystemExit("sessions record no layers; pass --server to read metadata")
    n_layers = layers[-1] + 1
    return {
        "model": {"model_id": model_id, "n_layers": n_layers, "d_model": None,
                  "mode": "presentation", "model_loaded": False},
        "lens": {"lens_loaded": True, "source_layers": layers[:-1], "n_prompts": None,
                 "d_model": None, "n_layers": n_layers,
                 "bands": bands or [], "bands_are_fallback": bands is None},
    }


def script_json(obj) -> str:
    # Safe inside <script>: no "</script>" or "<!--" can terminate the block.
    return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", type=Path,
                    help=f"session JSON files (default: every saved session in {SESSIONS_DIR})")
    ap.add_argument("--out", required=True, type=Path, help="output site directory")
    ap.add_argument("--clean", action="store_true",
                    help="delete the output directory before exporting")
    ap.add_argument("--min-layer", type=int, default=DEFAULT_MIN_LAYER,
                    help="drop grid layers below this one (default %(default)s: the sensory "
                    "band L0-L18; 0 keeps every layer)")
    ap.add_argument("--server", help="read model/lens metadata from this running server")
    ap.add_argument("--model-id", default="Qwen/Qwen3.8-27B",
                    help="model id shown in the UI when --server is not given")
    ap.add_argument("--bands", help="bands JSON (scripts/measure_bands.py output), or 'none'; "
                    f"default: the single file in {BANDS_DIR} if there is exactly one")
    args = ap.parse_args()

    paths = args.sessions or sorted(SESSIONS_DIR.glob("*.json"))
    paths = [p for p in paths if p.name != AUTOSAVE_ID]
    if not paths:
        raise SystemExit(f"no sessions to export (looked in {SESSIONS_DIR})")
    ids = [p.name for p in paths]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"duplicate session ids (file names): {sorted(dupes)}")

    out: Path = args.out
    if args.clean and out.exists():
        shutil.rmtree(out)
    sessions_out = out / "sessions"
    sessions_out.mkdir(parents=True, exist_ok=True)
    # Drop stale chunks so a session that shrank can't leave orphans behind.
    for old in sessions_out.glob("*.gz"):
        old.unlink()

    loaded: list[tuple[Path, dict]] = []
    for path in paths:
        data, lost_markups = drop_layers(json.loads(path.read_text()), args.min_layer)
        if lost_markups:
            print(f"  {path.name}: dropped {lost_markups} markup(s) below L{args.min_layer}",
                  file=sys.stderr)
        loaded.append((path, data))
    # Newest first, like the live server's listing.
    loaded.sort(key=lambda item: item[1].get("saved_at") or item[0].name, reverse=True)

    manifest = []
    total_bytes = 0
    for path, data in loaded:
        sid = path.name
        snapshots = data.get("snapshots") or []
        chunks = chunk_snapshots(snapshots)
        envelope = {**data, "snapshots": chunks[0],
                    "chunking": {"chunks": len(chunks), "snapshot_count": len(snapshots)}}
        sizes = [write_gz_json(sessions_out / f"{sid}.0.gz", envelope)]
        for k, chunk in enumerate(chunks[1:], start=1):
            sizes.append(write_gz_json(sessions_out / f"{sid}.{k}.gz", {"snapshots": chunk}))
        total_bytes += sum(sizes)
        manifest.append({
            "id": sid,
            "preview": session_preview(data),
            "created_at": data.get("created_at"),
            "saved_at": data.get("saved_at"),
            "message_count": len(data.get("messages") or []),
            "snapshot_count": len(snapshots),
        })
        print(f"  {sid}: {len(snapshots)} snapshots -> {len(chunks)} chunk(s), "
              f"{sum(sizes) / 1e6:.1f} MB (largest {max(sizes) / 1e6:.1f} MB)")

    if args.server:
        meta = fetch_server_meta(args.server)
        meta["model"]["mode"] = "presentation"
    else:
        meta = infer_meta([d for _, d in loaded], args.model_id, resolve_bands(args.bands))

    html = TEMPLATE.read_text()
    if html.count(MODE_LINE) != 1:
        raise SystemExit(f"{TEMPLATE}: expected exactly one {MODE_LINE!r}")
    html = html.replace(MODE_LINE, (
        'window.JLENS_MODE = "presentation";\n'
        f"  window.JLENS_META = {script_json(meta)};\n"
        f"  window.JLENS_SESSIONS = {script_json(manifest)};"
    ), 1)
    (out / "index.html").write_text(html)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    (sessions_out / "index.json").write_text(
        json.dumps({"sessions": manifest}, ensure_ascii=False, indent=2) + "\n")
    (out / ".nojekyll").write_text("")

    print(f"exported {len(manifest)} session(s) to {out}/ ({total_bytes / 1e6:.1f} MB of payload)")
    print(f"preview: python -m http.server -d {out} 8000  ->  http://127.0.0.1:8000/")


if __name__ == "__main__":
    main()
