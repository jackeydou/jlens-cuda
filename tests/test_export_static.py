"""scripts/export_static.py produces the layout presentation mode reads."""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("export_static", ROOT / "scripts" / "export_static.py")
export_static = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_static)

LAYERS = [0, 10, 19, 40, 63]


def _snap(snapshot_id: int, token_idx: int, **extra) -> dict:
    return {"snapshot_id": snapshot_id, "msg_idx": 1, "token_idx": token_idx, "role": "assistant",
            "grid": {"layers": LAYERS, "positions": [10 + token_idx],
                     "cells": {str(l): {"top_ids": [[1]], "top_tokens": [["x" * 50]],
                                        "top_scores": [[1.0]]} for l in LAYERS}}, **extra}


def _session(path: Path, n_tokens: int, prompt: str) -> dict:
    data = {
        "messages": [{"role": "user", "content": prompt, "tokens": []},
                     {"role": "assistant", "content": "", "tokens": []}],
        "snapshots": [_snap(1, -1)] + [_snap(k + 2, k) for k in range(n_tokens)],
        "layers": LAYERS,
        "markups": [{"pos": 11, "layer": 40, "token": "x"}, {"pos": 12, "layer": 10, "token": "y"}],
        "saved_at": f"2026-10-16T08:{n_tokens:02d}:00Z",
    }
    path.write_text(json.dumps(data))
    return data


def _read_gz(path: Path):
    return json.loads(gzip.decompress(path.read_bytes()))


def test_export_layout_chunks_and_baked_manifest(tmp_path, monkeypatch):
    src = tmp_path / "sessions"
    src.mkdir()
    big = _session(src / "20261016T090000Z-big.json", 30, "Write a story")
    _session(src / "20261016T080000Z-small.json", 2, "Hi </script><!-- there")
    (src / "autosave-latest.json").write_text(json.dumps(big))
    out = tmp_path / "site"
    (out / "sessions").mkdir(parents=True)
    (out / "sessions" / "stale.json.0.gz").write_bytes(b"old")

    monkeypatch.setattr(export_static, "FIRST_CHUNK_BYTES", 4000)
    monkeypatch.setattr(export_static, "CHUNK_BYTES", 8000)
    monkeypatch.setattr(export_static, "BANDS_DIR", tmp_path / "no-bands")
    monkeypatch.setattr("sys.argv", ["export_static.py", "--out", str(out), "--min-layer", "0",
                                     *map(str, sorted(src.glob("*.json")))])
    export_static.main()

    assert (out / ".nojekyll").exists()
    assert not (out / "sessions" / "stale.json.0.gz").exists()
    manifest = json.loads((out / "sessions" / "index.json").read_text())["sessions"]
    assert [s["id"] for s in manifest] == ["20261016T090000Z-big.json", "20261016T080000Z-small.json"]

    # Chunks reassemble into exactly the saved snapshots, chunk 0 first.
    first = _read_gz(out / "sessions" / "20261016T090000Z-big.json.0.gz")
    n = first["chunking"]["chunks"]
    assert n > 2
    snaps = list(first["snapshots"])
    for k in range(1, n):
        chunk = _read_gz(out / "sessions" / f"20261016T090000Z-big.json.{k}.gz")
        assert set(chunk) == {"snapshots"}
        snaps += chunk["snapshots"]
    assert snaps == big["snapshots"]
    assert first["markups"] == big["markups"] and first["messages"] == big["messages"]
    assert "omittedLayers" not in first

    html = (out / "index.html").read_text()
    assert 'window.JLENS_MODE = "presentation";' in html
    assert 'window.JLENS_MODE = "active"' not in html
    head = html[:html.index("</script>")]
    assert "<!--" not in head
    baked = json.loads(re.search(r"window\.JLENS_SESSIONS = (.*);", head).group(1))
    assert baked == manifest and baked[1]["preview"] == "Hi </script><!-- there"
    meta = json.loads(re.search(r"window\.JLENS_META = (.*);", head).group(1))
    assert meta["model"]["n_layers"] == 64
    assert meta["lens"]["source_layers"] == [0, 10, 19, 40]
    assert meta["lens"]["bands_are_fallback"] is True


def test_default_export_drops_the_sensory_band(tmp_path, monkeypatch):
    src = tmp_path / "s.json"
    saved = _session(src, 3, "Hi")
    out = tmp_path / "site"
    monkeypatch.setattr("sys.argv", ["export_static.py", "--out", str(out), "--bands", "none", str(src)])
    export_static.main()

    first = _read_gz(out / "sessions" / "s.json.0.gz")
    assert first["layers"] == [19, 40, 63]
    assert first["omittedLayers"] == {"start": 0, "end": 10}
    for snap in first["snapshots"]:
        assert snap["grid"]["layers"] == [19, 40, 63]
        assert list(snap["grid"]["cells"]) == ["19", "40", "63"]
    # Markups on dropped layers go with them; the rest survive.
    assert first["markups"] == [saved["markups"][0]]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["model"]["n_layers"] == 64
    assert meta["lens"]["source_layers"] == [19, 40]


def test_presentation_loader_contract():
    # The exporter's file names and globals are what web/index.html reads.
    html = (ROOT / "web" / "index.html").read_text()
    assert html.count(export_static.MODE_LINE) == 1
    assert "'sessions/' + encodeURIComponent(sessionId) + '.' + k + '.gz'" in html
    assert "first.chunking && first.chunking.chunks" in html
    assert "window.JLENS_SESSIONS" in html and "window.JLENS_META" in html
    assert "const omitted = saved.omittedLayers;" in html
