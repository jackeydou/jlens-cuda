"""Real-model gate on the H200 (port of upstream tests/test_intervention_stream.py).

Loads Qwen3.8-27B bf16 + a fitted lens and checks, end to end on CUDA:

- the cached StreamSession reproduces an uncached full forward (bf16
  tolerance) across prefill + single-token decode steps;
- empty edits are a no-op;
- the paper's France -> China swap redirects "Paris" through the streaming
  edit hook (the causal test for the lens + intervention path).

Heavy, so gated:

    JLENS_SLOW_TESTS=1 JLENS_PATH=data/lens/qwen38_27b_n1000.npz \
        uv run pytest tests/test_gpu_real_model.py -v -s
"""

from __future__ import annotations

import os

import pytest
import torch

LAYERS = [int(x) for x in os.environ.get("JLENS_TEST_LAYERS", "30,40,48").split(",")]
FRANCE_PROMPT = "Question: What is the capital of France?\nAnswer: The capital of France is"

pytestmark = pytest.mark.skipif(
    not os.environ.get("JLENS_SLOW_TESTS") or not torch.cuda.is_available(),
    reason="real-model GPU gate; set JLENS_SLOW_TESTS=1 on a GPU node",
)


@pytest.fixture(scope="session")
def model():
    from jlens_qwen.model import load

    return load()


@pytest.fixture(scope="session")
def lens():
    path = os.environ.get("JLENS_PATH", "data/lens/qwen38_27b_n1000.npz")
    if not os.path.exists(path):
        pytest.skip(f"lens missing at {path}")
    from jlens_qwen.lens import JacobianLens

    return JacobianLens.load(path, layers=LAYERS)


def _greedy(session, ids, n=6):
    logits, _ = session.extend(ids)
    out = []
    for _ in range(n):
        tok = int(torch.argmax(logits).item())
        out.append(tok)
        logits, _ = session.extend([tok])
    return out


def test_stream_matches_full_forward(model):
    ids = model.encode(FRANCE_PROMPT)[0].tolist()
    layers = [0, 20, 40, model.n_layers - 1]
    _, full = model.forward(torch.tensor([ids], device=model.device), capture_layers=layers)
    sess = model.make_stream(capture_layers=layers)
    parts = {l: [] for l in layers}
    for chunk in (ids[:-3], ids[-3:-2], ids[-2:-1], ids[-1:]):
        _, acts = sess.extend(chunk)
        for l in layers:
            parts[l].append(acts[l][0].float())
    for l in layers:
        got, ref = torch.cat(parts[l]), full[l][0].float()
        rel = ((got - ref).norm() / ref.norm()).item()
        print(f"L{l}: stream vs full rel err {rel:.2e}")
        assert rel < 3e-2, f"L{l} cached stream diverges from full forward (rel {rel:.3f})"


def test_empty_edits_identical(model):
    ids = model.encode(FRANCE_PROMPT)[0].tolist()
    plain = _greedy(model.make_stream(), ids)
    s = model.make_stream()
    s.set_edits([])
    assert _greedy(s, ids) == plain


def test_france_to_beijing_streaming(model, lens):
    from jlens_qwen.interventions import compile_edits

    ids = model.encode(FRANCE_PROMPT)[0].tolist()
    tok = model.tokenizer
    france_id = tok.encode(" France", add_special_tokens=False)[0]
    china_id = tok.encode(" China", add_special_tokens=False)[0]

    base_text = tok.decode(_greedy(model.make_stream(), ids))
    print(f"baseline: {base_text!r}")
    assert "paris" in base_text.lower(), f"unexpected baseline: {base_text!r}"

    edits = compile_edits(lens, model, mode="swap", layers=LAYERS,
                          token_id=france_id, target_id=china_id, alpha=1.0, from_pos=0)
    s = model.make_stream()
    s.set_edits(edits)
    text = tok.decode(_greedy(s, ids))
    print(f"swapped France->China at {LAYERS}: {text!r}")
    assert any(w in text.lower() for w in ("beijing", "china", "peking")), (
        f"swap did not redirect: {text!r} (baseline {base_text!r})")
