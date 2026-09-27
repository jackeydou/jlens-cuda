"""Numerical tests of the torch backend on a tiny random Qwen3.5 model.

No weights download, runs on CPU in seconds (the GDN layers use the
transformers torch fallback here; on the H200 the same code runs fla's
Triton kernels). Checks:

- StreamSession chunked, cached decoding == one uncached full forward
  (per-layer residuals and logits), for the hybrid GDN + attention cache.
- LayerEdit scoping: an edit applied through the cached stream equals the
  same edit applied in an uncached full forward.
- The Jacobian fit (Anthropic's jlens estimator, replicated batch +
  retained graph) matches an independent per-row autograd computation.
- JacobianLens .npz / .pt round trips and transport orientation.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

from jlens_qwen.lens import JacobianLens  # noqa: E402
from jlens_qwen.model import LayerEdit, TorchLensModel  # noqa: E402

VOCAB = 97
D = 64


class _Tok:
    """Minimal tokenizer: text is a space-separated list of ints."""

    eos_token_id = 0
    bos_token_id = None

    def encode(self, text, add_special_tokens=True):
        return [int(t) for t in text.split()]

    def decode(self, ids, **kw):
        return " ".join(str(int(i)) for i in ids)

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = self.encode(text)[:max_length]

        class _E:
            input_ids = torch.tensor([ids])

        return _E()


def _tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=D,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True},
        tie_word_embeddings=False,
    )


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    hf = Qwen3_5ForCausalLM(_tiny_config()).float().eval()
    # Random init leaves RMSNorm weights at 0 (=> scale 1) and tiny
    # projections; perturb everything so the test exercises real mixing.
    with torch.no_grad():
        for p in hf.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return TorchLensModel(hf, _Tok())


def _ids(n: int, seed: int = 1) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB, (n,), generator=g).tolist()


def test_stream_matches_full_forward(tiny):
    ids = _ids(23)
    layers = list(range(tiny.n_layers))
    final, full = tiny.forward(torch.tensor([ids]), capture_layers=layers)
    full_logits = tiny.unembed(final[0, -1])

    sess = tiny.make_stream(capture_layers=layers)
    got = {l: [] for l in layers}
    # Uneven chunks: prefill, single-token decode steps, another chunk.
    for chunk in (ids[:9], ids[9:10], ids[10:11], ids[11:18], ids[18:19], ids[19:]):
        logits, acts = sess.extend(chunk)
        for l in layers:
            got[l].append(acts[l][0])
    assert sess.n_consumed == len(ids)
    for l in layers:
        torch.testing.assert_close(torch.cat(got[l]), full[l][0], atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(logits, full_logits, atol=2e-4, rtol=2e-4)


def test_stream_edits_match_full_forward_edits(tiny):
    ids = _ids(20, seed=2)
    delta = torch.randn(D, generator=torch.Generator().manual_seed(3))
    edits = [
        LayerEdit(layer=1, fn=lambda h: h + 3.0 * delta, positions=(4, 12)),
        LayerEdit(layer=2, fn=lambda h: 0.5 * h, from_pos=15),
    ]
    ref_final = tiny.forward_with_edits(torch.tensor([ids]), edits)

    sess = tiny.make_stream(capture_layers=[3])
    sess.set_edits(edits)
    outs = []
    for chunk in (ids[:10], ids[10:14], ids[14:15], ids[15:]):
        _, acts = sess.extend(chunk)
        outs.append(acts[3][0])
    stream_final = tiny.final_norm(torch.cat(outs))
    torch.testing.assert_close(stream_final, ref_final[0], atol=2e-4, rtol=2e-4)

    # And the edit actually changed something.
    clean, _ = tiny.forward(torch.tensor([ids]))
    assert (clean[0] - ref_final[0]).abs().max() > 1e-2


def test_jlens_fit_matches_independent_autograd(tiny):
    """The replicated-batch / retained-graph estimator (jlens.fitting) vs a
    fresh forward + backward per output row, for a few rows."""
    jlens = pytest.importorskip("jlens")
    lm = jlens.from_hf(tiny.hf_model, tiny.tokenizer.hf)
    ids = _ids(24, seed=4)
    prompt = " ".join(map(str, ids))
    skip = 4
    src = [0, 2]
    J, seq_len, n_valid = jlens.jacobian_for_prompt(
        lm, prompt, src, dim_batch=8, max_seq_len=64, skip_first=skip)
    assert seq_len == len(ids) and n_valid == len(ids) - 1 - skip

    valid = torch.arange(skip, len(ids) - 1)
    for layer in src:
        for d in (0, 17, D - 1):
            acts = {}
            hooks = []

            def cap(i, root):
                def hook(m, a, out):
                    if root:
                        out.requires_grad_(True)
                    acts[i] = out
                return hook

            hooks.append(tiny.layers[layer].register_forward_hook(cap(layer, True)))
            hooks.append(tiny.layers[tiny.n_layers - 1].register_forward_hook(cap(-1, False)))
            with torch.enable_grad():
                tiny._text_module(input_ids=torch.tensor([ids]), use_cache=False)
                target = acts[-1][0, valid, d].sum()
                (g,) = torch.autograd.grad(target, acts[layer])
            for h in hooks:
                h.remove()
            row = g[0, valid].mean(0)
            torch.testing.assert_close(J[layer][d], row, atol=1e-4, rtol=1e-3)


def test_lens_roundtrip_and_transport(tiny, tmp_path):
    g = torch.Generator().manual_seed(5)
    Js = {0: torch.randn(D, D, generator=g), 2: torch.randn(D, D, generator=g)}
    lens = JacobianLens(Js, n_prompts=3, d_model=D, device="cpu")
    for name in ("l.npz", "l.pt"):
        path = str(tmp_path / name)
        lens.save(path)
        back = JacobianLens.load(path)
        assert back.source_layers == [0, 2] and back.n_prompts == 3
        torch.testing.assert_close(back.jacobians[2].float(), Js[2].half().float())
    h = torch.randn(5, D, generator=g)
    torch.testing.assert_close(lens.transport(h, 2), h @ Js[2].T, atol=1e-4, rtol=1e-4)
    out = lens.apply(tiny, " ".join(map(str, _ids(12))), layers=[0, 2])
    assert out["lens_logits"][0].shape == (12, VOCAB)
