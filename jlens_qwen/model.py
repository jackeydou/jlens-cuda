"""PyTorch / CUDA LensModel adapter for Qwen3.5-architecture models (bf16).

CUDA port of the upstream MLX adapter (WeZZard/jlens-qwen36,
``jlens_qwen/model.py``). Same public surface, so ``serve.py`` and the web
UI run unchanged on top of it:

- ``TorchLensModel.forward(input_ids, capture_layers)`` -> (final_normed, acts)
- ``TorchLensModel.unembed`` / ``final_norm`` / ``encode`` / ``generate``
- ``StreamSession.extend(ids)`` -> (last-position logits, per-layer acts)
  with residual-stream edits (``LayerEdit``) applied inside the forward.

Differences from the MLX version:

- The model is the stock HuggingFace ``Qwen3_5ForConditionalGeneration``
  in bf16 (Qwen/Qwen3.8-27B, Qwen/Qwen3.6-27B, ...). No forward rewrite:
  residuals are captured and edited with forward hooks on the decoder
  layers, and the model's own hybrid ``DynamicCache`` (KV for the 16
  full-attention layers, conv + recurrent state for the 48 Gated DeltaNet
  layers) gives O(T) cached decoding.
- Gated DeltaNet runs on flash-linear-attention's Triton kernels when the
  ``fla`` package is installed (transformers picks them up automatically);
  they have a backward pass, so fitting needs no custom kernel.
- Readout unembedding uses an fp32 copy of ``lm_head`` (TF32 matmul) so
  readout scores are not rounded to bf16; W_U is dense, no dequantization.

Row edits and readouts run under ``torch.inference_mode``; that context
is thread-local, so every entry point that serve.py calls through
``asyncio.to_thread`` enters it itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B"


@dataclass(frozen=True)
class LayerEdit:
    """A compiled residual-stream edit for the cached streaming path.

    At `layer`, the rows of the current chunk selected by the position
    scope are rewritten by ``fn`` ([n, D] fp32 -> [n, D] fp32). Position
    scopes are GLOBAL indices into the chat-templated token sequence
    (the same coordinates the readout grid uses); ``extend()`` maps them
    onto each chunk via its ``n_consumed`` offset.

    positions: explicit global positions, or None.
    from_pos: persistent scope — every global position >= from_pos, or None.
    The two scopes are unioned when both are set.
    """

    layer: int
    fn: Callable[[torch.Tensor], torch.Tensor]
    positions: tuple[int, ...] | None = None
    from_pos: int | None = None
    label: str = ""


def _chunk_local_indices(
    positions: Sequence[int] | None,
    from_pos: int | None,
    start: int,
    n: int,
) -> list[int]:
    """Map global position scopes onto a chunk covering ``[start, start+n)``.

    Returns sorted chunk-local row indices; empty when the scope misses
    the chunk entirely.
    """
    end = start + n
    sel: set[int] = set()
    if positions is not None:
        sel.update(p - start for p in positions if start <= p < end)
    if from_pos is not None and from_pos < end:
        sel.update(range(max(from_pos - start, 0), n))
    return sorted(sel)


def _apply_edits(
    hidden: torch.Tensor, edits: Sequence[LayerEdit], start: int
) -> torch.Tensor:
    """Apply same-layer edits to ``hidden`` [1, n, D], in list order.

    Gathers the selected rows, runs each edit fn in fp32, and scatters the
    result back into a copy (the layer's output tensor is left untouched).
    """
    n = hidden.shape[1]
    out = None
    for e in edits:
        local = _chunk_local_indices(e.positions, e.from_pos, start, n)
        if not local:
            continue
        if out is None:
            out = hidden.clone()
        idx = torch.tensor(local, device=hidden.device)
        rows = e.fn(out[0, idx].float())
        out[0, idx] = rows.to(out.dtype)
    return hidden if out is None else out


class ChatTokenizer:
    """Thin proxy over the HF tokenizer.

    transformers v5 returns a ``BatchEncoding`` from
    ``apply_chat_template(..., tokenize=True)``; serve.py (written against
    mlx-lm's wrapper) expects a flat ``list[int]``. Everything else is
    forwarded unchanged.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tok, name)

    def __len__(self) -> int:
        return len(self._tok)

    @property
    def hf(self) -> Any:
        """The wrapped HuggingFace tokenizer."""
        return self._tok

    def apply_chat_template(self, conversation, *args, **kwargs):
        out = self._tok.apply_chat_template(conversation, *args, **kwargs)
        if kwargs.get("tokenize", True) is False:
            return out
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if hasattr(out, "tolist"):
            out = out.tolist()
        if isinstance(out, (list, tuple)) and len(out) == 1 and isinstance(out[0], (list, tuple)):
            out = out[0]
        return [int(t) for t in out]


def _find_text_module(hf_model: torch.nn.Module) -> torch.nn.Module:
    """The bare text decoder (has .layers / .norm / .embed_tokens).

    Qwen3_5ForConditionalGeneration -> model.language_model;
    Qwen3_5ForCausalLM -> model.
    """
    for path in ("model.language_model", "model", "language_model"):
        obj = hf_model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if all(hasattr(obj, a) for a in ("layers", "norm", "embed_tokens")):
            return obj
    raise ValueError(f"cannot locate the text decoder inside {type(hf_model).__name__}")


class TorchLensModel:
    """A HuggingFace Qwen3.5-family model exposed through the upstream
    ``MLXLensModel`` interface.

    Attributes:
        n_layers: Number of residual blocks (64 for the 27B models).
        d_model: Residual-stream width (5120).
        layers: The decoder blocks (``text_model.layers``).
        tokenizer: ``ChatTokenizer`` over the HF tokenizer.
        device / dtype: Where and in what precision the weights live.
    """

    def __init__(self, hf_model: torch.nn.Module, tokenizer: Any, *,
                 unembed_fp32: bool = True) -> None:
        hf_model.eval()
        for p in hf_model.parameters():
            p.requires_grad_(False)
        self._hf_model = hf_model
        self.tokenizer = ChatTokenizer(tokenizer)
        self._text_module = _find_text_module(hf_model)
        self.layers = self._text_module.layers
        self.n_layers = len(self.layers)
        self.d_model = self._text_module.config.hidden_size
        self._lm_head = hf_model.lm_head
        self.device = self._text_module.embed_tokens.weight.device
        self.dtype = self._text_module.embed_tokens.weight.dtype
        # fp32 unembedding for readouts (~5 GB for the 248k vocab). TF32
        # keeps the matmul on tensor cores; outputs stay fp32.
        torch.backends.cuda.matmul.allow_tf32 = True
        self._W_U = self._lm_head.weight.float() if unembed_fp32 else None

    def __repr__(self) -> str:
        return (f"TorchLensModel({type(self._hf_model).__name__}, n_layers={self.n_layers}, "
                f"d_model={self.d_model}, device={self.device}, dtype={self.dtype})")

    @property
    def hf_model(self) -> torch.nn.Module:
        return self._hf_model

    @property
    def W_U(self) -> torch.Tensor:
        """Dense unembedding [vocab, d_model] fp32 (lm_head.weight)."""
        if self._W_U is None:
            self._W_U = self._lm_head.weight.float()
        return self._W_U

    # ------------------------------------------------------------------ io

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        """Tokenize ``text`` to ``input_ids`` of shape ``[1, seq_len]``."""
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > max_length:
            ids = ids[-max_length:]  # keep the tail (matches upstream)
        return torch.tensor([ids], device=self.device)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """LM head only: ``[..., d_model]`` -> fp32 logits ``[..., vocab]``.

        Callers apply ``final_norm`` first for the paper's
        ``W_U · norm(J_ℓ h_ℓ)`` readout (same contract as upstream).
        """
        with torch.inference_mode():
            if self._W_U is not None:
                return F.linear(residual.float(), self._W_U)
            return self._lm_head(residual.to(self.dtype)).float()

    def final_norm(self, residual: torch.Tensor) -> torch.Tensor:
        """The model's final pre-unembed RMSNorm (computed in fp32 for fp32 input)."""
        with torch.inference_mode():
            return self._text_module.norm(residual)

    # ------------------------------------------------------------- forward

    def _run(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Any = None,
        capture: set[int] | None = None,
        edits_by_layer: dict[int, list[LayerEdit]] | None = None,
        start: int = 0,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """One pass through the text decoder with capture/edit hooks.

        Returns (final-norm residual [B, S, D], {layer: residual after layer}).
        """
        capture = capture or set()
        edits_by_layer = edits_by_layer or {}
        acts: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(i: int):
            ed = edits_by_layer.get(i)

            def hook(module, args, output):
                is_t = torch.is_tensor(output)
                h = output if is_t else output[0]
                if ed:
                    h = _apply_edits(h, ed, start)
                if i in capture:
                    acts[i] = h
                if ed:
                    return h if is_t else (h, *output[1:])
                return None

            return hook

        if -1 in capture:
            handles.append(self._text_module.embed_tokens.register_forward_hook(
                lambda m, a, out: acts.__setitem__(-1, out)))
        for i in sorted(set(capture) | set(edits_by_layer)):
            if 0 <= i < self.n_layers:
                handles.append(self.layers[i].register_forward_hook(make_hook(i)))
        try:
            out = self._text_module(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=cache is not None,
            )
        finally:
            for h in handles:
                h.remove()
        return out.last_hidden_state, acts

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        capture_layers: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Full-sequence forward (no cache, no grad).

        Returns ``(final_residual, layer_acts)``: the post-final-norm
        residual ``[B, S, D]`` and ``layer_acts[i]`` = residual AFTER layer
        ``i`` (the embedding is keyed ``-1`` if requested).
        """
        with torch.inference_mode():
            return self._run(input_ids, capture=set(capture_layers or []))

    def forward_with_edits(
        self, input_ids: torch.Tensor, edits: Sequence[LayerEdit]
    ) -> torch.Tensor:
        """Uncached full-sequence forward with residual edits; returns final-norm residual."""
        by_layer: dict[int, list[LayerEdit]] = {}
        for e in edits:
            by_layer.setdefault(e.layer, []).append(e)
        with torch.inference_mode():
            final, _ = self._run(input_ids, edits_by_layer=by_layer)
        return final

    def forward_with_intervention(
        self,
        input_ids: torch.Tensor,
        intervene_layer: int,
        intervene_positions: list[int] | None,
        patched_h_fn,
    ) -> torch.Tensor:
        """Forward with ``patched_h_fn`` ([D] -> [D]) applied per position at one layer."""
        S = int(input_ids.shape[1])
        positions = tuple(intervene_positions) if intervene_positions is not None else tuple(range(S))

        def rowwise(rows: torch.Tensor) -> torch.Tensor:
            return torch.stack([patched_h_fn(r) for r in rows])

        edit = LayerEdit(layer=intervene_layer, fn=rowwise, positions=positions)
        return self.forward_with_edits(input_ids, [edit])

    # ------------------------------------------------------------ generate

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 64,
        temp: float = 0.0,
        intervene_layer: int | None = None,
        intervene_fn=None,
        intervene_positions: list[int] | None = None,
        intervene_each_step: bool = False,
    ) -> tuple[str, list[int]]:
        """Generate a continuation of `prompt` (greedy if temp == 0).

        Same semantics as upstream: with an intervention, it applies only on
        the first (prefill) pass unless ``intervene_each_step``.
        """
        input_ids = self.encode(prompt, max_length=512)
        generated: list[int] = []
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        def _sample(lf: torch.Tensor) -> int:
            lf = lf.float()
            if temp == 0:
                return int(torch.argmax(lf).item())
            probs = torch.softmax(lf / temp, dim=-1)
            return int(torch.multinomial(probs, 1).item())

        if intervene_layer is None or intervene_fn is None:
            session = self.make_stream()
            logits, _ = session.extend(input_ids[0].tolist())
            for _ in range(max_tokens):
                next_tok = _sample(logits)
                generated.append(next_tok)
                if eos_id is not None and next_tok == eos_id:
                    break
                logits, _ = session.extend([next_tok])
            return self.tokenizer.decode(generated), generated

        for step in range(max_tokens):
            if step == 0 or intervene_each_step:
                final = self.forward_with_intervention(
                    input_ids, intervene_layer, intervene_positions, intervene_fn)
            else:
                final, _ = self.forward(input_ids)
            next_tok = _sample(self.unembed(final[:, -1, :])[0])
            generated.append(next_tok)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_tok]], device=self.device)], dim=1)
            if eos_id is not None and next_tok == eos_id:
                break
        return self.tokenizer.decode(generated), generated

    def make_stream(self, capture_layers=None) -> "StreamSession":
        """Create a cached incremental-decoding session (see StreamSession)."""
        return StreamSession(self, capture_layers=capture_layers)


class StreamSession:
    """Cached incremental decoding with per-layer residual capture.

    Uses the model's own hybrid cache (KV for full-attention layers,
    conv + recurrent state for Gated DeltaNet layers), so each new chunk
    costs one pass over its own tokens — O(T) total decode.
    """

    def __init__(self, model: TorchLensModel, capture_layers=None) -> None:
        from transformers import DynamicCache

        self._model = model
        self.cache = DynamicCache(config=model._text_module.config)
        self.capture = sorted(set(capture_layers or []))
        self.n_consumed = 0
        self._edits_by_layer: dict[int, list[LayerEdit]] = {}

    def set_edits(self, edits: Sequence[LayerEdit] | None) -> None:
        """Install residual-stream edits applied inside every ``extend()``.

        Edits are grouped by layer and applied in list order within a
        layer, right after that layer's forward and BEFORE its residual
        is captured — so downstream layers (and their caches) consume the
        edited stream, and readouts show the written values.
        """
        by_layer: dict[int, list[LayerEdit]] = {}
        for e in edits or []:
            by_layer.setdefault(e.layer, []).append(e)
        self._edits_by_layer = by_layer

    def extend(self, ids) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Feed token ids (a Python list) after the tokens already consumed.

        Returns ``(logits, acts)``: the last position's fp32 vocab logits
        ``[vocab]`` and ``acts[l]`` = residual after layer ``l`` for the new
        tokens only, ``[1, len(ids), D]``.
        """
        m = self._model
        with torch.inference_mode():
            input_ids = torch.tensor([list(ids)], device=m.device)
            final, acts = m._run(
                input_ids,
                cache=self.cache,
                capture=set(self.capture),
                edits_by_layer=self._edits_by_layer,
                start=self.n_consumed,
            )
            logits = m.unembed(final[:, -1, :])[0]
        self.n_consumed += len(ids)
        return logits, acts


def load_hf(
    model_id: str | None = None,
    *,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the HF model + tokenizer (bf16, one device). Returns (hf_model, tokenizer)."""
    from transformers import AutoConfig, AutoTokenizer

    model_id = model_id or os.environ.get("JLENS_MODEL", DEFAULT_MODEL_ID)
    device = device or os.environ.get("JLENS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "vision_config"):
        from transformers import AutoModelForImageTextToText as AutoCls
    else:
        from transformers import AutoModelForCausalLM as AutoCls
    hf_model = AutoCls.from_pretrained(model_id, dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return hf_model, tokenizer


def load(model_id: str | None = None, **kwargs) -> TorchLensModel:
    """Load the model and wrap it as a TorchLensModel."""
    hf_model, tokenizer = load_hf(model_id, **kwargs)
    return TorchLensModel(hf_model, tokenizer)


# Upstream name, so ported scripts keep working.
MLXLensModel = TorchLensModel

__all__ = ["LayerEdit", "TorchLensModel", "MLXLensModel", "StreamSession",
           "ChatTokenizer", "load", "load_hf", "DEFAULT_MODEL_ID"]
