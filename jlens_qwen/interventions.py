"""J-space interventions: steer, swap, ablate.

Torch port of the upstream MLX ``interventions.py`` (same math, same API).
Each op modifies a residual-stream activation h_ℓ at one layer+position to
change what the model says.

J-lens vectors
--------------
For a token t, the J-lens vector at layer ℓ is

    v_t = J_ℓᵀ @ W_U[t, :]

so that <v_t, h> = <W_U[t], J_ℓ h>: adding v_t to h_ℓ makes the model more
likely to say t (averaged over contexts). W_U is the dense bf16 lm_head
weight (upcast to fp32) — no dequantization needed on the CUDA path.

- steer:  h += alpha * v_t              (alpha < 0 suppresses t)
- swap:   exchange the lens coordinate of s for t, orthogonal part unchanged
- ablate: remove span{v_t : t in S} from h (least-squares projection)
"""

from __future__ import annotations

import functools
from typing import Sequence

import torch

from .lens import JacobianLens
from .model import TorchLensModel


def get_unembedding_matrix(model: TorchLensModel) -> torch.Tensor:
    """Dense W_U [vocab, d_model] fp32 (shared with the readout path)."""
    return model.W_U


@functools.lru_cache(maxsize=4)
def j_lens_vectors(lens: JacobianLens, model: TorchLensModel, layer: int) -> torch.Tensor:
    """All J-lens vectors for layer ℓ: V = W_U @ J_ℓ, [vocab, d_model] fp32.

    SCRIPT-ONLY: ~5 GB per layer. The server path uses j_lens_vectors_lite().
    """
    if layer not in lens.jacobians:
        raise KeyError(f"layer {layer} not fitted (source_layers={lens.source_layers})")
    with torch.inference_mode():
        return get_unembedding_matrix(model) @ lens.jacobian_gpu(layer).float()


def j_lens_vector(lens: JacobianLens, model: TorchLensModel, layer: int, token_id: int) -> torch.Tensor:
    """The J-lens vector for a single token at a layer, [d_model]."""
    return j_lens_vectors_lite(lens, model, layer, [token_id])[0]


def j_lens_vector_for_text(lens, model, layer, text: str) -> torch.Tensor:
    """The J-lens vector for the first token of `text` at `layer`."""
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"text {text!r} encodes to no tokens")
    return j_lens_vector(lens, model, layer, ids[0])


def unembed_rows(model: TorchLensModel, token_ids: Sequence[int]) -> torch.Tensor:
    """Rows `token_ids` of W_U, [k, d_model] fp32."""
    idx = torch.tensor(list(token_ids), device=model.device)
    with torch.inference_mode():
        return model.W_U[idx]


def j_lens_vectors_lite(
    lens: JacobianLens | None,
    model: TorchLensModel,
    layer: int,
    token_ids: Sequence[int],
) -> torch.Tensor:
    """J-lens vectors v_t = J_ℓᵀ W_U[t] for a few tokens, [k, d_model] fp32.

    The final (unfitted) layer reads out through the plain logit lens, so
    there v_t = W_U[t].
    """
    W_rows = unembed_rows(model, token_ids)
    if lens is None or layer not in lens.jacobians:
        if layer != model.n_layers - 1:
            fitted = lens.source_layers if lens is not None else []
            raise KeyError(f"layer {layer} has no fitted Jacobian (source_layers={fitted})")
        return W_rows
    J = lens.jacobian_gpu(layer)
    with torch.inference_mode():
        return (W_rows.to(J.dtype) @ J).float()


def steer(h: torch.Tensor, v_t: torch.Tensor, alpha: float) -> torch.Tensor:
    """h += alpha * v_t."""
    return h + alpha * v_t.to(h.dtype)


def make_swap_basis(v_s: torch.Tensor, v_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """V = [v_s, v_t] [2, D] and inv(V Vᵀ) [2, 2]."""
    V = torch.stack([v_s, v_t], dim=0).float()
    inv = torch.linalg.inv(V @ V.T)
    return V, inv


def patch_swap(h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Swap the lens coordinates of s and t in h (single row [D])."""
    V, inv = make_swap_basis(v_s, v_t)
    return patch_swap_rows(h.float()[None, :], V, inv, alpha)[0].to(h.dtype)


def patch_swap_rows(
    h: torch.Tensor, V: torch.Tensor, VVt_inv: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """Batched patch_swap over rows: h [n, D] -> [n, D].

    c = (V Vᵀ)⁻¹ V h are the lens coordinates; they are swapped (scaled by
    alpha) and the difference written back along V.
    """
    co = VVt_inv @ (V @ h.T)                                     # [2, n]
    swapped = torch.stack([alpha * co[1], alpha * co[0]], dim=0)  # [2, n]
    return h + (swapped - co).T @ V                              # [n, D]


def gram_inv(V: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Ridge-regularized inverse of the Gram matrix V Vᵀ, [k, k] fp32."""
    G = (V @ V.T).double()
    k = G.shape[0]
    reg = ridge * (torch.trace(G) / max(k, 1))
    eye = torch.eye(k, dtype=G.dtype, device=G.device)
    return torch.linalg.inv(G + reg * eye).float()


def ablate_rows(
    h: torch.Tensor, V: torch.Tensor, G_inv: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """Remove the span{V} lens component from each row of h [n, D]."""
    co = G_inv @ (V @ h.T)          # [k, n]
    return h - alpha * (co.T @ V)   # [n, D]


def ablate_topk(
    h: torch.Tensor,
    lens: JacobianLens,
    model: TorchLensModel,
    layer: int,
    k: int = 16,
    n_iters: int = 5,
) -> torch.Tensor:
    """Remove the top-k J-space component of h [D] by greedy pursuit (script path)."""
    V = j_lens_vectors(lens, model, layer)                 # [vocab, D]
    V_norms = V.norm(dim=-1) + 1e-8
    residual = h.float()
    accumulated = torch.zeros_like(residual)
    chosen: list[int] = []
    for _ in range(min(k, n_iters * 4)):
        normalized = (V @ residual) / V_norms
        if chosen:
            normalized[torch.tensor(chosen, device=normalized.device)] = -1e9
        best = int(torch.argmax(normalized).item())
        chosen.append(best)
        v = V[best]
        component = (v @ residual) / (v @ v) * v
        accumulated = accumulated + component
        residual = residual - component
        if len(chosen) >= k:
            break
    return h.float() - accumulated


def compile_edits(
    lens: JacobianLens | None,
    model: TorchLensModel,
    *,
    mode: str,
    layers: Sequence[int],
    token_id: int | None = None,
    target_id: int | None = None,
    alpha: float = 1.0,
    positions: Sequence[int] | None = None,
    from_pos: int | None = None,
    ablate_token_ids: Sequence[int] | None = None,
    label: str = "",
) -> list:
    """Compile one intervention spec into per-layer LayerEdits.

    All vectors, bases and inverses are computed HERE, once per request —
    the returned closures are pure GPU matmuls, safe to run per decode step.
    """
    from .model import LayerEdit

    edits: list[LayerEdit] = []
    pos_t = tuple(positions) if positions is not None else None
    with torch.inference_mode():
        for layer in layers:
            if mode == "steer":
                if token_id is None:
                    raise ValueError("steer requires token_id")
                v = j_lens_vectors_lite(lens, model, layer, [token_id])[0]
                fn = lambda h, v=v, a=alpha: h + a * v
            elif mode == "swap":
                if token_id is None or target_id is None:
                    raise ValueError("swap requires token_id and target_id")
                vs = j_lens_vectors_lite(lens, model, layer, [token_id, target_id])
                V, inv = make_swap_basis(vs[0], vs[1])
                fn = lambda h, V=V, inv=inv, a=alpha: patch_swap_rows(h, V, inv, a)
            elif mode == "ablate":
                if not ablate_token_ids:
                    raise ValueError("ablate requires ablate_token_ids")
                V = j_lens_vectors_lite(lens, model, layer, list(ablate_token_ids))
                G_inv = gram_inv(V)
                fn = lambda h, V=V, G=G_inv, a=alpha: ablate_rows(h, V, G, a)
            else:
                raise ValueError(f"unknown intervention mode {mode!r}")
            edits.append(LayerEdit(
                layer=layer, fn=fn, positions=pos_t, from_pos=from_pos, label=label,
            ))
    return edits
