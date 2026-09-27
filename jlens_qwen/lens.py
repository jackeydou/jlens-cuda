"""JacobianLens: hold fitted J_l matrices, apply them to read out J-space tokens.

Torch port of the upstream MLX ``lens.py``. Reads and writes both on-disk
formats:

- ``.npz`` — this project's / upstream's schema: ``J_<l>`` fp16 arrays plus
  ``n_prompts`` and ``d_model`` (+ a ``.json`` sidecar).
- ``.pt``  — Anthropic ``jlens`` schema (what ``jlens.fit`` and Neuronpedia
  publish): ``{"J": {l: Tensor}, "n_prompts", "d_model", "source_layers"}``.

Both use the ``J @ h`` transport orientation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np
import torch

_DTYPES = {"float32": torch.float32, "fp32": torch.float32,
           "float16": torch.float16, "fp16": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


class JacobianLens:
    """A fitted Jacobian lens: per-layer J_l matrices and the readout method.

    ``jacobians`` holds CPU tensors (whatever precision was loaded);
    ``jacobian_gpu(l)`` memoizes a device copy in ``JLENS_J_DTYPE``
    (default float32 — 6.6 GB for 63 layers at d=5120, cheap on an H200).
    """

    def __init__(
        self,
        jacobians: dict[int, np.ndarray | torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
        device: str | torch.device | None = None,
    ) -> None:
        self.jacobians = {
            int(l): (J if torch.is_tensor(J) else torch.from_numpy(np.asarray(J)))
            for l, J in jacobians.items()
        }
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = int(n_prompts)
        self.d_model = int(d_model)
        self.device = torch.device(
            device or os.environ.get("JLENS_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gpu_dtype = _DTYPES[os.environ.get("JLENS_J_DTYPE", "float32")]
        self._gpu: dict[int, torch.Tensor] = {}

    def __repr__(self) -> str:
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)} layers))"
        )

    # ------------------------------------------------------------ disk io

    def save(self, path: str) -> None:
        """Save as ``.npz`` (upstream schema, fp16) or ``.pt`` (jlens schema)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".pt"):
            torch.save({
                "J": {l: J.to(torch.float16) for l, J in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            }, path)
            return
        J_fp16 = {f"J_{l}": J.to(torch.float16).numpy() for l, J in self.jacobians.items()}
        np.savez(path, **J_fp16, n_prompts=self.n_prompts, d_model=self.d_model)
        with open(path.replace(".npz", ".json"), "w") as f:
            json.dump({"n_prompts": self.n_prompts, "d_model": self.d_model,
                       "source_layers": self.source_layers}, f)

    @classmethod
    def load(cls, path: str, *, layers: Sequence[int] | None = None) -> "JacobianLens":
        """Load a ``.npz`` or ``.pt`` lens (optionally only some layers)."""
        want = set(layers) if layers is not None else None
        if path.endswith(".pt"):
            ck = torch.load(path, map_location="cpu", weights_only=True)
            if "J" not in ck:
                raise ValueError(f"{path} is not a JacobianLens file (keys {sorted(ck)})")
            J = {int(l): t for l, t in ck["J"].items() if want is None or int(l) in want}
            return cls(J, n_prompts=ck["n_prompts"], d_model=ck["d_model"])
        data = np.load(path, allow_pickle=False)
        J = {}
        for key in data.files:
            if key.startswith("J_"):
                l = int(key.split("_")[1])
                if want is None or l in want:
                    J[l] = torch.from_numpy(data[key])
        return cls(J, n_prompts=int(data["n_prompts"]), d_model=int(data["d_model"]))

    # ---------------------------------------------------------- transport

    def warm(self) -> None:
        """Upload every J_l to the device now (instead of on first request)."""
        for l in self.source_layers:
            self.jacobian_gpu(l)

    def jacobian_gpu(self, layer: int) -> torch.Tensor:
        """Memoized device copy of J_layer, [D, D]."""
        if layer not in self.jacobians:
            raise KeyError(f"layer {layer} not in source_layers {self.source_layers}")
        J = self._gpu.get(layer)
        if J is None:
            J = self.jacobians[layer].to(self.device, dtype=self.gpu_dtype)
            self._gpu[layer] = J
        return J

    # Upstream name (used by ported code).
    jacobian_mx = jacobian_gpu

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a residual at `layer` into the final-layer basis: ``J_l @ h``.

        residual: [..., D]. Returns fp32 [..., D].
        """
        J = self.jacobian_gpu(layer)
        with torch.inference_mode():
            return (residual.to(J.dtype) @ J.T).float()

    def apply(
        self,
        model,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> dict:
        """Run model on prompt, return lens logits at each requested layer.

        Returns dict with ``lens_logits`` ({layer: [S, vocab]}),
        ``model_logits`` ([S, vocab]), ``input_ids`` and ``token_strs``.
        """
        if layers is None:
            layers = self.source_layers
        if use_jacobian:
            unknown = set(layers) - set(self.source_layers)
            if unknown:
                raise ValueError(f"layers {sorted(unknown)} not fitted")
        final_layer = model.n_layers - 1
        input_ids = model.encode(prompt, max_length=max_seq_len)
        _, acts = model.forward(input_ids, capture_layers=list(layers) + [final_layer])

        lens_logits = {}
        for layer in layers:
            h = acts[layer][0].float()
            if use_jacobian and layer in self.jacobians:
                h = self.transport(h, layer)
            lens_logits[layer] = model.unembed(model.final_norm(h))
        model_logits = model.unembed(model.final_norm(acts[final_layer][0].float()))
        token_strs = [model.tokenizer.decode([int(t)]) for t in input_ids[0].tolist()]
        return {"lens_logits": lens_logits, "model_logits": model_logits,
                "input_ids": input_ids, "token_strs": token_strs}


def topk_tokens(logits: torch.Tensor, k: int = 10) -> tuple[list[int], list[float]]:
    """Top-k (ids, scores) from logits [vocab] or [seq, vocab] (last position)."""
    if logits.ndim == 2:
        logits = logits[-1]
    vals, ids = torch.topk(logits.float(), k)
    return ids.tolist(), vals.tolist()
