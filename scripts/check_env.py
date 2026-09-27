"""Check that this node can run the CUDA J-lens pipeline.

Prints GPU / library versions and — most importantly — which Gated
DeltaNet and causal-conv1d implementations transformers will use. Without
flash-linear-attention the 48 GDN layers fall back to transformers' pure
PyTorch reference path: correct, but >10x slower (fine for the web app,
painful for a 1000-prompt fit).

Run:  python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import sys


def _version(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed")
    except Exception as e:  # noqa: BLE001
        return f"NOT AVAILABLE ({type(e).__name__}: {e})"


def main() -> int:
    ok = True
    import torch

    print(f"python        {sys.version.split()[0]}")
    print(f"torch         {torch.__version__} (cuda {torch.version.cuda})")
    print(f"transformers  {_version('transformers')}")
    print(f"jlens         {_version('jlens')}")
    print(f"fla           {_version('fla')}")
    print(f"causal_conv1d {_version('causal_conv1d')}")
    print(f"triton        {_version('triton')}")

    if not torch.cuda.is_available():
        print("\n[FAIL] torch.cuda.is_available() is False — are you on a GPU node?")
        return 1
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"GPU {i}: {p.name}  {total / 1e9:.0f} GB total, {free / 1e9:.0f} GB free, "
              f"sm_{p.major}{p.minor}, bf16={torch.cuda.is_bf16_supported()}")

    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        fn = getattr(mq, name)
        # use_kernel_func_from_hub_with_fallback closes over the chosen
        # `implementation` and an `is_new_implementation` flag.
        free = dict(zip(fn.__code__.co_freevars,
                        (c.cell_contents for c in fn.__closure__ or [])))
        impl = free.get("implementation")
        chosen = impl if free.get("is_new_implementation") else None
        where = (f"{chosen.__module__}.{getattr(chosen, '__name__', chosen)}"
                 if chosen is not None else "transformers torch fallback")
        print(f"{name:34s} -> {where}")
        if name == "torch_chunk_gated_delta_rule" and chosen is None:
            ok = False

    if not ok:
        print("\n[WARN] Gated DeltaNet is on the slow torch fallback. "
              "`pip install flash-linear-attention` before fitting.")
    else:
        print("\n[OK] fla Triton kernels will be used for Gated DeltaNet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
