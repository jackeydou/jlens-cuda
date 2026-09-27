"""Verify the fitting path on the real model, and measure what a full fit costs.

Run this once on an H200 BEFORE launching the 1000-prompt fit:

1. Correctness: rows of J_l from the fitting estimator (replicated batch,
   retained graph, fla backward reused ceil(D/dim_batch) times) are compared
   against a fresh batch-1 forward + backward per row. bf16 kernels are not
   bit-identical across batch sizes, so the check is cosine similarity
   (expect > 0.99) and relative error (expect a few %).
2. Cost: the same full-depth prompt (63 source layers) with your --dim-batch,
   reporting seconds/prompt, peak GPU memory, and the projected wall time
   for N prompts on G GPUs.

Run:  python scripts/verify_fit.py --dim-batch 16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens_qwen.model import DEFAULT_MODEL_ID, load_hf  # noqa: E402

PROMPT = (
    "The Roman Empire was one of the largest and most influential civilizations in "
    "world history. At its peak, it controlled territories spanning from Britain to "
    "Egypt and from Spain to the Middle East, leaving a lasting legacy in law, "
    "language, architecture, and governance. Its roads, aqueducts and cities were "
    "copied for centuries, and Latin, the language of its administration, became "
    "the root of the Romance languages spoken by hundreds of millions of people "
    "today, from Portugal and Spain to France, Italy and Romania."
)


def independent_row(lm, input_ids, layer, target_layer, valid, d):
    """d-th row of J_layer from a fresh batch-1 forward + one backward."""
    acts = {}

    def cap(i, root):
        def hook(m, a, out):
            t = out if torch.is_tensor(out) else out[0]
            if root:
                t.requires_grad_(True)
            acts[i] = t
        return hook

    hs = [lm.layers[layer].register_forward_hook(cap(layer, True)),
          lm.layers[target_layer].register_forward_hook(cap(target_layer, False))]
    try:
        with torch.enable_grad():
            lm.forward(input_ids)
            target = acts[target_layer][0, valid, d].float().sum()
            (g,) = torch.autograd.grad(target, acts[layer])
    finally:
        for h in hs:
            h.remove()
    return g[0, valid].float().mean(0).cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("JLENS_MODEL", DEFAULT_MODEL_ID))
    ap.add_argument("--device", default="cuda", help="cuda (default); cpu only for smoke tests")
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--compile", action="store_true", help="torch.compile each decoder layer (as fit_lens.py --compile)")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--check-layers", default="0,31,62")
    ap.add_argument("--check-dims", default="0,1234,5119")
    ap.add_argument("--n-prompts", type=int, default=1000, help="for the time projection")
    ap.add_argument("--n-gpus", type=int, default=8, help="for the time projection")
    args = ap.parse_args()

    import jlens

    hf_model, tokenizer = load_hf(args.model, device=args.device)
    lm = jlens.from_hf(hf_model, tokenizer, compile=args.compile)
    print(lm, flush=True)
    target = lm.n_layers - 1
    check_layers = [int(x) for x in args.check_layers.split(",")]
    check_dims = [int(x) for x in args.check_dims.split(",")]

    from jlens.fitting import jacobian_for_prompt

    # ---- 1. one full-depth prompt through the fitting estimator (timed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    J, S, n_valid = jacobian_for_prompt(
        lm, PROMPT, list(range(target)), dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len, skip_first=args.skip_first)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
    print(f"full depth ({target} layers), seq_len={S}, valid={n_valid}, "
          f"dim_batch={args.dim_batch}: {sec:.0f} s/prompt, peak GPU memory {peak:.1f} GB "
          f"({math.ceil(lm.d_model / args.dim_batch)} backward passes)", flush=True)

    # ---- 2. correctness: estimator rows vs fresh batch-1 forward+backward
    input_ids = lm.encode(PROMPT, max_length=args.max_seq_len)
    valid = torch.arange(args.skip_first, input_ids.shape[1] - 1, device=input_ids.device)
    worst_cos = 1.0
    for l in check_layers:
        for d in check_dims:
            ref = independent_row(lm, input_ids, l, target, valid, d)
            got = J[l][d]
            cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
            rel = ((got - ref).norm() / ref.norm()).item()
            worst_cos = min(worst_cos, cos)
            print(f"  J_{l:<2d} row {d:<4d}  cos={cos:.5f}  rel_err={rel:.3%}  "
                  f"|row|={ref.norm():.3e}", flush=True)
    verdict = "OK" if worst_cos > 0.99 else "MISMATCH — do not fit until this is understood"
    print(f"correctness: worst cosine {worst_cos:.5f} -> {verdict}", flush=True)

    # ---- 3. projection
    total_h = sec * args.n_prompts / args.n_gpus / 3600
    print(f"\nprojection: {args.n_prompts} prompts on {args.n_gpus} GPUs ~ {total_h:.1f} h wall "
          "(+ model load per process)")
    print(f"host RAM per fit process: ~{2 * target * lm.d_model ** 2 * 4 / 1e9:.0f} GB "
          "(per-prompt J + running sum, fp32)")


if __name__ == "__main__":
    main()
