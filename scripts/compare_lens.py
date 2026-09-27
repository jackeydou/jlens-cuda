"""Compare two lenses fitted for the same model (e.g. ours vs Neuronpedia's).

Matrix level: per-layer relative Frobenius distance ||A - B|| / ||B|| and
cosine similarity of the flattened matrices. Two independent fits on
overlapping corpora agree to a few percent in the upper layers; the early
layers are noisier.

Readout level (with --model): top-1 agreement of the two lenses' readouts
over every (layer, position) of a few prompts — what the web UI would show.

Run:
    python scripts/compare_lens.py ours.pt theirs.pt
    python scripts/compare_lens.py ours.pt theirs.pt --model Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens_qwen.lens import JacobianLens  # noqa: E402

PROMPTS = [
    "Question: What is the capital of France?\nAnswer: The capital of France is",
    "Question: How many legs does a spider have?\nAnswer: A spider has",
    "The chemical symbol for water is",
    "Fact: The currency used in the country shaped like a boot is",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--model", default=None, help="also compare readouts on this model")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    A, B = JacobianLens.load(args.a), JacobianLens.load(args.b)
    print(f"A: {args.a}\n   {A}\nB: {args.b}\n   {B}")
    layers = sorted(set(A.source_layers) & set(B.source_layers))
    if A.d_model != B.d_model or not layers:
        raise SystemExit("lenses are not comparable (d_model / layers differ)")
    print(f"\n{'layer':>5} {'rel_fro(A-B)/B':>15} {'cosine':>8}")
    for l in layers:
        a, b = A.jacobians[l].float(), B.jacobians[l].float()
        rel = ((a - b).norm() / b.norm()).item()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        print(f"{l:5d} {rel:15.3%} {cos:8.4f}")

    if not args.model:
        return
    from jlens_qwen.model import load

    model = load(args.model, **({"device": args.device} if args.device else {}))
    A.device = B.device = model.device
    agree = total = 0
    per_layer = {l: [0, 0] for l in layers}
    for p in PROMPTS:
        ra = A.apply(model, p, layers=layers)["lens_logits"]
        rb = B.apply(model, p, layers=layers)["lens_logits"]
        for l in layers:
            same = (ra[l].argmax(-1) == rb[l].argmax(-1))
            per_layer[l][0] += int(same.sum())
            per_layer[l][1] += same.numel()
            agree += int(same.sum())
            total += same.numel()
    print(f"\nreadout top-1 agreement over {len(PROMPTS)} prompts: {agree}/{total} = {agree / total:.1%}")
    print("per layer: " + " ".join(f"L{l}:{a / max(n, 1):.0%}" for l, (a, n) in per_layer.items()))


if __name__ == "__main__":
    main()
