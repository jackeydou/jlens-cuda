"""Merge per-GPU lens shards into one lens (n_prompts-weighted mean).

Writes both formats: ``<out>.npz`` (what the web app loads, fp16) and
``<out>.pt`` (Anthropic jlens schema), plus ``<out>.provenance.json``.

Run:
    python scripts/merge_lens.py data/lens/shards/qwen38_27b_n1000_shard*.pt \
        --out data/lens/qwen38_27b_n1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--expect-n", type=int, default=None, help="fail unless total n_prompts matches")
    args = ap.parse_args()

    import jlens

    from jlens_qwen.lens import JacobianLens

    parts = []
    provs = []
    for p in sorted(args.shards):
        lens = jlens.JacobianLens.load(p)
        print(f"{p}: {lens}")
        parts.append(lens)
        prov = p.replace(".pt", ".provenance.json")
        if os.path.exists(prov):
            provs.append(json.load(open(prov)))
    models = {pv["model_id"] for pv in provs}
    if len(models) > 1:
        raise SystemExit(f"shards were fitted on different models: {models}")
    merged = jlens.JacobianLens.merge(parts)
    if args.expect_n is not None and merged.n_prompts != args.expect_n:
        raise SystemExit(f"merged n_prompts={merged.n_prompts}, expected {args.expect_n}")
    print(f"merged: {merged}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    merged.save(args.out + ".pt")  # fp16, jlens schema
    JacobianLens(merged.jacobians, n_prompts=merged.n_prompts,
                 d_model=merged.d_model, device="cpu").save(args.out + ".npz")
    prov = dict(provs[0]) if provs else {}
    prov.pop("shard", None)
    prov.pop("n_prompts_in_shard", None)
    prov.pop("fit_seconds", None)
    prov.update(n_prompts=merged.n_prompts, num_shards=len(parts),
                fit_seconds_per_shard=[pv.get("fit_seconds") for pv in provs])
    with open(args.out + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"wrote {args.out}.npz, {args.out}.pt, {args.out}.provenance.json")


if __name__ == "__main__":
    main()
