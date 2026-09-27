"""Fit one shard of a full-depth Jacobian lens on one GPU.

This is the CUDA replacement for upstream's ``fit_qwen38_n1000.py``. The
upstream MLX pipeline had to assemble per-layer Jacobians analytically and
chain-multiply them (J_{l-1} = J_l · M_l) because MLX's fused GDN kernel has
no backward. On CUDA the GDN layers run on flash-linear-attention's Triton
kernels, which DO have a backward, so we use Anthropic's reference
estimator from the ``jlens`` package directly — the paper's exact method,
and the one Neuronpedia's published lenses were fitted with:

    J_l = mean_prompts  mean_{source s}  sum_{target t >= s}  d h_final[t] / d h_l[s]

One forward on the prompt replicated ``dim_batch`` times, then
ceil(5120 / dim_batch) backward passes against the retained graph; each
pass yields ``dim_batch`` rows of every J_l (l = 0..62) at once.

Shard the corpus across GPUs (``--shard i --num-shards N``, one process per
GPU) and combine with ``scripts/merge_lens.py``. Each shard checkpoints and
resumes.

Run (one GPU):
    python scripts/fit_lens.py --n-prompts 1000 --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens_qwen.model import DEFAULT_MODEL_ID, load_hf  # noqa: E402
from jlens_qwen.prompts import load_prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("JLENS_MODEL", DEFAULT_MODEL_ID))
    ap.add_argument("--device", default="cuda", help="cuda (default); cpu only for smoke tests")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--source", choices=["wikitext", "c4"], default="wikitext",
                    help="corpus: WikiText-103 (Neuronpedia/paper-like) or c4 (upstream Qwen3.8 lens)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=128, help="paper: 128 tokens")
    ap.add_argument("--skip-first", type=int, default=16, help="paper: 16 sink positions")
    ap.add_argument("--dim-batch", type=int, default=16,
                    help="output dims per backward pass (memory ~ dim_batch x seq_len)")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--compile", action="store_true", help="torch.compile each decoder layer")
    ap.add_argument("--out-dir", default="data/lens/shards")
    ap.add_argument("--tag", default="qwen38_27b")
    args = ap.parse_args()

    import jlens

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.tag}_n{args.n_prompts}_shard{args.shard:02d}of{args.num_shards:02d}"
    out_path = os.path.join(args.out_dir, stem + ".pt")
    ckpt_path = os.path.join(args.out_dir, stem + ".ckpt.pt")
    if os.path.exists(out_path):
        print(f"{out_path} already exists — shard done, nothing to do")
        return

    prompts = load_prompts(n=args.n_prompts, min_chars=args.min_chars, source=args.source)
    if len(prompts) != args.n_prompts or len(set(prompts)) != args.n_prompts:
        raise SystemExit(
            f"corpus is not {args.n_prompts} unique prompts (got {len(prompts)}, "
            f"{len(set(prompts))} unique) — run scripts/prepare_corpus.py on a node "
            "with internet first")
    shard_prompts = prompts[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(shard_prompts)} prompts", flush=True)

    t0 = time.perf_counter()
    hf_model, tokenizer = load_hf(args.model, device=args.device)
    lm = jlens.from_hf(hf_model, tokenizer, compile=args.compile)
    print(f"loaded {lm} in {time.perf_counter() - t0:.0f}s", flush=True)

    source_layers = list(range(lm.n_layers - 1))  # L0..L62, target = L63 (pre-norm)
    t0 = time.perf_counter()
    lens = jlens.fit(
        lm,
        shard_prompts,
        source_layers=source_layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=ckpt_path,
        checkpoint_every=args.checkpoint_every,
        resume=True,
    )
    elapsed = time.perf_counter() - t0
    # fp32 shards: merge averages them, so keep full precision until the end.
    lens.save(out_path, dtype=torch.float32)
    print(f"saved {out_path}: {lens}  ({elapsed / 3600:.2f} h, "
          f"peak mem {(torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0):.1f} GB)", flush=True)
    with open(out_path.replace(".pt", ".provenance.json"), "w") as f:
        json.dump({
            "model_id": args.model,
            "n_prompts_in_shard": lens.n_prompts,
            "shard": args.shard, "num_shards": args.num_shards,
            "corpus": f"{args.source} n={args.n_prompts} min_chars={args.min_chars}",
            "estimator": "jlens.fit (Anthropic reference; sum over targets, mean over sources)",
            "max_seq_len": args.max_seq_len, "skip_first": args.skip_first,
            "dim_batch": args.dim_batch, "source_layers": source_layers,
            "target_layer": lm.n_layers - 1,
            "fit_seconds": round(elapsed), "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
