"""Download and cache the fitting corpus (run on a node WITH internet).

HPC compute nodes are often offline; run this once on the login node. It
writes data/corpus/prompts_{n}_{min_chars}.jsonl, which fit_lens.py reads
without touching the network.

Same loader as upstream (WikiText-103 first, c4 to top up). The paper fits
on 1000 sequences of 128 tokens; ~600 chars is enough for most WikiText
paragraphs to reach 128 Qwen tokens, hence the default --min-chars 600.

Run:  python scripts/prepare_corpus.py --n 1000 --min-chars 600
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens_qwen.prompts import load_prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--source", choices=["wikitext", "c4"], default="wikitext",
                    help="corpus: WikiText-103 (Neuronpedia/paper-like) or c4 (upstream Qwen3.8 lens)")
    ap.add_argument("--tokenizer", default=None,
                    help="optional HF model id / path: report token-length stats")
    args = ap.parse_args()

    prompts = load_prompts(n=args.n, min_chars=args.min_chars, source=args.source)
    uniq = len(set(prompts))
    print(f"{len(prompts)} prompts ({uniq} unique), "
          f"cached under data/corpus/ (source={args.source})")
    if uniq != args.n:
        print("WARNING: corpus is short or has duplicates (offline fallback?) — "
              "do not fit a research lens on it")
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        lens = sorted(len(tok(p).input_ids) for p in prompts)
        n128 = sum(l >= 128 for l in lens)
        print(f"token lengths: min {lens[0]}, median {lens[len(lens) // 2]}, "
              f"max {lens[-1]}; {n128}/{len(lens)} reach 128 tokens")


if __name__ == "__main__":
    main()
