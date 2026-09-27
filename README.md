# jlens-cuda — J-lens for Qwen3.8-27B (bf16) on NVIDIA H200

A CUDA / PyTorch port of [WeZZard/jlens-qwen36](https://github.com/WeZZard/jlens-qwen36)
(an MLX / Apple-Silicon visual debugger for the **Jacobian lens** from Anthropic's
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)),
retargeted to **`Qwen/Qwen3.8-27B` in bf16** on H200 GPUs.

**Step-by-step tutorial (中文): [docs/TUTORIAL.md](docs/TUTORIAL.md)**

## What changed vs. upstream

| | upstream (MLX) | this port (CUDA) |
|---|---|---|
| model | `mlx-community/Qwen3.x-27B-4bit` | `Qwen/Qwen3.8-27B`, bf16, HF transformers |
| GDN backward | custom Metal VJP kernel | flash-linear-attention Triton kernels |
| fit estimator | analytic per-layer `M_l`, chain `J_{l-1} = J_l M_l` | Anthropic `jlens.fit` (exact end-to-end Jacobian, paper's estimator) |
| fit cost | ~7–20 min / prompt on an M4 | measure with `scripts/verify_fit.py`; shards across GPUs |
| web app | `serve.py` + `web/index.html` | the same files; only the MLX touchpoints in `serve.py` changed |

Files:

```
jlens_qwen/model.py          TorchLensModel + StreamSession (hooks + HF hybrid cache)
jlens_qwen/lens.py           JacobianLens (loads .npz and jlens/Neuronpedia .pt)
jlens_qwen/interventions.py  steer / swap / ablate (J-lens vectors)
jlens_qwen/serve.py          upstream FastAPI server, ported
web/index.html               upstream UI, unchanged
scripts/check_env.py         GPU + kernel check
scripts/prepare_corpus.py    download the fitting corpus (login node)
scripts/verify_fit.py        correctness + cost check on one prompt
scripts/fit_lens.py          fit one shard on one GPU
scripts/merge_lens.py        merge shards -> .npz + .pt
scripts/readout_smoke.py     sanity readout of a fitted lens
scripts/measure_bands.py     sensory / workspace / motor bands for the UI
slurm/*.sbatch               Slurm templates (fit array, one-node fit, serve)
```

## Quick start

```bash
uv sync
uv run python scripts/check_env.py
uv run python scripts/prepare_corpus.py --n 1000 --min-chars 600   # needs internet
uv run python scripts/verify_fit.py --dim-batch 16                  # on a GPU node
sbatch slurm/fit_array.sbatch                                       # 8 shards
uv run python scripts/merge_lens.py data/lens/shards/*.pt --out data/lens/qwen38_27b_n1000 --expect-n 1000
JLENS_PATH=data/lens/qwen38_27b_n1000.npz uv run uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765
```

## Tests

`uv run pytest` runs on a laptop (no GPU, no weights): upstream's serve-logic
contract tests plus numerical tests of the torch backend on a tiny random
Qwen3.5 model.

## License

Apache-2.0 (as upstream). `web/index.html`, `jlens_qwen/serve.py` and
`jlens_qwen/prompts.py` are from WeZZard/jlens-qwen36; the fitting estimator is
Anthropic's [jacobian-lens](https://github.com/anthropics/jacobian-lens).
