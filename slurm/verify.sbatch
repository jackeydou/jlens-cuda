#!/bin/bash
# One-GPU sanity job: environment check, fit-path correctness + cost, CPU-free tests.
#   mkdir -p logs && DIM_BATCH=32 sbatch slurm/verify.sbatch
#SBATCH --job-name=jlens-verify
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_%j.out

source slurm/common.sh
uv run python scripts/check_env.py
uv run pytest -q
uv run python scripts/verify_fit.py --dim-batch "${DIM_BATCH:-16}" ${VERIFY_EXTRA_ARGS:-}
