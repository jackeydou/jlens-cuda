#!/bin/bash
# Fit the lens as an array of single-GPU shards (one prompt slice per task).
#   mkdir -p logs && sbatch slurm/fit_array.sbatch
#   N_PROMPTS=1000 DIM_BATCH=32 sbatch --array=0-15 slurm/fit_array.sbatch   # 16 shards
# Shards checkpoint every 10 prompts; a requeued / resubmitted task resumes.
#SBATCH --job-name=jlens-fit
#SBATCH --partition=CHANGE_ME            # your H200 partition
#SBATCH --gres=gpu:1                     # e.g. --gres=gpu:h200:1 on some clusters
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G                        # ~13 GB of fp32 Jacobian sums + model load
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --output=logs/fit_%A_%a.out

source slurm/common.sh
uv run python scripts/fit_lens.py \
    --n-prompts "${N_PROMPTS:-1000}" \
    --source "${CORPUS:-wikitext}" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT" \
    --dim-batch "${DIM_BATCH:-16}" \
    --tag "${TAG:-qwen38_27b}" \
    ${FIT_EXTRA_ARGS:-}
