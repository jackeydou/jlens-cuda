#!/bin/bash
# Fit the lens on ONE node with all its GPUs: one shard process per GPU.
#   mkdir -p logs && sbatch slurm/fit_node.sbatch
#SBATCH --job-name=jlens-fit-node
#SBATCH --partition=CHANGE_ME            # your H200 partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G                       # 8 x ~96 GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/fit_node_%j.out

source slurm/common.sh
NGPU=$(nvidia-smi -L | wc -l)
echo "launching $NGPU shards"
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run python scripts/fit_lens.py \
        --n-prompts "${N_PROMPTS:-1000}" \
        --source "${CORPUS:-wikitext}" \
        --shard "$i" --num-shards "$NGPU" \
        --dim-batch "${DIM_BATCH:-16}" \
        --tag "${TAG:-qwen38_27b}" \
        ${FIT_EXTRA_ARGS:-} > "logs/fit_node_${SLURM_JOB_ID}_shard${i}.log" 2>&1 &
    sleep 20   # stagger the 54 GB model loads
done
wait
echo "all shards finished"; ls -la data/lens/shards/
