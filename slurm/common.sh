# Sourced by every sbatch script. EDIT THE MARKED LINES FOR YOUR CLUSTER.
# Submit jobs from the project root (sbatch runs in the submit directory).
set -euo pipefail

# --- cluster-specific -------------------------------------------------------
# module load cuda/12.8        # CHANGE: use the cluster's actual module name.
#                              # H200 GDN backward needs nvcc for FLA's TileLang
#                              # workaround. Load it in hpc.env or uncomment here.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"   # CHANGE: fast shared storage, ~60 GB free
# ---------------------------------------------------------------------------

export HF_HUB_OFFLINE=1          # compute nodes are often offline; pre-download on the login node
export UV_OFFLINE=1 UV_FROZEN=1  # use the env prepared before submitting the job
export UV_NO_SYNC=1             # running a job must not reinstall a manually staged wheel
export UV_PYTHON="${UV_PYTHON:-3.11}"
export TOKENIZERS_PARALLELISM=false
export JLENS_MODEL="${JLENS_MODEL:-Qwen/Qwen3.8-27B}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs

echo "[$(date '+%F %T')] host=$(hostname) job=${SLURM_JOB_ID:-local} model=$JLENS_MODEL"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
