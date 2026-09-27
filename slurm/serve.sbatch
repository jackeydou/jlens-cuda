#!/bin/bash
# Run the web app on a GPU node; reach it from your laptop through an SSH tunnel.
#   mkdir -p logs && JLENS_PATH=data/lens/qwen38_27b_n1000.npz sbatch slurm/serve.sbatch
#   then read logs/serve_<jobid>.out for the exact ssh command.
#SBATCH --job-name=jlens-serve
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/serve_%j.out

source slurm/common.sh
export JLENS_PATH="${JLENS_PATH:-data/lens/qwen38_27b_n1000.npz}"
PORT="${PORT:-8765}"
NODE=$(hostname)
cat <<MSG
================================================================
 J-lens web app starting on ${NODE}:${PORT} (bound to 127.0.0.1)
 On your laptop run (replace LOGIN with your login host):
   ssh -N -L ${PORT}:localhost:${PORT} -J ${USER}@LOGIN ${USER}@${NODE}
 then open http://localhost:${PORT}/
================================================================
MSG
exec uv run uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port "$PORT"
