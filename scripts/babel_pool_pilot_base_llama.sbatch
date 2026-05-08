#!/bin/bash
#SBATCH --job-name=pool_pilot_base_llama
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=A6000|L40S|H100|A100_80GB|A100_40GB
#SBATCH --exclude=babel-n9-32,babel-m5-32,babel-p5-24
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/pool_pilot_base_llama_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/pool_pilot_base_llama_%j.err
#
# Base-Llama (non-Instruct) unsafe_compliance pool feasibility pilot.
# 100 dual_use prompts (held-out IDs excluded), 1 generation per prompt.

set -euo pipefail
REPO=/data/group_data/wehbelab/ahlayhel/safecomp-dpo
cd "$REPO"

module purge
module load cuda-12.4
export PATH="$HOME/miniforge3/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1

echo "[info] Host:    $(hostname)"
echo "[info] Date:    $(date -Iseconds)"
nvidia-smi || true

# Sample step is fast and CPU-only; running it here makes the job
# self-contained and idempotent (overwrites the gitignored sample file).
python -u scripts/build_pool_pilot_sample.py

python -u scripts/run_base_llama_pool_pilot.py \
    --config configs/generation/unsafe_compliance_base_llama_pilot.yaml

echo "[done]"
wc -l outputs/pool_pilot_base_llama/raw.jsonl
