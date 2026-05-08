#!/usr/bin/env bash
#SBATCH --job-name=pilot_v4
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --constraint=L40S|H100|A100_80GB|A100_40GB
#SBATCH --exclude=babel-n9-32,babel-m5-32,babel-p5-24
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/pilot_v4_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/pilot_v4_%j.err
#
# v4 base-Llama unsafe-pool feasibility pilot: domain-matched 2-shot priming.
# 200 prompts, exemplar cache at outputs/pool_pilot_base_llama_v4/exemplar_cache.jsonl.

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

python -u scripts/run_base_llama_pool_pilot.py \
    --config  configs/generation/unsafe_compliance_base_llama_pilot_v4.yaml

echo "[done]"
wc -l outputs/pool_pilot_base_llama_v4/raw.jsonl
