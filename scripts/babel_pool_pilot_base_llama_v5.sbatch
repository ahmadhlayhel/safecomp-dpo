#!/usr/bin/env bash
#SBATCH --job-name=pilot_v5_largepool
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --constraint=L40S|H100|A100_80GB|A100_40GB
#SBATCH --exclude=babel-n9-32,babel-m5-32,babel-p5-24
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/pilot_v5_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/pilot_v5_%j.err
#
# v5 large-pool generation: ~1,973 base-Llama generations under v4 methodology
# (domain-matched 2-shot, 9 domains, multi-sampled where unique <200).
# At ~8s/prompt on L40S that's roughly 4.5 hours; walltime padded to 6h.

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
    --config  configs/generation/unsafe_compliance_base_llama_pilot_v5.yaml

echo "[done]"
wc -l outputs/pool_pilot_base_llama_v5/raw.jsonl
