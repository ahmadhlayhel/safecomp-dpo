#!/bin/bash
#SBATCH --job-name=pool_pilot_base_llama_v2
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --constraint=A6000|L40S|H100|A100_80GB|A100_40GB
#SBATCH --exclude=babel-n9-32,babel-m5-32,babel-p5-24
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/pool_pilot_base_llama_v2_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/pool_pilot_base_llama_v2_%j.err
#
# v2 of the base-Llama unsafe_compliance pool feasibility pilot.
# Same 100 prompts as v1 (same seed, same stratification, same held-out
# exclusion). Different decoding settings to fight degeneracy.

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

# Same seed as v1 -> same 100 prompts -> clean v1/v2 comparison.
python -u scripts/build_pool_pilot_sample.py \
    --output outputs/pool_pilot_base_llama_v2/sampled_prompts.jsonl

python -u scripts/run_base_llama_pool_pilot.py \
    --config configs/generation/unsafe_compliance_base_llama_pilot_v2.yaml

echo "[done]"
wc -l outputs/pool_pilot_base_llama_v2/raw.jsonl
