#!/bin/bash
#SBATCH --job-name=safecomp_format_ablation
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=A6000|L40S|H100|A100_80GB|A100_40GB
#SBATCH --exclude=babel-n9-32,babel-m5-32,babel-p5-24
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/safecomp_format_ablation_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/safecomp_format_ablation_%j.err
#
# Format ablation: 30 dual_use prompts × 2 modes (raw, chat) × 3 variants
# (base, baseline, safecomp) = 180 generations.

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

python -u scripts/eval_format_ablation.py \
    --prompts hf_data/eval/format_ablation_prompts.jsonl \
    --output  outputs/eval/format_ablation/format_ablation_outputs.jsonl

echo "[done]"
wc -l outputs/eval/format_ablation/format_ablation_outputs.jsonl
