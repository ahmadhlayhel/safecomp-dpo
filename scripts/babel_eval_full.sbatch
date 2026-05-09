#!/usr/bin/env bash
#SBATCH --job-name=safecomp_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --constraint=A6000|L40S|H100|A100_80GB|A100_40GB
#SBATCH --exclude=babel-n9-32,babel-m5-32,babel-p5-24
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/safecomp_eval_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/safecomp_eval_%j.err

# Usage:
#   sbatch scripts/babel_eval_full.sbatch baseline
#   sbatch scripts/babel_eval_full.sbatch safecomp
#   sbatch scripts/babel_eval_full.sbatch dry-baseline
#   sbatch scripts/babel_eval_full.sbatch dry-safecomp

set -euo pipefail

REGIME="${1:-safecomp}"
REPO=/data/group_data/wehbelab/ahlayhel/safecomp-dpo

cd "$REPO"

case "$REGIME" in
  baseline)
    CONFIG=configs/eval/baseline_peft.yaml
    PROMPTS=hf_data/eval/held_out_prompts.jsonl
    DRY_RUN=""
    ;;
  safecomp)
    CONFIG=configs/eval/safecomp_peft.yaml
    PROMPTS=hf_data/eval/held_out_prompts.jsonl
    DRY_RUN=""
    ;;
  base)
    CONFIG=configs/eval/base.yaml
    PROMPTS=hf_data/eval/held_out_prompts.jsonl
    DRY_RUN=""
    ;;
  baseline_strict)
    CONFIG=configs/eval/baseline_strict_peft.yaml
    PROMPTS=hf_data/eval/held_out_prompts.jsonl
    DRY_RUN=""
    ;;
  safecomp_dualuse_x2)
    CONFIG=configs/eval/safecomp_dualuse_x2_peft.yaml
    PROMPTS=hf_data/eval/held_out_prompts.jsonl
    DRY_RUN=""
    ;;
  baseline_v5_basepool|safecomp_v5_basepool)
    CONFIG=configs/eval/${REGIME}_peft.yaml
    PROMPTS=hf_data/eval/held_out_prompts.jsonl
    DRY_RUN=""
    ;;
  dry-baseline)
    CONFIG=configs/eval/baseline_peft.yaml
    PROMPTS=hf_data/eval/dry_run_prompts.jsonl
    DRY_RUN="--dry-run"
    ;;
  dry-safecomp)
    CONFIG=configs/eval/safecomp_peft.yaml
    PROMPTS=hf_data/eval/dry_run_prompts.jsonl
    DRY_RUN="--dry-run"
    ;;
  *)
    echo "Unknown regime: $REGIME. Use baseline | safecomp | base | baseline_strict | safecomp_dualuse_x2 | baseline_v5_basepool | safecomp_v5_basepool | dry-baseline | dry-safecomp" >&2
    exit 1
    ;;
esac

mkdir -p /data/group_data/wehbelab/ahlayhel/logs

echo "=== safecomp_eval ==="
echo "Regime:  $REGIME"
echo "Config:  $CONFIG"
echo "Prompts: $PROMPTS"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date:    $(date)"
echo "=============================="

python scripts/evaluate_model.py \
    --prompts "$PROMPTS" \
    --config  "$CONFIG" \
    ${DRY_RUN}

echo "=== Done: $(date) ==="
