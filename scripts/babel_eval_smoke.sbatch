#!/bin/bash
#SBATCH --job-name=safecomp_eval_smoke
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --constraint=A6000|L40S|H100|A100_80GB|A100_40GB
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/safecomp_eval_smoke_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/safecomp_eval_smoke_%j.err
#
# Adapter inference smoke for SafeComp-DPO.
# Loads base + adapter for each regime, generates greedy responses for the
# 12-prompt smoke set, appends to outputs/eval/smoke/adapter_smoke_outputs.jsonl.
#
# Usage:
#   sbatch scripts/babel_eval_smoke.sbatch

set -euo pipefail

REPO=/data/group_data/wehbelab/ahlayhel/safecomp-dpo
cd "$REPO"

module purge
module load cuda-12.4
export PATH="$HOME/miniforge3/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1

PROMPTS=hf_data/eval/smoke_prompts.jsonl
OUT=outputs/eval/smoke/adapter_smoke_outputs.jsonl

# Start fresh each run to avoid mixing previous attempts.
rm -f "$OUT"

echo "[info] Host:    $(hostname)"
echo "[info] Date:    $(date -Iseconds)"
nvidia-smi || true

for REGIME in baseline safecomp; do
  ADAPTER="outputs/training/${REGIME}_babel/checkpoint"
  echo "[info] === regime=$REGIME adapter=$ADAPTER ==="
  python -u scripts/eval_adapter_smoke.py \
      --regime "$REGIME" \
      --adapter "$ADAPTER" \
      --prompts "$PROMPTS" \
      --output  "$OUT"
done

echo "[done] smoke output: $OUT"
wc -l "$OUT"
