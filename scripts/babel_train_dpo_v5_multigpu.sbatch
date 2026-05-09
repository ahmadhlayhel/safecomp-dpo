#!/bin/bash
#SBATCH --job-name=safecomp_dpo_v5
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --constraint=A6000|L40S|H100|A100_80GB|A100_40GB
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/safecomp_dpo_v5_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/safecomp_dpo_v5_%j.err
#
# 2-GPU DDP training for the v5 base-Llama rejected-pool DPO retrains.
#
# Usage:
#   sbatch scripts/babel_train_dpo_v5_multigpu.sbatch baseline_v5_basepool
#   sbatch scripts/babel_train_dpo_v5_multigpu.sbatch safecomp_v5_basepool
#
# Differs from babel_train_dpo.sbatch:
#   - --gres=gpu:2
#   - launches via `accelerate launch --num_processes 2` for DDP
#   - configs use grad_accum=2 (vs 4) so effective batch = 2*4*2 = 16,
#     identical to the single-GPU prior runs (1*4*4 = 16).
#
# train_dpo.py was patched to read LOCAL_RANK and pin each rank to its
# own GPU via device_map={"": local_rank} when LOCAL_RANK is set —
# preserves single-GPU behavior when launched directly.

set -euo pipefail

REGIME="${1:-}"
REPO=/data/group_data/wehbelab/ahlayhel/safecomp-dpo
cd "$REPO"

case "$REGIME" in
  baseline_v5_basepool|safecomp_v5_basepool)
    CONFIG="configs/training/${REGIME}_babel.yaml"
    DATASET="hf_data/dpo/${REGIME}_dpo_dataset.jsonl"
    ;;
  *)
    echo "Usage: sbatch $0 {baseline_v5_basepool|safecomp_v5_basepool}" >&2
    exit 2
    ;;
esac

module purge
module load cuda-12.4
export PATH="$HOME/miniforge3/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info
export BITSANDBYTES_NOWELCOME=1

echo "[info] Host:    $(hostname)"
echo "[info] Date:    $(date -Iseconds)"
echo "[info] Regime:  $REGIME"
echo "[info] Config:  $CONFIG"
echo "[info] Dataset: $DATASET"
nvidia-smi || true

test -f "$CONFIG"  || { echo "[error] config not found: $CONFIG"  >&2; exit 3; }
test -f "$DATASET" || { echo "[error] dataset not found: $DATASET" >&2; exit 3; }

# accelerate launch handles DDP setup: spawns one process per GPU,
# each with its own LOCAL_RANK. train_dpo.py reads LOCAL_RANK to pin
# the 4-bit model to the rank-specific device.
accelerate launch \
    --num_processes 2 \
    --multi_gpu \
    --mixed_precision bf16 \
    scripts/train_dpo.py \
    --config  "$CONFIG" \
    --dataset "$DATASET"

echo "[done] regime=$REGIME exit=$?"
