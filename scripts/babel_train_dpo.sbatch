#!/bin/bash
#SBATCH --job-name=safecomp_dpo
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --constraint=A6000|L40S|H100|A100_80GB|A100_40GB
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/safecomp_dpo_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/safecomp_dpo_%j.err
#
# Usage:
#   sbatch scripts/babel_train_dpo.sbatch baseline
#   sbatch scripts/babel_train_dpo.sbatch safecomp
#
# Trains Llama-3.1-8B-Instruct via TRL DPOTrainer + QLoRA (4-bit).
# Regime ("baseline" or "safecomp") selects the YAML config and dataset.
#
# Required environment (verify before submitting):
#   - python (miniforge3 base) with: trl, bitsandbytes, datasets
#     installed in addition to: torch, transformers, accelerate, peft,
#     huggingface-hub, pyyaml, pydantic
#   - HF_TOKEN with access to meta-llama/Llama-3.1-8B-Instruct
#     (huggingface-cli login already done, OR export HF_TOKEN below)

set -euo pipefail

REGIME="${1:-}"
REPO=/data/group_data/wehbelab/ahlayhel/safecomp-dpo
cd "$REPO"

case "$REGIME" in
  baseline|safecomp)
    CONFIG="configs/training/${REGIME}_babel.yaml"
    DATASET="hf_data/dpo/${REGIME}_dpo_dataset.jsonl"
    ;;
  smoke)
    CONFIG="configs/training/safecomp_smoke_babel.yaml"
    DATASET="hf_data/dpo/smoke_dpo_dataset.jsonl"
    ;;
  *)
    echo "Usage: sbatch $0 {baseline|safecomp|smoke}" >&2
    exit 2
    ;;
esac

# --- Modules / env ---
module purge
module load cuda-12.4

# Use miniforge3 base; switch to a project-specific venv/conda env if you create one.
export PATH="$HOME/miniforge3/bin:$PATH"

# HF cache and token (token already at ~/.cache/huggingface/token by default).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export HF_TOKEN=...   # only needed if not using huggingface-cli login

# Quiet TF/transformers warnings, force bf16-friendly attention impl.
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info
export BITSANDBYTES_NOWELCOME=1

echo "[info] Host:    $(hostname)"
echo "[info] Date:    $(date -Iseconds)"
echo "[info] Regime:  $REGIME"
echo "[info] Config:  $CONFIG"
echo "[info] Dataset: $DATASET"
echo "[info] CUDA:    $(nvcc --version | tail -1 || true)"
nvidia-smi || true

# Sanity checks before launch
test -f "$CONFIG"  || { echo "[error] config not found: $CONFIG"  >&2; exit 3; }
test -f "$DATASET" || { echo "[error] dataset not found: $DATASET" >&2; exit 3; }

python -u scripts/train_dpo.py \
    --config  "$CONFIG" \
    --dataset "$DATASET"

echo "[done] regime=$REGIME exit=$?"
