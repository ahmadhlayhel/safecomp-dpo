#!/usr/bin/env bash
#SBATCH --job-name=wildguard_v3v4
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --constraint=L40S|H100|A100_80GB|A100_40GB
#SBATCH --exclude=babel-n9-32,babel-m5-32,babel-p5-24
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/wildguard_v3v4_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/wildguard_v3v4_%j.err
#
# WildGuard cross-check judge over both pool pilots (v3 and v4).
# Loads allenai/wildguard once per pilot. ~7B bf16 fits comfortably on
# any of the constraint targets. Pure inference; no API spend.

set -euo pipefail
REPO=/data/group_data/wehbelab/ahlayhel/safecomp-dpo
cd "$REPO"

module purge
module load cuda-12.4
export PATH="$HOME/miniforge3/bin:$PATH"
# Keep HF_HOME at default ($HOME/.cache/huggingface) so the existing HF
# auth token is discovered. Redirect only the *model file cache* to project
# storage where there is ~1 TB free (home dir is over quota for the WildGuard
# 10 GB download). HF_HUB_CACHE overrides just the cache location.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE=/data/group_data/wehbelab/ahlayhel/hf_cache/hub
mkdir -p "$HF_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false

echo "[info] Host: $(hostname)  Date: $(date -Iseconds)"
nvidia-smi || true

for pilot in v3 v4; do
    DIR=outputs/pool_pilot_base_llama_${pilot}
    if [ ! -f "$DIR/raw.jsonl" ] || [ ! -f "$DIR/sampled_prompts.jsonl" ]; then
        echo "[skip] $pilot: missing raw.jsonl or sampled_prompts.jsonl"
        continue
    fi
    echo ""
    echo "=== WildGuard on $pilot ==="
    python -u scripts/run_pool_pilot_judge_wildguard.py \
        --raw     "$DIR/raw.jsonl" \
        --prompts "$DIR/sampled_prompts.jsonl" \
        --qc      "$DIR/qc.jsonl" \
        --output  "$DIR/judged_wildguard.jsonl"
done

echo ""
echo "[done]"
wc -l outputs/pool_pilot_base_llama_v3/judged_wildguard.jsonl 2>/dev/null || true
wc -l outputs/pool_pilot_base_llama_v4/judged_wildguard.jsonl 2>/dev/null || true
