#!/usr/bin/env bash
#SBATCH --job-name=safecomp_bench
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --constraint=A6000|L40S|H100|A100_80GB|A100_40GB
#SBATCH --exclude=babel-n9-32,babel-m5-32,babel-p5-24
#SBATCH --output=/data/group_data/wehbelab/ahlayhel/logs/safecomp_bench_%j.out
#SBATCH --error=/data/group_data/wehbelab/ahlayhel/logs/safecomp_bench_%j.err

# Usage:
#   sbatch scripts/babel_benchmark_eval.sbatch <benchmark> <model_variant>
#
# benchmark:     harmbench | xstest | or_bench | falsereject_test | dualuse_sample
# model_variant: baseline_v1 | safecomp_v1 | baseline_v5 | safecomp_v5 | base
#
# Examples:
#   sbatch scripts/babel_benchmark_eval.sbatch harmbench baseline_v5
#   sbatch scripts/babel_benchmark_eval.sbatch xstest safecomp_v5
#   sbatch scripts/babel_benchmark_eval.sbatch harmbench base

set -euo pipefail

BENCHMARK="${1:-harmbench}"
MODEL_VARIANT="${2:-baseline_v5}"
REPO=/data/group_data/wehbelab/ahlayhel/safecomp-dpo

cd "$REPO"

module purge
module load cuda-12.4
export PATH="$HOME/miniforge3/bin:$PATH"
# Unified HF cache: wildguard (native) + Llama (symlinked from huggingface_cache).
# Matches the pattern used by babel_judge_wildguard_pool_pilots.sbatch which works.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE=/data/group_data/wehbelab/ahlayhel/hf_cache/hub
export TOKENIZERS_PARALLELISM=false

# ── Adapter paths ──────────────────────────────────────────────────────────────
case "$MODEL_VARIANT" in
  baseline_v1)
    ADAPTER_PATH="outputs/training/baseline_babel/checkpoint"
    ;;
  safecomp_v1)
    ADAPTER_PATH="outputs/training/safecomp_babel/checkpoint"
    ;;
  baseline_v5)
    ADAPTER_PATH="outputs/training/baseline_v5_basepool_babel/checkpoint"
    ;;
  safecomp_v5)
    ADAPTER_PATH="outputs/training/safecomp_v5_basepool_babel/checkpoint"
    ;;
  base)
    ADAPTER_PATH="none"   # no adapter — raw Llama-3.1-8B-Instruct
    ;;
  *)
    echo "Unknown model_variant: $MODEL_VARIANT" >&2
    echo "Use: baseline_v1 | safecomp_v1 | baseline_v5 | safecomp_v5 | base" >&2
    exit 1
    ;;
esac

# ── Benchmark config ───────────────────────────────────────────────────────────
case "$BENCHMARK" in
  harmbench)
    CONFIG="configs/eval/benchmarks/harmbench_peft.yaml"
    ;;
  xstest)
    CONFIG="configs/eval/benchmarks/xstest_peft.yaml"
    ;;
  or_bench)
    CONFIG="configs/eval/benchmarks/or_bench_peft.yaml"
    ;;
  falsereject_test)
    CONFIG="configs/eval/benchmarks/falsereject_test_peft.yaml"
    ;;
  dualuse_sample)
    CONFIG="configs/eval/benchmarks/dualuse_sample_peft.yaml"
    ;;
  *)
    echo "Unknown benchmark: $BENCHMARK" >&2
    echo "Use: harmbench | xstest | or_bench | falsereject_test | dualuse_sample" >&2
    exit 1
    ;;
esac

RUN_ID="${BENCHMARK}_${MODEL_VARIANT}"
OUTPUT="outputs/eval/benchmarks/${RUN_ID}.jsonl"
REPORT="outputs/eval/benchmarks/${RUN_ID}_report.json"

mkdir -p outputs/eval/benchmarks hf_data/benchmarks
mkdir -p /data/group_data/wehbelab/ahlayhel/logs

# Ingest benchmark data if not already present.
# dualuse_sample is committed to git — no download needed.
DATA_FILE="hf_data/benchmarks/${BENCHMARK}.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    if [ "$BENCHMARK" = "dualuse_sample" ]; then
        echo "ERROR: dualuse_sample.jsonl not found — run git pull to get it." >&2
        exit 1
    fi
    echo "Ingesting ${BENCHMARK} dataset..."
    python3 scripts/ingest_benchmark_data.py --benchmark "$BENCHMARK"
else
    echo "Dataset already present: $DATA_FILE"
fi

echo "=== safecomp benchmark eval ==="
echo "Benchmark:     $BENCHMARK"
echo "Model variant: $MODEL_VARIANT"
echo "Adapter:       $ADAPTER_PATH"
echo "Config:        $CONFIG"
echo "Output:        $OUTPUT"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date:          $(date)"
echo "==============================="

python scripts/run_benchmark.py \
    --config       "$CONFIG" \
    --adapter-path "$ADAPTER_PATH" \
    --run-id       "$RUN_ID" \
    --output       "$OUTPUT" \
    --report-path  "$REPORT"

echo "=== Done: $(date) ==="
echo "Report: $REPORT"

# DEV NOTE: Before submitting 20 jobs, always validate with:
#   sbatch --wrap="bash scripts/babel_benchmark_eval.sbatch harmbench baseline_v5" --time=00:30:00
# or add --limit 5 support to run_benchmark.py for a 5-prompt smoke test.
