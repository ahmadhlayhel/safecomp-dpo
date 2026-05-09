"""
Standard benchmark evaluation entrypoint.

Contract
--------
Config: Benchmark YAML — benchmark name, model/scorer backends, paths
Output: BenchmarkRecord JSONL  (per-example results)
        Report JSON             (benchmark-specific metrics summary)

Supported benchmarks (--benchmark / config["benchmark"]):
    harmbench        Attack Success Rate (ASR)
    strongreject     StrongREJECT composite score
    do_not_answer    Percent harmful content
    xstest           ORR (safe split) + ASR (unsafe split)
    or_bench         Over-Refusal Rate (benign subset)
    falsereject_test Compliance / helpful-answer rate

Mock vs real
------------
All benchmarks run on mock data by default (no real dataset needed).
Set dataset_path in the config to use real benchmark data once downloaded.
Set model_backend / scorer_backend to non-mock values on BABEL for real runs.

Usage
-----
    python scripts/run_benchmark.py --config configs/eval/benchmarks/harmbench.yaml
    python scripts/run_benchmark.py --config configs/eval/benchmarks/xstest.yaml --dry-run
    python scripts/run_benchmark.py --config configs/eval/benchmarks/falsereject_test.yaml \\
        --output outputs/eval/benchmarks/falsereject_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

try:
    from safecomp_dpo.benchmarks import (
        get_adapter,
        get_model_backend,
        get_scorer_backend,
        run_benchmark,
    )
    from safecomp_dpo.io import write_jsonl
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.benchmarks import (
        get_adapter,
        get_model_backend,
        get_scorer_backend,
        run_benchmark,
    )
    from safecomp_dpo.io import write_jsonl


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a standard safety/over-refusal benchmark evaluation.",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to benchmark YAML config (e.g. configs/eval/benchmarks/harmbench_peft.yaml).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override output_path from config.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run benchmark in-memory and print metrics; do not write output files.",
    )
    # CLI overrides — let sbatch scripts parameterise a single base config
    p.add_argument(
        "--adapter-path",
        default=None,
        help="Override config adapter_path (use 'none' for base model, no adapter).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Override config run_id (e.g. harmbench_baseline_v5).",
    )
    p.add_argument(
        "--report-path",
        default=None,
        help="Override config report_path.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides (let sbatch scripts parameterise a single base config).
    if args.adapter_path is not None:
        # "none" sentinel means base model — no adapter.
        config["adapter_path"] = None if args.adapter_path.lower() == "none" else args.adapter_path
    if args.run_id is not None:
        config["run_id"] = args.run_id
    if args.output is not None:
        config["output_path"] = args.output
    if args.report_path is not None:
        config["report_path"] = args.report_path

    benchmark_name: str = config.get("benchmark", "")
    if not benchmark_name:
        print("ERROR: config must specify 'benchmark' key.", file=sys.stderr)
        sys.exit(1)

    run_id = config.get("run_id", "unknown_run")
    model_backend_name = config.get("model_backend", "mock")
    scorer_backend_name = config.get("scorer_backend", "mock")

    print(f"Benchmark:      {benchmark_name}")
    print(f"Run ID:         {run_id}")
    print(f"Model backend:  {model_backend_name}")
    print(f"Adapter path:   {config.get('adapter_path', 'none (base model)')}")
    print(f"Scorer backend: {scorer_backend_name}")
    if config.get("mock_behavior"):
        print(f"Mock behavior:  {config['mock_behavior']}")

    adapter = get_adapter(benchmark_name)
    model_backend = get_model_backend(model_backend_name, config)
    scorer = get_scorer_backend(scorer_backend_name)

    print(f"Loading {benchmark_name} prompts...")
    prompts = adapter.load(config)
    print(f"  {len(prompts)} prompts loaded")

    print("Running benchmark...")
    records = run_benchmark(prompts, model_backend, scorer, config, str(config_path))
    print(f"  {len(records)} records produced")

    print("Computing metrics...")
    metrics = adapter.compute_metrics(records)
    _print_metrics(metrics)

    if args.dry_run:
        print("Dry run — no output written.")
        return

    output_path = config.get("output_path") or args.output
    if output_path:
        write_jsonl(records, output_path)
        print(f"Wrote {len(records)} records to {output_path}")

    report_path = config.get("report_path")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, **metrics}, f, indent=2)
        print(f"Wrote report to {report_path}")

    print("Done.")


def _print_metrics(metrics: dict) -> None:
    """Print benchmark metrics in a readable format."""
    benchmark = metrics.get("benchmark", "")
    # XSTest has nested safe/unsafe structure
    if "safe" in metrics and "unsafe" in metrics:
        safe = metrics["safe"]
        unsafe = metrics["unsafe"]
        print(f"  {benchmark} safe  : n={safe['n']} | orr={safe.get('orr')}")
        print(f"  {benchmark} unsafe: n={unsafe['n']} | asr={unsafe.get('asr')}")
    else:
        parts = [f"{k}={v}" for k, v in metrics.items()
                 if k not in ("benchmark",)]
        print(f"  {benchmark}: {' | '.join(parts)}")


if __name__ == "__main__":
    main()
