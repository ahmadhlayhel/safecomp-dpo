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
        run_benchmark_from_responses,
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
        run_benchmark_from_responses,
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
    p.add_argument(
        "--rescore-from",
        default=None,
        metavar="JSONL",
        help=(
            "Skip generation entirely. Load (prompt, response) pairs from an "
            "existing BenchmarkRecord JSONL and re-score with the configured "
            "scorer. Useful for applying GPT-4o scoring to WildGuard-scored "
            "output without running inference again."
        ),
    )
    p.add_argument(
        "--scorer",
        default=None,
        help="Override scorer_backend from config (e.g. gpt4o, wildguard, regex).",
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
    if args.scorer is not None:
        config["scorer_backend"] = args.scorer

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

    if args.rescore_from:
        # ── Re-score mode: load existing responses, skip generation ───────────
        print(f"Re-score mode: loading responses from {args.rescore_from}")
        prompts, responses = _load_responses_jsonl(args.rescore_from)
        print(f"  {len(prompts)} (prompt, response) pairs loaded")
        # Use a mock model backend as a placeholder (no generation happens).
        from safecomp_dpo.benchmarks import MockBenchmarkModelBackend  # noqa: PLC0415
        model_backend = MockBenchmarkModelBackend()
    else:
        model_backend = get_model_backend(model_backend_name, config)
        print(f"Loading {benchmark_name} prompts...")
        prompts = adapter.load(config)
        print(f"  {len(prompts)} prompts loaded")

        # When using WildGuard as scorer alongside a PEFT generation backend, both
        # are 7B models.  Loading them simultaneously exceeds 40GB VRAM.  So we
        # generate all responses first, unload the generation model, then load
        # WildGuard for scoring.  For mock/regex scorers this has no effect.
        print("Generating responses...")
        responses = _generate_all(prompts, model_backend, config)
        print(f"  {len(responses)} responses generated")

        # Unload generation model before loading scorer (frees VRAM).
        _try_unload(model_backend)

    print(f"Loading scorer ({scorer_backend_name})...")
    scorer = get_scorer_backend(scorer_backend_name)

    print("Scoring responses...")
    records = run_benchmark_from_responses(
        prompts, responses, model_backend, scorer, config, str(config_path)
    )
    print(f"  {len(records)} records scored")

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


def _load_responses_jsonl(path: str):
    """Load (BenchmarkPrompt, response_str) pairs from an existing scored JSONL.

    Used by --rescore-from to skip generation and only re-run the scorer.
    """
    import json as _json  # noqa: PLC0415
    from safecomp_dpo.benchmarks import BenchmarkPrompt  # noqa: PLC0415
    prompts_out = []
    responses_out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            prompts_out.append(BenchmarkPrompt(
                prompt_id=obj.get("metadata", {}).get("prompt_id", obj.get("record_id", "")),
                benchmark=obj.get("benchmark", ""),
                prompt=obj["prompt"],
                split=obj.get("split"),
                metadata=obj.get("metadata", {}),
            ))
            responses_out.append(obj["response"])
    return prompts_out, responses_out


def _generate_all(prompts, model_backend, config) -> list[str]:
    """Generate responses for all prompts, printing progress every 50."""
    responses = []
    n = len(prompts)
    for i, prompt in enumerate(prompts, 1):
        call_cfg = dict(config)
        if prompt.split is not None:
            call_cfg["_mock_split"] = prompt.split
        responses.append(model_backend.generate(prompt.prompt, call_cfg))
        if i % 50 == 0 or i == n:
            print(f"  [gen] {i}/{n}", flush=True)
    return responses


def _try_unload(model_backend) -> None:
    """Delete cached model weights to free VRAM before loading the scorer."""
    import gc  # noqa: PLC0415
    if hasattr(model_backend, "_model") and model_backend._model is not None:
        try:
            import torch  # noqa: PLC0415
            del model_backend._model
            model_backend._model = None
            model_backend._tok = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[run_benchmark] generation model unloaded from VRAM", flush=True)
        except Exception:
            pass


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
