"""
Select a stratified subset of Alpaca benign prompts.

Reads the acquired PromptRecord JSONL pool and writes a downsampled subset
using two-phase stratified allocation over keyword-derived task buckets.

Task bucket assignment:
    Each record's prompt is tokenized (lowercase + strip surrounding punctuation),
    then the first normalized token (and first 15 tokens for coding) are matched
    against keyword sets to assign one of eight buckets:
        coding, math_reasoning, editing_rewriting, summarization,
        classification_analysis, generation, factual_instructional, other

    Precedence (first match wins):
        1. coding  — any coding keyword anywhere in first 15 tokens
        2. math_reasoning  — first token prefix-matches a math stem
        3. editing_rewriting  — first token in EDITING_VERBS
        4. summarization  — first token in SUMMARY_VERBS
        5. classification_analysis  — first token in CLASS_VERBS
        6. generation  — first token in GEN_VERBS
        7. factual_instructional  — first token in FACTUAL_WORDS
        8. other  — catch-all

    Bucket assignment is a selection-time artifact only.
    It is NOT written into PromptRecord metadata.

Two-phase allocation (shared with other selection scripts):
    Phase 1: base_i = min(count_i, min_per_category)
    Phase 2: distribute remaining budget proportionally using largest-remainder.

Contract:
    Input:  Acquired PromptRecord JSONL (alpaca_benign.jsonl)
    Output: Selected PromptRecord JSONL — unchanged schema, strict subset of input
            JSON selection report — bucket_allocation + has_input_summary

Usage:
    python scripts/select_prompts_alpaca.py --config configs/selection/alpaca.yaml
    python scripts/select_prompts_alpaca.py --config configs/selection/alpaca.yaml --dry-run
    python scripts/select_prompts_alpaca.py --config configs/selection/alpaca.yaml \\
        --input outputs/smoke/alpaca_benign.jsonl \\
        --output outputs/smoke/alpaca_selected.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from safecomp_dpo.io import load_prompt_records, write_jsonl
from safecomp_dpo.schemas import PromptRecord
from safecomp_dpo.selection import allocate


# ---------------------------------------------------------------------------
# Bucket keyword sets
# ---------------------------------------------------------------------------

_CODING_WORDS = frozenset({
    "code", "program", "implement", "debug", "algorithm", "script", "sql", "html", "css",
})

_MATH_PREFIXES = (
    "calculat", "equation", "formula", "mathemat", "probabilit", "statistic", "geometr",
)

_EDITING_VERBS = frozenset({
    "edit", "rewrite", "rephrase", "paraphrase", "revise", "proofread", "correct",
    "improve", "refine", "fix", "translate", "simplify", "shorten", "expand", "convert",
})

_SUMMARY_VERBS = frozenset({
    "summarize", "summarise", "condense", "outline", "abstract", "recap",
})

_CLASS_VERBS = frozenset({
    "classify", "categorize", "categorise", "identify", "detect", "label",
    "determine", "evaluate", "assess", "compare", "contrast", "analyze", "analyse",
    "rank", "rate", "score", "find",
})

_GEN_VERBS = frozenset({
    "write", "compose", "create", "generate", "draft", "design", "brainstorm",
    "recommend", "plan", "come", "make",
})

_FACTUAL_WORDS = frozenset({
    "what", "who", "when", "where", "why", "how", "which", "define",
    "describe", "explain", "name", "list", "give", "provide", "suggest", "tell",
})

TASK_BUCKETS = [
    "coding",
    "math_reasoning",
    "editing_rewriting",
    "summarization",
    "classification_analysis",
    "generation",
    "factual_instructional",
    "other",
]


# ---------------------------------------------------------------------------
# Token normalization and bucket assignment (pure — no I/O, fully testable)
# ---------------------------------------------------------------------------


def normalize_tokens(text: str, n: int = 15) -> list[str]:
    """Lowercase and strip surrounding punctuation from each token.

    Args:
        text: raw prompt string
        n: maximum number of tokens to return

    Returns:
        list of normalized tokens (length <= n), empty strings removed
    """
    tokens = text.strip().lower().split()[:n]
    return [t.strip(string.punctuation) for t in tokens]


def assign_task_bucket(prompt: str) -> str:
    """Assign a task bucket to a prompt using keyword matching.

    See module docstring for full precedence rules.

    Args:
        prompt: raw prompt string

    Returns:
        bucket name string (one of TASK_BUCKETS)
    """
    tokens = normalize_tokens(prompt, 15)
    w0 = tokens[0] if tokens else ""

    if any(w in _CODING_WORDS for w in tokens):
        return "coding"
    if any(w0.startswith(p) for p in _MATH_PREFIXES):
        return "math_reasoning"
    if w0 in _EDITING_VERBS:
        return "editing_rewriting"
    if w0 in _SUMMARY_VERBS:
        return "summarization"
    if w0 in _CLASS_VERBS:
        return "classification_analysis"
    if w0 in _GEN_VERBS:
        return "generation"
    if w0 in _FACTUAL_WORDS:
        return "factual_instructional"
    return "other"


# ---------------------------------------------------------------------------
# Selection (pure — no file I/O, fully testable)
# ---------------------------------------------------------------------------


def select_records(
    records: list[PromptRecord],
    target_n: int,
    seed: int,
    min_per_category: int,
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Stratified selection using keyword-derived task buckets.

    Bucket assignment is computed at selection time only; it is NOT written
    into PromptRecord metadata. Selected records are an unchanged strict subset
    of the input pool. Records within each bucket are sorted by prompt_id for
    stable, file-order-independent determinism.

    Args:
        records: full acquired pool
        target_n: number of records to select
        seed: RNG seed for reproducibility
        min_per_category: base floor per bucket (Phase 1)

    Returns:
        (selected, report) — selected is a strict subset of records (unchanged);
        report contains bucket_allocation and has_input_summary.
    """
    # Step 1: assign buckets and group records
    groups: dict[str, list[PromptRecord]] = defaultdict(list)
    for r in records:
        bucket = assign_task_bucket(r.prompt)
        groups[bucket].append(r)

    # Step 2: stable sort within each bucket by prompt_id
    for bucket in groups:
        groups[bucket].sort(key=lambda r: r.prompt_id)

    # Step 3: two-phase allocation
    bucket_counts = {bucket: len(group) for bucket, group in groups.items()}
    quota = allocate(bucket_counts, target_n, min_per_category)

    # Step 4: seeded sampling in sorted bucket order
    rng = random.Random(seed)
    selected: list[PromptRecord] = []
    bucket_alloc: list[dict[str, Any]] = []

    for bucket in sorted(groups.keys()):
        group = groups[bucket]
        q = min(quota[bucket], len(group))
        chosen = rng.sample(group, q)
        selected.extend(chosen)

        with_input = sum(1 for r in chosen if r.metadata.get("alpaca_has_input", False))
        bucket_alloc.append({
            "task_bucket": bucket,
            "source_count": len(group),
            "selected_count": q,
            "selection_rate": round(q / len(group), 4) if len(group) > 0 else 0.0,
            "has_input_in_selected": with_input,
            "has_input_rate_in_selected": round(with_input / q, 4) if q > 0 else 0.0,
        })

    # Step 5: overall has_input summary
    total_with_input = sum(1 for r in records if r.metadata.get("alpaca_has_input", False))
    selected_with_input = sum(1 for r in selected if r.metadata.get("alpaca_has_input", False))

    report: dict[str, Any] = {
        "source": "alpaca",
        "strategy": "stratified",
        "stratify_by": "task_bucket",
        "seed": seed,
        "min_per_category": min_per_category,
        "target_n": target_n,
        "actual_n": len(selected),
        "total_source_records": len(records),
        "n_buckets_in_pool": len(groups),
        "bucket_allocation": bucket_alloc,
        "has_input_summary": {
            "source_with_input": total_with_input,
            "source_without_input": len(records) - total_with_input,
            "selected_with_input": selected_with_input,
            "selected_without_input": len(selected) - selected_with_input,
            "selected_with_input_rate": (
                round(selected_with_input / len(selected), 4) if selected else 0.0
            ),
        },
    }
    return selected, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select stratified Alpaca benign prompts"
    )
    parser.add_argument("--config", required=True, help="Path to selection config YAML")
    parser.add_argument("--input", help="Override input_path from config")
    parser.add_argument("--output", help="Override output_path from config")
    parser.add_argument("--dry-run", action="store_true", help="Print allocation only; do not write")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    input_path = args.input or config["input_path"]
    output_path = args.output or config["output_path"]
    report_path = config["report_path"]

    records = load_prompt_records(input_path)
    selected, report = select_records(
        records=records,
        target_n=config["target_n"],
        seed=config["seed"],
        min_per_category=config["min_per_category"],
    )

    # Summary
    print(f"Source records:   {report['total_source_records']}")
    print(f"Buckets in pool:  {report['n_buckets_in_pool']}")
    print(f"Target:           {report['target_n']}")
    print(f"Actual selected:  {report['actual_n']}")

    hi = report["has_input_summary"]
    print(f"\nHas-input (source):   {hi['source_with_input']} with / {hi['source_without_input']} without")
    print(f"Has-input (selected): {hi['selected_with_input']} with / {hi['selected_without_input']} without")

    print(f"\nBucket allocation:")
    print(f"  {'Bucket':<30}  {'Source':>7}  {'Selected':>9}  {'Rate':>6}  {'w/input':>8}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*9}  {'-'*6}  {'-'*8}")
    for row in report["bucket_allocation"]:
        print(
            f"  {row['task_bucket']:<30}  "
            f"{row['source_count']:>7}  {row['selected_count']:>9}  "
            f"{row['selection_rate']:>6.3f}  {row['has_input_in_selected']:>8}"
        )

    if args.dry_run:
        print("\nDry run -- no output written.")
        return

    write_jsonl(selected, output_path)
    print(f"\nWrote {len(selected)} records to {output_path}")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
