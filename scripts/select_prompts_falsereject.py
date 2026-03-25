"""
Select a stratified subset of FalseReject benign_sensitive prompts.

Reads the acquired PromptRecord JSONL pool and writes a downsampled subset
using two-phase stratified allocation by fr_category_id.

Two-phase allocation:
    Phase 1 (base): assign each category min(count_i, min_per_category) records.
    Phase 2 (remainder): distribute the remaining budget proportionally over
                         remaining per-category capacity using largest-remainder
                         allocation to reach exactly target_n.

Contract:
    Input:  Acquired PromptRecord JSONL (falsereject_benign_sensitive.jsonl)
    Output: Selected PromptRecord JSONL — unchanged schema, strict subset of input
            JSON selection report — per-category allocation and summary stats

Usage:
    python scripts/select_prompts_falsereject.py --config configs/selection/falsereject.yaml
    python scripts/select_prompts_falsereject.py --config configs/selection/falsereject.yaml --dry-run
    python scripts/select_prompts_falsereject.py --config configs/selection/falsereject.yaml \\
        --input outputs/smoke/falsereject_benign_sensitive.jsonl \\
        --output outputs/smoke/falsereject_selected.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from safecomp_dpo.io import load_prompt_records, write_jsonl
from safecomp_dpo.schemas import PromptRecord
from safecomp_dpo.selection import allocate as _allocate


# ---------------------------------------------------------------------------
# Selection (pure — no file I/O, fully testable)
# ---------------------------------------------------------------------------


def select_records(
    records: list[PromptRecord],
    target_n: int,
    seed: int,
    min_per_category: int,
    stratify_by: str,
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Stratified selection with two-phase allocation.

    Categories are derived dynamically from the input records — no count is
    assumed. Records within each category are sorted by prompt_id for stable,
    file-order-independent determinism. Sampling uses random.Random(seed).

    Args:
        records: full acquired pool
        target_n: number of records to select
        seed: RNG seed for reproducibility
        min_per_category: base floor per category (Phase 1)
        stratify_by: metadata key to group on (e.g. 'fr_category_id')

    Returns:
        (selected, report) where selected is a list of PromptRecords and
        report is a dict with allocation details.
    """
    # Group by stratification key — derived dynamically
    groups: dict[Any, list[PromptRecord]] = defaultdict(list)
    for r in records:
        key = r.metadata[stratify_by]
        groups[key].append(r)

    # Stable sort within each category by prompt_id
    for key in groups:
        groups[key].sort(key=lambda r: r.prompt_id)

    category_counts = {k: len(v) for k, v in groups.items()}
    quota = _allocate(category_counts, target_n, min_per_category)

    # Seeded sampling in sorted category key order for full determinism
    rng = random.Random(seed)
    selected: list[PromptRecord] = []
    category_report: list[dict[str, Any]] = []

    for key in sorted(groups.keys()):
        group = groups[key]
        q = min(quota[key], len(group))  # safety cap for floating-point edge cases
        chosen = rng.sample(group, q)
        selected.extend(chosen)

        cat_text = group[0].metadata.get("fr_category_text", str(key))
        category_report.append({
            stratify_by: key,
            "fr_category_text": cat_text,
            "source_count": len(group),
            "selected_count": q,
            "selection_rate": round(q / len(group), 4) if len(group) > 0 else 0.0,
        })

    report: dict[str, Any] = {
        "source": "falsereject",
        "strategy": "stratified",
        "stratify_by": stratify_by,
        "seed": seed,
        "min_per_category": min_per_category,
        "target_n": target_n,
        "actual_n": len(selected),
        "total_source_records": len(records),
        "n_categories_in_pool": len(groups),
        "categories": category_report,
    }
    return selected, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select stratified FalseReject benign_sensitive prompts"
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
        stratify_by=config["stratify_by"],
    )

    # Summary
    print(f"Source records:     {report['total_source_records']}")
    print(f"Categories in pool: {report['n_categories_in_pool']}")
    print(f"Target:             {report['target_n']}")
    print(f"Actual selected:    {report['actual_n']}")
    print(f"\nPer-category allocation (sorted by category id):")
    print(f"  {'ID':<5}  {'Category':<52}  {'Source':>7}  {'Selected':>9}  {'Rate':>6}")
    print(f"  {'-'*5}  {'-'*52}  {'-'*7}  {'-'*9}  {'-'*6}")
    for cat in report["categories"]:
        print(
            f"  {cat[config['stratify_by']]:<5}  {cat['fr_category_text']:<52}  "
            f"{cat['source_count']:>7}  {cat['selected_count']:>9}  "
            f"{cat['selection_rate']:>6.3f}"
        )

    if args.dry_run:
        print("\nDry run — no output written.")
        return

    write_jsonl(selected, output_path)
    print(f"\nWrote {len(selected)} records to {output_path}")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
