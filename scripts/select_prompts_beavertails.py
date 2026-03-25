"""
Select a stratified subset of BeaverTails unsafe prompts.

Reads the acquired PromptRecord JSONL pool and writes a downsampled subset
using two-phase stratified allocation over derived primary harm labels.

Primary label assignment (least_frequent_active rule):
    Each record's bt_active_categories contains one or more atomic harm label
    strings (e.g. 'violence,aiding_and_abetting,incitement'). Labels are
    treated as opaque identifiers — never split on commas.

    pool_freq[k] = count of pool records where k appears in bt_active_categories
    bt_primary_category = argmin_{k in active_categories} pool_freq[k]
    Tie-break: alphabetical order (first alphabetically wins).

Two-phase allocation (shared with FalseReject selection):
    Phase 1: base_i = min(count_i, min_per_category)
    Phase 2: distribute remaining budget proportionally over remaining capacity
             using largest-remainder allocation.

Contract:
    Input:  Acquired PromptRecord JSONL (beavertails_unsafe.jsonl)
    Output: Selected PromptRecord JSONL — unchanged schema, strict subset of input
            JSON selection report — two views:
              primary_label_allocation  (stratification used for selection)
              active_category_coverage  (multi-label harm presence in selected subset)

Usage:
    python scripts/select_prompts_beavertails.py --config configs/selection/beavertails.yaml
    python scripts/select_prompts_beavertails.py --config configs/selection/beavertails.yaml --dry-run
    python scripts/select_prompts_beavertails.py --config configs/selection/beavertails.yaml \\
        --input outputs/smoke/beavertails_unsafe.jsonl \\
        --output outputs/smoke/beavertails_selected.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from safecomp_dpo.io import load_prompt_records, write_jsonl
from safecomp_dpo.schemas import PromptRecord
from safecomp_dpo.selection import allocate


# ---------------------------------------------------------------------------
# Primary label assignment (pure — no I/O, fully testable)
# ---------------------------------------------------------------------------


def compute_pool_freq(records: list[PromptRecord]) -> dict[str, int]:
    """Count how many pool records contain each category label.

    bt_active_categories entries are treated as atomic strings.
    They must never be split on commas.
    """
    freq: Counter[str] = Counter()
    for r in records:
        for label in r.metadata["bt_active_categories"]:
            freq[label] += 1
    return dict(freq)


def assign_primary_labels(
    records: list[PromptRecord],
    pool_freq: dict[str, int],
) -> dict[str, str]:
    """Assign one primary label per record using least_frequent_active.

    For each record, select the active category with the lowest pool_freq.
    Tie-break: alphabetical order (first alphabetically wins).

    Args:
        records: pool records
        pool_freq: mapping from label string to pool-wide occurrence count

    Returns:
        dict mapping prompt_id -> assigned primary label
    """
    assignment: dict[str, str] = {}
    for r in records:
        active = r.metadata["bt_active_categories"]
        primary = min(active, key=lambda label: (pool_freq[label], label))
        assignment[r.prompt_id] = primary
    return assignment


# ---------------------------------------------------------------------------
# Selection (pure — no file I/O, fully testable)
# ---------------------------------------------------------------------------


def select_records(
    records: list[PromptRecord],
    target_n: int,
    seed: int,
    min_per_category: int,
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Stratified selection with least_frequent_active primary label assignment.

    Categories (primary labels) are derived dynamically from the pool.
    Records within each category are sorted by prompt_id for stable ordering.
    Sampling uses random.Random(seed).

    Args:
        records: full acquired pool
        target_n: number of records to select
        seed: RNG seed for reproducibility
        min_per_category: base floor per primary label bucket (Phase 1)

    Returns:
        (selected, report) — selected is a strict subset of records (unchanged),
        report contains primary_label_allocation and active_category_coverage.
    """
    # Step 1: derive primary labels
    pool_freq = compute_pool_freq(records)
    assignment = assign_primary_labels(records, pool_freq)

    # Step 2: group by primary label; sort each group by prompt_id
    groups: dict[str, list[PromptRecord]] = defaultdict(list)
    for r in records:
        groups[assignment[r.prompt_id]].append(r)
    for label in groups:
        groups[label].sort(key=lambda r: r.prompt_id)

    # Step 3: two-phase allocation
    category_counts = {label: len(group) for label, group in groups.items()}
    quota = allocate(category_counts, target_n, min_per_category)

    # Step 4: seeded sampling in sorted label order
    rng = random.Random(seed)
    selected: list[PromptRecord] = []
    primary_alloc: list[dict[str, Any]] = []

    for label in sorted(groups.keys()):
        group = groups[label]
        q = min(quota[label], len(group))
        chosen = rng.sample(group, q)
        selected.extend(chosen)
        primary_alloc.append({
            "bt_primary_category": label,
            "source_count": len(group),
            "selected_count": q,
            "selection_rate": round(q / len(group), 4) if len(group) > 0 else 0.0,
        })

    # Step 5: active-category coverage over selected records
    selected_active_freq: Counter[str] = Counter()
    for r in selected:
        for label in r.metadata["bt_active_categories"]:
            selected_active_freq[label] += 1

    active_coverage: list[dict[str, Any]] = []
    for label in sorted(pool_freq.keys()):
        active_coverage.append({
            "bt_category": label,
            "records_with_category_in_source": pool_freq[label],
            "records_with_category_in_selected": selected_active_freq.get(label, 0),
            "coverage_rate": round(
                selected_active_freq.get(label, 0) / pool_freq[label], 4
            ) if pool_freq[label] > 0 else 0.0,
        })

    report: dict[str, Any] = {
        "source": "beavertails",
        "strategy": "stratified",
        "primary_label_rule": "least_frequent_active",
        "seed": seed,
        "min_per_category": min_per_category,
        "target_n": target_n,
        "actual_n": len(selected),
        "total_source_records": len(records),
        "n_categories_in_pool": len(groups),
        "primary_label_allocation": primary_alloc,
        "active_category_coverage": active_coverage,
    }
    return selected, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select stratified BeaverTails unsafe prompts"
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
    print(f"Source records:     {report['total_source_records']}")
    print(f"Categories in pool: {report['n_categories_in_pool']}")
    print(f"Target:             {report['target_n']}")
    print(f"Actual selected:    {report['actual_n']}")

    print(f"\nPrimary-label allocation:")
    print(f"  {'Primary label':<55}  {'Source':>7}  {'Selected':>9}  {'Rate':>6}")
    print(f"  {'-'*55}  {'-'*7}  {'-'*9}  {'-'*6}")
    for row in report["primary_label_allocation"]:
        print(
            f"  {row['bt_primary_category']:<55}  "
            f"{row['source_count']:>7}  {row['selected_count']:>9}  "
            f"{row['selection_rate']:>6.3f}"
        )

    print(f"\nActive-category coverage (selected subset):")
    print(f"  {'Category':<55}  {'In source':>9}  {'In selected':>12}  {'Rate':>6}")
    print(f"  {'-'*55}  {'-'*9}  {'-'*12}  {'-'*6}")
    for row in report["active_category_coverage"]:
        print(
            f"  {row['bt_category']:<55}  "
            f"{row['records_with_category_in_source']:>9}  "
            f"{row['records_with_category_in_selected']:>12}  "
            f"{row['coverage_rate']:>6.3f}"
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
