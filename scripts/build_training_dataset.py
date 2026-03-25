"""
Regime-specific training dataset construction.

Contract
--------
Input:  DPORecord JSONL — the full assembled dataset (all pair types, all categories)
        Produced by assemble_dpo_dataset.py from the merged pair artifact.
Config: Training-dataset YAML — regime name, include_rules, output paths
Output: Regime-specific DPORecord JSONL, sorted by pair_id

Why this stage exists
---------------------
Global pair_type filtering (e.g. filtering by pair_type=safe_completion) is
insufficient because:
  - Both training regimes share unsafe pairs (baseline_refusal) and
    benign_sensitive / benign pairs (safe_completion).
  - The safecomp regime includes ALL safe_completion dual_use pairs
    (two per prompt: safe>hard_refusal and safe>unsafe_compliance).
  - The baseline regime includes baseline_refusal dual_use pairs
    (one per prompt: hard_refusal>unsafe_compliance).
  - A single pair_type filter cannot distinguish which regime a pair belongs to.

This stage applies per-category selection rules to build an unambiguous,
regime-specific training set.

Regime-specific include_rules
------------------------------
Each rule specifies a (category, pair_type) combination to include.
A DPORecord is selected if its (category, pair_type) matches ANY rule.

Baseline regime:
  unsafe          baseline_refusal  (hard_refusal > unsafe_compliance)
  dual_use        baseline_refusal  (hard_refusal > unsafe_compliance)
  benign_sensitive safe_completion  (helpful_answer > hard_refusal)
  benign          safe_completion   (helpful_answer > hard_refusal)

SafeComp regime:
  unsafe          baseline_refusal  (hard_refusal > unsafe_compliance)  [shared]
  dual_use        safe_completion   (both p01 safe>hard and p02 safe>unsafe_compliance)
  benign_sensitive safe_completion  (helpful_answer > hard_refusal)
  benign          safe_completion   (helpful_answer > hard_refusal)

Usage
-----
    python scripts/build_training_dataset.py \\
        --config configs/training-datasets/baseline.yaml

    python scripts/build_training_dataset.py \\
        --config configs/training-datasets/safecomp.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

try:
    from safecomp_dpo.io import load_dpo_records, write_jsonl
    from safecomp_dpo.schemas import DPORecord
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_dpo_records, write_jsonl
    from safecomp_dpo.schemas import DPORecord


# ---------------------------------------------------------------------------
# Core selection logic (pure — no I/O)
# ---------------------------------------------------------------------------


def select_for_regime(
    records: list[DPORecord],
    include_rules: list[dict[str, str]],
) -> tuple[list[DPORecord], dict[str, Any]]:
    """Select DPO records matching the regime's include rules.

    Args:
        records:      full DPORecord list (all categories and pair types)
        include_rules: list of {category, pair_type} dicts.
                       A record is included if its (category, pair_type)
                       matches ANY rule.

    Returns:
        (selected, stats)
        selected: list of DPORecord matching the rules, sorted by pair_id
        stats:    selection statistics dict
    """
    if not include_rules:
        print(
            "  WARNING: include_rules is empty — no records will be selected.",
            file=sys.stderr,
        )

    allowed: set[tuple[str, str]] = {
        (rule["category"], rule["pair_type"])
        for rule in include_rules
    }

    selected = [
        r for r in records
        if (r.category.value, r.pair_type.value) in allowed
    ]
    selected.sort(key=lambda r: r.pair_id)

    n_total = len(records)
    n_selected = len(selected)

    category_dist: dict[str, int] = defaultdict(int)
    pair_type_dist: dict[str, int] = defaultdict(int)
    for r in selected:
        category_dist[r.category.value] += 1
        pair_type_dist[r.pair_type.value] += 1

    stats: dict[str, Any] = {
        "total_input_records": n_total,
        "selected_records": n_selected,
        "excluded_records": n_total - n_selected,
        "include_rules": include_rules,
        "category_distribution": dict(category_dist),
        "pair_type_distribution": dict(pair_type_dist),
    }
    return selected, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a regime-specific DPO training dataset from the full pair set.",
    )
    parser.add_argument(
        "--config", required=True, help="Path to training-dataset YAML config"
    )
    parser.add_argument("--input", help="Override input_path from config")
    parser.add_argument("--output", help="Override output_path from config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only; do not write output",
    )
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    regime: str = config.get("regime", "unknown")
    include_rules: list[dict[str, str]] = config.get("include_rules", [])

    input_path = args.input or config.get("input_path")
    if not input_path:
        print(
            "ERROR: specify --input or input_path in config.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Regime: {regime}")
    print(f"Loading DPO records from {input_path}")
    records = load_dpo_records(input_path)
    print(f"  {len(records)} records loaded")

    print(f"Applying {len(include_rules)} include rule(s)...")
    selected, stats = select_for_regime(records, include_rules)

    print(f"  Total input:    {stats['total_input_records']}")
    print(f"  Selected:       {stats['selected_records']}")
    print(f"  Excluded:       {stats['excluded_records']}")
    if stats["category_distribution"]:
        print(f"  Category dist:  {stats['category_distribution']}")
    if stats["pair_type_distribution"]:
        print(f"  Pair type dist: {stats['pair_type_distribution']}")

    if args.dry_run:
        print("Dry run — no output written.")
        return

    output_path = args.output or config.get("output_path")
    if not output_path:
        print("ERROR: specify --output or output_path in config.", file=sys.stderr)
        sys.exit(1)

    write_jsonl(selected, output_path)
    print(f"Wrote {len(selected)} records to {output_path}")

    sample_path = config.get("sample_path")
    sample_n = int(config.get("sample_n", 5))
    if sample_path and selected:
        write_jsonl(selected[:sample_n], sample_path)
        print(f"Wrote {min(sample_n, len(selected))} sample records to {sample_path}")

    report_path = config.get("report_path")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        stats["regime"] = regime
        stats["config_path"] = config_path
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote report to {report_path}")

    print("Done.")


if __name__ == "__main__":
    main()
