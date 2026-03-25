"""
Acquire FalseReject benign_sensitive prompts and write PromptRecord JSONL.

FalseReject (AmazonScience/FalseReject) is the v1 source pool for benign_sensitive
acquisition. Every row in the configured split is a candidate — the dataset is
constructed to contain only prompts that appear sensitive but are genuinely safe
to answer. Acquisition collects the full eligible pool; final curation and
stratified selection happen in a later stage.

Contract:
    Input:  AmazonScience/FalseReject HF dataset (configured split)
    Output: PromptRecord JSONL — one record per unique benign_sensitive prompt
    Schema: PromptRecord with category=benign_sensitive, created_at=null,
            fr_category_id/fr_category_text/fr_duplicate_* in metadata

Usage:
    python scripts/acquire_prompts_falsereject.py --config configs/acquisition/falsereject.yaml
    python scripts/acquire_prompts_falsereject.py --config configs/acquisition/falsereject.yaml --dry-run
    python scripts/acquire_prompts_falsereject.py --config configs/acquisition/falsereject.yaml \\
        --output outputs/smoke/falsereject_benign_sensitive.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from safecomp_dpo.io import write_jsonl
from safecomp_dpo.schemas import PromptCategory, PromptRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Canonical key for deduplication: strip + lowercase."""
    return text.strip().lower()


def _prompt_id(split: str, index: int) -> str:
    safe_split = split.replace("-", "_")
    return f"fr_{safe_split}_{index:06d}"


# ---------------------------------------------------------------------------
# Core transformation (pure — no HF dependency, fully testable)
# ---------------------------------------------------------------------------


def process_rows(
    rows: list[dict[str, Any]],
    split: str,
    min_prompt_chars: int,
    config_path: str,
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Filter, deduplicate, and build PromptRecords from raw FalseReject rows.

    All rows in the configured split are candidates (no label filter required).
    Only min_prompt_chars and exact-match deduplication are applied.

    Args:
        rows: raw dicts from the HF dataset (keys: prompt, category, category_text, ...)
        split: exact HF split name (used in prompt_id and metadata)
        min_prompt_chars: minimum character length after stripping
        config_path: path string stored in metadata for provenance

    Returns:
        (records, stats) where stats includes row counts and category distribution
    """
    # Stage 1: filter by min_prompt_chars (all rows are benign_sensitive candidates)
    eligible: list[tuple[int, dict[str, Any]]] = []
    n_short = 0
    for idx, row in enumerate(rows):
        text = row["prompt"].strip()
        if len(text) < min_prompt_chars:
            n_short += 1
            continue
        eligible.append((idx, row))

    # Stage 2: group by normalized prompt for deduplication
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in eligible:
        key = _normalize(row["prompt"])
        groups[key].append((idx, row))

    # Stage 3: build one PromptRecord per group
    records: list[PromptRecord] = []
    for key in sorted(groups):  # deterministic ordering
        group = sorted(groups[key], key=lambda x: x[0])
        canonical_idx, canonical_row = group[0]
        # Sort source ids for determinism; length must equal duplicate_count
        all_source_ids = sorted(str(i) for i, _ in group)

        record = PromptRecord(
            prompt_id=_prompt_id(split, canonical_idx),
            prompt=canonical_row["prompt"].strip(),
            category=PromptCategory.benign_sensitive,
            source="falsereject",
            source_id=str(canonical_idx),
            created_at=None,
            metadata={
                "fr_category_id": canonical_row["category"],
                "fr_category_text": canonical_row["category_text"],
                "fr_split": split,
                "fr_original_index": canonical_idx,
                "fr_duplicate_count": len(group),
                "fr_duplicate_source_ids": all_source_ids,
                "ingest_script": "acquire_prompts_falsereject.py",
                "ingest_config": config_path,
            },
        )
        records.append(record)

    # Category distribution over final unique records
    category_dist: dict[str, int] = {}
    cat_counter: Counter = Counter()
    for r in records:
        label = f"{r.metadata['fr_category_id']}: {r.metadata['fr_category_text']}"
        cat_counter[label] += 1
    category_dist = dict(sorted(cat_counter.items()))

    stats: dict[str, Any] = {
        "total_rows": len(rows),
        "after_min_chars_filter": len(rows) - n_short,
        "unique_after_dedup": len(records),
        "category_distribution": category_dist,
    }
    return records, stats


# ---------------------------------------------------------------------------
# Acquisition (loads from HF — not called in unit tests)
# ---------------------------------------------------------------------------


def acquire(config: dict[str, Any], config_path: str) -> tuple[list[PromptRecord], dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    split = config["split"]
    dataset = load_dataset(config["hf_dataset"], split=split)
    rows = [dict(row) for row in dataset]
    return process_rows(rows, split, config.get("min_prompt_chars", 20), config_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire FalseReject benign_sensitive prompts"
    )
    parser.add_argument("--config", required=True, help="Path to falsereject.yaml config")
    parser.add_argument("--output", help="Override output_path from config")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only; do not write")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    records, stats = acquire(config, config_path)

    print(f"Total rows in split:     {stats['total_rows']}")
    print(f"After min_prompt_chars:  {stats['after_min_chars_filter']}")
    print(f"Unique after dedup:      {stats['unique_after_dedup']}")
    print("\nCategory distribution (unique records):")
    for label, count in stats["category_distribution"].items():
        print(f"  {label:<60} {count}")

    if args.dry_run:
        print("\nDry run — no output written.")
        return

    output_path = args.output or config["output_path"]
    write_jsonl(records, output_path)
    print(f"\nWrote {len(records)} records to {output_path}")

    sample_path = config.get("sample_path")
    sample_n = int(config.get("sample_n", 5))
    if sample_path:
        write_jsonl(records[:sample_n], sample_path)
        print(f"Wrote {sample_n} sample records to {sample_path}")


if __name__ == "__main__":
    main()
