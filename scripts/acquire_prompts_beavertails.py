"""
Acquire BeaverTails unsafe prompts and write PromptRecord JSONL.

Contract:
    Input:  PKU-Alignment/BeaverTails HF dataset (configured split)
    Output: PromptRecord JSONL — one record per unique unsafe prompt
    Schema: PromptRecord with category=unsafe, created_at=null,
            bt_categories/bt_active_categories/bt_duplicate_* in metadata

Usage:
    python scripts/acquire_prompts_beavertails.py --config configs/acquisition/beavertails.yaml
    python scripts/acquire_prompts_beavertails.py --config configs/acquisition/beavertails.yaml --dry-run
    python scripts/acquire_prompts_beavertails.py --config configs/acquisition/beavertails.yaml \
        --output outputs/smoke/beavertails_unsafe.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
    return f"bt_{safe_split}_{index:06d}"


# ---------------------------------------------------------------------------
# Core transformation (pure — no HF dependency, fully testable)
# ---------------------------------------------------------------------------


def process_rows(
    rows: list[dict[str, Any]],
    split: str,
    min_prompt_chars: int,
    config_path: str,
) -> tuple[list[PromptRecord], dict[str, int]]:
    """Filter, deduplicate, and build PromptRecords from raw BeaverTails rows.

    Args:
        rows: raw dicts from the HF dataset (keys: prompt, response, category, is_safe)
        split: exact HF split name (used in prompt_id and metadata)
        min_prompt_chars: minimum character length after stripping
        config_path: path string stored in metadata for provenance

    Returns:
        (records, stats) where stats reports row counts at each filter stage
    """
    # Stage 1: filter
    eligible: list[tuple[int, dict[str, Any]]] = []
    n_unsafe = 0
    n_short = 0
    for idx, row in enumerate(rows):
        if row.get("is_safe", True):
            continue
        n_unsafe += 1
        text = row["prompt"].strip()
        if len(text) < min_prompt_chars:
            n_short += 1
            continue
        eligible.append((idx, row))

    # Stage 2: group by normalized prompt
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in eligible:
        key = _normalize(row["prompt"])
        groups[key].append((idx, row))

    # Stage 3: build one PromptRecord per group
    records: list[PromptRecord] = []
    for key in sorted(groups):  # deterministic ordering
        group = sorted(groups[key], key=lambda x: x[0])  # sort by original index
        canonical_idx, canonical_row = group[0]
        all_source_ids = [str(i) for i, _ in group]

        categories_raw: dict[str, bool] = dict(canonical_row["category"])
        active_categories = sorted(k for k, v in categories_raw.items() if v)

        record = PromptRecord(
            prompt_id=_prompt_id(split, canonical_idx),
            prompt=canonical_row["prompt"].strip(),
            category=PromptCategory.unsafe,
            source="beavertails",
            source_id=str(canonical_idx),
            created_at=None,
            metadata={
                "bt_categories": categories_raw,
                "bt_active_categories": active_categories,
                "bt_split": split,
                "bt_original_index": canonical_idx,
                "bt_duplicate_count": len(group),
                "bt_duplicate_source_ids": all_source_ids,
                "ingest_script": "acquire_prompts_beavertails.py",
                "ingest_config": config_path,
            },
        )
        records.append(record)

    stats = {
        "total_rows": len(rows),
        "after_is_safe_filter": n_unsafe,
        "after_min_chars_filter": n_unsafe - n_short,
        "unique_after_dedup": len(records),
    }
    return records, stats


# ---------------------------------------------------------------------------
# Acquisition (loads from HF — not called in unit tests)
# ---------------------------------------------------------------------------


def acquire(config: dict[str, Any], config_path: str) -> tuple[list[PromptRecord], dict[str, int]]:
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
    parser = argparse.ArgumentParser(description="Acquire BeaverTails unsafe prompts")
    parser.add_argument("--config", required=True, help="Path to beavertails.yaml config")
    parser.add_argument("--output", help="Override output_path from config")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only; do not write")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    records, stats = acquire(config, config_path)

    print(f"Total rows in split:     {stats['total_rows']}")
    print(f"After is_safe=False:     {stats['after_is_safe_filter']}")
    print(f"After min_prompt_chars:  {stats['after_min_chars_filter']}")
    print(f"Unique after dedup:      {stats['unique_after_dedup']}")

    if args.dry_run:
        print("Dry run — no output written.")
        return

    output_path = args.output or config["output_path"]
    write_jsonl(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")

    sample_path = config.get("sample_path")
    sample_n = int(config.get("sample_n", 5))
    if sample_path:
        write_jsonl(records[:sample_n], sample_path)
        print(f"Wrote {sample_n} sample records to {sample_path}")


if __name__ == "__main__":
    main()
