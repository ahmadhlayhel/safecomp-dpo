"""
Acquire Alpaca benign prompts and write PromptRecord JSONL.

tatsu-lab/alpaca is the v1 source pool for benign acquisition. The dataset
contains instruction-following tasks; all eligible rows are benign candidates.
Acquisition collects the full eligible pool after hygiene filtering and
deduplication. Task-diversity selection belongs to a later curation stage.

Prompt construction:
    If input is non-empty: "{instruction}\\n\\n{input}" (stripped)
    Otherwise:             "{instruction}" (stripped)

Contract:
    Input:  tatsu-lab/alpaca HF dataset (configured split)
    Output: PromptRecord JSONL — one record per unique benign prompt
    Schema: PromptRecord with category=benign, created_at=null,
            alpaca_has_input/alpaca_duplicate_* in metadata

Usage:
    python scripts/acquire_prompts_alpaca.py --config configs/acquisition/alpaca.yaml
    python scripts/acquire_prompts_alpaca.py --config configs/acquisition/alpaca.yaml --dry-run
    python scripts/acquire_prompts_alpaca.py --config configs/acquisition/alpaca.yaml \\
        --output outputs/smoke/alpaca_benign.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
    return f"alpaca_{safe_split}_{index:06d}"


def _build_prompt(instruction: str, input_text: str) -> str:
    """Construct the prompt string from Alpaca instruction and optional input."""
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


# ---------------------------------------------------------------------------
# Core transformation (pure — no HF dependency, fully testable)
# ---------------------------------------------------------------------------


def process_rows(
    rows: list[dict[str, Any]],
    split: str,
    min_prompt_chars: int,
    config_path: str,
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Filter, deduplicate, and build PromptRecords from raw Alpaca rows.

    Hygiene filters applied in order:
        1. Drop rows where instruction is empty after strip.
        2. Drop rows where the constructed prompt is below min_prompt_chars.
        3. Exact-match deduplication on normalized prompt text.

    Args:
        rows: raw dicts from the HF dataset (keys: instruction, input, output, text)
        split: exact HF split name (used in prompt_id and metadata)
        min_prompt_chars: minimum character length of the constructed prompt
        config_path: path string stored in metadata for provenance

    Returns:
        (records, stats) where stats reports row counts at each filter stage
    """
    n_empty_instruction = 0
    n_short = 0
    eligible: list[tuple[int, dict[str, Any]]] = []

    for idx, row in enumerate(rows):
        instruction = row.get("instruction", "")
        if not instruction.strip():
            n_empty_instruction += 1
            continue

        prompt = _build_prompt(instruction, row.get("input", ""))
        if len(prompt) < min_prompt_chars:
            n_short += 1
            continue

        eligible.append((idx, row))

    # Group by normalized prompt for deduplication
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in eligible:
        prompt = _build_prompt(row.get("instruction", ""), row.get("input", ""))
        key = _normalize(prompt)
        groups[key].append((idx, row))

    # Build one PromptRecord per group
    records: list[PromptRecord] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: x[0])
        canonical_idx, canonical_row = group[0]
        all_source_ids = sorted(str(i) for i, _ in group)

        instruction = canonical_row.get("instruction", "")
        input_text = canonical_row.get("input", "")
        prompt = _build_prompt(instruction, input_text)
        has_input = bool(input_text.strip())

        record = PromptRecord(
            prompt_id=_prompt_id(split, canonical_idx),
            prompt=prompt,
            category=PromptCategory.benign,
            source="alpaca",
            source_id=str(canonical_idx),
            created_at=None,
            metadata={
                "alpaca_has_input": has_input,
                "alpaca_split": split,
                "alpaca_original_index": canonical_idx,
                "alpaca_duplicate_count": len(group),
                "alpaca_duplicate_source_ids": all_source_ids,
                "ingest_script": "acquire_prompts_alpaca.py",
                "ingest_config": config_path,
            },
        )
        records.append(record)

    n_with_input = sum(1 for r in records if r.metadata["alpaca_has_input"])

    stats: dict[str, Any] = {
        "total_rows": len(rows),
        "after_empty_instruction_filter": len(rows) - n_empty_instruction,
        "after_min_prompt_chars_filter": len(rows) - n_empty_instruction - n_short,
        "unique_after_dedup": len(records),
        "unique_with_input": n_with_input,
        "unique_without_input": len(records) - n_with_input,
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
    parser = argparse.ArgumentParser(description="Acquire Alpaca benign prompts")
    parser.add_argument("--config", required=True, help="Path to alpaca.yaml config")
    parser.add_argument("--output", help="Override output_path from config")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only; do not write")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    records, stats = acquire(config, config_path)

    print(f"Total rows in split:              {stats['total_rows']}")
    print(f"After empty instruction filter:   {stats['after_empty_instruction_filter']}")
    print(f"After min_prompt_chars filter:    {stats['after_min_prompt_chars_filter']}")
    print(f"Unique after dedup:               {stats['unique_after_dedup']}")
    print(f"  with input field:               {stats['unique_with_input']}")
    print(f"  without input field:            {stats['unique_without_input']}")

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
