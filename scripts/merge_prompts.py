"""
Merge all four category prompt files into a single all_prompts.jsonl.

This is a prerequisite for running generate_responses.py on the full prompt
pool.  assembly/full.yaml also expects hf_data/prompts/all_prompts.jsonl.

Contract
--------
Inputs (hard-coded defaults, overridable via --config or individual flags):
    hf_data/prompts/selected/unsafe/beavertails_selected.jsonl
    hf_data/prompts/dual_use/dualuse_prompts.jsonl
    hf_data/prompts/selected/benign_sensitive/falsereject_selected.jsonl
    hf_data/prompts/selected/benign/alpaca_selected.jsonl

Output:
    hf_data/prompts/all_prompts.jsonl

Validation:
    - Each record must parse as a valid PromptRecord
    - No duplicate prompt_ids across sources (hard error)
    - Prints counts by category

Usage:
    python scripts/merge_prompts.py
    python scripts/merge_prompts.py --output hf_data/prompts/all_prompts.jsonl
    python scripts/merge_prompts.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from safecomp_dpo.io import load_prompt_records, write_jsonl
    from safecomp_dpo.schemas import PromptCategory, PromptRecord
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_prompt_records, write_jsonl
    from safecomp_dpo.schemas import PromptCategory, PromptRecord


# Default source paths in category order
DEFAULT_SOURCES: list[tuple[str, str]] = [
    ("unsafe",          "hf_data/prompts/selected/unsafe/beavertails_selected.jsonl"),
    ("dual_use",        "hf_data/prompts/dual_use/dualuse_prompts.jsonl"),
    ("benign_sensitive","hf_data/prompts/selected/benign_sensitive/falsereject_selected.jsonl"),
    ("benign",          "hf_data/prompts/selected/benign/alpaca_selected.jsonl"),
]

DEFAULT_OUTPUT = "hf_data/prompts/all_prompts.jsonl"


def merge_prompts(
    sources: list[tuple[str, str]],
    output_path: Path,
    dry_run: bool = False,
) -> dict:
    """Load, validate, dedup-check, and merge all prompt files.

    Returns a summary dict.
    """
    all_records: list[PromptRecord] = []
    seen_ids: set[str] = set()
    counts: dict[str, int] = {}
    errors: list[str] = []

    for label, path_str in sources:
        path = Path(path_str)
        if not path.exists():
            msg = f"Source file not found: {path}  (expected category: {label})"
            print(f"  ERROR: {msg}", file=sys.stderr)
            errors.append(msg)
            continue

        records = load_prompt_records(path)
        print(f"  {label}: {len(records)} records from {path}")

        for r in records:
            if r.prompt_id in seen_ids:
                msg = f"Duplicate prompt_id {r.prompt_id!r} (from {path})"
                print(f"  ERROR: {msg}", file=sys.stderr)
                errors.append(msg)
                continue
            seen_ids.add(r.prompt_id)
            all_records.append(r)
            cat = r.category.value
            counts[cat] = counts.get(cat, 0) + 1

    if errors:
        print(f"\n{len(errors)} error(s) found. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal: {len(all_records)} prompts")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")

    if dry_run:
        print("\nDry run — nothing written.")
    else:
        write_jsonl(all_records, output_path)
        print(f"\nWritten to {output_path}")

    return {
        "total": len(all_records),
        "counts_by_category": counts,
        "output": str(output_path),
        "dry_run": dry_run,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge all category prompt files into hf_data/prompts/all_prompts.jsonl."
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report counts without writing output.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    print("Merging prompt files...")
    for label, path in DEFAULT_SOURCES:
        print(f"  {label}: {path}")
    print()

    merge_prompts(DEFAULT_SOURCES, Path(args.output), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
