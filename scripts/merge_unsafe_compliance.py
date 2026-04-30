"""
Merge Pass-1 (GPT-4o-mini clean) and Pass-2 (Llama fallback) records
into the final hf_data/responses/unsafe_compliance_responses.jsonl.

Contract
--------
Inputs:
    --gpt-clean       ResponseRecord JSONL from filter_refusals.py
                      (Pass-1 records where likely_refusal=False)
    --llama-fallback  ResponseRecord JSONL from Pass-2 Llama generation
                      OPTIONAL — omit if Pass 2 was not needed or not yet run.
    --prompts         PromptRecord JSONL (all_prompts.jsonl) — used for coverage
                      validation and category breakdown.
    --output          Final merged ResponseRecord JSONL.
                      Default: hf_data/responses/unsafe_compliance_responses.jsonl
    --report          JSON summary report.

Outputs:
    <output>:  Final merged ResponseRecord JSONL, ready for merge_responses.py.
    <report>:  JSON with total, by-source-and-category breakdown, coverage stats,
               and list of any uncovered prompt_ids.

Validation:
    - All response_types must be "unsafe_compliance"
    - All prompt_ids must exist in all_prompts.jsonl
    - All prompt categories must be "unsafe" or "dual_use"
    - No duplicate prompt_ids across the two sources

Metadata written to each output record:
    GPT primary records:
        pass="primary", generator=<model>, likely_refusal=False
        (already set by filter_refusals.py — preserved here)
    Llama fallback records:
        pass="fallback", generator=<model>, fallback=True
        (added by this script)

Usage:
    # After Pass 1 only (no Llama fallback yet or needed):
    python scripts/merge_unsafe_compliance.py \
        --gpt-clean outputs/unsafe_compliance/gpt4o_clean.jsonl \
        --prompts   hf_data/prompts/all_prompts.jsonl \
        --output    hf_data/responses/unsafe_compliance_responses.jsonl \
        --report    outputs/unsafe_compliance/merge_report.json

    # Full two-pass merge:
    python scripts/merge_unsafe_compliance.py \
        --gpt-clean      outputs/unsafe_compliance/gpt4o_clean.jsonl \
        --llama-fallback outputs/unsafe_compliance/llama_fallback.jsonl \
        --prompts        hf_data/prompts/all_prompts.jsonl \
        --output         hf_data/responses/unsafe_compliance_responses.jsonl \
        --report         outputs/unsafe_compliance/merge_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
    from safecomp_dpo.schemas import PromptCategory, ResponseRecord, ResponseType
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
    from safecomp_dpo.schemas import PromptCategory, ResponseRecord, ResponseType


DEFAULT_OUTPUT = "hf_data/responses/unsafe_compliance_responses.jsonl"
DEFAULT_REPORT = "outputs/unsafe_compliance/merge_report.json"

_ELIGIBLE_CATEGORIES = {PromptCategory.unsafe, PromptCategory.dual_use}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge GPT-clean and Llama-fallback into the final unsafe_compliance JSONL."
    )
    p.add_argument("--gpt-clean", required=True, help="Pass-1 clean ResponseRecord JSONL")
    p.add_argument(
        "--llama-fallback",
        default=None,
        help="Pass-2 Llama ResponseRecord JSONL (optional — omit if not needed/ready)",
    )
    p.add_argument(
        "--prompts",
        required=True,
        help="PromptRecord JSONL (all_prompts.jsonl) for coverage validation",
    )
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--report", default=DEFAULT_REPORT)
    return p


def main() -> None:
    args = build_parser().parse_args()

    gpt_path = Path(getattr(args, "gpt_clean"))
    llama_path = Path(args.llama_fallback) if args.llama_fallback else None
    prompts_path = Path(args.prompts)
    output_path = Path(args.output)
    report_path = Path(args.report)

    print(f"Loading prompts from {prompts_path}")
    all_prompts = load_prompt_records(prompts_path)
    eligible_prompt_ids: set[str] = {
        pr.prompt_id for pr in all_prompts if pr.category in _ELIGIBLE_CATEGORIES
    }
    prompt_category: dict[str, str] = {
        pr.prompt_id: pr.category.value
        for pr in all_prompts
        if pr.category in _ELIGIBLE_CATEGORIES
    }
    print(f"  {len(eligible_prompt_ids)} eligible prompt_ids (unsafe + dual_use)")

    all_records: list[ResponseRecord] = []
    seen_prompt_ids: set[str] = set()
    errors: list[str] = []

    def ingest(records: list[ResponseRecord], source_label: str, is_fallback: bool) -> int:
        added = 0
        for rec in records:
            if rec.response_type != ResponseType.unsafe_compliance:
                errors.append(
                    f"{source_label}: unexpected response_type "
                    f"{rec.response_type!r} for {rec.response_id!r}"
                )
                continue
            if rec.prompt_id not in eligible_prompt_ids:
                errors.append(
                    f"{source_label}: prompt_id {rec.prompt_id!r} not in eligible set "
                    f"(must be unsafe or dual_use)"
                )
                continue
            if rec.prompt_id in seen_prompt_ids:
                errors.append(
                    f"{source_label}: duplicate prompt_id {rec.prompt_id!r}"
                )
                continue
            seen_prompt_ids.add(rec.prompt_id)

            if is_fallback:
                updated_meta = {
                    **rec.metadata,
                    "pass": "fallback",
                    "generator": rec.model,
                    "fallback": True,
                }
                rec = rec.model_copy(update={"metadata": updated_meta})

            all_records.append(rec)
            added += 1
        return added

    print(f"Loading GPT clean records from {gpt_path}")
    gpt_records = load_response_records(gpt_path)
    n_gpt = ingest(gpt_records, str(gpt_path), is_fallback=False)
    print(f"  {len(gpt_records)} loaded, {n_gpt} ingested")

    n_llama = 0
    if llama_path is not None:
        if llama_path.exists():
            print(f"Loading Llama fallback records from {llama_path}")
            llama_records = load_response_records(llama_path)
            n_llama = ingest(llama_records, str(llama_path), is_fallback=True)
            print(f"  {len(llama_records)} loaded, {n_llama} ingested")
        else:
            print(
                f"  WARNING: --llama-fallback path {llama_path} not found — skipping",
                file=sys.stderr,
            )

    if errors:
        print(f"\n{len(errors)} validation error(s):", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        sys.exit(1)

    missing_ids = eligible_prompt_ids - seen_prompt_ids

    # Coverage breakdown by source × category
    by_source_cat: dict[str, dict[str, int]] = {}
    for rec in all_records:
        pass_label = rec.metadata.get("pass", "primary")
        cat = prompt_category.get(rec.prompt_id, "unknown")
        by_source_cat.setdefault(pass_label, {})
        by_source_cat[pass_label][cat] = by_source_cat[pass_label].get(cat, 0) + 1

    coverage_pct = round(100 * len(all_records) / len(eligible_prompt_ids), 1) if eligible_prompt_ids else 0

    write_jsonl(all_records, output_path)

    report = {
        "output": str(output_path),
        "total": len(all_records),
        "gpt_primary": n_gpt,
        "llama_fallback": n_llama,
        "eligible_total": len(eligible_prompt_ids),
        "missing_count": len(missing_ids),
        "coverage_pct": coverage_pct,
        "by_source_and_category": by_source_cat,
        "missing_ids_sample": sorted(missing_ids)[:30],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nMerge complete.")
    print(f"  Total records:  {len(all_records)} / {len(eligible_prompt_ids)} eligible")
    print(f"  GPT primary:    {n_gpt}")
    print(f"  Llama fallback: {n_llama}")
    print(f"  Missing:        {len(missing_ids)}")
    print(f"  Coverage:       {coverage_pct}%")
    print(f"  By source+cat:  {by_source_cat}")
    print(f"\n  Output:  {output_path}")
    print(f"  Report:  {report_path}")

    if missing_ids:
        print(
            f"\n  NOTE: {len(missing_ids)} prompt_ids not covered. "
            f"Run Pass 2 (Llama fallback) on these if --llama-fallback was not provided, "
            f"or investigate if both passes were included."
        )


if __name__ == "__main__":
    main()
