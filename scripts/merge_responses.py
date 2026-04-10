"""
Assemble full_responses.jsonl from all generation artifacts.

This script:
  1. Takes dual_use safe_completion records from the existing converted file
     (keeps them, discards the old degraded hard_refusals from that file)
  2. Takes all records from hard_refusal_responses.jsonl
     (covers unsafe, dual_use, benign_sensitive, benign)
  3. Takes all records from helpful_answer_responses.jsonl
     (covers benign_sensitive, benign)
  4. Validates that response_ids are unique across all sources
  5. Writes hf_data/responses/full_responses.jsonl

This is the file expected by:
  - scripts/validate_responses.py (with configs/generation/full.yaml)
  - configs/assembly/full.yaml

Contract
--------
Inputs (overridable via CLI):
    --dualuse-sc       hf_data/responses/dual_use/dualuse_response_records.jsonl
                       (only safe_completion records are kept)
    --hard-refusal     hf_data/responses/hard_refusal_responses.jsonl
    --helpful-answer   hf_data/responses/helpful_answer_responses.jsonl
    --output           hf_data/responses/full_responses.jsonl
    --report           outputs/merge_reports/merge_responses_report.json

Output record counts (no unsafe_compliance — that is still deferred):
    dual_use safe_completion:            2502
    hard_refusal (all 4 categories):     7902
    helpful_answer (benign + bs):        3600
    Total:                              14004

Usage:
    python scripts/merge_responses.py
    python scripts/merge_responses.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from safecomp_dpo.io import load_response_records, write_jsonl
    from safecomp_dpo.schemas import ResponseRecord, ResponseType
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_response_records, write_jsonl
    from safecomp_dpo.schemas import ResponseRecord, ResponseType


DEFAULT_DUALUSE_SC   = "hf_data/responses/dual_use/dualuse_response_records.jsonl"
DEFAULT_HARD_REFUSAL = "hf_data/responses/hard_refusal_responses.jsonl"
DEFAULT_HELPFUL      = "hf_data/responses/helpful_answer_responses.jsonl"
DEFAULT_OUTPUT       = "hf_data/responses/full_responses.jsonl"
DEFAULT_REPORT       = "outputs/merge_reports/merge_responses_report.json"


def merge_responses(
    dualuse_sc_path: Path,
    hard_refusal_path: Path,
    helpful_answer_path: Path,
    output_path: Path,
    report_path: Path,
    dry_run: bool = False,
) -> dict:
    """Assemble full_responses.jsonl from three source files.

    Returns a summary dict.
    """
    all_records: list[ResponseRecord] = []
    seen_ids: set[str] = set()
    counts: dict[str, int] = {}
    errors: list[str] = []

    def add_records(
        records: list[ResponseRecord],
        source_label: str,
        filter_type: ResponseType | None = None,
    ) -> None:
        kept = 0
        dropped = 0
        for r in records:
            if filter_type is not None and r.response_type != filter_type:
                dropped += 1
                continue
            if r.response_id in seen_ids:
                msg = (
                    f"Duplicate response_id {r.response_id!r} "
                    f"(from {source_label})"
                )
                print(f"  ERROR: {msg}", file=sys.stderr)
                errors.append(msg)
                continue
            seen_ids.add(r.response_id)
            all_records.append(r)
            rt = r.response_type.value
            counts[rt] = counts.get(rt, 0) + 1
            kept += 1
        filter_note = f" (filter: {filter_type.value})" if filter_type else ""
        print(f"  {source_label}{filter_note}: kept={kept}, dropped={dropped}")

    # --- Source 1: dual_use safe_completion (keep only safe_completion) ---
    if not dualuse_sc_path.exists():
        msg = f"dual_use source not found: {dualuse_sc_path}"
        print(f"  ERROR: {msg}", file=sys.stderr)
        errors.append(msg)
    else:
        records = load_response_records(dualuse_sc_path)
        add_records(records, str(dualuse_sc_path), filter_type=ResponseType.safe_completion)

    # --- Source 2: hard_refusal (all categories) ---
    if not hard_refusal_path.exists():
        msg = f"hard_refusal source not found: {hard_refusal_path}"
        print(f"  WARNING: {msg} — run Pass 1 first", file=sys.stderr)
        # Non-fatal: allow partial merge so merge can be re-run after each pass
    else:
        records = load_response_records(hard_refusal_path)
        add_records(records, str(hard_refusal_path))

    # --- Source 3: helpful_answer ---
    if not helpful_answer_path.exists():
        msg = f"helpful_answer source not found: {helpful_answer_path}"
        print(f"  WARNING: {msg} — run Pass 2 first", file=sys.stderr)
    else:
        records = load_response_records(helpful_answer_path)
        add_records(records, str(helpful_answer_path))

    if errors:
        print(f"\n{len(errors)} hard error(s). Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal records: {len(all_records)}")
    for rt, n in sorted(counts.items()):
        print(f"  {rt}: {n}")

    report = {
        "output": str(output_path),
        "total": len(all_records),
        "counts_by_type": counts,
        "dry_run": dry_run,
    }

    if not dry_run:
        write_jsonl(all_records, output_path)
        print(f"\nWritten to {output_path}")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report: {report_path}")
    else:
        print("\nDry run — nothing written.")

    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge generation artifacts into full_responses.jsonl."
    )
    p.add_argument("--dualuse-sc",     default=DEFAULT_DUALUSE_SC)
    p.add_argument("--hard-refusal",   default=DEFAULT_HARD_REFUSAL)
    p.add_argument("--helpful-answer", default=DEFAULT_HELPFUL)
    p.add_argument("--output",         default=DEFAULT_OUTPUT)
    p.add_argument("--report",         default=DEFAULT_REPORT)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report counts without writing output.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    print("Merging response files...")
    merge_responses(
        dualuse_sc_path=Path(args.dualuse_sc),
        hard_refusal_path=Path(getattr(args, "hard_refusal")),
        helpful_answer_path=Path(getattr(args, "helpful_answer")),
        output_path=Path(args.output),
        report_path=Path(args.report),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
