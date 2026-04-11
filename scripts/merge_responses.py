"""
Assemble full_responses.jsonl from all generation artifacts.

This script:
  1. Takes dual_use safe_completion records from the existing converted file
     (keeps them, discards the old degraded hard_refusals from that file)
  2. Takes all records from hard_refusal_responses.jsonl
     (covers unsafe, dual_use, benign_sensitive, benign)
  3. Takes all records from helpful_answer_responses.jsonl
     (covers benign_sensitive, benign)
  4. Takes all records from unsafe_compliance_responses.jsonl  [optional]
     (covers unsafe + dual_use; omit until sourcing teammate delivers)
  5. Validates that response_ids are unique across all sources
  6. Writes hf_data/responses/full_responses.jsonl

This is the file expected by:
  - scripts/validate_responses.py (with configs/generation/full.yaml)
  - configs/assembly/full.yaml

Contract
--------
Inputs (overridable via CLI):
    --dualuse-sc          hf_data/responses/dual_use/dualuse_response_records.jsonl
                          (only safe_completion records are kept)
    --hard-refusal        hf_data/responses/hard_refusal_responses.jsonl
    --helpful-answer      hf_data/responses/helpful_answer_responses.jsonl
    --unsafe-compliance   hf_data/responses/unsafe_compliance_responses.jsonl
                          OPTIONAL — omit until sourcing stage is done.
                          Warns but does not abort if missing.
    --output              hf_data/responses/full_responses.jsonl
    --report              outputs/merge_reports/merge_responses_report.json

Expected record counts:
    Without unsafe_compliance (generation passes only):
        dual_use safe_completion:            2502
        hard_refusal (all 4 categories):     7902
        helpful_answer (benign + bs):        3600
        Total:                              14004

    With unsafe_compliance (full pipeline):
        + unsafe_compliance (unsafe + dual_use): 4302
        Total:                                  18306

Usage:
    # After generation passes only (no unsafe_compliance yet):
    python scripts/merge_responses.py

    # Full merge once unsafe_compliance is delivered:
    python scripts/merge_responses.py \
        --unsafe-compliance hf_data/responses/unsafe_compliance_responses.jsonl

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


DEFAULT_DUALUSE_SC        = "hf_data/responses/dual_use/dualuse_response_records.jsonl"
DEFAULT_HARD_REFUSAL      = "hf_data/responses/hard_refusal_responses.jsonl"
DEFAULT_HELPFUL           = "hf_data/responses/helpful_answer_responses.jsonl"
DEFAULT_UNSAFE_COMPLIANCE = "hf_data/responses/unsafe_compliance_responses.jsonl"
DEFAULT_OUTPUT            = "hf_data/responses/full_responses.jsonl"
DEFAULT_REPORT            = "outputs/merge_reports/merge_responses_report.json"


def merge_responses(
    dualuse_sc_path: Path,
    hard_refusal_path: Path,
    helpful_answer_path: Path,
    output_path: Path,
    report_path: Path,
    unsafe_compliance_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Assemble full_responses.jsonl from source files.

    unsafe_compliance_path is optional — pass None to skip (sourcing deferred).
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
        print(
            f"  WARNING: hard_refusal not found: {hard_refusal_path} "
            f"-- run Pass 1 first",
            file=sys.stderr,
        )
    else:
        records = load_response_records(hard_refusal_path)
        add_records(records, str(hard_refusal_path))

    # --- Source 3: helpful_answer ---
    if not helpful_answer_path.exists():
        print(
            f"  WARNING: helpful_answer not found: {helpful_answer_path} "
            f"-- run Pass 2 first",
            file=sys.stderr,
        )
    else:
        records = load_response_records(helpful_answer_path)
        add_records(records, str(helpful_answer_path))

    # --- Source 4: unsafe_compliance (optional — deferred until sourcing done) ---
    if unsafe_compliance_path is None:
        print(
            "  NOTE: --unsafe-compliance not provided — "
            "baseline_refusal pairs for unsafe/dual_use will be incomplete "
            "until sourcing teammate delivers this file."
        )
    elif not unsafe_compliance_path.exists():
        print(
            f"  WARNING: unsafe_compliance file not found: {unsafe_compliance_path} "
            f"-- check path or omit --unsafe-compliance to skip",
            file=sys.stderr,
        )
    else:
        records = load_response_records(unsafe_compliance_path)
        add_records(records, str(unsafe_compliance_path))

    if errors:
        print(f"\n{len(errors)} hard error(s). Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal records: {len(all_records)}")
    for rt, n in sorted(counts.items()):
        print(f"  {rt}: {n}")

    uc_included = unsafe_compliance_path is not None and unsafe_compliance_path.exists()
    if not uc_included:
        print(
            "\n  [unsafe_compliance not included — "
            "baseline_refusal pairs will be skipped in build_pairs.py "
            "until this file is added]"
        )

    report = {
        "output": str(output_path),
        "total": len(all_records),
        "counts_by_type": counts,
        "unsafe_compliance_included": uc_included,
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
        print("\nDry run -- nothing written.")

    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge generation artifacts into full_responses.jsonl."
    )
    p.add_argument("--dualuse-sc",          default=DEFAULT_DUALUSE_SC)
    p.add_argument("--hard-refusal",        default=DEFAULT_HARD_REFUSAL)
    p.add_argument("--helpful-answer",      default=DEFAULT_HELPFUL)
    p.add_argument(
        "--unsafe-compliance",
        default=None,
        help=(
            "Path to unsafe_compliance ResponseRecord JSONL "
            "(optional — omit until sourcing teammate delivers). "
            f"Default location when ready: {DEFAULT_UNSAFE_COMPLIANCE}"
        ),
    )
    p.add_argument("--output",  default=DEFAULT_OUTPUT)
    p.add_argument("--report",  default=DEFAULT_REPORT)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report counts without writing output.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    uc_path: Path | None = None
    if args.unsafe_compliance is not None:
        uc_path = Path(args.unsafe_compliance)

    print("Merging response files...")
    merge_responses(
        dualuse_sc_path=Path(args.dualuse_sc),
        hard_refusal_path=Path(getattr(args, "hard_refusal")),
        helpful_answer_path=Path(getattr(args, "helpful_answer")),
        output_path=Path(args.output),
        report_path=Path(args.report),
        unsafe_compliance_path=uc_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
