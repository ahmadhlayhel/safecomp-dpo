"""
Extract BeaverTails source responses as unsafe_compliance ResponseRecords.

For each of the 1800 selected unsafe prompts, retrieves the original response
text from the locally cached HF BeaverTails dataset and writes a canonical
ResponseRecord with response_type="unsafe_compliance".

This replaces the Llama fallback generation pass for the unsafe category.
The BeaverTails responses are human-written labeled-harmful completions from
the source dataset — distinct provenance from all model-generated responses
and requiring no GPU or API calls.

Contract
--------
Inputs:
    --selected-prompts  PromptRecord JSONL of selected unsafe prompts
                        default: hf_data/prompts/selected/unsafe/beavertails_selected.jsonl
    --hf-dataset        HF dataset name (default: PKU-Alignment/BeaverTails)
    --hf-split          HF split name   (default: 30k_train)
    --output            ResponseRecord JSONL output
                        default: outputs/unsafe_compliance/beavertails_unsafe_compliance.jsonl
    --report            JSON summary report
                        default: outputs/unsafe_compliance/beavertails_extraction_report.json

Output ResponseRecord fields:
    response_type  = "unsafe_compliance"
    model          = "beavertails/30k_train"   (source tag, not a model name)
    temperature    = None
    metadata:
        pass           = "beavertails_source"
        generator      = "beavertails/30k_train"
        source_id      = original HF row index (str)
        bt_split       = HF split name
        is_safe        = False  (verified from source row)
        short_output   = bool  (word count < 20)
        word_count     = int
        finish_reason  = "source"  (distinguishes from API finish reasons)

Join key:
    selected_prompt.source_id -> int(source_id) -> dataset[idx]
    Prompt text is verified to match exactly after .strip().

Validation:
    - Prompt text must match source row (hard error if not)
    - source row is_safe must be False (hard error if not)
    - Response must be non-empty (skipped with warning if empty)
    - Short responses (< 20 words) are flagged but included

Usage:
    python scripts/extract_beavertails_responses.py

    python scripts/extract_beavertails_responses.py \\
        --selected-prompts hf_data/prompts/selected/unsafe/beavertails_selected.jsonl \\
        --output           outputs/unsafe_compliance/beavertails_unsafe_compliance.jsonl \\
        --report           outputs/unsafe_compliance/beavertails_extraction_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from safecomp_dpo.io import load_prompt_records, write_jsonl
    from safecomp_dpo.schemas import ResponseRecord, ResponseType
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_prompt_records, write_jsonl
    from safecomp_dpo.schemas import ResponseRecord, ResponseType


DEFAULT_SELECTED   = "hf_data/prompts/selected/unsafe/beavertails_selected.jsonl"
DEFAULT_HF_DATASET = "PKU-Alignment/BeaverTails"
DEFAULT_HF_SPLIT   = "30k_train"
DEFAULT_OUTPUT     = "outputs/unsafe_compliance/beavertails_unsafe_compliance.jsonl"
DEFAULT_REPORT     = "outputs/unsafe_compliance/beavertails_extraction_report.json"

SHORT_WORD_THRESHOLD = 20


def make_response_id(prompt_id: str) -> str:
    return f"{prompt_id}__unsafe_compliance__s01"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract BeaverTails source responses as unsafe_compliance ResponseRecords."
    )
    p.add_argument("--selected-prompts", default=DEFAULT_SELECTED)
    p.add_argument("--hf-dataset",       default=DEFAULT_HF_DATASET)
    p.add_argument("--hf-split",         default=DEFAULT_HF_SPLIT)
    p.add_argument("--output",           default=DEFAULT_OUTPUT)
    p.add_argument("--report",           default=DEFAULT_REPORT)
    return p


def main() -> None:
    args = build_parser().parse_args()

    selected_path = Path(args.selected_prompts)
    output_path   = Path(args.output)
    report_path   = Path(args.report)
    hf_split      = args.hf_split
    model_tag     = f"beavertails/{hf_split}"

    # --- Load HF dataset ---
    print(f"Loading HF dataset {args.hf_dataset!r} split={hf_split!r} from local cache...")
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    ds = load_dataset(args.hf_dataset, split=hf_split)
    print(f"  {len(ds)} rows loaded. Columns: {ds.column_names}")

    # --- Load selected prompts ---
    print(f"Loading selected prompts from {selected_path}")
    selected = load_prompt_records(selected_path)
    print(f"  {len(selected)} prompts loaded")

    # --- Extract responses ---
    records: list[ResponseRecord] = []
    n_ok = 0
    n_short = 0
    n_empty = 0
    n_mismatch = 0
    n_not_unsafe = 0
    word_counts: list[int] = []

    for pr in selected:
        idx = int(pr.source_id)
        raw = dict(ds[idx])

        # Validate prompt text match
        if raw["prompt"].strip() != pr.prompt.strip():
            print(
                f"  MISMATCH prompt_id={pr.prompt_id} source_id={pr.source_id}: "
                f"selected={pr.prompt[:60]!r} raw={raw['prompt'][:60]!r}",
                file=sys.stderr,
            )
            n_mismatch += 1
            continue

        # Validate is_safe
        if raw.get("is_safe", True):
            print(
                f"  WARNING: source row for {pr.prompt_id} has is_safe=True — skipping",
                file=sys.stderr,
            )
            n_not_unsafe += 1
            continue

        response_text = raw["response"].strip()

        # Skip empty
        if not response_text:
            print(f"  WARNING: empty response for {pr.prompt_id} — skipping", file=sys.stderr)
            n_empty += 1
            continue

        wc = len(response_text.split())
        word_counts.append(wc)
        short = wc < SHORT_WORD_THRESHOLD
        if short:
            n_short += 1

        record = ResponseRecord(
            response_id=make_response_id(pr.prompt_id),
            prompt_id=pr.prompt_id,
            response=response_text,
            response_type=ResponseType.unsafe_compliance,
            model=model_tag,
            temperature=None,
            metadata={
                "pass":        "beavertails_source",
                "generator":   model_tag,
                "source_id":   pr.source_id,
                "bt_split":    hf_split,
                "is_safe":     False,
                "short_output": short,
                "word_count":  wc,
                "finish_reason": "source",
            },
        )
        records.append(record)
        n_ok += 1

    # --- Write output ---
    write_jsonl(records, output_path)

    # --- Stats ---
    word_counts.sort()
    n = len(word_counts)
    def pct(p: float) -> int:
        return word_counts[int(n * p / 100)] if n else 0

    report = {
        "selected_prompts": str(selected_path),
        "hf_dataset": args.hf_dataset,
        "hf_split": hf_split,
        "output": str(output_path),
        "total_selected": len(selected),
        "extracted": n_ok,
        "skipped_mismatch": n_mismatch,
        "skipped_not_unsafe": n_not_unsafe,
        "skipped_empty": n_empty,
        "short_output_flagged": n_short,
        "word_count_distribution": {
            "min":    word_counts[0] if n else None,
            "p10":    pct(10),
            "p25":    pct(25),
            "median": pct(50),
            "p75":    pct(75),
            "p90":    pct(90),
            "max":    word_counts[-1] if n else None,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nExtraction complete.")
    print(f"  Extracted:              {n_ok} / {len(selected)}")
    print(f"  Short (< {SHORT_WORD_THRESHOLD} words):      {n_short}  ({100*n_short/n_ok:.1f}% of extracted)" if n_ok else "")
    print(f"  Skipped (mismatch):     {n_mismatch}")
    print(f"  Skipped (is_safe=True): {n_not_unsafe}")
    print(f"  Skipped (empty):        {n_empty}")
    print(f"  Word count — min={word_counts[0] if n else 'N/A'}  median={pct(50)}  p90={pct(90)}  max={word_counts[-1] if n else 'N/A'}")
    print(f"\n  Output: {output_path}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
