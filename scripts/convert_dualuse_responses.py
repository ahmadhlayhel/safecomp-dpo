"""
Convert dual_use generation artifact to canonical ResponseRecord JSONL.

The generation-time file (hf_data/responses/dual_use/dualuse_responses.jsonl)
uses a custom nested schema:

    {"raw_id": "A_osint_0001", "prompt": "...", "domain": "...",
     "responses": {"safe_completion": "...", "hard_refusal": "...",
                   "unsafe_compliance": null}}

This script converts it to flat ResponseRecord JSONL that the rest of the
pipeline (validate_responses.py, build_pairs.py, ...) can consume:

    {"response_id": "du_A_osint_0001__safe_completion__s01",
     "prompt_id":   "du_A_osint_0001",
     "response":    "...",
     "response_type": "safe_completion",
     "model": "...",
     ...}

Contract
--------
Inputs:
    --responses   Custom-format JSONL (default: hf_data/responses/dual_use/dualuse_responses.jsonl)
    --prompts     PromptRecord JSONL used to resolve raw_id -> prompt_id
                  (default: hf_data/prompts/dual_use/dualuse_prompts.jsonl)
    --output      ResponseRecord JSONL output
                  (default: hf_data/responses/dual_use/dualuse_response_records.jsonl)
    --model       Model name to embed in metadata (default: unknown_api)
    --report      JSON report path (default: outputs/conversion_reports/dualuse_conversion_report.json)

Outputs:
    ResponseRecord JSONL — one record per non-null (raw_id, response_type) pair.
    Null response types (e.g. unsafe_compliance) are skipped and counted.

response_id convention:
    {prompt_id}__{response_type}__s01
    Example: du_A_osint_0001__safe_completion__s01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from safecomp_dpo.schemas import ResponseRecord, ResponseType
    from safecomp_dpo.io import write_jsonl
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.schemas import ResponseRecord, ResponseType
    from safecomp_dpo.io import write_jsonl


# Response types present in the custom format that we want to convert.
# unsafe_compliance is intentionally omitted — it is null for all records.
CONVERT_TYPES: list[str] = ["safe_completion", "hard_refusal"]


def build_source_id_to_prompt_id(prompts_path: Path) -> dict[str, str]:
    """Return {source_id: prompt_id} from a PromptRecord JSONL file."""
    mapping: dict[str, str] = {}
    with prompts_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            source_id = rec.get("source_id")
            prompt_id = rec.get("prompt_id")
            if not source_id or not prompt_id:
                print(
                    f"  WARNING: prompts line {lineno} missing source_id or prompt_id — skipped",
                    file=sys.stderr,
                )
                continue
            mapping[source_id] = prompt_id
    return mapping


def convert(
    responses_path: Path,
    prompts_path: Path,
    output_path: Path,
    report_path: Path,
    model: str,
) -> dict:
    """Convert the custom dual_use response file to ResponseRecord JSONL.

    Returns a summary dict suitable for writing as a JSON report.
    """
    source_id_to_prompt_id = build_source_id_to_prompt_id(prompts_path)
    print(f"  Loaded {len(source_id_to_prompt_id)} source_id -> prompt_id mappings")

    records: list[ResponseRecord] = []

    n_raw = 0
    n_skipped_no_mapping = 0
    n_skipped_null = 0
    n_converted = 0

    type_counts: dict[str, int] = {rt: 0 for rt in CONVERT_TYPES}

    with responses_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n_raw += 1
            src = json.loads(line)

            raw_id: str = src.get("raw_id", "")
            responses: dict = src.get("responses", {})

            prompt_id = source_id_to_prompt_id.get(raw_id)
            if not prompt_id:
                print(
                    f"  WARNING: line {lineno} raw_id={raw_id!r} has no matching prompt_id — skipped",
                    file=sys.stderr,
                )
                n_skipped_no_mapping += 1
                continue

            for rt_name in CONVERT_TYPES:
                text = responses.get(rt_name)
                if text is None:
                    n_skipped_null += 1
                    continue
                if not text.strip():
                    print(
                        f"  WARNING: line {lineno} raw_id={raw_id!r} response_type={rt_name!r} "
                        f"is empty — skipped",
                        file=sys.stderr,
                    )
                    n_skipped_null += 1
                    continue

                response_type = ResponseType(rt_name)
                response_id = f"{prompt_id}__{rt_name}__s01"

                rec = ResponseRecord(
                    response_id=response_id,
                    prompt_id=prompt_id,
                    response=text,
                    response_type=response_type,
                    model=model,
                    temperature=None,
                    metadata={
                        "backend": "api",
                        "source_file": str(responses_path),
                        "raw_id": raw_id,
                        "type_source": "explicit_api",
                    },
                )
                records.append(rec)
                n_converted += 1
                type_counts[rt_name] += 1

    write_jsonl(records, output_path)

    report = {
        "source_file": str(responses_path),
        "prompts_file": str(prompts_path),
        "output_file": str(output_path),
        "model": model,
        "n_raw_source_records": n_raw,
        "n_skipped_no_mapping": n_skipped_no_mapping,
        "n_skipped_null_or_empty": n_skipped_null,
        "n_converted": n_converted,
        "response_type_counts": type_counts,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert dual_use generation artifact to ResponseRecord JSONL."
    )
    p.add_argument(
        "--responses",
        default="hf_data/responses/dual_use/dualuse_responses.jsonl",
        help="Path to custom-format dual_use response JSONL.",
    )
    p.add_argument(
        "--prompts",
        default="hf_data/prompts/dual_use/dualuse_prompts.jsonl",
        help="Path to dual_use PromptRecord JSONL (for raw_id -> prompt_id mapping).",
    )
    p.add_argument(
        "--output",
        default="hf_data/responses/dual_use/dualuse_response_records.jsonl",
        help="Output path for ResponseRecord JSONL.",
    )
    p.add_argument(
        "--model",
        default="unknown_api",
        help="Model name to embed in metadata (e.g. claude-3-5-sonnet-20241022).",
    )
    p.add_argument(
        "--report",
        default="outputs/conversion_reports/dualuse_conversion_report.json",
        help="Path to write JSON conversion report.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    responses_path = Path(args.responses)
    prompts_path = Path(args.prompts)
    output_path = Path(args.output)
    report_path = Path(args.report)

    print(f"Reading responses from {responses_path}")
    print(f"Reading prompts from   {prompts_path}")
    print(f"Writing output to      {output_path}")

    report = convert(responses_path, prompts_path, output_path, report_path, args.model)

    print()
    print("Done.")
    print(f"  Source records:         {report['n_raw_source_records']}")
    print(f"  Skipped (no mapping):   {report['n_skipped_no_mapping']}")
    print(f"  Skipped (null/empty):   {report['n_skipped_null_or_empty']}")
    print(f"  Converted records:      {report['n_converted']}")
    for rt, count in report["response_type_counts"].items():
        print(f"    {rt}: {count}")
    print(f"  Report written to {report_path}")


if __name__ == "__main__":
    main()
