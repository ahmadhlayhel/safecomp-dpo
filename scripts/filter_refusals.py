"""
Detect and separate likely-refused responses from a Pass-1 generation run.

Contract
--------
Inputs:
    --responses       ResponseRecord JSONL from Pass 1 (GPT-4o-mini output).
                      All records must have response_type="unsafe_compliance".
    --prompts         PromptRecord JSONL (all_prompts.jsonl) — used to emit
                      refused PromptRecords for the Pass-2 Llama fallback.
    --clean-out       Output path for non-refused ResponseRecords.
                      Each record gains: likely_refusal=false, pass="primary",
                      generator=<model> in metadata.
    --refused-prompts Output PromptRecord JSONL for prompts whose Pass-1 response
                      was flagged as a refusal. Feed this to generate_responses.py
                      as --prompts for the Llama fallback pass.
    --report          Path to write a JSON summary report.

Outputs:
    <clean-out>:       ResponseRecord JSONL — records that passed the refusal check.
    <refused-prompts>: PromptRecord JSONL — prompts needing Pass-2 generation.
    <report>:          JSON with total/clean/refused counts broken down by category.

Refusal detection
-----------------
A response is flagged as likely_refusal=True if its text matches one or more
first-person declining patterns. Length alone does not trigger a flag —
a short but unambiguous answer is kept. The patterns are conservative: they
target GPT-4o-mini's characteristic refusal phrasing rather than content words
(e.g. "harmful") that could also appear in a genuine compliant response.

Review the refusal_report.json and spot-check refused_prompts.jsonl before
running Pass 2. Manual overrides can be applied by editing refused_prompts.jsonl
before feeding it to the Llama fallback.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # First-person inability / unwillingness
        r"I(?:'m| am) (?:sorry|unable|not able|afraid)\b",
        r"\bI (?:can'?t|cannot|won'?t|will not) (?:help|assist|provide|generate|create|write|fulfill|support|do that|complete)\b",
        r"\bI(?:'m| am) not (?:able|going to|in a position|willing)\b",
        r"\bI (?:do not|don'?t) (?:assist|help|provide|support|generate|create)\b",
        r"\bI (?:must|have to|need to) (?:decline|refuse)\b",
        r"\bI (?:apologize|must decline|am going to decline)\b",
        # Request-framing refusals
        r"\b(?:This|That|Your) (?:request|question|prompt|task) (?:is|asks|would|goes|violates|isn'?t)\b",
        r"\bnot (?:something|a task|a request) I(?:'m| am) (?:able|willing|going)\b",
        # Common GPT opener phrases that signal a refusal is coming
        r"^I understand (?:that|your|the) .{0,80}but\b",
        r"^I appreciate (?:that|your|the) .{0,80}but\b",
        r"\bI(?:'m| am) (?:designed|here) to (?:be helpful|help with beneficial|assist with safe)\b",
    ]
]


def is_likely_refusal(text: str) -> bool:
    """Return True if the response text matches a refusal pattern."""
    return any(p.search(text) for p in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Tag likely refusals in a Pass-1 unsafe_compliance response JSONL "
            "and emit refused PromptRecords for the Llama fallback pass."
        )
    )
    p.add_argument("--responses", required=True, help="Pass-1 ResponseRecord JSONL")
    p.add_argument(
        "--prompts",
        required=True,
        help="PromptRecord JSONL (all_prompts.jsonl) — source of truth for PromptRecords",
    )
    p.add_argument("--clean-out", required=True, help="Output: clean ResponseRecord JSONL")
    p.add_argument(
        "--refused-prompts",
        required=True,
        help="Output: PromptRecord JSONL for refused subset (Pass-2 input)",
    )
    p.add_argument("--report", required=True, help="Output: JSON summary report")
    return p


def main() -> None:
    args = build_parser().parse_args()

    responses_path = Path(args.responses)
    prompts_path = Path(args.prompts)
    clean_out_path = Path(getattr(args, "clean_out"))
    refused_prompts_path = Path(getattr(args, "refused_prompts"))
    report_path = Path(args.report)

    print(f"Loading responses from {responses_path}")
    responses = load_response_records(responses_path)
    print(f"  {len(responses)} records loaded")

    print(f"Loading prompts from {prompts_path}")
    all_prompts = load_prompt_records(prompts_path)
    prompt_map: dict[str, PromptRecord] = {pr.prompt_id: pr for pr in all_prompts}
    print(f"  {len(prompt_map)} prompts loaded")

    clean_records: list[ResponseRecord] = []
    refused_prompt_ids: list[str] = []
    by_category_clean: dict[str, int] = {}
    by_category_refused: dict[str, int] = {}
    n_missing_prompt = 0

    for rec in responses:
        flagged = is_likely_refusal(rec.response)
        pr = prompt_map.get(rec.prompt_id)
        cat = pr.category.value if pr else "unknown"

        if flagged:
            refused_prompt_ids.append(rec.prompt_id)
            by_category_refused[cat] = by_category_refused.get(cat, 0) + 1
        else:
            updated_meta = {
                **rec.metadata,
                "likely_refusal": False,
                "pass": "primary",
                "generator": rec.model,
            }
            clean_records.append(rec.model_copy(update={"metadata": updated_meta}))
            by_category_clean[cat] = by_category_clean.get(cat, 0) + 1

    refused_prompts: list[PromptRecord] = []
    for pid in refused_prompt_ids:
        pr = prompt_map.get(pid)
        if pr is None:
            print(
                f"  WARNING: refused prompt_id {pid!r} not found in prompts file",
                file=sys.stderr,
            )
            n_missing_prompt += 1
        else:
            refused_prompts.append(pr)

    write_jsonl(clean_records, clean_out_path)
    write_jsonl(refused_prompts, refused_prompts_path)

    n_clean = len(clean_records)
    n_refused = len(refused_prompt_ids)
    total = len(responses)

    report = {
        "responses_input": str(responses_path),
        "total_input": total,
        "clean": n_clean,
        "refused": n_refused,
        "refusal_rate_pct": round(100 * n_refused / total, 1) if total else 0,
        "missing_prompt_records": n_missing_prompt,
        "clean_by_category": by_category_clean,
        "refused_by_category": by_category_refused,
        "clean_output": str(clean_out_path),
        "refused_prompts_output": str(refused_prompts_path),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults:")
    print(f"  total input:  {total}")
    print(f"  clean (kept): {n_clean}")
    print(f"  refused:      {n_refused}  ({report['refusal_rate_pct']}%)")
    print(f"  clean by category:   {by_category_clean}")
    print(f"  refused by category: {by_category_refused}")
    print(f"\nOutputs:")
    print(f"  clean records:   {clean_out_path}  ({n_clean} records)")
    print(f"  refused prompts: {refused_prompts_path}  ({len(refused_prompts)} records)")
    print(f"  report:          {report_path}")

    if n_refused > 0:
        print(
            f"\nNext: review {refused_prompts_path}, then run Pass 2 "
            f"(unsafe_compliance_llama.yaml) on BABEL."
        )
    else:
        print(
            "\nNo refusals detected — Pass 2 not needed. "
            "Run merge_unsafe_compliance.py with --gpt-clean only."
        )


if __name__ == "__main__":
    main()
