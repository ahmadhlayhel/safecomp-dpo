"""
Quality-control pipeline for unsafe_compliance ResponseRecords.

Runs between the Llama fallback pass and the final merge.  Three stages:

  1. Automatic checks  — fast, no API calls; flags known failure modes
  2. LLM judge pass    — GPT-4o-mini classifies each (prompt, response) pair
                         as KEEP / BORDERLINE / DROP  (optional, --judge flag)
  3. Spot-check sample — stratified sample across category × source for
                         mandatory human review before merge

Contract
--------
Inputs:
    --responses  <JSONL>    ResponseRecord JSONL to audit.  Repeat to pass
                            multiple files (e.g. GPT-clean + Llama fallback).
    --prompts    <JSONL>    all_prompts.jsonl — used for prompt text in the
                            judge pass and for stratum labels in spot-check.
    --passed     <JSONL>    Output: records that passed QC (KEEP).
                            Feed this to merge_unsafe_compliance.py instead
                            of the raw gpt4o_clean / llama_fallback files.
    --review     <JSONL>    Output: BORDERLINE + DROP records for human review.
    --spot-check <JSONL>    Output: stratified sample for mandatory spot-check.
    --report     <JSON>     Summary report.

Options:
    --judge                 Run the LLM judge pass (requires OPENAI_API_KEY).
    --judge-model <model>   Model for judge (default: gpt-4o-mini).
    --judge-cache <JSONL>   Cache file for judge results.  Allows safe resumption
                            if the judge pass is interrupted.  Defaults to
                            <report-dir>/qc_judge_cache.jsonl.
    --spot-check-n <int>    Records per stratum (default: 25).
    --seed <int>            Random seed for spot-check sampling (default: 42).

Auto-check flags (added to metadata.qc.auto):
    short_output     — word count < 20
    likely_refusal   — matches first-person declining phrases
    model_confusion  — model talks about itself / meta-commentary
    duplicate        — first 25 words match a previously seen record
    auto_drop        — any of the above is True (summary flag)

Judge fields (added to metadata.qc, only if --judge):
    judge_label    — "KEEP" | "BORDERLINE" | "DROP"
    judge_reason   — one-sentence explanation from the judge

Final disposition (metadata.qc.disposition):
    Without --judge: "KEEP" if not auto_drop, else "AUTO_DROP"
    With --judge:    judge_label value (KEEP / BORDERLINE / DROP)

Manual spot-check plan
----------------------
After this script runs, open --spot-check and review records in each stratum:

  Stratum                       Target N   What to verify
  ───────────────────────────── ──────     ─────────────────────────────────────
  unsafe × primary (GPT)          25       Actually responds to the unsafe request?
                                           Not a refusal? Would enable misuse?
  dual_use × primary (GPT)        25       Provides the requested dual-use content?
                                           Not a generic safety response?
  unsafe × fallback (Llama)       25       Same as unsafe GPT, plus: no Llama
                                           boilerplate ("as a responsible AI...")?
  dual_use × fallback (Llama)     25       Same as dual_use GPT; check Llama-specific
                                           hedging patterns that patterns may miss.

  Total mandatory review:        ≤100 records (fewer if a stratum is small)

  Threshold: if > 15% of any stratum looks wrong, flag before merging.
  The review JSONL contains the full prompt + response + QC metadata.

Usage
-----
  # Auto-checks only (no API calls):
  python scripts/qc_responses.py \\
      --responses outputs/unsafe_compliance/gpt4o_clean.jsonl \\
      --responses outputs/unsafe_compliance/llama_fallback.jsonl \\
      --prompts   hf_data/prompts/all_prompts.jsonl \\
      --passed    outputs/unsafe_compliance/qc_passed.jsonl \\
      --review    outputs/unsafe_compliance/qc_review.jsonl \\
      --spot-check outputs/unsafe_compliance/qc_spot_check.jsonl \\
      --report    outputs/unsafe_compliance/qc_report.json

  # With LLM judge (recommended before final merge):
  python scripts/qc_responses.py \\
      --responses outputs/unsafe_compliance/gpt4o_clean.jsonl \\
      --responses outputs/unsafe_compliance/llama_fallback.jsonl \\
      --prompts   hf_data/prompts/all_prompts.jsonl \\
      --passed    outputs/unsafe_compliance/qc_passed.jsonl \\
      --review    outputs/unsafe_compliance/qc_review.jsonl \\
      --spot-check outputs/unsafe_compliance/qc_spot_check.jsonl \\
      --report    outputs/unsafe_compliance/qc_report.json \\
      --judge
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord


# ---------------------------------------------------------------------------
# Auto-check: refusal patterns (mirrors filter_refusals.py)
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"I(?:'m| am) (?:sorry|unable|not able|afraid)\b",
        r"\bI (?:can'?t|cannot|won'?t|will not) (?:help|assist|provide|generate|create|write|fulfill|support|do that|complete)\b",
        r"\bI(?:'m| am) not (?:able|going to|in a position|willing)\b",
        r"\bI (?:do not|don'?t) (?:assist|help|provide|support|generate|create)\b",
        r"\bI (?:must|have to|need to) (?:decline|refuse)\b",
        r"\bI (?:apologize|must decline|am going to decline)\b",
        r"\b(?:This|That|Your) (?:request|question|prompt|task) (?:is|asks|would|goes|violates|isn'?t)\b",
        r"\bnot (?:something|a task|a request) I(?:'m| am) (?:able|willing|going)\b",
        r"^I understand (?:that|your|the) .{0,80}but\b",
        r"^I appreciate (?:that|your|the) .{0,80}but\b",
        r"\bI(?:'m| am) (?:designed|here) to (?:be helpful|help with beneficial|assist with safe)\b",
    ]
]

# ---------------------------------------------------------------------------
# Auto-check: model self-identification / meta-commentary patterns
# ---------------------------------------------------------------------------

_MODEL_CONFUSION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bAs an? (?:AI|language model|AI language model|AI assistant|LLM|large language model)\b",
        r"\bI(?:'m| am) an? (?:AI|language model|AI language model|AI assistant|LLM|large language model)\b",
        r"\bI(?:'m| am) (?:Claude|ChatGPT|GPT-\d|Llama|Gemini|Bard|Mistral)\b",
        r"\b(?:created|developed|trained|built) by (?:Anthropic|OpenAI|Meta|Google|Microsoft|Mistral)\b",
        r"\bmy training (?:data|cutoff|knowledge)\b",
        r"\bAs a responsible AI\b",
        r"\bI don'?t have (?:the ability to access|access to real-time|personal (?:opinions|beliefs|feelings))\b",
        r"\bI(?:'m| am) just an? (?:AI|language model|program|bot|chatbot)\b",
    ]
]

_SHORT_WORD_THRESHOLD = 20
_DUPLICATE_FINGERPRINT_WORDS = 25


# ---------------------------------------------------------------------------
# Auto-check runner
# ---------------------------------------------------------------------------


def run_auto_checks(rec: ResponseRecord, seen_fingerprints: set[str]) -> dict[str, Any]:
    """Return auto QC flags for a single record.

    seen_fingerprints is mutated in-place to track duplicates across calls.
    """
    text = rec.response
    words = text.split()

    short_output = len(words) < _SHORT_WORD_THRESHOLD
    likely_refusal = any(p.search(text) for p in _REFUSAL_PATTERNS)
    model_confusion = any(p.search(text) for p in _MODEL_CONFUSION_PATTERNS)

    fp = " ".join(words[:_DUPLICATE_FINGERPRINT_WORDS]).lower()
    duplicate = fp in seen_fingerprints
    if not duplicate:
        seen_fingerprints.add(fp)

    auto_drop = short_output or likely_refusal or model_confusion or duplicate

    return {
        "auto": {
            "short_output": short_output,
            "likely_refusal": likely_refusal,
            "model_confusion": model_confusion,
            "duplicate": duplicate,
            "auto_drop": auto_drop,
            "word_count": len(words),
        }
    }


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are auditing rejected-example training data for AI safety research.
Each record is an unsafe_compliance response — a model output that complies
with a harmful or dual-use request. These serve as the REJECTED side of
DPO preference pairs; the model being trained will learn NOT to produce them.

Your task: verify that each response is usable as a rejected example.\
"""

_JUDGE_USER_TEMPLATE = """\
PROMPT:
{prompt}

RESPONSE:
{response}

Rate the response using exactly one of the labels below, then give one sentence explaining why.

KEEP       — clearly complies with the request; actually answers what was asked;
             provides harmful or dual-use content; is not a refusal, filler,
             generic disclaimer, or meta-commentary
BORDERLINE — partially complies or is ambiguous; hedges but still provides some
             useful content; has minor quality issues; warrants human review
DROP       — is a refusal; is generic filler unrelated to the prompt; is malformed;
             discusses the prompt at the meta-level rather than answering it;
             is clearly off-topic; or is a model self-identification message

First line: KEEP, BORDERLINE, or DROP
Second line: one sentence reason.\
"""


def run_judge(
    prompt: str,
    response: str,
    client: Any,
    model: str,
    max_retries: int = 4,
) -> tuple[str, str]:
    """Call the LLM judge. Returns (label, reason). label in {KEEP, BORDERLINE, DROP}."""
    user_content = _JUDGE_USER_TEMPLATE.format(prompt=prompt, response=response)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            raw = (resp.choices[0].message.content or "").strip()
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = 2.0 * (2 ** attempt)
                print(f"  Judge API error (attempt {attempt + 1}): {exc} — retrying in {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
            else:
                raise

    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    first = lines[0].upper() if lines else ""

    if "KEEP" in first:
        label = "KEEP"
    elif "DROP" in first:
        label = "DROP"
    else:
        label = "BORDERLINE"

    reason = lines[1] if len(lines) > 1 else lines[0] if lines else raw
    return label, reason


# ---------------------------------------------------------------------------
# Spot-check sampling
# ---------------------------------------------------------------------------


def sample_spot_check(
    records: list[ResponseRecord],
    prompt_map: dict[str, PromptRecord],
    n_per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Stratified sample across {category} × {pass label} from the KEEP set.

    Strata keys: e.g. "unsafe__primary", "dual_use__fallback".
    Each sampled record is returned as a flat dict with prompt text included
    for easy human review.
    """
    rng = random.Random(seed)
    strata: dict[str, list[ResponseRecord]] = defaultdict(list)

    for rec in records:
        pr = prompt_map.get(rec.prompt_id)
        cat = pr.category.value if pr else "unknown"
        pass_label = rec.metadata.get("pass", "primary")
        strata[f"{cat}__{pass_label}"].append(rec)

    samples: list[dict[str, Any]] = []
    for stratum_key in sorted(strata):
        pool = strata[stratum_key]
        chosen = rng.sample(pool, min(n_per_stratum, len(pool)))
        for rec in chosen:
            pr = prompt_map.get(rec.prompt_id)
            samples.append(
                {
                    "stratum": stratum_key,
                    "response_id": rec.response_id,
                    "prompt_id": rec.prompt_id,
                    "prompt": pr.prompt if pr else "[prompt not found]",
                    "response": rec.response,
                    "model": rec.model,
                    "qc": rec.metadata.get("qc", {}),
                }
            )

    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QC pipeline for unsafe_compliance ResponseRecords."
    )
    p.add_argument(
        "--responses",
        action="append",
        required=True,
        metavar="JSONL",
        help="ResponseRecord JSONL to audit. Repeat for multiple files.",
    )
    p.add_argument("--prompts", required=True, help="PromptRecord JSONL (all_prompts.jsonl)")
    p.add_argument("--passed", required=True, help="Output: KEEP records")
    p.add_argument("--review", required=True, help="Output: BORDERLINE + DROP records")
    p.add_argument("--spot-check", required=True, help="Output: stratified spot-check sample")
    p.add_argument("--report", required=True, help="Output: JSON summary report")
    p.add_argument(
        "--judge",
        action="store_true",
        help="Run the LLM judge pass (requires OPENAI_API_KEY).",
    )
    p.add_argument("--judge-model", default="gpt-4o-mini", help="Model for judge calls")
    p.add_argument(
        "--judge-cache",
        default=None,
        help=(
            "JSONL cache for judge results — safe resumption if interrupted. "
            "Defaults to <report-dir>/qc_judge_cache.jsonl."
        ),
    )
    p.add_argument(
        "--spot-check-n",
        type=int,
        default=25,
        help="Records per stratum in the spot-check sample (default: 25)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for spot-check sampling")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # --- Load records ---
    all_records: list[ResponseRecord] = []
    for path_str in args.responses:
        path = Path(path_str)
        recs = load_response_records(path)
        all_records.extend(recs)
        print(f"Loaded {len(recs):>5} records from {path}")
    print(f"Total: {len(all_records)} records")

    # --- Load prompts ---
    print(f"Loading prompts from {args.prompts}")
    all_prompts = load_prompt_records(args.prompts)
    prompt_map: dict[str, PromptRecord] = {pr.prompt_id: pr for pr in all_prompts}
    print(f"  {len(prompt_map)} prompts loaded")

    # --- Stage 1: Auto-checks ---
    print("\nRunning auto-checks...")
    seen_fingerprints: set[str] = set()
    qc_map: dict[str, dict[str, Any]] = {}

    n_auto_drop = 0
    auto_flag_counts: dict[str, int] = defaultdict(int)

    for rec in all_records:
        qc = run_auto_checks(rec, seen_fingerprints)
        qc_map[rec.response_id] = qc
        auto = qc["auto"]
        if auto["auto_drop"]:
            n_auto_drop += 1
        for flag in ("short_output", "likely_refusal", "model_confusion", "duplicate"):
            if auto[flag]:
                auto_flag_counts[flag] += 1

    print(f"  auto_drop: {n_auto_drop} / {len(all_records)}")
    for flag, count in sorted(auto_flag_counts.items()):
        print(f"    {flag}: {count}")

    # --- Stage 2: LLM judge pass (optional) ---
    if args.judge:
        import os

        try:
            import openai
        except ImportError:
            print("ERROR: openai package required for --judge. pip install openai>=1.0", file=sys.stderr)
            sys.exit(1)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
            sys.exit(1)

        client = openai.OpenAI(api_key=api_key)

        # Determine cache path
        report_dir = Path(args.report).parent
        cache_path = Path(args.judge_cache) if args.judge_cache else report_dir / "qc_judge_cache.jsonl"

        # Load existing cache
        judge_cache: dict[str, tuple[str, str]] = {}
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        judge_cache[entry["response_id"]] = (
                            entry["judge_label"],
                            entry["judge_reason"],
                        )
            print(f"\nJudge cache loaded: {len(judge_cache)} existing results from {cache_path}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_f = cache_path.open("a", encoding="utf-8")

        n_to_judge = sum(1 for rec in all_records if rec.response_id not in judge_cache)
        print(f"\nRunning judge pass on {n_to_judge} records (skipping {len(judge_cache)} cached)...")

        n_judged = 0
        try:
            for rec in all_records:
                if rec.response_id in judge_cache:
                    label, reason = judge_cache[rec.response_id]
                else:
                    pr = prompt_map.get(rec.prompt_id)
                    prompt_text = pr.prompt if pr else rec.prompt_id
                    label, reason = run_judge(prompt_text, rec.response, client, args.judge_model)
                    judge_cache[rec.response_id] = (label, reason)
                    cache_f.write(
                        json.dumps(
                            {
                                "response_id": rec.response_id,
                                "judge_label": label,
                                "judge_reason": reason,
                            }
                        )
                        + "\n"
                    )
                    cache_f.flush()
                    n_judged += 1
                    if n_judged % 100 == 0:
                        print(f"  {n_judged} / {n_to_judge} judged...")

                qc_map[rec.response_id]["judge_label"] = label
                qc_map[rec.response_id]["judge_reason"] = reason
        finally:
            cache_f.close()

        print(f"  Judge pass complete. {n_judged} new calls, {len(judge_cache)} total cached.")

    # --- Assign final disposition ---
    for rec in all_records:
        qc = qc_map[rec.response_id]
        if args.judge:
            qc["disposition"] = qc["judge_label"]
        else:
            qc["disposition"] = "AUTO_DROP" if qc["auto"]["auto_drop"] else "KEEP"

    # --- Annotate records and split ---
    passed_records: list[ResponseRecord] = []
    review_records: list[ResponseRecord] = []

    for rec in all_records:
        qc = qc_map[rec.response_id]
        updated_meta = {**rec.metadata, "qc": qc}
        annotated = rec.model_copy(update={"metadata": updated_meta})

        if qc["disposition"] == "KEEP":
            passed_records.append(annotated)
        else:
            review_records.append(annotated)

    # --- Stage 3: Spot-check sample ---
    spot_check = sample_spot_check(passed_records, prompt_map, args.spot_check_n, args.seed)

    # --- Write outputs ---
    passed_path = Path(args.passed)
    review_path = Path(args.review)
    spot_path = Path(getattr(args, "spot_check"))
    report_path = Path(args.report)

    write_jsonl(passed_records, passed_path)
    write_jsonl(review_records, review_path)

    spot_path.parent.mkdir(parents=True, exist_ok=True)
    with spot_path.open("w", encoding="utf-8") as f:
        for item in spot_check:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # --- Build report ---
    disposition_counts: dict[str, int] = defaultdict(int)
    by_stratum_disposition: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rec in all_records:
        qc = qc_map[rec.response_id]
        disp = qc["disposition"]
        disposition_counts[disp] += 1
        pr = prompt_map.get(rec.prompt_id)
        cat = pr.category.value if pr else "unknown"
        pass_label = rec.metadata.get("pass", "primary")
        stratum = f"{cat}__{pass_label}"
        by_stratum_disposition[stratum][disp] += 1

    spot_strata = defaultdict(int)
    for item in spot_check:
        spot_strata[item["stratum"]] += 1

    report = {
        "total_input": len(all_records),
        "passed": len(passed_records),
        "review": len(review_records),
        "judge_run": args.judge,
        "judge_model": args.judge_model if args.judge else None,
        "disposition_counts": dict(disposition_counts),
        "auto_flag_counts": dict(auto_flag_counts),
        "by_stratum": {k: dict(v) for k, v in sorted(by_stratum_disposition.items())},
        "spot_check_total": len(spot_check),
        "spot_check_by_stratum": dict(spot_strata),
        "outputs": {
            "passed": str(passed_path),
            "review": str(review_path),
            "spot_check": str(spot_path),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # --- Summary ---
    print(f"\nQC complete.")
    print(f"  passed (KEEP):  {len(passed_records)}")
    print(f"  review:         {len(review_records)}")
    print(f"  dispositions:   {dict(disposition_counts)}")
    print(f"  spot-check:     {len(spot_check)} records across {len(spot_strata)} strata")
    print(f"\n  passed     : {passed_path}")
    print(f"  review     : {review_path}")
    print(f"  spot-check : {spot_path}")
    print(f"  report     : {report_path}")
    print(f"\nNext: review {spot_path} manually, then run merge_unsafe_compliance.py with --gpt-clean / --llama-fallback pointing at {passed_path}.")


if __name__ == "__main__":
    main()
