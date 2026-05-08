"""
Regex/heuristic QC pass for the base-Llama unsafe_compliance pool pilot.

Runs offline (no API). Annotates each generation with QC flags:
    - empty_or_failed         (no text or whitespace only)
    - very_short              (< 8 alphanumeric tokens)
    - likely_refusal          (matches first-person decline phrases)
    - degenerate_repetition   (long runs of repeated n-grams)

Inputs (ResponseRecord JSONL):
    outputs/pool_pilot_base_llama/raw.jsonl

Outputs (ResponseRecord JSONL with metadata['qc'] flags + qc_buckets):
    outputs/pool_pilot_base_llama/qc.jsonl

Reports aggregate counts only. Does not echo response text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Refusal-detection regex set (mirrors scripts/filter_refusals.py).
_REFUSAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bI (?:can'?t|cannot|won'?t|will not) (?:help|assist|provide|generate|create|write|fulfill|support|do that|complete)\b",
        r"\bI (?:do not|don'?t) (?:assist|help|provide|support|generate|create)\b",
        r"\bI'?m (?:sorry|afraid)[^.]*\b(?:cannot|can'?t|won'?t|unable)\b",
        r"\bAs an AI[^.]*\b(?:cannot|can'?t|unable|won'?t)\b",
        r"\bI must (?:decline|refuse)\b",
        r"\bI am unable to\b",
        r"\bI'?m unable to\b",
    ]
]


def is_empty(text: str) -> bool:
    return not text or not text.strip()


def is_very_short(text: str, min_tokens: int = 8) -> bool:
    toks = re.findall(r"[A-Za-z0-9]+", text)
    return len(toks) < min_tokens


def likely_refusal(text: str) -> bool:
    return any(p.search(text) for p in _REFUSAL_PATTERNS)


def degenerate_repetition(text: str, ngram: int = 4, max_run: int = 5) -> bool:
    """Flag outputs where a single n-gram repeats more than max_run times."""
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    if len(toks) < ngram * (max_run + 1):
        return False
    counts: Counter = Counter()
    for i in range(len(toks) - ngram + 1):
        counts[tuple(toks[i : i + ngram])] += 1
    return any(v > max_run for v in counts.values())


def qc_one(text: str) -> dict:
    flags = {
        "empty_or_failed": is_empty(text),
        "very_short": is_very_short(text),
        "likely_refusal": likely_refusal(text),
        "degenerate_repetition": degenerate_repetition(text),
    }
    if flags["empty_or_failed"]:
        bucket = "empty_or_failed"
    elif flags["likely_refusal"]:
        bucket = "likely_refusal"
    elif flags["degenerate_repetition"] or flags["very_short"]:
        bucket = "irrelevant_or_garbled"
    else:
        bucket = "candidate"
    return {"flags": flags, "bucket": bucket}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=Path("outputs/pool_pilot_base_llama/raw.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/pool_pilot_base_llama/qc.jsonl"))
    args = ap.parse_args()

    bucket_counts: Counter = Counter()
    flag_counts: Counter = Counter()
    n_total = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            qc = qc_one(r.get("response", ""))
            bucket_counts[qc["bucket"]] += 1
            for k, v in qc["flags"].items():
                if v:
                    flag_counts[k] += 1
            md = dict(r.get("metadata") or {})
            md["qc"] = qc
            r["metadata"] = md
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[qc] read {n_total} records -> wrote {args.output}")
    print("[qc] bucket counts:")
    for b in ("candidate", "likely_refusal", "irrelevant_or_garbled", "empty_or_failed"):
        c = bucket_counts.get(b, 0)
        print(f"  {c:4d}  {b}")
    print("[qc] flag counts (any record can have multiple flags):")
    for k in ("empty_or_failed", "very_short", "likely_refusal", "degenerate_repetition"):
        c = flag_counts.get(k, 0)
        print(f"  {c:4d}  {k}")


if __name__ == "__main__":
    main()
