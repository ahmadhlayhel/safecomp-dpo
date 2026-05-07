"""
Side-by-side baseline vs safecomp generations on dual_use prompts.

Pulls 30 paired dual_use eval records (10 per stratification bucket) and
writes a single markdown file for human review. The point is to *see* what
the two adapters actually produce on the category where the SafeComp
hypothesis lives, instead of relying on aggregate judge labels.

Stratification (10 each):
  A) LLM-judge disagrees between regimes — the predicted differentiation
  B) Both regimes labeled `safe_completion` by the LLM judge — what does
     "safe_completion" look like in practice?
  C) Both regimes labeled `hard_refusal` by the LLM judge — what does the
     model refuse to do, and how?

Inputs:
  outputs/eval/baseline_peft_eval.jsonl                       (regex judge)
  outputs/eval/safecomp_peft_eval.jsonl                       (regex judge)
  outputs/eval/llm_judge/baseline_peft_eval.llm_judged.jsonl  (LLM judge)
  outputs/eval/llm_judge/safecomp_peft_eval.llm_judged.jsonl  (LLM judge)

Output:
  outputs/eval/inspect/dual_use_inspection.md

Usage:
  python scripts/inspect_dual_use_pairs.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REGEX_BASE = Path("outputs/eval/baseline_peft_eval.jsonl")
REGEX_SAFE = Path("outputs/eval/safecomp_peft_eval.jsonl")
LLM_BASE = Path("outputs/eval/llm_judge/baseline_peft_eval.llm_judged.jsonl")
LLM_SAFE = Path("outputs/eval/llm_judge/safecomp_peft_eval.llm_judged.jsonl")

OUT_PATH = Path("outputs/eval/inspect/dual_use_inspection.md")
PER_BUCKET = 10
SEED = 0
RESPONSE_TRUNC = 1200


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def main() -> None:
    rb = load_jsonl(REGEX_BASE)
    rs = load_jsonl(REGEX_SAFE)
    lb = load_jsonl(LLM_BASE)
    ls = load_jsonl(LLM_SAFE)

    pids_dualuse = sorted(
        pid for pid in rb if rb[pid]["category"] == "dual_use" and pid in rs and pid in lb and pid in ls
    )
    print(f"dual_use paired prompts: {len(pids_dualuse)}")

    rows = []
    for pid in pids_dualuse:
        rows.append(
            {
                "pid": pid,
                "prompt": rb[pid]["prompt"],
                "base_resp": rb[pid]["response"],
                "safe_resp": rs[pid]["response"],
                "base_regex": rb[pid]["judge_label"],
                "safe_regex": rs[pid]["judge_label"],
                "base_llm": lb[pid]["judge_label"],
                "safe_llm": ls[pid]["judge_label"],
            }
        )

    bucket_diff = [r for r in rows if r["base_llm"] != r["safe_llm"]]
    bucket_safe_safe = [
        r for r in rows
        if r["base_llm"] == "safe_completion" and r["safe_llm"] == "safe_completion"
    ]
    bucket_refuse_refuse = [
        r for r in rows
        if r["base_llm"] == "hard_refusal" and r["safe_llm"] == "hard_refusal"
    ]

    print(f"  bucket A (LLM disagrees between regimes): {len(bucket_diff)}")
    print(f"  bucket B (both safe_completion):           {len(bucket_safe_safe)}")
    print(f"  bucket C (both hard_refusal):              {len(bucket_refuse_refuse)}")

    rng = random.Random(SEED)
    picks_A = rng.sample(bucket_diff, k=min(PER_BUCKET, len(bucket_diff)))
    picks_B = rng.sample(bucket_safe_safe, k=min(PER_BUCKET, len(bucket_safe_safe)))
    picks_C = rng.sample(bucket_refuse_refuse, k=min(PER_BUCKET, len(bucket_refuse_refuse)))

    cross = {}
    for r in rows:
        key = (r["base_llm"], r["safe_llm"])
        cross[key] = cross.get(key, 0) + 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Dual_use side-by-side inspection — baseline vs safecomp\n\n")
        f.write(f"Total dual_use prompts paired: **{len(rows)}**\n\n")
        f.write("## LLM-judge cross-tab (rows = baseline label, cols = safecomp label)\n\n")
        labels = ["hard_refusal", "safe_completion", "unsafe_compliance", "helpful_answer"]
        f.write("| baseline \\\\ safecomp | " + " | ".join(labels) + " |\n")
        f.write("|" + "|".join(["---"] * (len(labels) + 1)) + "|\n")
        for bl in labels:
            row_vals = [str(cross.get((bl, sl), 0)) for sl in labels]
            f.write(f"| **{bl}** | " + " | ".join(row_vals) + " |\n")
        f.write("\n")

        for label, picks, desc in [
            ("A", picks_A, "LLM-judge labels DIFFER between regimes — the predicted differentiation"),
            ("B", picks_B, "Both regimes labeled `safe_completion` by the LLM judge"),
            ("C", picks_C, "Both regimes labeled `hard_refusal` by the LLM judge"),
        ]:
            f.write(f"\n---\n\n# Bucket {label}: {desc}\n")
            f.write(f"Sampled {len(picks)} of {len(bucket_diff if label == 'A' else (bucket_safe_safe if label == 'B' else bucket_refuse_refuse))}.\n\n")

            for i, r in enumerate(picks, 1):
                bresp = r["base_resp"]
                sresp = r["safe_resp"]
                if len(bresp) > RESPONSE_TRUNC:
                    bresp = bresp[:RESPONSE_TRUNC] + " [...truncated]"
                if len(sresp) > RESPONSE_TRUNC:
                    sresp = sresp[:RESPONSE_TRUNC] + " [...truncated]"

                f.write(f"\n## {label}{i}. `{r['pid']}`\n\n")
                f.write(f"**Prompt:**\n\n> {r['prompt']}\n\n")
                f.write(
                    f"**Baseline** (llm=`{r['base_llm']}`, regex=`{r['base_regex']}`):\n\n"
                )
                f.write("```\n" + bresp.strip() + "\n```\n\n")
                f.write(
                    f"**SafeComp** (llm=`{r['safe_llm']}`, regex=`{r['safe_regex']}`):\n\n"
                )
                f.write("```\n" + sresp.strip() + "\n```\n\n")

    print(f"\nWrote {OUT_PATH}")
    print(f"Length: {OUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
