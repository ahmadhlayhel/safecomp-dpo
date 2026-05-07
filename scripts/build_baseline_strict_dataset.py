"""
Build hf_data/dpo/baseline_strict_dpo_dataset.jsonl.

Differs from the baseline regime ONLY on dual_use:

  baseline (current):  chosen=hard_refusal vs rejected=unsafe_compliance
                       — but the "unsafe_compliance" pool for dual_use was
                         sourced via gpt-4o-mini and is *not actually unsafe*.
                         It's well-curated helpful analytical content.
  baseline_strict:     chosen=hard_refusal vs rejected=safe_completion
                       — directly trains the model to prefer refusal even over
                         the safest, most professional engagement style.
                         Skips the broken unsafe_compliance pool entirely.

Shared categories (unsafe / benign_sensitive / benign) are unchanged from
the existing baseline_dpo_dataset.jsonl, so the only delta vs baseline is
the dual_use rejected text.

Sources:
  - hf_data/dpo/baseline_dpo_dataset.jsonl  (carries the chosen=hard_refusal
                                              text we want to keep)
  - hf_data/responses/dual_use/dualuse_response_records.jsonl
        (provides per-prompt safe_completion text we substitute as rejected)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BASELINE_DPO = Path("hf_data/dpo/baseline_dpo_dataset.jsonl")
DUALUSE_RESPONSES = Path("hf_data/responses/dual_use/dualuse_response_records.jsonl")
OUT_PATH = Path("hf_data/dpo/baseline_strict_dpo_dataset.jsonl")


def main() -> None:
    # Index dual_use safe_completion text by prompt_id (1 per prompt).
    safe_by_pid: dict[str, dict] = {}
    with DUALUSE_RESPONSES.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("response_type") != "safe_completion":
                continue
            safe_by_pid[r["prompt_id"]] = r
    print(f"Indexed {len(safe_by_pid)} dual_use safe_completion responses")

    n_total = 0
    n_dual_replaced = 0
    n_dual_missing = 0
    by_cat = Counter()
    by_cat_pt = Counter()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_fh = OUT_PATH.open("w", encoding="utf-8")
    try:
        with BASELINE_DPO.open() as f:
            for line in f:
                r = json.loads(line)
                n_total += 1
                cat = r.get("category")
                by_cat[cat] += 1

                if cat == "dual_use":
                    safe = safe_by_pid.get(r["prompt_id"])
                    if safe is None:
                        n_dual_missing += 1
                        continue  # drop this prompt
                    new = dict(r)
                    new["pair_type"] = "strict_refusal"
                    new["pair_id"] = f"{r['prompt_id']}__strict_refusal__p01"
                    new["rejected"] = safe["response"]
                    new["rejected_id"] = safe["response_id"]
                    out_fh.write(json.dumps(new, ensure_ascii=False) + "\n")
                    n_dual_replaced += 1
                    by_cat_pt[(cat, "strict_refusal")] += 1
                else:
                    out_fh.write(line if line.endswith("\n") else line + "\n")
                    by_cat_pt[(cat, r.get("pair_type"))] += 1
    finally:
        out_fh.close()

    print(f"\nWrote {OUT_PATH}")
    print(f"  source rows:           {n_total}")
    print(f"  dual_use replaced:     {n_dual_replaced}")
    print(f"  dual_use missing safe: {n_dual_missing}")
    print(f"  category counts (output):")
    for cat in ("unsafe", "dual_use", "benign_sensitive", "benign"):
        print(f"    {cat}: {by_cat_pt[(cat, 'baseline_refusal')] + by_cat_pt[(cat, 'strict_refusal')] + by_cat_pt[(cat, 'safe_completion')]}")
    print(f"  (category, pair_type):")
    for k in sorted(by_cat_pt):
        print(f"    {k}: {by_cat_pt[k]}")

    # Sanity-load via the pair-data lens
    n_lines = sum(1 for _ in OUT_PATH.open())
    print(f"  total rows in output:  {n_lines}")


if __name__ == "__main__":
    main()
