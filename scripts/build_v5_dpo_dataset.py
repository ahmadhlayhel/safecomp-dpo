"""
Build the v5-basepool DPO datasets for the baseline and safecomp regimes.

Replaces the contaminated gpt-4o-mini-sourced dual_use rejected slice with
base-Llama v5 generations (domain-matched 2-shot, 9 high-yield domains,
eval-excluded — see outputs/pool_pilot_base_llama_v5/).

Per CLAUDE.md's locked v1 pair logic:
  baseline regime, dual_use:
      chosen   = hard_refusal (curated, dualuse_response_records.jsonl)
      rejected = v5 base-Llama generation
      pair_type = baseline_refusal
  safecomp regime, dual_use:
      pair A: chosen=safe_completion, rejected=hard_refusal       (kept from old)
      pair B: chosen=safe_completion, rejected=v5 generation      (replaced)

Inputs:
  outputs/pool_pilot_base_llama_v5/raw.jsonl                 1,973 records
  outputs/pool_pilot_base_llama_v5/sampled_prompts.jsonl     1,973 records
  hf_data/responses/dual_use/dualuse_response_records.jsonl  5,004 records
                                                             (2,502 prompts × {safe_completion, hard_refusal})
  hf_data/dpo/baseline_dpo_dataset.jsonl                     7,875 (kept: shared categories)
  hf_data/dpo/safecomp_dpo_dataset.jsonl                    10,377 (kept: shared cats + safe>hard)
  hf_data/eval/held_out_prompts.jsonl                          800 (200 dual_use heldout)

Outputs:
  hf_data/dpo/baseline_v5_basepool_dpo_dataset.jsonl
  hf_data/dpo/safecomp_v5_basepool_dpo_dataset.jsonl

Aggregate-only stdout. No prompt or response text echoed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    # --- Load v5 generations + matching prompt text ----------------------
    v5_raw = load_jsonl(Path("outputs/pool_pilot_base_llama_v5/raw.jsonl"))
    v5_prompts = {r["prompt_id"]: r["prompt"]
                  for r in load_jsonl(
                      Path("outputs/pool_pilot_base_llama_v5/sampled_prompts.jsonl"))}

    # --- Load curated dual_use chosens -----------------------------------
    dual_resp = load_jsonl(
        Path("hf_data/responses/dual_use/dualuse_response_records.jsonl"))
    chosen_by_pid: dict[str, dict[str, dict]] = {}
    for r in dual_resp:
        chosen_by_pid.setdefault(r["prompt_id"], {})[r["response_type"]] = r

    # --- Heldout eval IDs (must not appear in training) ------------------
    heldout = set()
    for r in load_jsonl(Path("hf_data/eval/held_out_prompts.jsonl")):
        if r.get("category") == "dual_use":
            heldout.add(r["prompt_id"])
    print(f"[init] heldout dual_use eval IDs: {len(heldout)} (excluded from all training data)")

    # --- Build new dual_use pairs from v5 --------------------------------
    new_baseline_du = []
    new_safecomp_du = []
    n_skipped_no_chosen = 0
    n_skipped_heldout = 0
    for v in v5_raw:
        pid = v["prompt_id"]
        if pid in heldout:
            n_skipped_heldout += 1
            continue
        chosens = chosen_by_pid.get(pid)
        if not chosens or "hard_refusal" not in chosens or "safe_completion" not in chosens:
            n_skipped_no_chosen += 1
            continue
        prompt_text = v5_prompts.get(pid)
        if not prompt_text:
            n_skipped_no_chosen += 1
            continue

        sample_idx = int(v.get("sample_index", 1))
        rejected_id = v["response_id"]
        # baseline regime: chosen=hard_refusal
        hr = chosens["hard_refusal"]
        new_baseline_du.append({
            "pair_id": f"{pid}__base_v5_basepool__s{sample_idx:02d}",
            "prompt_id": pid,
            "category": "dual_use",
            "pair_type": "baseline_refusal",
            "prompt": prompt_text,
            "chosen": hr["response"],
            "rejected": v["response"],
            "chosen_id": hr["response_id"],
            "rejected_id": rejected_id,
        })
        # safecomp regime, replacement pair: chosen=safe_completion
        sc = chosens["safe_completion"]
        new_safecomp_du.append({
            "pair_id": f"{pid}__safe_v5_basepool__s{sample_idx:02d}",
            "prompt_id": pid,
            "category": "dual_use",
            "pair_type": "safe_completion",
            "prompt": prompt_text,
            "chosen": sc["response"],
            "rejected": v["response"],
            "chosen_id": sc["response_id"],
            "rejected_id": rejected_id,
        })

    print(f"[v5] generations read:                {len(v5_raw)}")
    print(f"[v5] skipped (heldout):               {n_skipped_heldout}")
    print(f"[v5] skipped (no curated chosen):     {n_skipped_no_chosen}")
    print(f"[v5] new baseline dual_use pairs:     {len(new_baseline_du)}")
    print(f"[v5] new safecomp dual_use pairs:     {len(new_safecomp_du)}")

    # --- Reuse shared-category pairs from old datasets -------------------
    old_baseline = load_jsonl(Path("hf_data/dpo/baseline_dpo_dataset.jsonl"))
    old_safecomp = load_jsonl(Path("hf_data/dpo/safecomp_dpo_dataset.jsonl"))

    # baseline: keep all NON-dual_use; drop heldout from non-dual_use too if any
    base_shared = [r for r in old_baseline
                   if r["category"] != "dual_use"
                   and r["prompt_id"] not in heldout]
    print(f"[shared] baseline non-dual_use kept:  {len(base_shared)}  "
          f"({Counter(r['category'] for r in base_shared)})")

    # safecomp: keep all NON-dual_use, AND keep dual_use rows where the
    # rejected is hard_refusal (i.e. the safe>hard pair_type is preserved).
    # Drop dual_use rows where rejected is unsafe_compliance (replaced by v5).
    # Drop heldout in all cases.
    safe_shared = []
    safe_du_kept = []
    for r in old_safecomp:
        if r["prompt_id"] in heldout:
            continue
        if r["category"] != "dual_use":
            safe_shared.append(r)
            continue
        # dual_use: distinguish by rejected response type via id
        rid = r.get("rejected_id", "")
        if "hard_refusal" in rid:
            safe_du_kept.append(r)
        # rows where "unsafe_compliance" in rid get replaced by v5 -> drop here
    print(f"[shared] safecomp non-dual_use kept:  {len(safe_shared)}  "
          f"({Counter(r['category'] for r in safe_shared)})")
    print(f"[shared] safecomp dual_use safe>hard kept: {len(safe_du_kept)}")

    # --- Concatenate -----------------------------------------------------
    baseline_v5 = base_shared + new_baseline_du
    safecomp_v5 = safe_shared + safe_du_kept + new_safecomp_du

    print(f"\n[final] baseline_v5_basepool total:   {len(baseline_v5)}")
    print(f"        breakdown by category: "
          f"{Counter(r['category'] for r in baseline_v5)}")
    print(f"        breakdown by pair_type: "
          f"{Counter(r['pair_type'] for r in baseline_v5)}")
    print(f"[final] safecomp_v5_basepool total:   {len(safecomp_v5)}")
    print(f"        breakdown by category: "
          f"{Counter(r['category'] for r in safecomp_v5)}")
    print(f"        breakdown by pair_type: "
          f"{Counter(r['pair_type'] for r in safecomp_v5)}")

    # --- Eval-leak sanity check -----------------------------------------
    bl_pids = set(r["prompt_id"] for r in baseline_v5)
    sc_pids = set(r["prompt_id"] for r in safecomp_v5)
    bl_leak = bl_pids & heldout
    sc_leak = sc_pids & heldout
    print(f"\n[sanity] baseline ∩ heldout = {len(bl_leak)}  (must be 0)")
    print(f"[sanity] safecomp ∩ heldout = {len(sc_leak)}  (must be 0)")
    assert len(bl_leak) == 0 and len(sc_leak) == 0, "eval-leak detected!"

    # --- Write -----------------------------------------------------------
    out_b = Path("hf_data/dpo/baseline_v5_basepool_dpo_dataset.jsonl")
    out_s = Path("hf_data/dpo/safecomp_v5_basepool_dpo_dataset.jsonl")
    write_jsonl(out_b, baseline_v5)
    write_jsonl(out_s, safecomp_v5)
    print(f"\n[write] {out_b}")
    print(f"[write] {out_s}")


if __name__ == "__main__":
    main()
