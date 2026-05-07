"""
Build hf_data/dpo/safecomp_dualuse_x2_dpo_dataset.jsonl.

Same as the safecomp regime, but each dual_use pair is duplicated 2x.
Tests whether the safe_completion signal was being drowned by shared-category
pairs (unsafe / benign_sensitive / benign make up 5,373 of 10,377 = 51.8% of
the existing safecomp dataset; dual_use is 5,004 = 48.2%).

After 2x oversampling:
  dual_use (×2):     10,008 pairs  (66%)
  shared (no change): 5,373 pairs  (34%)
  total:             15,381 pairs

Source:
  hf_data/dpo/safecomp_dpo_dataset.jsonl  (the existing safecomp dataset)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SAFECOMP_DPO = Path("hf_data/dpo/safecomp_dpo_dataset.jsonl")
OUT_PATH = Path("hf_data/dpo/safecomp_dualuse_x2_dpo_dataset.jsonl")
DUPLICATE_FACTOR = 2


def main() -> None:
    n_in = 0
    n_dual_in = 0
    n_out = 0
    by_cat_pt = Counter()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_fh = OUT_PATH.open("w", encoding="utf-8")
    try:
        with SAFECOMP_DPO.open() as f:
            for line in f:
                r = json.loads(line)
                n_in += 1
                cat = r.get("category")
                pt = r.get("pair_type")
                if cat == "dual_use":
                    n_dual_in += 1
                    for k in range(DUPLICATE_FACTOR):
                        copy = dict(r)
                        copy["pair_id"] = f"{r['pair_id']}__dup{k}"
                        out_fh.write(json.dumps(copy, ensure_ascii=False) + "\n")
                        n_out += 1
                        by_cat_pt[(cat, pt)] += 1
                else:
                    out_fh.write(line if line.endswith("\n") else line + "\n")
                    n_out += 1
                    by_cat_pt[(cat, pt)] += 1
    finally:
        out_fh.close()

    print(f"Wrote {OUT_PATH}")
    print(f"  source rows:                  {n_in}")
    print(f"  source dual_use rows:         {n_dual_in}")
    print(f"  duplicate factor (dual_use):  {DUPLICATE_FACTOR}x")
    print(f"  output rows:                  {n_out}")
    print(f"  (category, pair_type):")
    for k in sorted(by_cat_pt):
        print(f"    {k}: {by_cat_pt[k]}")


if __name__ == "__main__":
    main()
