"""
Stratified sampler for the base-Llama unsafe_compliance pool feasibility pilot.

Samples N prompts from hf_data/prompts/dual_use/dualuse_prompts.jsonl, balanced
across (du_author, du_domain) combos, with all held-out eval prompt_ids excluded.

Reproducible via --seed. Output is a PromptRecord-shaped JSONL that the pilot
generation runner consumes.

Aggregate-only console output: counts per combo, total written. Never prints
prompt text.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def held_out_dual_use_ids(held_out_path: Path) -> set[str]:
    ids: set[str] = set()
    for r in load_jsonl(held_out_path):
        if r.get("category") == "dual_use":
            ids.add(r["prompt_id"])
    return ids


def stratified_sample(
    prompts: list[dict],
    n_total: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)

    by_combo: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in prompts:
        md = r.get("metadata") or {}
        key = (md.get("du_author", "?"), md.get("du_domain", "?"))
        by_combo[key].append(r)

    combos = sorted(by_combo)
    base = n_total // len(combos)
    extra = n_total - base * len(combos)
    # Deterministic which combos get the +1: first `extra` in sorted order
    quotas = {c: base + (1 if i < extra else 0) for i, c in enumerate(combos)}

    sampled: list[dict] = []
    for combo, q in quotas.items():
        pool = by_combo[combo]
        if q > len(pool):
            raise ValueError(
                f"Combo {combo} has only {len(pool)} eligible prompts, asked for {q}"
            )
        rng.shuffle(pool)
        sampled.extend(pool[:q])

    return sampled


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path,
                   default=Path("hf_data/prompts/dual_use/dualuse_prompts.jsonl"))
    p.add_argument("--held-out", type=Path,
                   default=Path("hf_data/eval/held_out_prompts.jsonl"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/pool_pilot_base_llama/sampled_prompts.jsonl"))
    p.add_argument("-n", "--n-total", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260508)
    args = p.parse_args()

    all_prompts = load_jsonl(args.prompts)
    held_ids = held_out_dual_use_ids(args.held_out)

    eligible = [r for r in all_prompts if r["prompt_id"] not in held_ids]
    n_excluded = len(all_prompts) - len(eligible)

    print(f"[sample] dual_use canonical prompts: {len(all_prompts)}")
    print(f"[sample] held-out dual_use eval ids: {len(held_ids)}")
    print(f"[sample] excluded held-out ids:      {n_excluded}")
    print(f"[sample] eligible candidates:        {len(eligible)}")

    sampled = stratified_sample(eligible, args.n_total, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for r in sampled:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_combo: dict[tuple[str, str], int] = defaultdict(int)
    for r in sampled:
        md = r.get("metadata") or {}
        by_combo[(md.get("du_author", "?"), md.get("du_domain", "?"))] += 1

    print(f"[sample] wrote {len(sampled)} prompts -> {args.output}")
    print("[sample] per-combo counts:")
    for combo, c in sorted(by_combo.items()):
        print(f"  {c:3d}  author={combo[0]:1s}  domain={combo[1]}")


if __name__ == "__main__":
    main()
