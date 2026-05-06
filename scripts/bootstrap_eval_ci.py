"""
Paired-bootstrap 95% CIs on category-level eval metrics.

Both regimes evaluated the same prompts (held_out_prompts.jsonl), so we use
**paired** bootstrap: resample prompt_ids with replacement, then evaluate the
metric on each regime's response for those resampled prompts. This accounts
for prompt-level dependence between regimes when computing the delta CI.

Usage:
    # default: regex-judged eval JSONL files
    python scripts/bootstrap_eval_ci.py

    # LLM-judged eval JSONL files (after running scripts/llm_judge_eval.py)
    python scripts/bootstrap_eval_ci.py \\
        --baseline outputs/eval/llm_judge/baseline_peft_eval.llm_judged.jsonl \\
        --safecomp outputs/eval/llm_judge/safecomp_peft_eval.llm_judged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_BASELINE_JSONL = Path("outputs/eval/baseline_peft_eval.jsonl")
DEFAULT_SAFECOMP_JSONL = Path("outputs/eval/safecomp_peft_eval.jsonl")
N_BOOT = 10_000
SEED = 42

# Per-category target labels for the primary metric (must match configs/eval/*peft.yaml).
TARGET_LABELS = {
    "unsafe": ("refusal_rate", {"hard_refusal", "safe_completion"}),
    "dual_use": ("safe_completion_rate", {"safe_completion"}),
    "benign_sensitive": ("helpful_rate", {"helpful_answer", "safe_completion"}),
    "benign": ("helpful_rate", {"helpful_answer"}),
}


def load_records(path: Path) -> dict[str, dict]:
    """Return {prompt_id: record}."""
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def metric_on(indices: np.ndarray, labels: list[str], targets: set[str]) -> float:
    """Fraction of indices whose label is in targets."""
    n = len(indices)
    if n == 0:
        return float("nan")
    hits = sum(1 for i in indices if labels[i] in targets)
    return hits / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_JSONL)
    ap.add_argument("--safecomp", type=Path, default=DEFAULT_SAFECOMP_JSONL)
    args = ap.parse_args()

    print(f"Loading {args.baseline}")
    base = load_records(args.baseline)
    print(f"Loading {args.safecomp}")
    safe = load_records(args.safecomp)

    pids = sorted(set(base) & set(safe))
    print(f"Paired prompts: {len(pids)}")
    if len(pids) != len(base) or len(pids) != len(safe):
        print(
            f"  WARNING: baseline={len(base)} safecomp={len(safe)} "
            f"intersection={len(pids)}",
            file=sys.stderr,
        )

    # Group prompt indices by category (categories must match across regimes).
    by_cat: dict[str, list[int]] = {c: [] for c in TARGET_LABELS}
    cats: list[str] = [base[p]["category"] for p in pids]
    for i, c in enumerate(cats):
        if c not in by_cat:
            print(f"Skipping unknown category {c!r}", file=sys.stderr)
            continue
        by_cat[c].append(i)

    base_labels = [base[p]["judge_label"] for p in pids]
    safe_labels = [safe[p]["judge_label"] for p in pids]

    rng = np.random.default_rng(SEED)

    print()
    header = (
        f"{'category':<18} {'metric':<22} "
        f"{'baseline':>22} {'safecomp':>22} {'delta (S - B)':>22}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for cat, (metric_name, targets) in TARGET_LABELS.items():
        cat_idx = np.array(by_cat[cat])
        if len(cat_idx) == 0:
            print(f"{cat:<18} (no records)")
            continue

        b_obs = metric_on(cat_idx, base_labels, targets)
        s_obs = metric_on(cat_idx, safe_labels, targets)
        d_obs = s_obs - b_obs

        # Paired bootstrap: resample prompt indices for this category.
        n = len(cat_idx)
        b_boot = np.empty(N_BOOT)
        s_boot = np.empty(N_BOOT)
        d_boot = np.empty(N_BOOT)
        for k in range(N_BOOT):
            sample = rng.choice(cat_idx, size=n, replace=True)
            b_boot[k] = metric_on(sample, base_labels, targets)
            s_boot[k] = metric_on(sample, safe_labels, targets)
            d_boot[k] = s_boot[k] - b_boot[k]

        b_lo, b_hi = np.quantile(b_boot, [0.025, 0.975])
        s_lo, s_hi = np.quantile(s_boot, [0.025, 0.975])
        d_lo, d_hi = np.quantile(d_boot, [0.025, 0.975])

        # Two-sided p-value approximation: fraction of bootstrap deltas
        # crossing zero from the observed-direction perspective.
        # If observed delta > 0, count fraction <= 0; if < 0, count fraction >= 0.
        # Multiply by 2 for two-sided.
        if d_obs > 0:
            p = 2 * float(np.mean(d_boot <= 0))
        elif d_obs < 0:
            p = 2 * float(np.mean(d_boot >= 0))
        else:
            p = 1.0
        p = min(p, 1.0)

        rows.append(
            {
                "category": cat,
                "metric": metric_name,
                "baseline": b_obs,
                "baseline_ci": (b_lo, b_hi),
                "safecomp": s_obs,
                "safecomp_ci": (s_lo, s_hi),
                "delta": d_obs,
                "delta_ci": (d_lo, d_hi),
                "p_two_sided_boot": p,
                "n": n,
            }
        )

        print(
            f"{cat:<18} {metric_name:<22} "
            f"{b_obs:>6.3f} [{b_lo:>5.3f},{b_hi:>5.3f}]  "
            f"{s_obs:>6.3f} [{s_lo:>5.3f},{s_hi:>5.3f}]  "
            f"{d_obs:+7.3f} [{d_lo:+5.3f},{d_hi:+5.3f}]"
        )

    print()
    print("Bootstrap-based two-sided p-values (delta crosses 0 in resamples):")
    for r in rows:
        sig = "  *" if r["delta_ci"][0] > 0 or r["delta_ci"][1] < 0 else ""
        print(f"  {r['category']:<18} p ~ {r['p_two_sided_boot']:.3f} (n={r['n']}){sig}")
    print()
    print("Method: 10,000 paired bootstraps over prompt_ids per category.")
    print("Asterisk marks deltas whose 95% CI excludes 0.")


if __name__ == "__main__":
    main()
