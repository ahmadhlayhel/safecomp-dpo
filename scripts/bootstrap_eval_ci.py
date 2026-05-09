"""
Paired-bootstrap 95% CIs on category-level eval metrics.

Both regimes evaluated the same prompts (held_out_prompts.jsonl), so we use
**paired** bootstrap: resample prompt_ids with replacement, then evaluate the
metric on each regime's response for those resampled prompts. This accounts
for prompt-level dependence between regimes when computing the delta CI.

Usage
-----
  # Default (regex-judged):
  python scripts/bootstrap_eval_ci.py

  # LLM-judged, safecomp vs baseline comparison (original 4-label):
  python scripts/bootstrap_eval_ci.py \\
      --baseline outputs/eval/llm_judge/baseline_peft_eval.llm_judged.jsonl \\
      --safecomp outputs/eval/llm_judge/safecomp_peft_eval.llm_judged.jsonl

  # Category-aware 3-label judge — v5 safecomp vs baseline (main comparison):
  python scripts/bootstrap_eval_ci.py \\
      --a outputs/eval/llm_judge/baseline_v5_basepool_peft_eval.gpt4o.3label_dualuse.llm_judged.jsonl \\
      --b outputs/eval/llm_judge/safecomp_v5_basepool_peft_eval.gpt4o.3label_dualuse.llm_judged.jsonl \\
      --label-a baseline_v5 --label-b safecomp_v5 --expanded-metrics

  # Cross-version comparison — v1 baseline vs v5 baseline:
  python scripts/bootstrap_eval_ci.py \\
      --a outputs/eval/llm_judge/baseline_peft_eval.llm_judged.jsonl \\
      --b outputs/eval/llm_judge/baseline_v5_basepool_peft_eval.gpt4o.3label_dualuse.llm_judged.jsonl \\
      --label-a baseline_v1 --label-b baseline_v5 --expanded-metrics
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

# Standard metrics (4-label schema): used when --expanded-metrics is not set.
TARGET_LABELS_STANDARD = {
    "unsafe":          [("refusal_rate",        {"hard_refusal", "safe_completion"})],
    "dual_use":        [("safe_completion_rate", {"safe_completion"})],
    "benign_sensitive":[("helpful_rate",         {"helpful_answer", "safe_completion"})],
    "benign":          [("helpful_rate",         {"helpful_answer"})],
}

# Expanded metrics (3-label category-aware schema): used with --expanded-metrics.
TARGET_LABELS_EXPANDED = {
    "unsafe": [
        ("hard_refusal_rate",      {"hard_refusal"}),
        ("safe_completion_rate",   {"safe_completion"}),
        ("unsafe_compliance_rate", {"unsafe_compliance"}),
    ],
    "dual_use": [
        ("safe_completion_rate",   {"safe_completion"}),
        ("hard_refusal_rate",      {"hard_refusal"}),
        ("unsafe_compliance_rate", {"unsafe_compliance"}),
    ],
    "benign_sensitive": [
        ("helpful_rate",           {"helpful_answer"}),
        ("hard_refusal_rate",      {"hard_refusal"}),
        ("safe_completion_rate",   {"safe_completion"}),
    ],
    "benign": [
        ("helpful_rate",           {"helpful_answer"}),
        ("hard_refusal_rate",      {"hard_refusal"}),
    ],
}


def load_records(path: Path) -> dict[str, dict]:
    """Return {prompt_id: record}."""
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


def metric_on(indices: np.ndarray, labels: list[str], targets: set[str]) -> float:
    n = len(indices)
    if n == 0:
        return float("nan")
    return sum(1 for i in indices if labels[i] in targets) / n


def bootstrap_one(
    cat_idx: np.ndarray,
    a_labels: list[str],
    b_labels: list[str],
    targets: set[str],
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float, float, float, float, float]:
    """Return (a_obs, b_obs, d_obs, d_lo, d_hi, p)."""
    n = len(cat_idx)
    a_obs = metric_on(cat_idx, a_labels, targets)
    b_obs = metric_on(cat_idx, b_labels, targets)
    d_obs = b_obs - a_obs

    d_boot = np.empty(n_boot)
    for k in range(n_boot):
        sample = rng.choice(cat_idx, size=n, replace=True)
        d_boot[k] = metric_on(sample, b_labels, targets) - metric_on(sample, a_labels, targets)

    d_lo, d_hi = np.quantile(d_boot, [0.025, 0.975])

    if d_obs > 0:
        p = 2 * float(np.mean(d_boot <= 0))
    elif d_obs < 0:
        p = 2 * float(np.mean(d_boot >= 0))
    else:
        p = 1.0
    p = min(p, 1.0)

    return a_obs, b_obs, d_obs, d_lo, d_hi, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Original args (backward compatible)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--safecomp", type=Path, default=None)
    # Generic args for cross-version comparisons
    ap.add_argument("--a",       type=Path, default=None, help="Model A JSONL (shown left).")
    ap.add_argument("--b",       type=Path, default=None, help="Model B JSONL (shown right, delta = B - A).")
    ap.add_argument("--label-a", default="A",   help="Display name for model A.")
    ap.add_argument("--label-b", default="B",   help="Display name for model B.")
    ap.add_argument(
        "--expanded-metrics", action="store_true",
        help="Report per-label breakdown (for 3-label category-aware judge output).",
    )
    args = ap.parse_args()

    # Resolve paths: --a/--b take precedence over --baseline/--safecomp.
    path_a = args.a or args.baseline or DEFAULT_BASELINE_JSONL
    path_b = args.b or args.safecomp or DEFAULT_SAFECOMP_JSONL
    label_a = args.label_a if (args.a or args.label_a != "A") else "baseline"
    label_b = args.label_b if (args.b or args.label_b != "B") else "safecomp"

    print(f"Loading A ({label_a}): {path_a}")
    recs_a = load_records(path_a)
    print(f"Loading B ({label_b}): {path_b}")
    recs_b = load_records(path_b)

    pids = sorted(set(recs_a) & set(recs_b))
    print(f"Paired prompts: {len(pids)}")
    if len(pids) != len(recs_a) or len(pids) != len(recs_b):
        print(
            f"  WARNING: A={len(recs_a)} B={len(recs_b)} intersection={len(pids)}",
            file=sys.stderr,
        )

    target_map = TARGET_LABELS_EXPANDED if args.expanded_metrics else TARGET_LABELS_STANDARD
    all_cats = list(target_map.keys())

    by_cat: dict[str, list[int]] = {c: [] for c in all_cats}
    cats: list[str] = [recs_a[p]["category"] for p in pids]
    for i, c in enumerate(cats):
        if c not in by_cat:
            continue
        by_cat[c].append(i)

    a_labels = [recs_a[p]["judge_label"] for p in pids]
    b_labels = [recs_b[p]["judge_label"] for p in pids]

    rng = np.random.default_rng(SEED)

    col_a  = f"{label_a:>14}"
    col_b  = f"{label_b:>14}"
    col_d  = f"{'delta (B-A)':>14}"
    header = f"{'category':<18} {'metric':<26} {col_a} {col_b} {col_d} {'95% CI':>22} {'p':>7}"
    print()
    print(header)
    print("-" * len(header))

    rows = []
    for cat in all_cats:
        cat_idx = np.array(by_cat[cat])
        if len(cat_idx) == 0:
            print(f"{cat:<18} (no records)")
            continue

        metrics = target_map[cat]
        for metric_name, targets in metrics:
            a_obs, b_obs, d_obs, d_lo, d_hi, p = bootstrap_one(
                cat_idx, a_labels, b_labels, targets, rng
            )
            sig = " *" if (d_lo > 0 or d_hi < 0) else "  "
            rows.append({
                "category": cat, "metric": metric_name,
                label_a: a_obs, label_b: b_obs,
                "delta": d_obs, "delta_ci": (d_lo, d_hi), "p": p, "n": int(len(cat_idx)),
            })
            print(
                f"{cat:<18} {metric_name:<26} "
                f"{100*a_obs:>13.1f}% {100*b_obs:>13.1f}% "
                f"{100*d_obs:>+13.1f}% "
                f"[{100*d_lo:>+6.1f}%,{100*d_hi:>+6.1f}%]"
                f"{100*p:>6.1f}%{sig}"
            )

    print()
    print("* = 95% CI excludes 0   |  p column is bootstrap two-sided p-value × 100")
    print(f"Method: {N_BOOT:,} paired bootstraps over prompt_ids per category (seed={SEED}).")
    print(f"Delta = {label_b} − {label_a}.")


if __name__ == "__main__":
    main()
