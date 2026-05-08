"""
Compare pool-pilot judges across (gpt-4o-mini, WildGuard) and across
(v3, v4) pilots. Aggregate-only; no raw text printed.

Inputs (all optional — script reports on whatever is present):
    --gpt-v3    outputs/pool_pilot_base_llama_v3/judged.jsonl
    --gpt-v4    outputs/pool_pilot_base_llama_v4/judged.jsonl
    --wg-v3     outputs/pool_pilot_base_llama_v3/judged_wildguard.jsonl
    --wg-v4     outputs/pool_pilot_base_llama_v4/judged_wildguard.jsonl
    --qc-v3     outputs/pool_pilot_base_llama_v3/qc.jsonl
    --qc-v4     outputs/pool_pilot_base_llama_v4/qc.jsonl

Outputs:
    --summary-json outputs/pool_pilot_base_llama_v4/judge_comparison_summary.json
    Stdout: per-judge label tables, agreement matrix, Cohen kappa,
            per-domain unsafe_compliance rates, boundary-disagreement
            breakdown.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CANONICAL = ("hard_refusal", "safe_completion", "unsafe_compliance")


def load_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_by_pid(rows: list[dict]) -> dict[str, dict]:
    return {r["prompt_id"]: r for r in rows}


def pct(n: int, d: int) -> float:
    return (n / d * 100.0) if d else 0.0


def label_distribution(rows: list[dict]) -> dict:
    n = len(rows)
    c = Counter(r.get("judge_label") for r in rows)
    return {
        "n": n,
        "counts": {k: c.get(k, 0) for k in CANONICAL},
        "rates": {k: pct(c.get(k, 0), n) for k in CANONICAL},
        "parse_fail": sum(1 for r in rows if r.get("judge_label") not in CANONICAL),
    }


def confusion_matrix(idx_a: dict[str, dict], idx_b: dict[str, dict]) -> dict:
    """A = rows, B = cols. Reports observed counts on overlapping prompt_ids."""
    common = sorted(set(idx_a) & set(idx_b))
    mat: dict[str, dict[str, int]] = {a: {b: 0 for b in CANONICAL} for a in CANONICAL}
    other = 0
    for pid in common:
        la = idx_a[pid].get("judge_label")
        lb = idx_b[pid].get("judge_label")
        if la in CANONICAL and lb in CANONICAL:
            mat[la][lb] += 1
        else:
            other += 1
    return {"n_common": len(common), "matrix": mat, "outside_canonical": other}


def cohen_kappa(idx_a: dict[str, dict], idx_b: dict[str, dict]) -> float | None:
    common = [pid for pid in set(idx_a) & set(idx_b)
              if idx_a[pid].get("judge_label") in CANONICAL
              and idx_b[pid].get("judge_label") in CANONICAL]
    n = len(common)
    if n == 0:
        return None
    a = [idx_a[pid]["judge_label"] for pid in common]
    b = [idx_b[pid]["judge_label"] for pid in common]
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    counts_a = Counter(a); counts_b = Counter(b)
    p_e = sum(counts_a[k] * counts_b[k] for k in CANONICAL) / (n * n)
    if abs(1 - p_e) < 1e-12:
        return 0.0
    return (p_o - p_e) / (1 - p_e)


def per_domain_unsafe_rate(rows: list[dict]) -> dict[str, dict]:
    by: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        d = r.get("du_domain") or (r.get("metadata") or {}).get("du_domain") or "?"
        by[d][r.get("judge_label")] += 1
    out = {}
    for d, c in by.items():
        n = sum(c.values())
        out[d] = {
            "n": n,
            "unsafe": pct(c.get("unsafe_compliance", 0), n),
            "safe":   pct(c.get("safe_completion", 0), n),
            "hard":   pct(c.get("hard_refusal", 0), n),
        }
    return out


def boundary_disagreement(idx_a: dict, idx_b: dict, name_a: str, name_b: str) -> dict:
    """Count cases where one judge says safe_completion and the other says
    unsafe_compliance — the DPO-relevant boundary."""
    common = sorted(set(idx_a) & set(idx_b))
    a_safe_b_unsafe = []
    a_unsafe_b_safe = []
    for pid in common:
        la = idx_a[pid].get("judge_label")
        lb = idx_b[pid].get("judge_label")
        if la == "safe_completion" and lb == "unsafe_compliance":
            a_safe_b_unsafe.append(pid)
        elif la == "unsafe_compliance" and lb == "safe_completion":
            a_unsafe_b_safe.append(pid)
    n = len(common)
    return {
        "n_common": n,
        f"{name_a}_safe__{name_b}_unsafe": len(a_safe_b_unsafe),
        f"{name_a}_unsafe__{name_b}_safe": len(a_unsafe_b_safe),
        "boundary_disagreement_total": len(a_safe_b_unsafe) + len(a_unsafe_b_safe),
        "boundary_disagreement_rate":
            pct(len(a_safe_b_unsafe) + len(a_unsafe_b_safe), n),
    }


def print_label_table(name: str, dist: dict) -> None:
    print(f"\n[{name}]  n={dist['n']}  parse_fail={dist['parse_fail']}")
    for k in CANONICAL:
        c = dist["counts"][k]; r = dist["rates"][k]
        print(f"  {c:4d}  {k:20s}  {r:5.1f}%")


def print_confusion(name_a: str, name_b: str, cm: dict) -> None:
    print(f"\n[{name_a} (rows) × {name_b} (cols)]  n_common={cm['n_common']}  "
          f"outside_canonical={cm['outside_canonical']}")
    header = " " * 22 + "  ".join(f"{c:>18s}" for c in CANONICAL)
    print(header)
    for a in CANONICAL:
        row = f"  {a:20s}  " + "  ".join(f"{cm['matrix'][a][b]:18d}" for b in CANONICAL)
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpt-v3", type=Path,
                    default=Path("outputs/pool_pilot_base_llama_v3/judged.jsonl"))
    ap.add_argument("--gpt-v4", type=Path,
                    default=Path("outputs/pool_pilot_base_llama_v4/judged.jsonl"))
    ap.add_argument("--wg-v3", type=Path,
                    default=Path("outputs/pool_pilot_base_llama_v3/judged_wildguard.jsonl"))
    ap.add_argument("--wg-v4", type=Path,
                    default=Path("outputs/pool_pilot_base_llama_v4/judged_wildguard.jsonl"))
    ap.add_argument("--summary-json", type=Path,
                    default=Path("outputs/pool_pilot_base_llama_v4/judge_comparison_summary.json"))
    args = ap.parse_args()

    files = {
        "gpt_v3": args.gpt_v3, "gpt_v4": args.gpt_v4,
        "wg_v3": args.wg_v3,   "wg_v4": args.wg_v4,
    }
    rows = {k: load_jsonl(v) for k, v in files.items()}
    idx  = {k: index_by_pid(v) for k, v in rows.items()}
    print("[avail] " + "  ".join(f"{k}={len(v)}" for k, v in rows.items()))

    summary: dict = {"distributions": {}, "agreements": {}, "per_domain": {},
                     "boundary": {}}

    for k, r in rows.items():
        if r:
            d = label_distribution(r)
            summary["distributions"][k] = d
            print_label_table(k, d)

    pairs = [
        ("gpt_v3", "wg_v3"),
        ("gpt_v4", "wg_v4"),
    ]
    for a, b in pairs:
        if rows[a] and rows[b]:
            cm = confusion_matrix(idx[a], idx[b])
            kappa = cohen_kappa(idx[a], idx[b])
            bd = boundary_disagreement(idx[a], idx[b], a, b)
            print_confusion(a, b, cm)
            print(f"  Cohen kappa = {kappa:.3f}" if kappa is not None
                  else "  Cohen kappa = (no overlap)")
            print(f"  boundary disagreement: "
                  f"{bd['boundary_disagreement_total']}/{bd['n_common']} "
                  f"({bd['boundary_disagreement_rate']:.1f}%)")
            summary["agreements"][f"{a}_vs_{b}"] = {
                "n_common": cm["n_common"],
                "matrix": cm["matrix"],
                "cohen_kappa": kappa,
                "boundary": bd,
            }
        else:
            print(f"\n[skip {a} vs {b}] missing data: "
                  f"{a}={len(rows[a])} {b}={len(rows[b])}")

    # v3 vs v4 under WildGuard (head-to-head under a single fixed schema).
    if rows["wg_v3"] and rows["wg_v4"]:
        d3 = label_distribution(rows["wg_v3"])
        d4 = label_distribution(rows["wg_v4"])
        print(f"\n[v3 vs v4 under WildGuard]  v3.n={d3['n']}  v4.n={d4['n']}")
        for k in CANONICAL:
            print(f"  {k:20s}  v3={d3['rates'][k]:5.1f}%  v4={d4['rates'][k]:5.1f}%  "
                  f"Δ={d4['rates'][k]-d3['rates'][k]:+5.1f}pp")
        summary["v3_vs_v4_wildguard"] = {"v3": d3["rates"], "v4": d4["rates"]}

    # Per-domain unsafe rate by judge.
    for key in ("gpt_v3", "gpt_v4", "wg_v3", "wg_v4"):
        if rows[key]:
            pd = per_domain_unsafe_rate(rows[key])
            summary["per_domain"][key] = pd
            print(f"\n[per-domain unsafe% under {key}]")
            for d, s in sorted(pd.items(), key=lambda kv: -kv[1]["unsafe"]):
                print(f"  {d:32s}  n={s['n']:4d}  unsafe={s['unsafe']:5.1f}%  "
                      f"safe={s['safe']:5.1f}%  hard={s['hard']:5.1f}%")

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] -> {args.summary_json}")


if __name__ == "__main__":
    main()
