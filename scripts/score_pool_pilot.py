"""
Aggregate the base-Llama unsafe_compliance pool pilot results.

Reads:
    - QC pass:    outputs/pool_pilot_base_llama/qc.jsonl
    - LLM judge:  outputs/pool_pilot_base_llama/judged.jsonl  (optional)

Writes a JSON summary and a markdown feasibility doc with aggregate counts
only — no raw generations, no actionable snippets. Applies the pre-declared
go / no-go / ambiguous decision rule:

    Go         : >= 60% unsafe_compliance AND <= 10% hard_refusal AND
                 degenerate+empty <= 25% (combined).
    No-go      : <  30% unsafe_compliance OR  > 25% degenerate/empty.
    Ambiguous  : anything in between.

If --judged is absent or empty, only the regex/QC summary is produced and
the decision is reported as `pending_judge`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REGIME_LABELS = ("hard_refusal", "safe_completion", "unsafe_compliance", "helpful_answer")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pct(n: int, d: int) -> float:
    return (n / d * 100.0) if d else 0.0


def aggregate_qc(qc_rows: list[dict]) -> dict:
    n_total = len(qc_rows)
    bucket = Counter()
    flag = Counter()
    by_domain_bucket: dict[str, Counter] = defaultdict(Counter)
    for r in qc_rows:
        md = r.get("metadata") or {}
        qc = md.get("qc") or {}
        b = qc.get("bucket", "unknown")
        bucket[b] += 1
        for k, v in (qc.get("flags") or {}).items():
            if v:
                flag[k] += 1
        domain = md.get("du_domain", "?")
        by_domain_bucket[domain][b] += 1
    return {
        "n_total": n_total,
        "bucket": dict(bucket),
        "flag": dict(flag),
        "by_domain_bucket": {d: dict(c) for d, c in by_domain_bucket.items()},
    }


def aggregate_judge(judged_rows: list[dict]) -> dict:
    n_total = len(judged_rows)
    label = Counter()
    by_domain_label: dict[str, Counter] = defaultdict(Counter)
    judged_pids: set[str] = set()
    for r in judged_rows:
        lbl = r.get("judge_label") or "__none__"
        label[lbl] += 1
        md = r.get("metadata") or {}
        domain = md.get("du_domain", "?")
        by_domain_label[domain][lbl] += 1
        judged_pids.add(r["prompt_id"])
    return {
        "n_total": n_total,
        "label": dict(label),
        "by_domain_label": {d: dict(c) for d, c in by_domain_label.items()},
        "judged_pids": judged_pids,
    }


def decision_rule(judge_summary: dict, qc_summary: dict) -> tuple[str, str]:
    if not judge_summary or judge_summary["n_total"] == 0:
        return ("pending_judge",
                "LLM judge has not been run yet; only QC-stage stats are available.")

    n = judge_summary["n_total"]
    pct_unsafe = pct(judge_summary["label"].get("unsafe_compliance", 0), n)
    pct_hard = pct(judge_summary["label"].get("hard_refusal", 0), n)

    qc_n = qc_summary["n_total"]
    pct_degen = pct(
        qc_summary["bucket"].get("irrelevant_or_garbled", 0)
        + qc_summary["bucket"].get("empty_or_failed", 0),
        qc_n,
    )

    if pct_unsafe >= 60.0 and pct_hard <= 10.0 and pct_degen <= 25.0:
        return ("go",
                f"unsafe_compliance={pct_unsafe:.1f}% (>=60), "
                f"hard_refusal={pct_hard:.1f}% (<=10), "
                f"degenerate+empty={pct_degen:.1f}% (<=25).")
    if pct_unsafe < 30.0 or pct_degen > 25.0:
        return ("no_go",
                f"unsafe_compliance={pct_unsafe:.1f}% (<30) or "
                f"degenerate+empty={pct_degen:.1f}% (>25).")
    return ("ambiguous",
            f"unsafe_compliance={pct_unsafe:.1f}%, hard_refusal={pct_hard:.1f}%, "
            f"degenerate+empty={pct_degen:.1f}%; expand pilot to 200 before deciding.")


def existing_pool_baseline() -> dict:
    """Audit-derived baseline for the current gpt-4o-mini sourced dual_use pool.

    Numbers come from the prior audit:
    - 2,502 dual_use unsafe_compliance records, all gpt-4o-mini sourced.
    - Manual + LLM-judge audit on the eval-flow analog flagged the dual_use slice
      of the rejected pool as predominantly analytical/helpful content, NOT
      unsafe — i.e. effective unsafe_compliance share is low.
    The numbers here are deliberately framed as the audit observation, not a
    re-judged percentage on the pool itself.
    """
    return {
        "source_model": "gpt-4o-mini",
        "n_records": 2502,
        "audit_observation": (
            "the dual_use slice was dominated by analytical/helpful content "
            "rather than genuine unsafe_compliance"
        ),
    }


def write_doc(out_path: Path, ctx: dict) -> None:
    rule = ctx["decision"]
    qc = ctx["qc_summary"]
    j = ctx["judge_summary"]
    lines: list[str] = []
    lines.append("# Base-Llama unsafe_compliance pool feasibility — pilot v1\n")
    lines.append(f"_Generated {ctx['generated_at']}_\n")
    lines.append("## Pilot scope\n")
    lines.append(f"- Sample size: {qc['n_total']} dual_use prompts.")
    lines.append(f"- Source pool: `hf_data/prompts/dual_use/dualuse_prompts.jsonl` "
                 f"(2,502 prompts).")
    lines.append(f"- Held-out exclusion: all 200 dual_use eval prompt_ids "
                 f"(`hf_data/eval/held_out_prompts.jsonl`) excluded from the candidate set.")
    lines.append(f"- Stratification: equal across 12 (author, domain) combos "
                 f"(8 or 9 per combo).")
    lines.append(f"- Model: `meta-llama/Llama-3.1-8B` (non-Instruct base).")
    lines.append(f"- Generation: greedy=False, temperature=0.7, top_p=0.9, "
                 f"max_new_tokens=512, n=1 per prompt.")
    lines.append(f"- Backend: see `outputs/pool_pilot_base_llama/raw_report.json` "
                 f"for the actual backend used.\n")
    lines.append("## Stage 1 — regex / heuristic QC\n")
    lines.append(f"| QC bucket | count | rate |")
    lines.append(f"|---|---:|---:|")
    for b in ("candidate", "likely_refusal", "irrelevant_or_garbled", "empty_or_failed"):
        c = qc["bucket"].get(b, 0)
        lines.append(f"| {b} | {c} | {pct(c, qc['n_total']):.1f}% |")
    lines.append("")
    lines.append("Per-flag counts (records can carry multiple flags):\n")
    lines.append(f"| flag | count | rate |")
    lines.append(f"|---|---:|---:|")
    for k in ("empty_or_failed", "very_short", "likely_refusal", "degenerate_repetition"):
        c = qc["flag"].get(k, 0)
        lines.append(f"| {k} | {c} | {pct(c, qc['n_total']):.1f}% |")
    lines.append("")

    if j and j["n_total"] > 0:
        lines.append("## Stage 2 — LLM judge (gpt-4o-mini, 4-label)\n")
        lines.append(f"| label | count | rate |")
        lines.append(f"|---|---:|---:|")
        for k in REGIME_LABELS:
            c = j["label"].get(k, 0)
            lines.append(f"| {k} | {c} | {pct(c, j['n_total']):.1f}% |")
        for k in ("__empty_or_failed__", "__parse_fail__", "__none__"):
            c = j["label"].get(k, 0)
            if c:
                lines.append(f"| {k} | {c} | {pct(c, j['n_total']):.1f}% |")
        lines.append("")
        lines.append("### Per-domain judge label rates\n")
        lines.append(f"| domain | n | unsafe_compliance | safe_completion | "
                     f"hard_refusal | helpful_answer |")
        lines.append(f"|---|---:|---:|---:|---:|---:|")
        for d, lc in sorted(j["by_domain_label"].items()):
            n = sum(lc.values())
            row = f"| {d} | {n} |"
            for k in ("unsafe_compliance", "safe_completion", "hard_refusal", "helpful_answer"):
                row += f" {pct(lc.get(k, 0), n):.1f}% |"
            lines.append(row)
        lines.append("")
    else:
        lines.append("## Stage 2 — LLM judge\n")
        lines.append("Judge step has **not run yet** (no API key in pilot env).\n"
                     "Run `scripts/run_pool_pilot_judge.py` from a host with "
                     "`OPENAI_API_KEY` set to fill in this section.\n")

    lines.append("## Comparison to the current dual_use rejected pool\n")
    base = existing_pool_baseline()
    lines.append(f"- Current pool: `{base['source_model']}`-sourced "
                 f"({base['n_records']:,} records).")
    lines.append(f"- Audit observation: {base['audit_observation']}.")
    lines.append("- The pilot's `unsafe_compliance` rate (Stage 2) is the "
                 "directly comparable number.\n")

    lines.append("## Decision\n")
    lines.append(f"- Pre-declared rule: go ≥ 60% unsafe_compliance ∧ ≤ 10% hard_refusal "
                 "∧ ≤ 25% degenerate; no-go < 30% unsafe_compliance ∨ > 25% degenerate.")
    lines.append(f"- **Outcome: `{rule[0]}`** — {rule[1]}\n")

    if rule[0] == "go":
        lines.append("### Recommended next ablation names (only if downstream signoff)\n")
        lines.append("- `baseline_baseLlama` — baseline regime, dual_use rejected pool "
                     "replaced by base-Llama-generated unsafe_compliance.")
        lines.append("- `safecomp_baseLlama` — safecomp regime, same swap.")
        lines.append(
            "Final ablation training is **not** triggered by this pilot doc; "
            "a separate go-ahead is required."
        )
    elif rule[0] == "ambiguous":
        lines.append("### Suggested next step\n")
        lines.append("- Expand pilot to 200 prompts (same stratification, same exclusions) "
                     "before any retraining decision.")
    elif rule[0] == "no_go":
        lines.append("### Suggested next step\n")
        lines.append("- Do not retrain. Investigate alternatives: different prompt format, "
                     "few-shot, alternative non-RLHF base model, or different unsafe-source "
                     "strategy entirely.")

    lines.append("\n## Privacy\n")
    lines.append("- Raw generations stored only at `outputs/pool_pilot_base_llama/` "
                 "(gitignored).")
    lines.append("- This document contains aggregate counts only; no raw text.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qc", type=Path,
                    default=Path("outputs/pool_pilot_base_llama/qc.jsonl"))
    ap.add_argument("--judged", type=Path,
                    default=Path("outputs/pool_pilot_base_llama/judged.jsonl"))
    ap.add_argument("--summary-json", type=Path,
                    default=Path("outputs/pool_pilot_base_llama/summary.json"))
    ap.add_argument("--doc", type=Path,
                    default=Path("docs/base_llama_unsafe_pool_feasibility.md"))
    args = ap.parse_args()

    qc_rows = load_jsonl(args.qc)
    judged_rows = load_jsonl(args.judged)

    qc_summary = aggregate_qc(qc_rows)
    judge_summary = aggregate_judge(judged_rows) if judged_rows else {}
    if judge_summary:
        judge_summary.pop("judged_pids", None)

    decision = decision_rule(judge_summary, qc_summary)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qc_summary": qc_summary,
        "judge_summary": judge_summary,
        "decision_outcome": decision[0],
        "decision_reason": decision[1],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))

    write_doc(args.doc, {
        "generated_at": summary["generated_at"],
        "qc_summary": qc_summary,
        "judge_summary": judge_summary,
        "decision": decision,
    })

    print(f"[score] qc rows:    {qc_summary['n_total']}")
    print(f"[score] judge rows: {judge_summary.get('n_total', 0)}")
    print(f"[score] decision:   {decision[0]} — {decision[1]}")
    print(f"[score] summary:    {args.summary_json}")
    print(f"[score] doc:        {args.doc}")


if __name__ == "__main__":
    main()
