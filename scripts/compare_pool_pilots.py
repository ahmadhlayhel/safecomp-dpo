"""
Compare pool-pilot QC stages (v1 vs v2) and rewrite the feasibility doc.

Inputs:
    --v1-qc  outputs/pool_pilot_base_llama/qc.jsonl
    --v2-qc  outputs/pool_pilot_base_llama_v2/qc.jsonl
    --doc    docs/base_llama_unsafe_pool_feasibility.md
    --v1-config configs/generation/unsafe_compliance_base_llama_pilot.yaml
    --v2-config configs/generation/unsafe_compliance_base_llama_pilot_v2.yaml

Output: aggregate-only markdown comparison. No raw text is included.

v2 decision rule (caller-declared):
    - no_go         : v2 degenerate_repetition rate > 25%
    - judge_v2      : v2 degenerate <= 25% AND v2 candidate rate >= 70%
    - in_between    : neither (suggest v3 only if there's an obvious lever)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def aggregate(qc_rows: list[dict]) -> dict:
    n_total = len(qc_rows)
    bucket = Counter()
    flag = Counter()
    by_domain_bucket: dict[str, Counter] = defaultdict(Counter)
    new_tokens: list[int] = []
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
        if "n_new_tokens" in md:
            new_tokens.append(int(md["n_new_tokens"]))
    avg_new = sum(new_tokens) / max(len(new_tokens), 1)
    return {
        "n_total": n_total,
        "bucket": dict(bucket),
        "flag": dict(flag),
        "by_domain_bucket": {d: dict(c) for d, c in by_domain_bucket.items()},
        "avg_new_tokens": round(avg_new, 1),
    }


def pct(n: int, d: int) -> float:
    return (n / d * 100.0) if d else 0.0


def decision_v2(v2_summary: dict) -> tuple[str, str]:
    n = v2_summary["n_total"]
    if n == 0:
        return ("v2_missing", "v2 QC file is empty.")
    degen = pct(v2_summary["flag"].get("degenerate_repetition", 0), n)
    cand = pct(v2_summary["bucket"].get("candidate", 0), n)
    if degen > 25.0:
        return ("no_go",
                f"v2 degenerate_repetition = {degen:.1f}% (>25%); base-Llama "
                f"unsafe-pool generation is not viable with these decoding "
                f"settings.")
    if degen <= 25.0 and cand >= 70.0:
        return ("judge_v2",
                f"v2 degenerate = {degen:.1f}% (<=25), candidate = {cand:.1f}% "
                f"(>=70). Proceed to LLM-judge step on v2.")
    return ("in_between",
            f"v2 degenerate = {degen:.1f}%, candidate = {cand:.1f}%. Mixed; "
            "consider one final v3 only if there is an obvious additional "
            "lever, otherwise stop.")


def cfg_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text()) or {}
    g = cfg.get("generation") or {}
    return {
        "model":               cfg.get("model", "?"),
        "temperature":         g.get("temperature"),
        "top_p":               g.get("top_p"),
        "max_new_tokens":      g.get("max_new_tokens"),
        "repetition_penalty":  g.get("repetition_penalty"),
        "no_repeat_ngram_size": g.get("no_repeat_ngram_size"),
        "trim_at":             g.get("trim_at"),
        "stop":                g.get("stop"),
    }


def render_setting(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, list):
        return ", ".join(repr(x) for x in v) if v else "—"
    return str(v)


def write_doc(out_path: Path, ctx: dict) -> None:
    v1 = ctx["v1"]
    v2 = ctx["v2"]
    cfg1 = ctx["cfg1"]
    cfg2 = ctx["cfg2"]
    decision = ctx["decision"]
    lines: list[str] = []
    lines.append("# Base-Llama unsafe_compliance pool feasibility — pilot v1 + v2\n")
    lines.append(f"_Generated {ctx['generated_at']}_\n")
    lines.append("Aggregate counts only. No raw generations are included in this "
                 "document, in commits, or in any console output during the pilot.\n")

    lines.append("## Pilot scope (shared across v1 and v2)\n")
    lines.append("- 100 dual_use training prompts.")
    lines.append("- Source: `hf_data/prompts/dual_use/dualuse_prompts.jsonl` (2,502 prompts).")
    lines.append("- Held-out exclusion: all 200 dual_use eval prompt_ids "
                 "(`hf_data/eval/held_out_prompts.jsonl`) excluded.")
    lines.append("- Stratification: equal across 12 (author, domain) combos "
                 "(8 or 9 per combo).")
    lines.append("- Same random seed in both versions, so v1 and v2 see the "
                 "same 100 prompts.")
    lines.append("- Model: `meta-llama/Llama-3.1-8B` (non-Instruct base) in both.")
    lines.append("- Backend: HF transformers (vLLM not available in this env).\n")

    lines.append("## Decoding settings (v1 vs v2)\n")
    lines.append("| setting | v1 | v2 |")
    lines.append("|---|---|---|")
    for k in ("temperature", "top_p", "max_new_tokens",
              "repetition_penalty", "no_repeat_ngram_size",
              "trim_at", "stop"):
        lines.append(f"| {k} | {render_setting(cfg1.get(k))} | "
                     f"{render_setting(cfg2.get(k))} |")
    lines.append("")

    lines.append("## QC aggregate (v1 vs v2)\n")
    lines.append("| QC bucket | v1 count | v1 rate | v2 count | v2 rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in ("candidate", "likely_refusal", "irrelevant_or_garbled", "empty_or_failed"):
        c1 = v1["bucket"].get(b, 0); c2 = v2["bucket"].get(b, 0)
        lines.append(f"| {b} | {c1} | {pct(c1, v1['n_total']):.1f}% "
                     f"| {c2} | {pct(c2, v2['n_total']):.1f}% |")
    lines.append("")

    lines.append("Per-flag counts (records can carry multiple flags):\n")
    lines.append("| flag | v1 count | v1 rate | v2 count | v2 rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in ("empty_or_failed", "very_short", "likely_refusal", "degenerate_repetition"):
        c1 = v1["flag"].get(k, 0); c2 = v2["flag"].get(k, 0)
        lines.append(f"| {k} | {c1} | {pct(c1, v1['n_total']):.1f}% "
                     f"| {c2} | {pct(c2, v2['n_total']):.1f}% |")
    lines.append("")

    lines.append("Token usage:\n")
    lines.append("| metric | v1 | v2 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| avg new_tokens | {v1['avg_new_tokens']} | {v2['avg_new_tokens']} |")
    lines.append("")

    lines.append("## Per-domain QC rates\n")
    domains = sorted(set(v1["by_domain_bucket"]) | set(v2["by_domain_bucket"]))
    lines.append("| domain | v1 n | v1 candidate | v1 degen | v2 n | v2 candidate | v2 degen |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in domains:
        b1 = v1["by_domain_bucket"].get(d, {})
        b2 = v2["by_domain_bucket"].get(d, {})
        n1 = sum(b1.values()); n2 = sum(b2.values())
        c1 = b1.get("candidate", 0); c2 = b2.get("candidate", 0)
        g1 = b1.get("irrelevant_or_garbled", 0); g2 = b2.get("irrelevant_or_garbled", 0)
        lines.append(
            f"| {d} | {n1} | {pct(c1, n1):.1f}% | {pct(g1, n1):.1f}% "
            f"| {n2} | {pct(c2, n2):.1f}% | {pct(g2, n2):.1f}% |"
        )
    lines.append("")

    lines.append("## Decision rule (pre-declared)\n")
    lines.append("- **no_go**     — v2 `degenerate_repetition` rate > 25%")
    lines.append("- **judge_v2**  — v2 degenerate <= 25% AND v2 candidate rate >= 70%")
    lines.append("- **in_between** — neither (consider one v3 only if there is "
                 "an obvious lever; otherwise stop)\n")

    lines.append(f"### Outcome: `{decision[0]}`\n")
    lines.append(f"{decision[1]}\n")

    if decision[0] == "no_go":
        lines.append("### Recommendation\n")
        lines.append("- **Do not** retrain DPO on a base-Llama-sourced pool with "
                     "the current prompt format.")
        lines.append("- **Do not** request LLM-judge credit spend on either v1 or v2.")
        lines.append("- The 0% refusal property is real but not enough on its own; "
                     "a usable pool requires a non-degenerate continuation regime, "
                     "which neither v1 nor v2 achieves.")
        lines.append("- If a future v3 is considered, the dominant lever to "
                     "explore is prompt format (e.g. multi-shot exemplars or a "
                     "minimal instruction-style stub) before any further decoding "
                     "tweaks.")
    elif decision[0] == "judge_v2":
        lines.append("### Recommendation\n")
        lines.append("- Send v2 raw + qc + sampled_prompts to a host with "
                     "`OPENAI_API_KEY` set and run "
                     "`scripts/run_pool_pilot_judge.py` "
                     "(input: `outputs/pool_pilot_base_llama_v2/qc.jsonl`, "
                     "prompts: `outputs/pool_pilot_base_llama_v2/sampled_prompts.jsonl`, "
                     "output: `outputs/pool_pilot_base_llama_v2/judged.jsonl`).")
        lines.append("- After judge runs, compare against the 60% / 30% "
                     "unsafe_compliance bars from the v1 decision rule; only "
                     "then consider a retraining ablation.")
    else:
        lines.append("### Recommendation\n")
        lines.append("- Pause and decide whether a single v3 is worth running. "
                     "If yes, the obvious next lever is prompt format (few-shot "
                     "exemplars), not further decoding tweaks.")

    lines.append("\n## Privacy\n")
    lines.append("- Raw v1 + v2 generations stored only at "
                 "`outputs/pool_pilot_base_llama{,_v2}/` (gitignored).")
    lines.append("- This document and its summary JSON contain aggregate counts "
                 "only.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1-qc", type=Path,
                    default=Path("outputs/pool_pilot_base_llama/qc.jsonl"))
    ap.add_argument("--v2-qc", type=Path,
                    default=Path("outputs/pool_pilot_base_llama_v2/qc.jsonl"))
    ap.add_argument("--v1-config", type=Path,
                    default=Path("configs/generation/unsafe_compliance_base_llama_pilot.yaml"))
    ap.add_argument("--v2-config", type=Path,
                    default=Path("configs/generation/unsafe_compliance_base_llama_pilot_v2.yaml"))
    ap.add_argument("--doc", type=Path,
                    default=Path("docs/base_llama_unsafe_pool_feasibility.md"))
    ap.add_argument("--summary-json", type=Path,
                    default=Path("outputs/pool_pilot_base_llama_v2/comparison_summary.json"))
    args = ap.parse_args()

    v1 = aggregate(load_jsonl(args.v1_qc))
    v2 = aggregate(load_jsonl(args.v2_qc))
    cfg1 = cfg_summary(args.v1_config)
    cfg2 = cfg_summary(args.v2_config)
    decision = decision_v2(v2)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "v1": v1,
        "v2": v2,
        "v1_config": cfg1,
        "v2_config": cfg2,
        "decision_outcome": decision[0],
        "decision_reason": decision[1],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))

    write_doc(args.doc, {
        "generated_at": summary["generated_at"],
        "v1": v1, "v2": v2, "cfg1": cfg1, "cfg2": cfg2,
        "decision": decision,
    })

    print(f"[compare] v1 n={v1['n_total']} cand={pct(v1['bucket'].get('candidate',0), v1['n_total']):.1f}%  "
          f"degen_flag={pct(v1['flag'].get('degenerate_repetition',0), v1['n_total']):.1f}%")
    print(f"[compare] v2 n={v2['n_total']} cand={pct(v2['bucket'].get('candidate',0), v2['n_total']):.1f}%  "
          f"degen_flag={pct(v2['flag'].get('degenerate_repetition',0), v2['n_total']):.1f}%")
    print(f"[compare] decision: {decision[0]} — {decision[1]}")
    print(f"[compare] summary: {args.summary_json}")
    print(f"[compare] doc:     {args.doc}")


if __name__ == "__main__":
    main()
