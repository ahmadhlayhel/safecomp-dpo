# Base-Llama unsafe_compliance pool feasibility — pilot v1

_Generated 2026-05-08T11:39:01.886933+00:00_

## Pilot scope

- Sample size: 100 dual_use prompts.
- Source pool: `hf_data/prompts/dual_use/dualuse_prompts.jsonl` (2,502 prompts).
- Held-out exclusion: all 200 dual_use eval prompt_ids (`hf_data/eval/held_out_prompts.jsonl`) excluded from the candidate set.
- Stratification: equal across 12 (author, domain) combos (8 or 9 per combo).
- Model: `meta-llama/Llama-3.1-8B` (non-Instruct base).
- Generation: greedy=False, temperature=0.7, top_p=0.9, max_new_tokens=512, n=1 per prompt.
- Backend: see `outputs/pool_pilot_base_llama/raw_report.json` for the actual backend used.

## Stage 1 — regex / heuristic QC

| QC bucket | count | rate |
|---|---:|---:|
| candidate | 59 | 59.0% |
| likely_refusal | 0 | 0.0% |
| irrelevant_or_garbled | 41 | 41.0% |
| empty_or_failed | 0 | 0.0% |

Per-flag counts (records can carry multiple flags):

| flag | count | rate |
|---|---:|---:|
| empty_or_failed | 0 | 0.0% |
| very_short | 2 | 2.0% |
| likely_refusal | 0 | 0.0% |
| degenerate_repetition | 39 | 39.0% |

### QC-stage interpretation (cautious)

- **Refusal hurdle cleared.** 0/100 generations match any refusal regex.
  The non-Instruct base model has no RLHF refusal priors — exactly the
  property we wanted relative to the Instruct-fallback config that the
  prior unsafe_compliance pool used.
- **Degeneracy is the dominant failure mode.** 39% of generations carry
  the `degenerate_repetition` flag (4-gram repeated >5 times), which is
  the well-known base-model continuation failure. This is *above* the
  pre-declared `>25 % degenerate/empty ⇒ no-go` threshold, before the
  LLM judge has even seen the data.
- **Best-case ceiling on usable yield is ~59%.** Only 59 records survive
  QC as "candidate"; even if the LLM judge labeled every candidate as
  `unsafe_compliance`, the overall `unsafe_compliance` rate would be
  ≈59 % — just under the `≥60 %` go bar. Realistically the judge will
  reclassify some candidates as `safe_completion` or `helpful_answer`,
  pushing the achievable rate lower.
- **Net read.** The QC stats already make a clean **go** unlikely. The
  most plausible outcomes are **ambiguous** (worth expanding to 200 to
  resolve) or **no-go** (degeneracy + low yield kill the approach for
  this prompt set with these decoding settings). Worth running the LLM
  judge to confirm.

## Stage 2 — LLM judge

Judge step has **not run yet** (no API key in pilot env).
Run `scripts/run_pool_pilot_judge.py` from a host with `OPENAI_API_KEY` set to fill in this section.

## Comparison to the current dual_use rejected pool

- Current pool: `gpt-4o-mini`-sourced (2,502 records).
- Audit observation: the dual_use slice was dominated by analytical/helpful content rather than genuine unsafe_compliance.
- The pilot's `unsafe_compliance` rate (Stage 2) is the directly comparable number.

## Decision

- Pre-declared rule: go ≥ 60% unsafe_compliance ∧ ≤ 10% hard_refusal ∧ ≤ 25% degenerate; no-go < 30% unsafe_compliance ∨ > 25% degenerate.
- **Outcome: `pending_judge`** — LLM judge has not been run yet; only QC-stage stats are available.


## Privacy

- Raw generations stored only at `outputs/pool_pilot_base_llama/` (gitignored).
- This document contains aggregate counts only; no raw text.
