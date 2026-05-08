# Base-Llama unsafe_compliance pool feasibility — pilot v1 + v2

_Generated 2026-05-08T11:54:46.127686+00:00_

Aggregate counts only. No raw generations are included in this document, in commits, or in any console output during the pilot.

## Pilot scope (shared across v1 and v2)

- 100 dual_use training prompts.
- Source: `hf_data/prompts/dual_use/dualuse_prompts.jsonl` (2,502 prompts).
- Held-out exclusion: all 200 dual_use eval prompt_ids (`hf_data/eval/held_out_prompts.jsonl`) excluded.
- Stratification: equal across 12 (author, domain) combos (8 or 9 per combo).
- Same random seed in both versions, so v1 and v2 see the same 100 prompts.
- Model: `meta-llama/Llama-3.1-8B` (non-Instruct base) in both.
- Backend: HF transformers (vLLM not available in this env).

## Decoding settings (v1 vs v2)

| setting | v1 | v2 |
|---|---|---|
| temperature | 0.7 | 0.8 |
| top_p | 0.9 | 0.95 |
| max_new_tokens | 512 | 384 |
| repetition_penalty | — | 1.15 |
| no_repeat_ngram_size | — | 4 |
| trim_at | — | '\nQuestion:', '\n\nQuestion:' |
| stop | — | '\nQuestion:', '\n\nQuestion:' |

## QC aggregate (v1 vs v2)

| QC bucket | v1 count | v1 rate | v2 count | v2 rate |
|---|---:|---:|---:|---:|
| candidate | 59 | 59.0% | 98 | 98.0% |
| likely_refusal | 0 | 0.0% | 0 | 0.0% |
| irrelevant_or_garbled | 41 | 41.0% | 2 | 2.0% |
| empty_or_failed | 0 | 0.0% | 0 | 0.0% |

Per-flag counts (records can carry multiple flags):

| flag | v1 count | v1 rate | v2 count | v2 rate |
|---|---:|---:|---:|---:|
| empty_or_failed | 0 | 0.0% | 0 | 0.0% |
| very_short | 2 | 2.0% | 2 | 2.0% |
| likely_refusal | 0 | 0.0% | 0 | 0.0% |
| degenerate_repetition | 39 | 39.0% | 0 | 0.0% |

Token usage:

| metric | v1 | v2 |
|---|---:|---:|
| avg new_tokens | 295.8 | 238.5 |

## Per-domain QC rates

| domain | v1 n | v1 candidate | v1 degen | v2 n | v2 candidate | v2 degen |
|---|---:|---:|---:|---:|---:|---:|
| biorisks_biosafety | 9 | 33.3% | 66.7% | 9 | 100.0% | 0.0% |
| chemical_hazards | 8 | 75.0% | 25.0% | 8 | 100.0% | 0.0% |
| disinformation | 8 | 50.0% | 50.0% | 8 | 100.0% | 0.0% |
| financial_crime | 8 | 50.0% | 50.0% | 8 | 100.0% | 0.0% |
| network_intelligence | 9 | 55.6% | 44.4% | 9 | 100.0% | 0.0% |
| offensive_cybersecurity | 9 | 66.7% | 33.3% | 9 | 100.0% | 0.0% |
| pharmacology_medication_safety | 8 | 87.5% | 12.5% | 8 | 87.5% | 12.5% |
| physical_security | 8 | 62.5% | 37.5% | 8 | 100.0% | 0.0% |
| privacy_opsec | 9 | 77.8% | 22.2% | 9 | 88.9% | 11.1% |
| social_engineering | 8 | 62.5% | 37.5% | 8 | 100.0% | 0.0% |
| surveillance_countersurveillance | 8 | 62.5% | 37.5% | 8 | 100.0% | 0.0% |
| weapons_ballistics_compliance | 8 | 25.0% | 75.0% | 8 | 100.0% | 0.0% |

## Decision rule (pre-declared)

- **no_go**     — v2 `degenerate_repetition` rate > 25%
- **judge_v2**  — v2 degenerate <= 25% AND v2 candidate rate >= 70%
- **in_between** — neither (consider one v3 only if there is an obvious lever; otherwise stop)

### Outcome: `judge_v2`

v2 degenerate = 0.0% (<=25), candidate = 98.0% (>=70). Proceed to LLM-judge step on v2.

### Recommendation

- Send v2 raw + qc + sampled_prompts to a host with `OPENAI_API_KEY` set and run `scripts/run_pool_pilot_judge.py` (input: `outputs/pool_pilot_base_llama_v2/qc.jsonl`, prompts: `outputs/pool_pilot_base_llama_v2/sampled_prompts.jsonl`, output: `outputs/pool_pilot_base_llama_v2/judged.jsonl`).
- After judge runs, compare against the 60% / 30% unsafe_compliance bars from the v1 decision rule; only then consider a retraining ablation.

## Privacy

- Raw v1 + v2 generations stored only at `outputs/pool_pilot_base_llama{,_v2}/` (gitignored).
- This document and its summary JSON contain aggregate counts only.
