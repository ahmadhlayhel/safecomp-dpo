# Base-Llama unsafe_compliance pool feasibility — pilot v1 + v2

_Generated 2026-05-08T12:17:47.601568+00:00_

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

## Stage 2 — LLM judge on v2 (gpt-4o-mini, four-label)

- Records judged: 100
- Fallbacks: 0    invalid labels: 0

| label | count | rate |
|---|---:|---:|
| unsafe_compliance | 14 | 14.0% |
| safe_completion | 68 | 68.0% |
| helpful_answer | 17 | 17.0% |
| hard_refusal | 1 | 1.0% |

### Per-domain LLM-judge label rates (v2)

| domain | n | unsafe% | safe% | helpful% | hard% |
|---|---:|---:|---:|---:|---:|
| biorisks_biosafety | 9 | 0.0% | 88.9% | 11.1% | 0.0% |
| chemical_hazards | 8 | 12.5% | 87.5% | 0.0% | 0.0% |
| disinformation | 8 | 12.5% | 62.5% | 25.0% | 0.0% |
| financial_crime | 8 | 50.0% | 37.5% | 12.5% | 0.0% |
| network_intelligence | 9 | 22.2% | 44.4% | 33.3% | 0.0% |
| offensive_cybersecurity | 9 | 0.0% | 66.7% | 22.2% | 11.1% |
| pharmacology_medication_safety | 8 | 12.5% | 75.0% | 12.5% | 0.0% |
| physical_security | 8 | 0.0% | 87.5% | 12.5% | 0.0% |
| privacy_opsec | 9 | 11.1% | 66.7% | 22.2% | 0.0% |
| social_engineering | 8 | 0.0% | 62.5% | 37.5% | 0.0% |
| surveillance_countersurveillance | 8 | 12.5% | 87.5% | 0.0% | 0.0% |
| weapons_ballistics_compliance | 8 | 37.5% | 50.0% | 12.5% | 0.0% |

## QC-stage decision rule (pre-declared, v1->v2 gate)

- **no_go**     — v2 `degenerate_repetition` rate > 25%
- **judge_v2**  — v2 degenerate <= 25% AND v2 candidate rate >= 70%
- **in_between** — neither (consider one v3 only if there is an obvious lever; otherwise stop)

**QC-stage outcome: `judge_v2`** — v2 degenerate = 0.0% (<=25), candidate = 98.0% (>=70). Proceed to LLM-judge step on v2.

## Final decision rule (pre-declared, original v1 bars)

- **go**        — >=60% unsafe_compliance AND <=10% hard_refusal AND <=25% degenerate
- **no_go**     — <30% unsafe_compliance OR >25% degenerate
- **ambiguous** — anything in between

### Final outcome: `no_go`

v2 unsafe_compliance=14.0% (<30) — base-Llama pool is dominated by safety-framed and helpful content even without RLHF refusal training.

### Recommendation

- **Do not** retrain DPO on a base-Llama-sourced unsafe pool. The non-Instruct base model defaults to safety-framed content on dual_use prompts even without any RLHF training — pretraining itself carries enough safety bias to yield predominantly `safe_completion` rather than `unsafe_compliance`.
- **Do not** request additional LLM-judge credit on this approach (v1 was QC-gated out; v2 is judge-gated out).
- This closes the base-Llama line of attack on the dual_use rejected-pool problem. The next experimentally productive direction is independent of the base-model swap: e.g. a second-judge pass on the existing five-model eval set, or a structurally different rejected-pool source.

## Privacy

- Raw v1 + v2 generations stored only at `outputs/pool_pilot_base_llama{,_v2}/` (gitignored).
- This document and its summary JSON contain aggregate counts only.
