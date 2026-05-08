# LLM-judge results summary — five-model comparison

_Generated 2026-05-08. Aggregate counts only; no raw responses or prompts are
included._

## Provenance

| File | Rows | Source |
|---|---:|---|
| `outputs/eval/llm_judge/base_model_eval.llm_judged.jsonl` | 800 | new (this pull) |
| `outputs/eval/llm_judge/baseline_peft_eval.llm_judged.jsonl` | 800 | prior pass |
| `outputs/eval/llm_judge/safecomp_peft_eval.llm_judged.jsonl` | 800 | prior pass |
| `outputs/eval/llm_judge/baseline_strict_eval.llm_judged.jsonl` | 800 | new (this pull) |
| `outputs/eval/llm_judge/safecomp_dualuse_x2_eval.llm_judged.jsonl` | 800 | new (this pull) |

Judge: `gpt-4o-mini`, four-label schema (`hard_refusal | safe_completion |
unsafe_compliance | helpful_answer`). All five files validated:
- 800 rows / 800 unique prompt_ids each
- 200 prompts per category × 4 categories per file
- 0 fallbacks, 0 invalid labels
- 800 prompt_ids common across all 5 models (full pairing)

## LLM-judge metrics (n=200 per category)

| metric | base | baseline | safecomp | baseline_strict | safecomp_dualuse_x2 |
|---|---:|---:|---:|---:|---:|
| unsafe.refusal_rate           | 45.5% | 70.0% | **76.0%** | 68.0% | **78.0%** |
| dual_use.safe_completion_rate | 54.5% | **67.5%** | 64.5% | **67.5%** | 65.0% |
| dual_use.hard_refusal_rate    | 1.0%  | 0.5%  | 0.5%  | 0.0% | 0.5% |
| dual_use.unsafe_compliance_rate | 9.5% | 7.0% | 7.0% | 7.0% | 7.0% |
| benign_sensitive.helpful_rate | 65.0% | 71.0% | 65.0% | **79.0%** | 58.0% |
| benign.helpful_rate           | 99.5% | 97.5% | 98.5% | 98.5% | 98.0% |

## Paired bootstrap CIs (B = 10,000, Δ = A − B, percentage points)

### base vs baseline
| metric | Δ | 95% CI | p |
|---|---:|---|---:|
| unsafe.refusal_rate | −24.5 | [−32.0, −17.5] | <0.001 * |
| dual_use.safe_completion | −13.0 | [−20.5, −5.5] | 0.001 * |
| dual_use.hard_refusal | +0.5 | [−1.0, +2.5] | 0.775 |
| dual_use.unsafe_compliance | +2.5 | [−2.0, +7.5] | 0.346 |
| benign_sensitive.helpful | −6.0 | [−13.0, +1.0] | 0.102 |
| benign.helpful | +2.0 | [+0.5, +4.0] | 0.033 * |

### base vs safecomp
| metric | Δ | 95% CI | p |
|---|---:|---|---:|
| unsafe.refusal_rate | −30.5 | [−37.5, −23.5] | <0.001 * |
| dual_use.safe_completion | −10.0 | [−18.0, −2.0] | 0.016 * |
| dual_use.hard_refusal | +0.5 | [−1.0, +2.5] | 0.775 |
| dual_use.unsafe_compliance | +2.5 | [−2.0, +7.5] | 0.343 |
| benign_sensitive.helpful | +0.0 | [−7.5, +7.5] | 1.000 |
| benign.helpful | +1.0 | [−1.0, +3.0] | 0.458 |

### baseline vs safecomp
| metric | Δ | 95% CI | p |
|---|---:|---|---:|
| unsafe.refusal_rate | −6.0 | [−11.0, −1.5] | 0.012 * |
| dual_use.safe_completion | +3.0 | [−4.0, +10.0] | 0.439 |
| dual_use.hard_refusal | +0.0 | [+0.0, +0.0] | 1.000 |
| dual_use.unsafe_compliance | +0.0 | [−4.5, +4.5] | 1.000 |
| benign_sensitive.helpful | +6.0 | [−0.5, +12.5] | 0.082 |
| benign.helpful | −1.0 | [−2.5, +0.0] | 0.276 |

### baseline_strict vs baseline
| metric | Δ | 95% CI | p |
|---|---:|---|---:|
| unsafe.refusal_rate | −2.0 | [−7.5, +3.5] | 0.528 |
| dual_use.safe_completion | +0.0 | [−7.5, +7.5] | 1.000 |
| dual_use.hard_refusal | −0.5 | [−1.5, +0.0] | 0.732 |
| dual_use.unsafe_compliance | +0.0 | [−4.5, +4.5] | 1.000 |
| **benign_sensitive.helpful** | **+8.0** | **[+1.5, +14.5]** | **0.015 *** |
| benign.helpful | +1.0 | [−1.0, +3.0] | 0.460 |

### safecomp_dualuse_x2 vs safecomp
| metric | Δ | 95% CI | p |
|---|---:|---|---:|
| unsafe.refusal_rate | +2.0 | [−1.0, +5.0] | 0.264 |
| dual_use.safe_completion | +0.5 | [−6.0, +6.5] | 0.927 |
| dual_use.hard_refusal | +0.0 | [+0.0, +0.0] | 1.000 |
| dual_use.unsafe_compliance | +0.0 | [−4.0, +4.0] | 1.000 |
| **benign_sensitive.helpful** | **−7.0** | **[−12.0, −2.0]** | **0.009 *** |
| benign.helpful | −0.5 | [−2.5, +1.0] | 0.765 |

### safecomp_dualuse_x2 vs baseline_strict
| metric | Δ | 95% CI | p |
|---|---:|---|---:|
| **unsafe.refusal_rate** | **+10.0** | **[+4.5, +16.0]** | **0.001 *** |
| dual_use.safe_completion | −2.5 | [−9.5, +4.5] | 0.522 |
| dual_use.hard_refusal | +0.5 | [+0.0, +1.5] | 0.719 |
| dual_use.unsafe_compliance | +0.0 | [−4.5, +4.0] | 1.000 |
| **benign_sensitive.helpful** | **−21.0** | **[−28.5, −13.5]** | **<0.001 *** |
| benign.helpful | −0.5 | [−3.0, +1.5] | 0.835 |

(* = 95% CI excludes 0.)

## How LLM-judge changes the regex story

Direct regex-vs-LLM divergences (regex numbers from the prior synthesis,
2026-05-07):

| Comparison | Metric | Regex Δ (p) | LLM Δ (p) | Interpretation change |
|---|---|---:|---:|---|
| base → baseline | unsafe.refusal | −6.0 (0.038) | **−24.5 (<0.001)** | DPO improvement on unsafe is ~4× larger than regex implied |
| base → safecomp | unsafe.refusal | −3.0 (n.s.) | **−30.5 (<0.001)** | Same — was hidden by regex |
| baseline_strict → baseline | dual_use.hard_refusal | **+16.0 (<0.001)** | +0.0 (1.000) | **Regex "repair" effect disappears** |
| baseline_strict → baseline | dual_use.unsafe_compliance | −7.5 (0.021) | +0.0 (1.000) | Same — regex artifact |
| safecomp_dualuse_x2 → baseline_strict | benign_sensitive.helpful | −7.0 (0.020) | **−21.0 (<0.001)** | Overrefusal cost is ~3× larger than regex implied |
| baseline → safecomp | dual_use.safe_completion | −2.5 (0.623) | +3.0 (0.439) | Both null — confirmed |

The pattern: **regex over-counts dual_use hard_refusal** and **under-counts
true unsafe-prompt non-refusal**, because keyword-based refusal detection
fires on safety-framed dual_use content while missing actual harmful answers
that lack refusal phrasing. LLM judge sees these correctly.

## Updated scientific status

### Verified at LLM-judge level
- **DPO works on unsafe prompts.** Baseline +24.5 pp, SafeComp +30.5 pp refusal_rate over base (both p<0.001).
- **DPO works on dual_use safe_completion.** Baseline +13.0 pp, SafeComp +10.0 pp over base (p≤0.016).
- **DPO has minimal cost on benign helpful_rate.** Worst case −2 pp; not material.

### Still null at LLM-judge level (the original differentiating hypothesis)
- **SafeComp does not beat Baseline on dual_use.safe_completion** under either judge:
  - Regex Δ = −2.5 pp (p=0.623). LLM Δ = +3.0 pp (p=0.439).
  - Both deltas have CIs spanning ~±10 pp, so we are not powered to detect
    a small positive effect, but we *can* rule out a large one.

### Repaired ablations did not repair the experiment
- **`baseline_strict`** (cleaner rejected pool for dual_use): the regex
  signal of "+16 pp hard_refusal, −7.5 pp unsafe_compliance" was a regex
  artifact. Under LLM-judge, all dual_use deltas vs baseline are CI-zero.
  The only LLM-confirmed effect is +8 pp `benign_sensitive.helpful`
  (p=0.015) — a side effect, not the targeted repair.
- **`safecomp_dualuse_x2`** (oversampled dual_use pairs): no LLM-judge
  effect on dual_use vs safecomp; large overrefusal cost on
  benign_sensitive (−7 pp vs safecomp, −21 pp vs baseline_strict).
- **Conclusion:** neither audit-driven ablation produced the
  hypothesized SafeComp–over–Baseline differentiation on dual_use.

### Notable side-finding
- `safecomp_dualuse_x2` does score +10 pp on `unsafe.refusal_rate` over
  `baseline_strict` (p=0.001) — but pays −21 pp on `benign_sensitive.helpful`.
  Net safety↔overrefusal trade is unfavorable for any "we made it safer"
  framing.

## Caveats (do not over-interpret)

- Single judge (`gpt-4o-mini`). A stricter judge (`gpt-4o`,
  Claude Opus/Sonnet) could change marginal calls, especially on the
  `safe_completion`↔`unsafe_compliance` boundary, where dual_use n=200
  and the unsafe_compliance rate is only ~7–10 % — wide CI territory.
- The `dual_use.hard_refusal ≈ 0 %` reading across all five regimes is
  striking; it could partly reflect this judge's tendency to label any
  substantive engagement as `safe_completion`. Worth a second-judge
  sanity check before any structural conclusion is drawn.
- All numbers are on a single 800-prompt held-out set; n=200 per category
  bounds resolution.
- The held-out dual_use prompts overlap with the training pool by
  prompt_id (see `docs/base_llama_unsafe_pool_feasibility.md` once the
  pilot completes). This was the eval set used at training time and
  inherits that limitation.

## Remaining tasks

1. **Optional second-opinion judge pass** (gpt-4o or Claude) on the same five
   files, restricted to `dual_use` records, to check whether the
   SafeComp-vs-Baseline null is judge-specific.
2. **Base-Llama unsafe-pool feasibility pilot** (job 7832566 was running
   at the time of this pull) — independent of this LLM-judge analysis.
   Its result determines whether a cleaner unsafe_compliance pool is even
   plausible; only then would another retraining pass be on the table.
3. **No further retraining is recommended yet.** The two audit-driven
   ablations did not move the LLM-judged needle on dual_use.

## Final headline

The cleaner LLM judge sharpens — but does not change — the central
finding: SafeComp-DPO does not produce a measurably better dual_use
safe-completion behavior than Baseline-DPO, on this prompt set, with
this judge. Both DPO regimes do work vs the base model, especially on
unsafe prompts where the effect is much stronger than regex suggested.
The two audit-driven ablations are now best read as null repairs at
the LLM-judge level, not as confirmations of the original hypothesis.
