# SafeComp-DPO Repo Instructions

## Purpose
This repo implements an end-to-end research pipeline for comparing:

- Baseline Refusal-DPO
- Experimental SafeComp-DPO

on Llama 3.1 8B-Instruct.

The goal is to build the full system so it runs end to end on dummy data now, and on real data later with minimal changes.

## Current repo state
The repo has a full mock pipeline through all stages, including evaluation, plus partially real backend/data layers in some places.

Implemented stages:
1. prompt acquisition / ingestion
2. prompt selection
3. acceptable-response generation
4. rejected-response sourcing
5. response validation
6. pair construction
7. DPO dataset assembly
8. regime-specific training-dataset construction
9. training
10. custom model evaluation
11. standard benchmark evaluation

All stages run end-to-end on mock data. 1071 tests, all passing.

Standard benchmarks implemented and real datasets wired locally:
- HarmBench (400 prompts, ASR)
- StrongREJECT (313 prompts, mean score)
- Do-Not-Answer (939 prompts, percent harmful)
- XSTest (250 safe / 200 unsafe, ORR + ASR)
- OR-Bench (1319 prompts, ORR)
- FalseReject-Test (1187 prompts, compliance rate)

Real datasets live in `hf_data/benchmarks/` (gitignored). Mock fallback is automatic if a dataset file is absent.

Current missing / not-yet-runtime-validated real backends:
- real generation backend (vllm on BABEL)
- real benchmark model backend (vllm on BABEL)
- real benchmark scoring backend (WildGuard / Llama Guard / StrongREJECT classifier)

Training:
- the TRL / QLoRA training backend is implemented
- it still needs real BABEL/runtime validation

## Real data status (as of 2026-05-02)

The full pipeline now runs end-to-end on real data. All response artifacts are present.

| Artifact | File | Count | Status |
|---|---|---|---|
| All prompts | `hf_data/prompts/all_prompts.jsonl` | 7 902 | DONE — committed |
| dual_use safe_completion + hard_refusal | `hf_data/responses/dual_use/dualuse_response_records.jsonl` | 5 004 | DONE — committed |
| hard_refusal (all 4 categories) | `hf_data/responses/hard_refusal_responses.jsonl` | 7 902 | DONE — committed |
| helpful_answer (benign + benign_sensitive) | `hf_data/responses/helpful_answer_responses.jsonl` | 3 600 | DONE — committed |
| unsafe_compliance (unsafe + dual_use) | `hf_data/responses/unsafe_compliance_responses.jsonl` | 4 275 | DONE — committed |
| Full merged responses | `hf_data/responses/full_responses.jsonl` | 18 279 | DONE — local |
| Full DPO dataset | `hf_data/dpo/full_dpo_dataset.jsonl` | 12 879 | DONE — local |
| Baseline training dataset | `hf_data/dpo/baseline_dpo_dataset.jsonl` | 7 875 | DONE — local |
| SafeComp training dataset | `hf_data/dpo/safecomp_dpo_dataset.jsonl` | 10 377 | DONE — local |

unsafe_compliance sourcing: 2 502 GPT-4o-mini records (dual_use) + 1 773 BeaverTails source records (unsafe), 27 garbage records dropped (≤5 words).

## Workflow
Prefer:
1. explore only the files needed
2. plan
3. implement
4. verify

For substantial tasks:
- understand the current contracts first
- do not broadly re-explore the whole repo unless necessary
- keep changes scoped
- preserve existing stage boundaries

## Development principles
- Pilot first, then full pipeline
- Keep local Windows setup lightweight
- Move GPU-heavy work to BABEL later
- Keep code modular and config-driven
- Preserve provenance at every stage
- Prefer scripts over notebooks for core pipeline logic
- Use JSONL for datasets and YAML for configs
- Keep `CLAUDE.md` compact and high-signal

## Canonical prompt categories
Use exactly:
- unsafe
- dual_use
- benign_sensitive
- benign

## Canonical response types
Use exactly:
- hard_refusal
- safe_completion
- unsafe_compliance
- helpful_answer

Do not silently rename these.

## Core record types
The repo revolves around:
- PromptRecord
- ResponseRecord
- ValidationRecord
- PreferencePair
- DPORecord

Prefer explicit enums and explicit metadata.

## Locked v1 pair logic
Use this as the current source of truth unless explicitly changed.

Shared by both training regimes:

### unsafe
- chosen = hard_refusal
- rejected = unsafe_compliance
- pair_type = baseline_refusal

### benign_sensitive
- chosen = helpful_answer
- rejected = hard_refusal
- pair_type = safe_completion

### benign
- chosen = helpful_answer
- rejected = hard_refusal
- pair_type = safe_completion

Different only on dual_use:

### dual_use for baseline model
- chosen = hard_refusal
- rejected = unsafe_compliance
- pair_type = baseline_refusal

### dual_use for SafeComp model
- chosen = safe_completion
- rejected = hard_refusal
- pair_type = safe_completion

### dual_use for SafeComp model
- chosen = safe_completion
- rejected = unsafe_compliance
- pair_type = safe_completion

## Training-dataset rule
Do not build final training datasets by global `pair_type` filtering alone.

Final datasets are regime-specific:

- baseline dataset
- safecomp dataset

Shared-category pairs go into both datasets.
Only `dual_use` differs between regimes.

Use the regime-specific training-dataset construction stage for this logic.

## Current project direction
Important current decisions:

- The main SafeComp vs baseline divergence lives in `dual_use`
- `unsafe_compliance` does not come from acceptable-response generation
- acceptable-response generation excludes `unsafe_compliance`
- `unsafe_compliance` is expected to come from a separate rejected-response sourcing stage
- the exact unsafe-compliance sourcing technique is still open unless explicitly locked later
- do not treat older pilot behavior as final design if newer code and configs disagree

## V1 prompt-acquisition policy
Current v1 prompt target is about 8000 clean prompts.

Category targets:
- unsafe: about 1800
- dual_use: about 2600
- benign_sensitive: about 2000
- benign: about 1600

Current v1 source policy:
- unsafe: BeaverTails only
- benign_sensitive: FalseReject train only
- benign: Alpaca only
- dual_use: project-created / contributor-provided in v1

Current exclusions / simplifications:
- Do not use HH-RLHF in v1 prompt acquisition
- Do not use BeaverTails confidence / decision-boundary ideas
- Do not require manual mining and relabeling of dual_use prompts from existing datasets in v1

## Current phase clarification
For the three dataset-backed categories:
- unsafe from BeaverTails
- benign_sensitive from FalseReject train
- benign from Alpaca

the prompt-source policy, acquisition logic, and selection logic are already decided and implemented.

The main pending data/execution work is:
- ~~assembling the real dual_use prompt set~~ DONE
- ~~running real acceptable-response generation~~ DONE (hard_refusal + helpful_answer + dual_use safe_completion)
- ~~sourcing unsafe_compliance~~ DONE (4 275 records in hf_data/responses/unsafe_compliance_responses.jsonl)
- ~~full end-to-end pipeline run on real data~~ DONE (12 879 pairs, baseline 7 875, safecomp 10 377)
- **real BABEL/runtime validation for training and evaluation** — this is the current active task

## Current implementation priorities
When extending the repo, prioritize this order:

1. keep current pair logic and regime logic correct
2. add real backends for generation, rejected-response sourcing, training, and evaluation
3. support real BABEL runs
4. report results cleanly

Do not rewrite stable stages without a clear reason.

## Data and storage rules
- Code, configs, docs, schemas, and tiny samples belong in Git
- Large datasets and artifacts belong in the private Hugging Face dataset repo
- Raw harmful generations must remain private
- Do not commit secrets, tokens, caches, checkpoints, or large raw artifacts

Selected `hf_data/` files are now tracked directly in the GitHub repo (the
nested `.git` from the HF dataset clone was removed). The gitignore uses
`hf_data/**` with explicit `!` exceptions for the files listed above. See
`.gitignore` for the current whitelist.

## Folder intent
- `src/safecomp_dpo/`: main package
- `configs/`: YAML configs
- `scripts/`: runnable entrypoints and utilities
- `tests/`: unit tests
- `data/samples/`: tiny sample JSONL only
- `outputs/`, `results/`: local generated artifacts
- `hf_data/`: local clone of the private HF dataset repo

## Local environment constraints
Keep local Windows setup lightweight.

Do not add GPU-only dependencies by default:
- vllm
- bitsandbytes
- flash-attn
- full multi-GPU training stack

These belong in the BABEL environment and should be added only when implementing real backends.

## Coding style
- Prefer simple readable Python
- Use type hints
- Use Pydantic for schemas
- Use small focused modules
- Keep CLI code explicit and testable
- Avoid overengineering

## Pipeline rule
Before implementing any stage, make the contract explicit:
- inputs
- outputs
- schema
- validation checks
- config knobs

Do not rely on hidden assumptions.

## What to avoid
- Do not generate large datasets unless explicitly asked
- Do not re-explore the whole repo without a reason
- Do not assume harmful outputs are meant for public release
- Do not add unnecessary framework complexity
- Do not rely on global `pair_type` filtering as the final regime-selection logic

## Reference docs
Use these when relevant:
- `README.md`
- `PROJECT_PLAN.md`

If docs and code disagree, flag it instead of guessing.
Prefer current code, configs, and tests over stale prose.