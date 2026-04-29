# Unsafe Compliance Handoff

## Purpose of this document

This document is a handoff for the next conversation working on
`unsafe_compliance` sourcing. It captures the current state of the pipeline,
exactly what is missing, and the delivery contract your teammate must meet.

---

## Current pipeline state (as of 2026-04-29)

### Done

| Artifact | Location | Count |
|---|---|---|
| All prompts merged | `hf_data/prompts/all_prompts.jsonl` | 7 902 PromptRecords |
| Dual-use safe_completion + hard_refusal | `hf_data/responses/dual_use/dualuse_response_records.jsonl` | 5 004 ResponseRecords |
| hard_refusal (all 4 categories) | `hf_data/responses/hard_refusal_responses.jsonl` | 7 902 ResponseRecords |
| helpful_answer (benign + benign_sensitive) | `hf_data/responses/helpful_answer_responses.jsonl` | 3 600 ResponseRecords |

Generation configs are in `configs/generation/` (hard_refusal.yaml, helpful_answer.yaml).

### The one missing piece

`hf_data/responses/unsafe_compliance_responses.jsonl` — **4 302 records** covering:
- `unsafe`: 1 800 prompts
- `dual_use`: 2 502 prompts

This file is the only thing blocking the full pipeline run.

---

## Delivery contract for unsafe_compliance

### File to deliver

```
hf_data/responses/unsafe_compliance_responses.jsonl
```

### Schema — ResponseRecord (Pydantic)

Every line must be a valid JSON object matching:

```python
class ResponseRecord(BaseModel):
    response_id: str        # see convention below
    prompt_id: str          # must match a prompt_id in all_prompts.jsonl
    response: str           # the unsafe/compliant text
    response_type: str      # must be exactly "unsafe_compliance"
    model: str              # model used (e.g. "gpt-4o-mini", "llama-3.1-8b-instruct")
    temperature: float
    metadata: dict          # at minimum: {"finish_reason": "stop"}
```

### response_id convention

```
{prompt_id}__unsafe_compliance__s01
```

Example: if `prompt_id` is `unsafe_0042`, then:
```
unsafe_0042__unsafe_compliance__s01
```

### Required coverage

- All 1 800 `unsafe` prompts from `all_prompts.jsonl`
- All 2 502 `dual_use` prompts from `all_prompts.jsonl`
- Total: **4 302 records** exactly (one per prompt)

To get the list of prompt_ids to cover:

```bash
python - <<'EOF'
import json
with open("hf_data/prompts/all_prompts.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["category"] in ("unsafe", "dual_use"):
            print(r["prompt_id"])
EOF
```

### What the response should look like

A realistic model output that **complies** with the unsafe or dual-use request —
i.e., it provides the harmful information or follows through on the request
without refusing. This is the rejected response in the pair logic.

Do NOT produce refusals, hedges, or partial answers. The record must contain
the actual compliant generation.

---

## How to validate once delivered

```bash
python scripts/validate_responses.py \
    --responses hf_data/responses/unsafe_compliance_responses.jsonl \
    --prompts   hf_data/prompts/all_prompts.jsonl \
    --config    configs/generation/full.yaml
```

Or run a quick sanity check:

```bash
python - <<'EOF'
import json
from collections import Counter

path = "hf_data/responses/unsafe_compliance_responses.jsonl"
records = [json.loads(l) for l in open(path)]
cats = Counter()
for r in records:
    # derive category from prompt_id prefix
    cats[r["prompt_id"].split("_")[0]] += 1

print(f"Total records: {len(records)}")
print(dict(cats))
# Expect: unsafe=1800, dual=2502 (or dual_use prefix depending on prompt_id scheme)
EOF
```

---

## What runs immediately after delivery

### Step 1 — Merge all responses

```bash
python scripts/merge_responses.py \
    --unsafe-compliance hf_data/responses/unsafe_compliance_responses.jsonl
```

Expected output:
```
safe_completion: 2502
hard_refusal:    7902
helpful_answer:  3600
unsafe_compliance: 4302
Total: 18306
```

Output file: `hf_data/responses/full_responses.jsonl`

### Step 2 — Validate full response set

```bash
python scripts/validate_responses.py \
    --responses hf_data/responses/full_responses.jsonl \
    --prompts   hf_data/prompts/all_prompts.jsonl \
    --config    configs/generation/full.yaml
```

### Step 3 — Build preference pairs

```bash
python scripts/build_pairs.py \
    --responses hf_data/responses/full_responses.jsonl \
    --output    hf_data/pairs/preference_pairs.jsonl
```

### Step 4 — Assemble DPO datasets (regime-specific)

```bash
python scripts/assemble_dataset.py \
    --pairs   hf_data/pairs/preference_pairs.jsonl \
    --regime  baseline \
    --output  hf_data/datasets/baseline_dpo.jsonl

python scripts/assemble_dataset.py \
    --pairs   hf_data/pairs/preference_pairs.jsonl \
    --regime  safecomp \
    --output  hf_data/datasets/safecomp_dpo.jsonl
```

### Step 5 — Training (BABEL)

See `configs/training/` and `PROJECT_PLAN.md` for BABEL training configs.

---

## Key pair logic reminder

The unsafe_compliance records feed into these pairs:

| Category | Regime | chosen | rejected | pair_type |
|---|---|---|---|---|
| unsafe | both | hard_refusal | unsafe_compliance | baseline_refusal |
| dual_use | baseline | hard_refusal | unsafe_compliance | baseline_refusal |
| dual_use | safecomp | safe_completion | unsafe_compliance | safe_completion |

Without unsafe_compliance, the `build_pairs.py` step will skip all
`baseline_refusal` pairs and the safecomp `safe_completion vs unsafe_compliance`
pairs — so training datasets will be incomplete.

---

## Files to read before starting

- `src/safecomp_dpo/schemas.py` — ResponseRecord, ResponseType enum
- `src/safecomp_dpo/io.py` — load_response_records, write_jsonl
- `scripts/generate_responses.py` — reference OpenAI API backend implementation
- `configs/generation/hard_refusal.yaml` — example generation config
- `hf_data/prompts/all_prompts.jsonl` — source of truth for prompt_ids

---

## git / storage reminder

`hf_data/responses/unsafe_compliance_responses.jsonl` is already whitelisted in
`.gitignore` — you can commit it directly once it exists.

The file contains harmful model outputs — treat it as private. The GitHub repo
is private; do not share the file outside the project.
