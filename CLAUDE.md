# SafeComp-DPO Repo Instructions

## Purpose
This repo implements a research pipeline for safe-completion DPO on Llama 3.1 8B-Instruct.

The project compares:
- Baseline Refusal-DPO
- Experimental SafeComp-DPO

## Current stage
We are in repository foundation and pilot design.

Work in this order:
1. docs and repo rules
2. schemas
3. pilot configs
4. local utilities
5. pilot pipeline
6. full data pipeline
7. BABEL training and eval

Do not jump to full extraction, generation, or training unless explicitly asked.

## Core workflow
Prefer:
1. explore
2. plan
3. implement
4. verify

For substantial tasks, understand the current files before writing code.

## Development principles
- Pilot first
- Keep local Windows setup lightweight
- Move GPU-heavy work to BABEL later
- Keep code modular and config-driven
- Preserve provenance at every stage
- Prefer scripts over notebooks for core pipeline logic
- Use JSONL for datasets and YAML for configs

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

## Expected responses by category
- unsafe: hard_refusal, safe_completion, unsafe_compliance
- dual_use: hard_refusal, safe_completion, unsafe_compliance
- benign_sensitive: hard_refusal, helpful_answer
- benign: helpful_answer

## Core record types
The repo should revolve around:
- PromptRecord
- ResponseRecord
- ValidationRecord
- PreferencePair

Prefer explicit enums and explicit metadata.

## Data and storage rules
- Code, configs, docs, schemas, and tiny samples belong in Git
- Large datasets and artifacts belong in the private Hugging Face dataset repo
- Raw harmful generations must remain private
- Do not commit secrets, tokens, caches, checkpoints, or large raw artifacts

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
- Do not generate large synthetic datasets unless explicitly asked
- Do not implement training before schemas and pilot configs are stable
- Do not assume harmful outputs are meant for public release
- Do not add unnecessary framework complexity

## Immediate priorities
If asked to help with setup, prioritize:
1. docs
2. `pyproject.toml`
3. schemas
4. JSONL utilities
5. pilot configs
6. CLI stubs

## Reference docs
Use these first when present:
- `README.md`
- `PROJECT_PLAN.md`

If docs and code disagree, flag it instead of guessing.