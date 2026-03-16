# Project Plan

## Stage 0: Repository foundation
- finalize CLAUDE.md
- finalize README.md
- finalize local folder structure
- define schemas
- define pilot configs

## Stage 1: Pilot pipeline
- create a tiny manually curated pilot prompt set
- test record formats
- test response-generation interfaces
- test validation flow
- test preference-pair construction

## Stage 2: Full data pipeline
- collect prompts from source datasets
- deduplicate and filter
- generate responses by category
- validate outputs
- construct baseline and safe-completion preference pairs

## Stage 3: Training
- baseline refusal DPO
- experimental safe-completion DPO

## Stage 4: Evaluation
- safety benchmarks
- over-refusal benchmarks
- dual-use custom evaluation
- analysis and ablations

## Current rule
Do not scale before the pilot pipeline works end to end.