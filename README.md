# SafeComp-DPO

Research repository for a safe-completion DPO project using Llama 3.1 8B-Instruct.

## Goal
Compare standard refusal-based DPO against safe-completion DPO on unsafe, dual-use, benign-sensitive, and benign prompts.

## Main idea
Instead of only learning to refuse unsafe prompts, the experimental model learns to provide the most helpful safe response possible when appropriate.

## Repository scope
This repo contains:
- code
- configs
- schemas
- lightweight examples
- documentation

Large datasets and generated artifacts are stored separately in a private Hugging Face dataset repo.

## Current status
Early setup and pipeline design.

Current priorities:
- clean repo structure
- explicit data schemas
- pilot pipeline
- later scaling on BABEL

## Planned pipeline
1. prompt collection and curation
2. response generation
3. safety validation and filtering
4. preference pair construction
5. baseline refusal DPO
6. safe-completion DPO
7. evaluation and analysis

## Storage policy
- GitHub: code and documentation
- Hugging Face dataset repo: prompts, responses, validated outputs, preference pairs
- BABEL: training and large-scale inference

## Note
Raw harmful generations must remain private and should not be prepared for public release.