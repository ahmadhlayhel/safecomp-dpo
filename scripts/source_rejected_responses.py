"""
Rejected-response sourcing script.

Contract
--------
Input:  PromptRecord JSONL (produced by prompt acquisition / merge stage)
Config: Sourcing YAML — categories to source, backend, output paths
Output: ResponseRecord JSONL — one record per sourced prompt,
        response_type = unsafe_compliance

Purpose
-------
unsafe_compliance responses are the rejected side of baseline_refusal pairs
for unsafe and dual_use prompts.  They must NOT come from the acceptable
generation pipeline (generate_responses.py) because that pipeline only
produces responses the model is willing to give.

This stage sources compliant-but-unsafe responses from a separate backend
(real: a jailbreak or uncensored model; mock: deterministic canned text).

The output ResponseRecords are validated by validate_responses.py using
configs/sourcing/rejected.yaml as the config (which includes unsafe_compliance
in expected_response_types for unsafe and dual_use).

response_id convention (same as generate_responses.py):
    {prompt_id}__unsafe_compliance__s{sample:02d}
    Example: u_001__unsafe_compliance__s01

Backend protocol:
    source(prompt: str, config: dict) -> str
    Returns the unsafe compliance text for the given prompt.

Mock backend:
    Returns a clearly-labeled canned string. No GPU or real model required.
    All output is synthetic and clearly marked as mock.

Usage
-----
    python scripts/source_rejected_responses.py \\
        --prompts hf_data/prompts/all_prompts.jsonl \\
        --config configs/sourcing/rejected.yaml

    python scripts/source_rejected_responses.py \\
        --prompts hf_data/prompts/all_prompts.jsonl \\
        --config configs/sourcing/rejected.yaml \\
        --output outputs/smoke/rejected_responses.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Protocol

import yaml

try:
    from safecomp_dpo.io import load_prompt_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord, ResponseType
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_prompt_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord, ResponseType


# ---------------------------------------------------------------------------
# Backend protocol — seam for future real sourcing backend
# ---------------------------------------------------------------------------


class SourcingBackend(Protocol):
    name: str
    type_source_label: str

    def source(self, prompt: str, config: dict) -> str:
        """Return an unsafe_compliance response text for the given prompt."""
        ...


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockSourcingBackend:
    """GPU-free mock sourcing backend for pipeline development and testing.

    Returns a clearly-labeled canned string per prompt.
    The first 60 characters of the prompt are embedded so output records are
    individually identifiable in the JSONL file.

    All output is synthetic and must not be used as real harmful content.
    """

    name: str = "mock"
    type_source_label: str = "mock_sourcing"

    def source(self, prompt: str, config: dict) -> str:
        prefix = prompt[:60].rstrip()
        if len(prompt) > 60:
            prefix += "..."
        return (
            f"[MOCK unsafe_compliance] This is a synthetic compliance response "
            f'for pipeline testing only. | prompt: "{prefix}"'
        )


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


_BACKENDS: dict[str, type] = {
    "mock": MockSourcingBackend,
}


def get_backend(name: str) -> SourcingBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown sourcing backend {name!r}. Available: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_output_path(config_path: str | Path) -> Path:
    stem = Path(config_path).stem
    return Path("outputs") / "sourcing" / f"{stem}_responses.jsonl"


def make_response_id(prompt_id: str, sample: int = 1) -> str:
    """Build a deterministic response ID for an unsafe_compliance response.

    Convention: {prompt_id}__unsafe_compliance__s{sample:02d}
    """
    return f"{prompt_id}__unsafe_compliance__s{sample:02d}"


# ---------------------------------------------------------------------------
# Core sourcing logic (pure — no I/O)
# ---------------------------------------------------------------------------


def source_rejected_responses(
    prompts: list[PromptRecord],
    config: dict,
    backend: SourcingBackend,
    config_path: str | Path,
) -> list[ResponseRecord]:
    """Source unsafe_compliance ResponseRecords for the configured categories.

    Args:
        prompts:     full PromptRecord list (all categories)
        config:      sourcing config dict
        backend:     SourcingBackend instance
        config_path: stored in metadata for provenance

    Returns:
        list of ResponseRecords with response_type=unsafe_compliance,
        one per prompt that belongs to a configured category.
        Prompts in other categories are silently skipped.
    """
    categories: list[str] = config.get("categories", [])
    samples_per_prompt: int = int(config.get("samples_per_prompt", 1))
    model_name: str = config.get("model", "mock")

    if not categories:
        print(
            "  WARNING: no categories specified in config — no responses will be sourced.",
            file=sys.stderr,
        )

    category_set = set(categories)
    records: list[ResponseRecord] = []

    for prompt in prompts:
        if prompt.category.value not in category_set:
            continue

        for sample in range(1, samples_per_prompt + 1):
            response_text = backend.source(prompt.prompt, config)
            response_id = make_response_id(prompt.prompt_id, sample)

            record = ResponseRecord(
                response_id=response_id,
                prompt_id=prompt.prompt_id,
                response=response_text,
                response_type=ResponseType.unsafe_compliance,
                model=model_name,
                metadata={
                    "backend": backend.name,
                    "config_path": str(config_path),
                    "type_source": backend.type_source_label,
                },
            )
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Source unsafe_compliance ResponseRecords for unsafe and dual_use prompts."
        ),
    )
    p.add_argument(
        "--prompts",
        default="hf_data/prompts/all_prompts.jsonl",
        help="Path to PromptRecord JSONL file.",
    )
    p.add_argument(
        "--config",
        default="configs/sourcing/rejected.yaml",
        help="Path to sourcing YAML config.",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for ResponseRecord JSONL. "
            "Defaults to outputs/sourcing/<config_stem>_responses.jsonl."
        ),
    )
    p.add_argument(
        "--backend",
        default="mock",
        choices=sorted(_BACKENDS),
        help="Backend to use for sourcing (default: mock).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    prompts_path = Path(args.prompts)
    output_path = Path(args.output) if args.output else default_output_path(config_path)

    print(f"Loading prompts from {prompts_path}")
    prompts = load_prompt_records(prompts_path)
    print(f"  {len(prompts)} prompts loaded")

    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    categories: list[str] = config.get("categories", [])
    print(f"  Sourcing categories: {categories}")

    backend = get_backend(args.backend)
    print(f"Backend: {backend.name}")

    print("Sourcing rejected responses...")
    records = source_rejected_responses(prompts, config, backend, str(config_path))

    n_by_category: dict[str, int] = {}
    for r in records:
        # prompt_id prefix doesn't carry category — count from config categories
        pass
    print(f"  {len(records)} ResponseRecords sourced (unsafe_compliance)")

    print(f"Writing to {output_path}")
    write_jsonl(records, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
