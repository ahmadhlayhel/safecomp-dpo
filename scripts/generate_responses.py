"""
Pilot response-generation script.

Contract
--------
Inputs:
    --prompts   PromptRecord JSONL (default: data/samples/pilot_prompts.jsonl)
    --config    Generation YAML config (default: configs/generation/pilot.yaml)
    --output    Output ResponseRecord JSONL
                (default: outputs/responses/<config_stem>_responses.jsonl)
    --backend   Backend selector string (default: mock)

Outputs:
    ResponseRecord JSONL — one record per (prompt, response_type) pair.

response_id convention:
    {prompt_id}__{response_type}__s{sample:02d}
    Example: pilot_d_001__safe_completion__s01

    The sample index is fixed at s01 for single-sample runs.  Multiple samples
    per (prompt, response_type) use s02, s03, etc.  This keeps IDs stable and
    FK-safe when multi-sample generation is added later.

response_type assignment policy:
    The script generates exactly the response types listed under
    expected_response_types[category] in the config.  The type is passed
    explicitly to the backend — it is never inferred from the generated text
    at this stage.  metadata.type_source records how the type was assigned.

Mock backend:
    Returns a canned, clearly-labeled string per (prompt, response_type).
    No GPU or model weights required.  type_source = "explicit_mock".

    The backend callable is the seam for future real model backends.
    Backend signature:
        generate(prompt: str, response_type: ResponseType, config: dict) -> str

Validation:
    Each ResponseRecord is validated via model_validate before writing.
    A validation error aborts with a message identifying the failing
    (prompt_id, response_type).

Output path:
    Default: outputs/responses/<config_stem>_responses.jsonl
    Override with --output.
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
    # Allow running from project root without installing the package.
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_prompt_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord, ResponseType


# ---------------------------------------------------------------------------
# Backend protocol — the seam for future real model backends
# ---------------------------------------------------------------------------


class Backend(Protocol):
    name: str
    type_source_label: str

    def generate(self, prompt: str, response_type: ResponseType, config: dict) -> str:
        ...


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

_MOCK_TEMPLATES: dict[ResponseType, str] = {
    ResponseType.hard_refusal: (
        "[MOCK hard_refusal] I'm sorry, I can't help with that request."
    ),
    ResponseType.safe_completion: (
        "[MOCK safe_completion] Here is a safe and informative response to your question."
    ),
    ResponseType.helpful_answer: (
        "[MOCK helpful_answer] Here is a helpful and complete answer to your question."
    ),
    # unsafe_compliance is included in the template table only so the mock
    # backend is complete as a stub.  It must not appear in pilot generation
    # output — the config's expected_response_types enforces this.
    ResponseType.unsafe_compliance: (
        "[MOCK unsafe_compliance] This type should not appear in pilot generation."
    ),
}


class MockBackend:
    """GPU-free mock backend for pipeline development and testing.

    Returns a canned, clearly-labeled string for each (prompt, response_type).
    The first 60 characters of the prompt are embedded so output records are
    individually identifiable in the JSONL file.
    """

    name: str = "mock"
    type_source_label: str = "explicit_mock"

    def generate(self, prompt: str, response_type: ResponseType, config: dict) -> str:
        template = _MOCK_TEMPLATES[response_type]
        prefix = prompt[:60].rstrip()
        if len(prompt) > 60:
            prefix += "..."
        return f'{template} | prompt: "{prefix}"'


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type] = {
    "mock": MockBackend,
}


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[name]()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_response_id(
    prompt_id: str, response_type: ResponseType, sample: int = 1
) -> str:
    """Build a deterministic response ID.

    Convention: {prompt_id}__{response_type}__s{sample:02d}
    Example:    pilot_d_001__safe_completion__s01
    """
    return f"{prompt_id}__{response_type.value}__s{sample:02d}"


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_output_path(config_path: str | Path) -> Path:
    stem = Path(config_path).stem
    return Path("outputs") / "responses" / f"{stem}_responses.jsonl"


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------


def generate_responses(
    prompts: list[PromptRecord],
    config: dict,
    backend: Backend,
    config_path: str | Path,
) -> list[ResponseRecord]:
    """Generate ResponseRecords for all (prompt, response_type) pairs.

    The set of response types per prompt is driven by
    config['expected_response_types'][category].  Each record is validated
    against the ResponseRecord schema before being appended.
    """
    expected: dict[str, list[str]] = config.get("expected_response_types", {})
    model_name: str = config.get("model", "mock")
    temperature: float | None = config.get("temperature")

    records: list[ResponseRecord] = []

    for prompt in prompts:
        category = prompt.category.value
        response_type_names = expected.get(category, [])

        if not response_type_names:
            print(
                f"  WARNING: no expected_response_types for category {category!r} "
                f"(prompt_id={prompt.prompt_id!r}) — skipping",
                file=sys.stderr,
            )
            continue

        for rt_name in response_type_names:
            response_type = ResponseType(rt_name)
            response_text = backend.generate(prompt.prompt, response_type, config)
            response_id = make_response_id(prompt.prompt_id, response_type)

            record = ResponseRecord(
                response_id=response_id,
                prompt_id=prompt.prompt_id,
                response=response_text,
                response_type=response_type,
                model=model_name,
                temperature=temperature,
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
        description="Generate ResponseRecord JSONL from a PromptRecord JSONL file.",
    )
    p.add_argument(
        "--prompts",
        default="data/samples/pilot_prompts.jsonl",
        help="Path to PromptRecord JSONL file.",
    )
    p.add_argument(
        "--config",
        default="configs/generation/pilot.yaml",
        help="Path to generation YAML config.",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for ResponseRecord JSONL. "
            "Defaults to outputs/responses/<config_stem>_responses.jsonl."
        ),
    )
    p.add_argument(
        "--backend",
        default="mock",
        choices=sorted(_BACKENDS),
        help="Backend to use for generation (default: mock).",
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

    backend = get_backend(args.backend)
    print(f"Backend: {backend.name}")

    print("Generating responses...")
    records = generate_responses(prompts, config, backend, str(config_path))
    print(f"  {len(records)} ResponseRecords generated")

    print(f"Writing to {output_path}")
    write_jsonl(records, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
