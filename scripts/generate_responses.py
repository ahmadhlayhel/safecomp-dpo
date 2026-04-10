"""
Response-generation script for the SafeComp-DPO pipeline.

Contract
--------
Inputs:
    --prompts   PromptRecord JSONL (default: data/samples/pilot_prompts.jsonl)
    --config    Generation YAML config (default: configs/generation/pilot.yaml)
    --output    Output ResponseRecord JSONL
                (default: outputs/responses/<config_stem>_responses.jsonl)
    --backend   Backend selector string (default: mock)
    --resume    If the output file already exists, skip already-generated
                response_ids and append new records.  Safe to re-run after
                a crash or interruption.
    --report    Optional path to write a JSON summary report.

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

Per-response-type settings:
    The config may include a response_type_settings section:

        response_type_settings:
          hard_refusal:
            system_prompt: "..."
            temperature: 0.3
            max_tokens: 256
          helpful_answer:
            system_prompt: "..."
            temperature: 0.7
            max_tokens: 512

    These override the top-level temperature/max_tokens for each type.
    The openai_api backend reads the system_prompt from this section.
    The mock backend ignores it.

Mock backend:
    Returns a canned, clearly-labeled string per (prompt, response_type).
    No GPU or model weights required.  type_source = "explicit_mock".

OpenAI API backend:
    Calls the OpenAI Chat Completions API.  Requires OPENAI_API_KEY env var.
    Retries on rate-limit errors with exponential backoff.
    type_source = "explicit_api".

Validation:
    Each ResponseRecord is validated via model_validate before writing.
    A validation error aborts with a message identifying the failing
    (prompt_id, response_type).

Output path:
    Default: outputs/responses/<config_stem>_responses.jsonl
    Override with --output.

Resume:
    Pass --resume to skip already-generated response_ids and append to the
    existing output file.  Safe to re-run after a crash.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import yaml

try:
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord, ResponseType
except ImportError:
    # Allow running from project root without installing the package.
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
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
# OpenAI API backend
# ---------------------------------------------------------------------------


class OpenAIAPIBackend:
    """OpenAI Chat Completions backend for production response generation.

    One API call per (prompt, response_type).  Reads per-response-type
    system prompts and generation settings from config['response_type_settings'].

    After each call, self.last_usage is updated with:
        finish_reason, prompt_tokens, completion_tokens

    These are picked up by main() and written into ResponseRecord.metadata.

    Retries on RateLimitError with exponential backoff (up to 5 attempts).

    Requires:
        OPENAI_API_KEY environment variable
        pip install openai>=1.0
    """

    name: str = "openai_api"
    type_source_label: str = "explicit_api"

    def __init__(self) -> None:
        try:
            import openai as _openai  # noqa: PLC0415
            self._openai = _openai
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for the openai_api backend.\n"
                "Install it with:  pip install openai>=1.0\n"
                "Or:               pip install '.[generation]'"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Set it with:  export OPENAI_API_KEY=sk-..."
            )

        self.client = self._openai.OpenAI(api_key=api_key)
        self.last_usage: dict[str, Any] = {}

    def generate(self, prompt: str, response_type: ResponseType, config: dict) -> str:
        rt_name = response_type.value
        rt_settings: dict = config.get("response_type_settings", {}).get(rt_name, {})

        system_prompt: str = rt_settings.get(
            "system_prompt", "You are a helpful assistant."
        )
        temperature: float = float(
            rt_settings.get("temperature", config.get("temperature", 0.7))
        )
        max_tokens: int = int(
            rt_settings.get("max_tokens", config.get("max_tokens", 512))
        )
        model: str = config.get("model", "gpt-4o-2024-11-20")

        max_retries = 5
        backoff = 2.0
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.last_usage = {
                    "finish_reason": resp.choices[0].finish_reason,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                    "completion_tokens": (
                        resp.usage.completion_tokens if resp.usage else None
                    ),
                }
                return resp.choices[0].message.content or ""

            except self._openai.RateLimitError as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = backoff * (2 ** attempt)
                    print(
                        f"  Rate limit hit — waiting {wait:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})...",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                else:
                    raise

        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type] = {
    "mock": MockBackend,
    "openai_api": OpenAIAPIBackend,
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
# Core generation logic (used by tests — signature must remain stable)
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

    Note: this function accumulates all records in memory.  For production
    runs use main() which streams records to disk and supports --resume.
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
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous run: load already-generated response_ids from the "
            "output file and skip them.  Appends new records to the existing file."
        ),
    )
    p.add_argument(
        "--report",
        default=None,
        help="Optional path to write a JSON generation report.",
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

    # Build the skip set if resuming
    skip_ids: set[str] = set()
    file_mode = "w"
    if args.resume and output_path.exists():
        existing = load_response_records(output_path)
        skip_ids = {r.response_id for r in existing}
        file_mode = "a"
        print(f"Resume mode: {len(skip_ids)} already done, will skip and append.")

    backend = get_backend(args.backend)
    print(f"Backend: {backend.name}")

    # Per-response-type settings (for accurate ResponseRecord fields)
    rt_settings_map: dict[str, dict] = config.get("response_type_settings", {})
    top_model: str = config.get("model", "unknown")
    top_temperature: float | None = config.get("temperature")

    expected: dict[str, list[str]] = config.get("expected_response_types", {})

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_generated = 0
    n_skipped = 0
    n_errors = 0
    counts_by_type: dict[str, int] = {}
    counts_by_category: dict[str, int] = {}

    print(f"Generating responses -> {output_path}")
    with output_path.open(file_mode, encoding="utf-8") as out_f:
        for prompt in prompts:
            category = prompt.category.value
            rt_names = expected.get(category, [])

            if not rt_names:
                continue

            for rt_name in rt_names:
                response_type = ResponseType(rt_name)
                response_id = make_response_id(prompt.prompt_id, response_type)

                if response_id in skip_ids:
                    n_skipped += 1
                    continue

                rt_cfg = rt_settings_map.get(rt_name, {})
                effective_model = rt_cfg.get("model", top_model)
                effective_temp: float | None = (
                    float(rt_cfg["temperature"])
                    if "temperature" in rt_cfg
                    else top_temperature
                )

                try:
                    response_text = backend.generate(
                        prompt.prompt, response_type, config
                    )
                except Exception as exc:
                    print(
                        f"\n  ERROR generating {response_id}: {exc}",
                        file=sys.stderr,
                    )
                    n_errors += 1
                    continue

                usage_meta: dict[str, Any] = getattr(backend, "last_usage", {})

                record = ResponseRecord(
                    response_id=response_id,
                    prompt_id=prompt.prompt_id,
                    response=response_text,
                    response_type=response_type,
                    model=effective_model,
                    temperature=effective_temp,
                    metadata={
                        "backend": backend.name,
                        "config_path": str(config_path),
                        "type_source": backend.type_source_label,
                        **usage_meta,
                    },
                )
                out_f.write(record.model_dump_json() + "\n")
                out_f.flush()

                n_generated += 1
                counts_by_type[rt_name] = counts_by_type.get(rt_name, 0) + 1
                counts_by_category[category] = (
                    counts_by_category.get(category, 0) + 1
                )

                if n_generated % 100 == 0:
                    total_done = n_generated + n_skipped
                    print(
                        f"  {n_generated} generated, {n_skipped} skipped"
                        f", {n_errors} errors  (total processed: {total_done})"
                    )

    print(
        f"\nDone.  generated={n_generated}  skipped={n_skipped}  errors={n_errors}"
    )
    print(f"  by type:     {counts_by_type}")
    print(f"  by category: {counts_by_category}")
    print(f"  output:      {output_path}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config": str(config_path),
            "prompts": str(prompts_path),
            "output": str(output_path),
            "backend": backend.name,
            "model": top_model,
            "n_generated": n_generated,
            "n_skipped": n_skipped,
            "n_errors": n_errors,
            "counts_by_type": counts_by_type,
            "counts_by_category": counts_by_category,
        }
        with report_path.open("w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)
        print(f"  report:      {report_path}")


if __name__ == "__main__":
    main()
