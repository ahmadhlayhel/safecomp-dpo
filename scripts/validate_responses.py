"""
Pilot response-validation script.

Contract
--------
Inputs:
    --responses  ResponseRecord JSONL (default: outputs/responses/pilot_responses.jsonl)
    --prompts    PromptRecord JSONL (default: data/samples/pilot_prompts.jsonl)
    --config     Generation YAML config (default: configs/generation/pilot.yaml)
    --output     ValidationRecord JSONL
                 (default: outputs/validations/<config_stem>_validations.jsonl)

Outputs:
    ValidationRecord JSONL — one record per ResponseRecord.
    Summary printed to stdout.
    Exit 0 if all valid, exit 1 if any invalid.

validation_id convention:
    {response_id}__val
    Example: pilot_d_001__safe_completion__s01__val

validator field:
    "rule_based_pilot"

Checks (pilot stage, rule-based only — no GPU, no classifiers):
    prompt_id_not_found       — response.prompt_id not in the prompts index
    response_type_not_expected — response_type not in expected_response_types[category]
    empty_response            — response text is empty or whitespace-only
    response_id_inconsistent  — the prompt_id and response_type embedded in the
                                response_id do not match the record's declared fields
                                (stronger than format-only: checks actual values)
    duplicate_response_id     — response_id appears more than once in the file;
                                ALL colliding records are marked invalid because
                                FK ambiguity affects the full collision set

Semantic / safety checks (classifier-based) are deferred to the full pipeline.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

try:
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord, ValidationRecord
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import load_prompt_records, load_response_records, write_jsonl
    from safecomp_dpo.schemas import PromptRecord, ResponseRecord, ValidationRecord


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALIDATOR = "rule_based_pilot"

# Matches the __s{nn} sample suffix at the end of a response_id.
# Used to strip the suffix before splitting prompt_id and response_type
# with rfind('__'), which handles underscores within segment values cleanly.
_SAMPLE_SUFFIX_RE = re.compile(r"__s\d{2}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_validation_id(response_id: str) -> str:
    return f"{response_id}__val"


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_output_path(config_path: str | Path) -> Path:
    stem = Path(config_path).stem
    return Path("outputs") / "validations" / f"{stem}_validations.jsonl"


def find_duplicate_response_ids(responses: list[ResponseRecord]) -> set[str]:
    """Return all response_ids that appear more than once.

    All members of a collision set are returned — not just the later duplicates.
    The caller marks every record whose response_id is in this set as invalid.
    """
    counts = Counter(r.response_id for r in responses)
    return {rid for rid, n in counts.items() if n > 1}


def is_response_id_consistent(record: ResponseRecord) -> bool:
    """Return True if the response_id encodes the record's declared prompt_id
    and response_type.

    Strategy: strip the __s{nn} suffix, then split on the last __ to recover
    the prompt_id and response_type segments.  Using rfind rather than a
    single regex avoids ambiguity caused by underscores within segment values
    (e.g. helpful_answer, hard_refusal).
    """
    m = _SAMPLE_SUFFIX_RE.search(record.response_id)
    if not m:
        return False
    base = record.response_id[: m.start()]
    sep = base.rfind("__")
    if sep == -1:
        return False
    embedded_prompt_id = base[:sep]
    embedded_response_type = base[sep + 2:]
    return (
        embedded_prompt_id == record.prompt_id
        and embedded_response_type == record.response_type.value
    )


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------


def validate_responses(
    responses: list[ResponseRecord],
    prompts: list[PromptRecord],
    config: dict,
) -> list[ValidationRecord]:
    """Validate each ResponseRecord and return one ValidationRecord per response.

    Checks (in order):
        1. prompt_id_not_found
        2. response_type_not_expected
        3. empty_response
        4. response_id_inconsistent
        5. duplicate_response_id
    """
    prompt_index: dict[str, PromptRecord] = {p.prompt_id: p for p in prompts}
    expected: dict[str, list[str]] = config.get("expected_response_types", {})
    duplicate_ids: set[str] = find_duplicate_response_ids(responses)

    records: list[ValidationRecord] = []

    for response in responses:
        issues: list[str] = []

        prompt = prompt_index.get(response.prompt_id)
        if prompt is None:
            issues.append("prompt_id_not_found")

        if prompt is not None:
            allowed = expected.get(prompt.category.value, [])
            if response.response_type.value not in allowed:
                issues.append("response_type_not_expected")

        if not response.response.strip():
            issues.append("empty_response")

        if not is_response_id_consistent(response):
            issues.append("response_id_inconsistent")

        if response.response_id in duplicate_ids:
            issues.append("duplicate_response_id")

        records.append(ValidationRecord(
            validation_id=make_validation_id(response.response_id),
            response_id=response.response_id,
            prompt_id=response.prompt_id,
            is_valid=not issues,
            issues=issues,
            validator=VALIDATOR,
        ))

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate ResponseRecord JSONL and emit ValidationRecord JSONL.",
    )
    p.add_argument(
        "--responses",
        default="outputs/responses/pilot_responses.jsonl",
        help="Path to ResponseRecord JSONL file.",
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
            "Output path for ValidationRecord JSONL. "
            "Defaults to outputs/validations/<config_stem>_validations.jsonl."
        ),
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    responses_path = Path(args.responses)
    prompts_path = Path(args.prompts)
    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else default_output_path(config_path)

    print(f"Loading prompts from {prompts_path}")
    prompts = load_prompt_records(prompts_path)
    print(f"  {len(prompts)} prompts loaded")

    print(f"Loading responses from {responses_path}")
    responses = load_response_records(responses_path)
    print(f"  {len(responses)} responses loaded")

    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    print("Validating...")
    validations = validate_responses(responses, prompts, config)

    n_valid = sum(1 for v in validations if v.is_valid)
    n_invalid = len(validations) - n_valid
    print(f"  {len(validations)} checked, {n_valid} valid, {n_invalid} invalid")

    if n_invalid > 0:
        for v in validations:
            if not v.is_valid:
                print(f"  INVALID {v.response_id}: {v.issues}", file=sys.stderr)

    print(f"Writing to {output_path}")
    write_jsonl(validations, output_path)
    print("Done.")

    sys.exit(0 if n_invalid == 0 else 1)


if __name__ == "__main__":
    main()
