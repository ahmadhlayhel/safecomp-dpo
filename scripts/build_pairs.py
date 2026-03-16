"""
Pilot preference-pair construction script.

Contract
--------
Inputs:
    --responses    ResponseRecord JSONL (default: outputs/responses/pilot_responses.jsonl)
    --validations  ValidationRecord JSONL (default: outputs/validations/pilot_validations.jsonl)
    --prompts      PromptRecord JSONL (default: data/samples/pilot_prompts.jsonl)
    --config       Pair-rules YAML (default: configs/pairs/pilot.yaml)
    --output       PreferencePair JSONL (default: outputs/pairs/pilot_pairs.jsonl)

Outputs:
    PreferencePair JSONL — one record per (prompt, rule) where both required
    response types are available as valid responses.

pair_id convention:
    {prompt_id}__{pair_type}__p01
    Example: pilot_d_001__safe_completion__p01
    p{nn} parallels the s{nn} sample index in response_id.

Invalid responses:
    Responses absent from the validation index, or with is_valid=False, are
    excluded from both chosen and rejected positions.  If either required type
    is missing or invalid for a prompt, that rule is skipped with a warning.

Config-driven:
    Pair rules are defined per category in the YAML config.  An empty list for
    a category is an explicit skip — not silent omission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

try:
    from safecomp_dpo.io import (
        load_preference_pairs,
        load_prompt_records,
        load_response_records,
        load_validation_records,
        write_jsonl,
    )
    from safecomp_dpo.schemas import (
        PairType,
        PreferencePair,
        PromptRecord,
        ResponseRecord,
        ValidationRecord,
    )
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.io import (
        load_preference_pairs,
        load_prompt_records,
        load_response_records,
        load_validation_records,
        write_jsonl,
    )
    from safecomp_dpo.schemas import (
        PairType,
        PreferencePair,
        PromptRecord,
        ResponseRecord,
        ValidationRecord,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_output_path(config_path: str | Path) -> Path:
    stem = Path(config_path).stem
    return Path("outputs") / "pairs" / f"{stem}_pairs.jsonl"


def build_valid_response_index(
    responses: list[ResponseRecord],
    validations: list[ValidationRecord],
) -> dict[str, dict[str, str]]:
    """Build a nested index: prompt_id → {response_type_value → response_id}.

    Only responses with is_valid=True in the validation set are included.
    Responses with no validation record are excluded (conservative).
    """
    valid_ids: set[str] = {v.response_id for v in validations if v.is_valid}
    index: dict[str, dict[str, str]] = {}
    for r in responses:
        if r.response_id in valid_ids:
            index.setdefault(r.prompt_id, {})[r.response_type.value] = r.response_id
    return index


def make_pair_id(prompt_id: str, pair_type: PairType, index: int = 1) -> str:
    """Build a deterministic pair ID.

    Convention: {prompt_id}__{pair_type}__p{index:02d}
    Example:    pilot_d_001__safe_completion__p01
    """
    return f"{prompt_id}__{pair_type.value}__p{index:02d}"


# ---------------------------------------------------------------------------
# Core pair-construction logic
# ---------------------------------------------------------------------------


def build_pairs(
    prompts: list[PromptRecord],
    responses: list[ResponseRecord],
    validations: list[ValidationRecord],
    config: dict,
    config_path: str | Path,
) -> list[PreferencePair]:
    """Construct PreferencePairs from validated responses.

    For each prompt, iterates over the pair_rules defined for its category.
    A pair is emitted only when both the chosen and rejected response types
    are present as valid responses for that prompt.
    """
    prompt_index: dict[str, PromptRecord] = {p.prompt_id: p for p in prompts}
    valid_response_index = build_valid_response_index(responses, validations)
    pair_rules: dict[str, list[dict]] = config.get("pair_rules", {})

    pairs: list[PreferencePair] = []

    for prompt in prompts:
        category = prompt.category.value
        rules = pair_rules.get(category, [])
        available = valid_response_index.get(prompt.prompt_id, {})

        for i, rule in enumerate(rules, start=1):
            pair_type = PairType(rule["pair_type"])
            chosen_type: str = rule["chosen_type"]
            rejected_type: str = rule["rejected_type"]

            chosen_id = available.get(chosen_type)
            rejected_id = available.get(rejected_type)

            if chosen_id is None:
                print(
                    f"  WARNING: {prompt.prompt_id}: no valid {chosen_type!r} response"
                    f" for {pair_type.value} pair — skipping",
                    file=sys.stderr,
                )
                continue

            if rejected_id is None:
                print(
                    f"  WARNING: {prompt.prompt_id}: no valid {rejected_type!r} response"
                    f" for {pair_type.value} pair — skipping",
                    file=sys.stderr,
                )
                continue

            pairs.append(PreferencePair(
                pair_id=make_pair_id(prompt.prompt_id, pair_type, i),
                prompt_id=prompt.prompt_id,
                chosen_id=chosen_id,
                rejected_id=rejected_id,
                pair_type=pair_type,
                metadata={
                    "config_path": str(config_path),
                    "chosen_type": chosen_type,
                    "rejected_type": rejected_type,
                },
            ))

    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Construct PreferencePair JSONL from validated ResponseRecords.",
    )
    p.add_argument(
        "--responses",
        default="outputs/responses/pilot_responses.jsonl",
        help="Path to ResponseRecord JSONL file.",
    )
    p.add_argument(
        "--validations",
        default="outputs/validations/pilot_validations.jsonl",
        help="Path to ValidationRecord JSONL file.",
    )
    p.add_argument(
        "--prompts",
        default="data/samples/pilot_prompts.jsonl",
        help="Path to PromptRecord JSONL file.",
    )
    p.add_argument(
        "--config",
        default="configs/pairs/pilot.yaml",
        help="Path to pair-rules YAML config.",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for PreferencePair JSONL. "
            "Defaults to outputs/pairs/<config_stem>_pairs.jsonl."
        ),
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    responses_path = Path(args.responses)
    validations_path = Path(args.validations)
    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else default_output_path(config_path)

    print(f"Loading prompts from {prompts_path}")
    prompts = load_prompt_records(prompts_path)
    print(f"  {len(prompts)} prompts loaded")

    print(f"Loading responses from {responses_path}")
    responses = load_response_records(responses_path)
    print(f"  {len(responses)} responses loaded")

    print(f"Loading validations from {validations_path}")
    validations = load_validation_records(validations_path)
    n_valid = sum(1 for v in validations if v.is_valid)
    print(f"  {len(validations)} validation records ({n_valid} valid)")

    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    pair_rules: dict[str, list] = config.get("pair_rules", {})

    print("Building pairs...")
    pairs = build_pairs(prompts, responses, validations, config, config_path)

    # Per-category summary
    prompt_index = {p.prompt_id: p for p in prompts}
    from collections import defaultdict
    category_counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        cat = prompt_index[pair.prompt_id].category.value
        category_counts[cat] += 1

    for category in ["unsafe", "dual_use", "benign_sensitive", "benign"]:
        rules = pair_rules.get(category, [])
        count = category_counts.get(category, 0)
        if not rules:
            print(f"  {category}: no rules — skipping")
        else:
            print(f"  {category}: {count} pair(s)")

    print(f"  {len(pairs)} total pairs constructed")
    print(f"Writing to {output_path}")
    write_jsonl(pairs, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
