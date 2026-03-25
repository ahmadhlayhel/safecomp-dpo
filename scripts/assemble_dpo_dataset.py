"""
Assemble DPO-ready training records from pipeline artifacts.

Contract:
    Inputs:
        PromptRecord JSONL     — one record per prompt
        ResponseRecord JSONL   — one record per (prompt, response_type, sample)
        ValidationRecord JSONL — one validation judgment per response
        PreferencePair JSONL   — one pair per (prompt, pair_type, index)

    Output:
        DPORecord JSONL — flat records sorted by pair_id, ready for DPO training

    FK requirements:
        PreferencePair.chosen_id   → ResponseRecord  (hard error if missing)
        PreferencePair.rejected_id → ResponseRecord  (hard error if missing)
        PreferencePair.prompt_id   → PromptRecord    (hard error if missing)
        Both chosen_id and rejected_id must have is_valid=True in ValidationRecord.
        Pairs that fail this filter are skipped and counted in stats.

Schema:
    DPORecord: pair_id, prompt_id, category, pair_type, prompt,
               chosen, rejected, chosen_id, rejected_id

Usage:
    python scripts/assemble_dpo_dataset.py --config configs/assembly/full.yaml
    python scripts/assemble_dpo_dataset.py --config configs/assembly/full.yaml --dry-run
    python scripts/assemble_dpo_dataset.py --config configs/assembly/full.yaml \\
        --output outputs/smoke/dpo_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from safecomp_dpo.io import (
    load_preference_pairs,
    load_prompt_records,
    load_response_records,
    load_validation_records,
    write_jsonl,
)
from safecomp_dpo.schemas import (
    DPORecord,
    PreferencePair,
    PromptRecord,
    ResponseRecord,
    ValidationRecord,
)


# ---------------------------------------------------------------------------
# Core transformation (pure — no file I/O, fully testable)
# ---------------------------------------------------------------------------


def assemble(
    pairs: list[PreferencePair],
    prompts: list[PromptRecord],
    responses: list[ResponseRecord],
    validations: list[ValidationRecord],
) -> tuple[list[DPORecord], dict[str, Any]]:
    """Join pipeline artifacts into flat DPO training records.

    Args:
        pairs:       PreferencePair list (defines chosen/rejected IDs and pair_type)
        prompts:     PromptRecord list (source of prompt text and category)
        responses:   ResponseRecord list (source of response text)
        validations: ValidationRecord list (filter: only is_valid=True responses)

    Returns:
        (records, stats)
        records: DPORecord list sorted by pair_id
        stats:   assembly statistics dict

    Raises:
        ValueError: if a pair references a response_id or prompt_id not present
                    in the supplied lists (broken FK — pipeline integrity error).
    """
    # ── Build lookup dicts ───────────────────────────────────────────────────
    prompts_by_id: dict[str, PromptRecord] = {p.prompt_id: p for p in prompts}
    responses_by_id: dict[str, ResponseRecord] = {r.response_id: r for r in responses}

    # One ValidationRecord per response_id: last by validation_id (alphabetical).
    # Warn when duplicates exist so the caller knows the pipeline state is unusual.
    validations_by_response: dict[str, ValidationRecord] = {}
    for v in sorted(validations, key=lambda x: x.validation_id):
        if v.response_id in validations_by_response:
            print(
                f"  WARNING: duplicate ValidationRecord for response_id={v.response_id!r}; "
                "keeping last by validation_id.",
                file=sys.stderr,
            )
        validations_by_response[v.response_id] = v

    valid_response_ids: frozenset[str] = frozenset(
        response_id
        for response_id, v in validations_by_response.items()
        if v.is_valid
    )

    # ── Process pairs ────────────────────────────────────────────────────────
    n_skipped_not_fully_valid = 0
    records: list[DPORecord] = []

    for pair in sorted(pairs, key=lambda p: p.pair_id):
        # Hard FK checks — abort on broken pipeline integrity
        if pair.chosen_id not in responses_by_id:
            raise ValueError(
                f"Broken FK: pair {pair.pair_id!r} chosen_id={pair.chosen_id!r} "
                "not found in responses. Check pipeline integrity."
            )
        if pair.rejected_id not in responses_by_id:
            raise ValueError(
                f"Broken FK: pair {pair.pair_id!r} rejected_id={pair.rejected_id!r} "
                "not found in responses. Check pipeline integrity."
            )
        if pair.prompt_id not in prompts_by_id:
            raise ValueError(
                f"Broken FK: pair {pair.pair_id!r} prompt_id={pair.prompt_id!r} "
                "not found in prompts. Check pipeline integrity."
            )

        # Validation filter — skip pairs where either side is not fully valid
        chosen_valid = pair.chosen_id in valid_response_ids
        rejected_valid = pair.rejected_id in valid_response_ids
        if not chosen_valid or not rejected_valid:
            n_skipped_not_fully_valid += 1
            print(
                f"  SKIP: pair {pair.pair_id!r} — response(s) not fully validated. "
                f"chosen={pair.chosen_id!r} valid={chosen_valid}, "
                f"rejected={pair.rejected_id!r} valid={rejected_valid}",
                file=sys.stderr,
            )
            continue

        prompt_rec = prompts_by_id[pair.prompt_id]
        chosen_rec = responses_by_id[pair.chosen_id]
        rejected_rec = responses_by_id[pair.rejected_id]

        records.append(DPORecord(
            pair_id=pair.pair_id,
            prompt_id=pair.prompt_id,
            category=prompt_rec.category,
            pair_type=pair.pair_type,
            prompt=prompt_rec.prompt,
            chosen=chosen_rec.response,
            rejected=rejected_rec.response,
            chosen_id=pair.chosen_id,
            rejected_id=pair.rejected_id,
        ))

    # ── Stats ────────────────────────────────────────────────────────────────
    category_dist: dict[str, int] = defaultdict(int)
    pair_type_dist: dict[str, int] = defaultdict(int)
    for r in records:
        category_dist[r.category.value] += 1
        pair_type_dist[r.pair_type.value] += 1

    stats: dict[str, Any] = {
        "total_pairs": len(pairs),
        "pairs_skipped_not_fully_valid": n_skipped_not_fully_valid,
        "assembled_records": len(records),
        "category_distribution": dict(category_dist),
        "pair_type_distribution": dict(pair_type_dist),
    }
    return records, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble DPO training records from pipeline artifacts",
    )
    parser.add_argument("--config", required=True, help="Path to assembly YAML config")
    parser.add_argument("--output", help="Override output_path from config")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print stats only; do not write"
    )
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pairs = load_preference_pairs(config["pairs_path"])
    prompts = load_prompt_records(config["prompts_path"])
    responses = load_response_records(config["responses_path"])
    validations = load_validation_records(config["validations_path"])

    records, stats = assemble(pairs, prompts, responses, validations)

    print(f"Total pairs:          {stats['total_pairs']}")
    print(f"Skipped (validation): {stats['pairs_skipped_not_fully_valid']}")
    print(f"Assembled records:    {stats['assembled_records']}")
    if stats["category_distribution"]:
        print(f"Category dist:        {stats['category_distribution']}")
    if stats["pair_type_distribution"]:
        print(f"Pair type dist:       {stats['pair_type_distribution']}")

    if args.dry_run:
        print("Dry run — no output written.")
        return

    output_path = args.output or config["output_path"]
    write_jsonl(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")

    sample_path = config.get("sample_path")
    sample_n = int(config.get("sample_n", 5))
    if sample_path and records:
        write_jsonl(records[:sample_n], sample_path)
        print(f"Wrote {sample_n} sample records to {sample_path}")

    report_path = config.get("report_path")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
