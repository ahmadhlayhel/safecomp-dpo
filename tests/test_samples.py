"""Schema-consistency tests for committed sample JSONL files.

Ensures that files under data/samples/ remain parseable against the current
schemas as the pipeline evolves, and that FK chains are intact across the
pilot sample set.
"""

from __future__ import annotations

from pathlib import Path

from safecomp_dpo.io import (
    load_preference_pairs,
    load_prompt_records,
    load_response_records,
    load_validation_records,
)

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


# ---------------------------------------------------------------------------
# Each sample file parses without error
# ---------------------------------------------------------------------------


def test_pilot_prompts_parses():
    records = load_prompt_records(SAMPLES / "pilot_prompts.jsonl")
    assert len(records) > 0


def test_pilot_responses_sample_parses():
    records = load_response_records(SAMPLES / "pilot_responses_sample.jsonl")
    assert len(records) > 0


def test_pilot_validations_sample_parses():
    records = load_validation_records(SAMPLES / "pilot_validations_sample.jsonl")
    assert len(records) > 0


def test_pilot_pairs_sample_parses():
    records = load_preference_pairs(SAMPLES / "pilot_pairs_sample.jsonl")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# FK chains are intact across the pilot sample set
# ---------------------------------------------------------------------------


def test_responses_sample_prompt_ids_exist():
    prompt_ids = {p.prompt_id for p in load_prompt_records(SAMPLES / "pilot_prompts.jsonl")}
    for r in load_response_records(SAMPLES / "pilot_responses_sample.jsonl"):
        assert r.prompt_id in prompt_ids, (
            f"{r.response_id}: prompt_id {r.prompt_id!r} not in pilot_prompts.jsonl"
        )


def test_validations_sample_response_ids_exist():
    response_ids = {r.response_id for r in load_response_records(SAMPLES / "pilot_responses_sample.jsonl")}
    for v in load_validation_records(SAMPLES / "pilot_validations_sample.jsonl"):
        assert v.response_id in response_ids, (
            f"{v.validation_id}: response_id {v.response_id!r} not in pilot_responses_sample.jsonl"
        )


def test_pairs_sample_response_ids_exist():
    response_ids = {r.response_id for r in load_response_records(SAMPLES / "pilot_responses_sample.jsonl")}
    for p in load_preference_pairs(SAMPLES / "pilot_pairs_sample.jsonl"):
        assert p.chosen_id in response_ids, (
            f"{p.pair_id}: chosen_id {p.chosen_id!r} not in pilot_responses_sample.jsonl"
        )
        assert p.rejected_id in response_ids, (
            f"{p.pair_id}: rejected_id {p.rejected_id!r} not in pilot_responses_sample.jsonl"
        )
