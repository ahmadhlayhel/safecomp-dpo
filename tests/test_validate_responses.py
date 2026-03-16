"""Tests for scripts/validate_responses.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    from validate_responses import (
        VALIDATOR,
        default_output_path,
        find_duplicate_response_ids,
        is_response_id_consistent,
        make_validation_id,
        validate_responses,
    )
except ImportError:
    _scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(_scripts_dir))
    from validate_responses import (
        VALIDATOR,
        default_output_path,
        find_duplicate_response_ids,
        is_response_id_consistent,
        make_validation_id,
        validate_responses,
    )

from safecomp_dpo.schemas import (
    PromptCategory,
    PromptRecord,
    ResponseRecord,
    ResponseType,
    ValidationRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_prompt(prompt_id: str, category: PromptCategory) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=f"Test prompt for {category.value}",
        category=category,
        source="test",
    )


def make_response(
    prompt_id: str = "pilot_b_001",
    response_type: ResponseType = ResponseType.helpful_answer,
    response: str = "A helpful answer.",
    response_id: str | None = None,
) -> ResponseRecord:
    rid = response_id or f"{prompt_id}__{response_type.value}__s01"
    return ResponseRecord(
        response_id=rid,
        prompt_id=prompt_id,
        response=response,
        response_type=response_type,
        model="test-model",
    )


MINIMAL_CONFIG = {
    "model": "test-model",
    "temperature": 0.7,
    "expected_response_types": {
        "unsafe": ["hard_refusal", "safe_completion"],
        "dual_use": ["hard_refusal", "safe_completion"],
        "benign_sensitive": ["helpful_answer"],
        "benign": ["helpful_answer"],
    },
}

PROMPTS = [
    make_prompt("pilot_u_001", PromptCategory.unsafe),
    make_prompt("pilot_d_001", PromptCategory.dual_use),
    make_prompt("pilot_bs_001", PromptCategory.benign_sensitive),
    make_prompt("pilot_b_001", PromptCategory.benign),
]


# ---------------------------------------------------------------------------
# make_validation_id
# ---------------------------------------------------------------------------


def test_make_validation_id():
    assert make_validation_id("pilot_b_001__helpful_answer__s01") == (
        "pilot_b_001__helpful_answer__s01__val"
    )


def test_make_validation_id_deterministic():
    rid = "pilot_d_001__safe_completion__s01"
    assert make_validation_id(rid) == make_validation_id(rid)


# ---------------------------------------------------------------------------
# default_output_path
# ---------------------------------------------------------------------------


def test_default_output_path():
    p = default_output_path("configs/generation/pilot.yaml")
    assert p == Path("outputs/validations/pilot_validations.jsonl")


# ---------------------------------------------------------------------------
# find_duplicate_response_ids
# ---------------------------------------------------------------------------


def test_find_duplicates_empty():
    assert find_duplicate_response_ids([]) == set()


def test_find_duplicates_no_duplicates():
    responses = [
        make_response("pilot_b_001", ResponseType.helpful_answer),
        make_response("pilot_u_001", ResponseType.hard_refusal),
    ]
    assert find_duplicate_response_ids(responses) == set()


def test_find_duplicates_one_collision():
    r = make_response("pilot_b_001", ResponseType.helpful_answer)
    responses = [r, r]
    assert find_duplicate_response_ids(responses) == {r.response_id}


def test_find_duplicates_all_colliding_ids_returned():
    """All members of a collision set are returned, not just later ones."""
    rid = "pilot_b_001__helpful_answer__s01"
    r1 = make_response(response_id=rid)
    r2 = make_response(response_id=rid)
    result = find_duplicate_response_ids([r1, r2])
    assert rid in result


def test_find_duplicates_multiple_collision_sets():
    rid_a = "pilot_b_001__helpful_answer__s01"
    rid_b = "pilot_u_001__hard_refusal__s01"
    responses = [
        make_response(response_id=rid_a),
        make_response(response_id=rid_a),
        make_response(response_id=rid_b),
        make_response(response_id=rid_b),
    ]
    assert find_duplicate_response_ids(responses) == {rid_a, rid_b}


def test_find_duplicates_unique_not_included():
    rid_dup = "pilot_b_001__helpful_answer__s01"
    rid_ok = "pilot_u_001__hard_refusal__s01"
    responses = [
        make_response(response_id=rid_dup),
        make_response(response_id=rid_dup),
        make_response(response_id=rid_ok),
    ]
    assert rid_ok not in find_duplicate_response_ids(responses)


# ---------------------------------------------------------------------------
# is_response_id_consistent
# ---------------------------------------------------------------------------


def test_consistent_id_valid():
    r = make_response("pilot_b_001", ResponseType.helpful_answer)
    assert is_response_id_consistent(r) is True


def test_consistent_id_all_response_types():
    for rt in [
        ResponseType.hard_refusal,
        ResponseType.safe_completion,
        ResponseType.helpful_answer,
        ResponseType.unsafe_compliance,
    ]:
        r = make_response("pilot_x_001", rt)
        assert is_response_id_consistent(r) is True, f"Failed for {rt}"


def test_consistent_id_wrong_prompt_id():
    r = make_response(
        prompt_id="pilot_b_001",
        response_type=ResponseType.helpful_answer,
        response_id="pilot_OTHER__helpful_answer__s01",
    )
    assert is_response_id_consistent(r) is False


def test_consistent_id_wrong_response_type():
    r = make_response(
        prompt_id="pilot_b_001",
        response_type=ResponseType.helpful_answer,
        response_id="pilot_b_001__hard_refusal__s01",
    )
    assert is_response_id_consistent(r) is False


def test_consistent_id_missing_sample_suffix():
    r = make_response(
        prompt_id="pilot_b_001",
        response_type=ResponseType.helpful_answer,
        response_id="pilot_b_001__helpful_answer",
    )
    assert is_response_id_consistent(r) is False


def test_consistent_id_no_delimiters():
    r = make_response(
        prompt_id="pilot_b_001",
        response_type=ResponseType.helpful_answer,
        response_id="somethinggarbled",
    )
    assert is_response_id_consistent(r) is False


def test_consistent_id_higher_sample_number():
    r = make_response(
        prompt_id="pilot_b_001",
        response_type=ResponseType.helpful_answer,
        response_id="pilot_b_001__helpful_answer__s03",
    )
    assert is_response_id_consistent(r) is True


# ---------------------------------------------------------------------------
# validate_responses — happy path
# ---------------------------------------------------------------------------


def test_valid_response_is_valid():
    responses = [make_response("pilot_b_001", ResponseType.helpful_answer)]
    validations = validate_responses(responses, PROMPTS, MINIMAL_CONFIG)
    assert len(validations) == 1
    assert validations[0].is_valid is True
    assert validations[0].issues == []


def test_one_validation_per_response():
    responses = [
        make_response("pilot_b_001", ResponseType.helpful_answer),
        make_response("pilot_u_001", ResponseType.hard_refusal),
    ]
    validations = validate_responses(responses, PROMPTS, MINIMAL_CONFIG)
    assert len(validations) == len(responses)


def test_validation_id_convention():
    r = make_response("pilot_b_001", ResponseType.helpful_answer)
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert validations[0].validation_id == f"{r.response_id}__val"


def test_validator_field():
    r = make_response("pilot_b_001", ResponseType.helpful_answer)
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert validations[0].validator == VALIDATOR


def test_prompt_id_denormalized():
    r = make_response("pilot_b_001", ResponseType.helpful_answer)
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert validations[0].prompt_id == r.prompt_id


def test_response_id_preserved():
    r = make_response("pilot_b_001", ResponseType.helpful_answer)
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert validations[0].response_id == r.response_id


# ---------------------------------------------------------------------------
# validate_responses — check: prompt_id_not_found
# ---------------------------------------------------------------------------


def test_unknown_prompt_id_flagged():
    r = make_response("unknown_prompt", ResponseType.helpful_answer)
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert validations[0].is_valid is False
    assert "prompt_id_not_found" in validations[0].issues


def test_known_prompt_id_not_flagged():
    r = make_response("pilot_b_001", ResponseType.helpful_answer)
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "prompt_id_not_found" not in validations[0].issues


# ---------------------------------------------------------------------------
# validate_responses — check: response_type_not_expected
# ---------------------------------------------------------------------------


def test_unexpected_response_type_flagged():
    # benign only allows helpful_answer; hard_refusal is unexpected
    r = make_response(
        "pilot_b_001",
        ResponseType.hard_refusal,
        response_id="pilot_b_001__hard_refusal__s01",
    )
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "response_type_not_expected" in validations[0].issues


def test_expected_response_type_not_flagged():
    r = make_response("pilot_u_001", ResponseType.hard_refusal)
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "response_type_not_expected" not in validations[0].issues


def test_response_type_not_expected_skipped_when_prompt_missing():
    """If prompt_id is unknown, response_type check is skipped (no category to check against)."""
    r = make_response("unknown_prompt", ResponseType.helpful_answer)
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "response_type_not_expected" not in validations[0].issues


# ---------------------------------------------------------------------------
# validate_responses — check: empty_response
# ---------------------------------------------------------------------------


def test_empty_response_flagged():
    r = make_response("pilot_b_001", ResponseType.helpful_answer, response="")
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "empty_response" in validations[0].issues


def test_whitespace_only_response_flagged():
    r = make_response("pilot_b_001", ResponseType.helpful_answer, response="   \n  ")
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "empty_response" in validations[0].issues


def test_non_empty_response_not_flagged():
    r = make_response("pilot_b_001", ResponseType.helpful_answer, response="Hello.")
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "empty_response" not in validations[0].issues


# ---------------------------------------------------------------------------
# validate_responses — check: response_id_inconsistent
# ---------------------------------------------------------------------------


def test_inconsistent_response_id_flagged():
    r = make_response(
        prompt_id="pilot_b_001",
        response_type=ResponseType.helpful_answer,
        response_id="pilot_OTHER__helpful_answer__s01",
    )
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "response_id_inconsistent" in validations[0].issues


def test_consistent_response_id_not_flagged():
    r = make_response("pilot_b_001", ResponseType.helpful_answer)
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "response_id_inconsistent" not in validations[0].issues


def test_response_id_wrong_type_segment_flagged():
    r = make_response(
        prompt_id="pilot_b_001",
        response_type=ResponseType.helpful_answer,
        response_id="pilot_b_001__hard_refusal__s01",
    )
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    assert "response_id_inconsistent" in validations[0].issues


# ---------------------------------------------------------------------------
# validate_responses — check: duplicate_response_id (all colliders invalid)
# ---------------------------------------------------------------------------


def test_duplicate_id_both_records_invalid():
    """Both records in a collision set must be marked invalid."""
    rid = "pilot_b_001__helpful_answer__s01"
    r1 = make_response(response_id=rid)
    r2 = make_response(response_id=rid)
    validations = validate_responses([r1, r2], PROMPTS, MINIMAL_CONFIG)
    for v in validations:
        assert "duplicate_response_id" in v.issues
        assert v.is_valid is False


def test_non_duplicate_id_not_flagged():
    responses = [
        make_response("pilot_b_001", ResponseType.helpful_answer),
        make_response("pilot_u_001", ResponseType.hard_refusal),
    ]
    validations = validate_responses(responses, PROMPTS, MINIMAL_CONFIG)
    for v in validations:
        assert "duplicate_response_id" not in v.issues


def test_three_way_collision_all_invalid():
    rid = "pilot_b_001__helpful_answer__s01"
    responses = [make_response(response_id=rid) for _ in range(3)]
    validations = validate_responses(responses, PROMPTS, MINIMAL_CONFIG)
    assert all("duplicate_response_id" in v.issues for v in validations)


# ---------------------------------------------------------------------------
# validate_responses — multiple issues on one record
# ---------------------------------------------------------------------------


def test_multiple_issues_accumulated():
    r = make_response(
        prompt_id="unknown_prompt",
        response_type=ResponseType.helpful_answer,
        response="",
        response_id="bad_id_no_format",
    )
    validations = validate_responses([r], PROMPTS, MINIMAL_CONFIG)
    issues = validations[0].issues
    assert "prompt_id_not_found" in issues
    assert "empty_response" in issues
    assert "response_id_inconsistent" in issues
    assert validations[0].is_valid is False


# ---------------------------------------------------------------------------
# validate_responses — output is ValidationRecord instances
# ---------------------------------------------------------------------------


def test_output_are_validation_records():
    responses = [make_response("pilot_b_001", ResponseType.helpful_answer)]
    validations = validate_responses(responses, PROMPTS, MINIMAL_CONFIG)
    assert all(isinstance(v, ValidationRecord) for v in validations)
