"""Tests for scripts/assemble_dpo_dataset.py."""

import sys
from pathlib import Path

import pytest

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from assemble_dpo_dataset import assemble
from safecomp_dpo.schemas import (
    DPORecord,
    PairType,
    PreferencePair,
    PromptCategory,
    PromptRecord,
    ResponseRecord,
    ResponseType,
    ValidationRecord,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _prompt(
    prompt_id: str = "p001",
    prompt: str = "How does SQL injection work?",
    category: PromptCategory = PromptCategory.dual_use,
) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        category=category,
        source="test_source",
        source_id=prompt_id,
    )


def _response(
    response_id: str = "p001__safe_completion__s01",
    prompt_id: str = "p001",
    response: str = "SQL injection exploits unparameterized queries...",
    response_type: ResponseType = ResponseType.safe_completion,
) -> ResponseRecord:
    return ResponseRecord(
        response_id=response_id,
        prompt_id=prompt_id,
        response=response,
        response_type=response_type,
        model="test-model",
    )


def _validation(
    validation_id: str = "p001__safe_completion__s01__val",
    response_id: str = "p001__safe_completion__s01",
    prompt_id: str = "p001",
    is_valid: bool = True,
) -> ValidationRecord:
    return ValidationRecord(
        validation_id=validation_id,
        response_id=response_id,
        prompt_id=prompt_id,
        is_valid=is_valid,
        validator="test_validator",
    )


def _pair(
    pair_id: str = "p001__safe_completion__p01",
    prompt_id: str = "p001",
    chosen_id: str = "p001__safe_completion__s01",
    rejected_id: str = "p001__hard_refusal__s01",
    pair_type: PairType = PairType.safe_completion,
) -> PreferencePair:
    return PreferencePair(
        pair_id=pair_id,
        prompt_id=prompt_id,
        chosen_id=chosen_id,
        rejected_id=rejected_id,
        pair_type=pair_type,
    )


def _default_fixtures() -> tuple[
    list[PreferencePair],
    list[PromptRecord],
    list[ResponseRecord],
    list[ValidationRecord],
]:
    """Return a minimal valid set of fixtures for one pair."""
    prompt = _prompt()
    chosen = _response(
        response_id="p001__safe_completion__s01",
        prompt_id="p001",
        response="Safe completion text.",
        response_type=ResponseType.safe_completion,
    )
    rejected = _response(
        response_id="p001__hard_refusal__s01",
        prompt_id="p001",
        response="I cannot help with that.",
        response_type=ResponseType.hard_refusal,
    )
    chosen_val = _validation(
        validation_id="p001__safe_completion__s01__val",
        response_id="p001__safe_completion__s01",
    )
    rejected_val = _validation(
        validation_id="p001__hard_refusal__s01__val",
        response_id="p001__hard_refusal__s01",
    )
    pair = _pair()
    return [pair], [prompt], [chosen, rejected], [chosen_val, rejected_val]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_assemble_happy_path():
    pairs, prompts, responses, validations = _default_fixtures()
    records, stats = assemble(pairs, prompts, responses, validations)
    assert len(records) == 1
    assert stats["assembled_records"] == 1
    assert stats["pairs_skipped_not_fully_valid"] == 0


def test_assemble_returns_dpo_records():
    pairs, prompts, responses, validations = _default_fixtures()
    records, _ = assemble(pairs, prompts, responses, validations)
    assert isinstance(records[0], DPORecord)


def test_assemble_record_fields():
    pairs, prompts, responses, validations = _default_fixtures()
    records, _ = assemble(pairs, prompts, responses, validations)
    r = records[0]
    assert r.pair_id == "p001__safe_completion__p01"
    assert r.prompt_id == "p001"
    assert r.prompt == "How does SQL injection work?"
    assert r.chosen == "Safe completion text."
    assert r.rejected == "I cannot help with that."
    assert r.chosen_id == "p001__safe_completion__s01"
    assert r.rejected_id == "p001__hard_refusal__s01"


def test_assemble_category_propagated():
    pairs, prompts, responses, validations = _default_fixtures()
    records, _ = assemble(pairs, prompts, responses, validations)
    assert records[0].category == PromptCategory.dual_use


def test_assemble_pair_type_propagated():
    pairs, prompts, responses, validations = _default_fixtures()
    records, _ = assemble(pairs, prompts, responses, validations)
    assert records[0].pair_type == PairType.safe_completion


def test_assemble_empty_inputs():
    records, stats = assemble([], [], [], [])
    assert records == []
    assert stats["assembled_records"] == 0
    assert stats["total_pairs"] == 0


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------


def test_assemble_sorted_by_pair_id():
    prompt_a = _prompt(prompt_id="aaa", prompt="Prompt AAA")
    prompt_z = _prompt(prompt_id="zzz", prompt="Prompt ZZZ")
    resp_a_chosen = _response(
        response_id="aaa__safe_completion__s01", prompt_id="aaa",
        response="chosen a", response_type=ResponseType.safe_completion,
    )
    resp_a_rejected = _response(
        response_id="aaa__hard_refusal__s01", prompt_id="aaa",
        response="rejected a", response_type=ResponseType.hard_refusal,
    )
    resp_z_chosen = _response(
        response_id="zzz__safe_completion__s01", prompt_id="zzz",
        response="chosen z", response_type=ResponseType.safe_completion,
    )
    resp_z_rejected = _response(
        response_id="zzz__hard_refusal__s01", prompt_id="zzz",
        response="rejected z", response_type=ResponseType.hard_refusal,
    )
    vals = [
        _validation("aaa__safe_completion__s01__val", "aaa__safe_completion__s01", "aaa"),
        _validation("aaa__hard_refusal__s01__val", "aaa__hard_refusal__s01", "aaa"),
        _validation("zzz__safe_completion__s01__val", "zzz__safe_completion__s01", "zzz"),
        _validation("zzz__hard_refusal__s01__val", "zzz__hard_refusal__s01", "zzz"),
    ]
    # Submit pairs in reverse order
    pairs = [
        _pair("zzz__safe_completion__p01", "zzz", "zzz__safe_completion__s01", "zzz__hard_refusal__s01"),
        _pair("aaa__safe_completion__p01", "aaa", "aaa__safe_completion__s01", "aaa__hard_refusal__s01"),
    ]
    records, _ = assemble(
        pairs,
        [prompt_a, prompt_z],
        [resp_a_chosen, resp_a_rejected, resp_z_chosen, resp_z_rejected],
        vals,
    )
    assert records[0].pair_id == "aaa__safe_completion__p01"
    assert records[1].pair_id == "zzz__safe_completion__p01"


# ---------------------------------------------------------------------------
# FK violations — hard errors
# ---------------------------------------------------------------------------


def test_assemble_raises_on_missing_chosen_response():
    pairs, prompts, responses, validations = _default_fixtures()
    # Remove the chosen response
    responses = [r for r in responses if r.response_id != "p001__safe_completion__s01"]
    with pytest.raises(ValueError, match="chosen_id"):
        assemble(pairs, prompts, responses, validations)


def test_assemble_raises_on_missing_rejected_response():
    pairs, prompts, responses, validations = _default_fixtures()
    responses = [r for r in responses if r.response_id != "p001__hard_refusal__s01"]
    with pytest.raises(ValueError, match="rejected_id"):
        assemble(pairs, prompts, responses, validations)


def test_assemble_raises_on_missing_prompt():
    pairs, prompts, responses, validations = _default_fixtures()
    with pytest.raises(ValueError, match="prompt_id"):
        assemble(pairs, [], responses, validations)


def test_assemble_error_message_includes_pair_id():
    pairs, prompts, responses, validations = _default_fixtures()
    responses = [r for r in responses if r.response_id != "p001__safe_completion__s01"]
    with pytest.raises(ValueError, match="p001__safe_completion__p01"):
        assemble(pairs, prompts, responses, validations)


# ---------------------------------------------------------------------------
# Validation filter — skip, not error
# ---------------------------------------------------------------------------


def test_assemble_skips_pair_when_chosen_invalid(capsys):
    pairs, prompts, responses, validations = _default_fixtures()
    # Mark chosen response as invalid
    validations = [
        v if v.response_id != "p001__safe_completion__s01"
        else _validation(
            "p001__safe_completion__s01__val",
            "p001__safe_completion__s01",
            "p001",
            is_valid=False,
        )
        for v in validations
    ]
    records, stats = assemble(pairs, prompts, responses, validations)
    assert len(records) == 0
    assert stats["pairs_skipped_not_fully_valid"] == 1
    captured = capsys.readouterr()
    assert "SKIP" in captured.err


def test_assemble_skips_pair_when_rejected_invalid(capsys):
    pairs, prompts, responses, validations = _default_fixtures()
    validations = [
        v if v.response_id != "p001__hard_refusal__s01"
        else _validation(
            "p001__hard_refusal__s01__val",
            "p001__hard_refusal__s01",
            "p001",
            is_valid=False,
        )
        for v in validations
    ]
    records, stats = assemble(pairs, prompts, responses, validations)
    assert len(records) == 0
    assert stats["pairs_skipped_not_fully_valid"] == 1


def test_assemble_skips_pair_when_no_validation_record(capsys):
    pairs, prompts, responses, _ = _default_fixtures()
    # Pass empty validation list
    records, stats = assemble(pairs, prompts, responses, [])
    assert len(records) == 0
    assert stats["pairs_skipped_not_fully_valid"] == 1
    captured = capsys.readouterr()
    assert "SKIP" in captured.err


def test_assemble_skips_only_invalid_pair_keeps_valid():
    # Two pairs: one valid, one with invalid chosen
    prompt_a = _prompt(prompt_id="aaa", prompt="Prompt A")
    prompt_b = _prompt(prompt_id="bbb", prompt="Prompt B")
    resp_a_chosen = _response("aaa__safe_completion__s01", "aaa", "chosen a", ResponseType.safe_completion)
    resp_a_rejected = _response("aaa__hard_refusal__s01", "aaa", "rejected a", ResponseType.hard_refusal)
    resp_b_chosen = _response("bbb__safe_completion__s01", "bbb", "chosen b", ResponseType.safe_completion)
    resp_b_rejected = _response("bbb__hard_refusal__s01", "bbb", "rejected b", ResponseType.hard_refusal)
    validations = [
        _validation("aaa__safe_completion__s01__val", "aaa__safe_completion__s01", "aaa", is_valid=False),
        _validation("aaa__hard_refusal__s01__val", "aaa__hard_refusal__s01", "aaa"),
        _validation("bbb__safe_completion__s01__val", "bbb__safe_completion__s01", "bbb"),
        _validation("bbb__hard_refusal__s01__val", "bbb__hard_refusal__s01", "bbb"),
    ]
    pairs = [
        _pair("aaa__safe_completion__p01", "aaa", "aaa__safe_completion__s01", "aaa__hard_refusal__s01"),
        _pair("bbb__safe_completion__p01", "bbb", "bbb__safe_completion__s01", "bbb__hard_refusal__s01"),
    ]
    records, stats = assemble(
        pairs,
        [prompt_a, prompt_b],
        [resp_a_chosen, resp_a_rejected, resp_b_chosen, resp_b_rejected],
        validations,
    )
    assert len(records) == 1
    assert records[0].pair_id == "bbb__safe_completion__p01"
    assert stats["pairs_skipped_not_fully_valid"] == 1
    assert stats["assembled_records"] == 1


# ---------------------------------------------------------------------------
# Duplicate validation records
# ---------------------------------------------------------------------------


def test_assemble_warns_on_duplicate_validation(capsys):
    pairs, prompts, responses, validations = _default_fixtures()
    # Add a second (duplicate) validation for chosen
    validations.append(
        _validation(
            "p001__safe_completion__s01__val_v2",
            "p001__safe_completion__s01",
            "p001",
            is_valid=True,
        )
    )
    records, stats = assemble(pairs, prompts, responses, validations)
    assert len(records) == 1  # still assembles correctly
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_keys_present():
    _, _, _, _ = _default_fixtures()
    _, stats = assemble([], [], [], [])
    assert "total_pairs" in stats
    assert "pairs_skipped_not_fully_valid" in stats
    assert "assembled_records" in stats
    assert "category_distribution" in stats
    assert "pair_type_distribution" in stats


def test_stats_total_pairs():
    pairs, prompts, responses, validations = _default_fixtures()
    _, stats = assemble(pairs, prompts, responses, validations)
    assert stats["total_pairs"] == 1


def test_stats_category_distribution():
    pairs, prompts, responses, validations = _default_fixtures()
    _, stats = assemble(pairs, prompts, responses, validations)
    assert stats["category_distribution"]["dual_use"] == 1


def test_stats_pair_type_distribution():
    pairs, prompts, responses, validations = _default_fixtures()
    _, stats = assemble(pairs, prompts, responses, validations)
    assert stats["pair_type_distribution"]["safe_completion"] == 1


# ---------------------------------------------------------------------------
# Schema roundtrip
# ---------------------------------------------------------------------------


def test_dpo_record_schema_roundtrip():
    pairs, prompts, responses, validations = _default_fixtures()
    records, _ = assemble(pairs, prompts, responses, validations)
    serialized = records[0].model_dump(mode="json")
    reparsed = DPORecord.model_validate(serialized)
    assert reparsed.pair_id == records[0].pair_id
    assert reparsed.category == PromptCategory.dual_use
    assert reparsed.pair_type == PairType.safe_completion
