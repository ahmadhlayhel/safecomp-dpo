"""Tests for safecomp_dpo.schemas."""

import pytest
from pydantic import ValidationError

from safecomp_dpo.schemas import (
    PairType,
    PreferencePair,
    PromptCategory,
    PromptRecord,
    ResponseRecord,
    ResponseType,
    ValidationRecord,
)


# ---------------------------------------------------------------------------
# Enum membership
# ---------------------------------------------------------------------------


def test_prompt_category_values():
    assert set(PromptCategory) == {
        PromptCategory.unsafe,
        PromptCategory.dual_use,
        PromptCategory.benign_sensitive,
        PromptCategory.benign,
    }


def test_response_type_values():
    assert set(ResponseType) == {
        ResponseType.hard_refusal,
        ResponseType.safe_completion,
        ResponseType.unsafe_compliance,
        ResponseType.helpful_answer,
    }


def test_pair_type_values():
    assert set(PairType) == {PairType.safe_completion, PairType.baseline_refusal}


# ---------------------------------------------------------------------------
# PromptRecord
# ---------------------------------------------------------------------------


def test_prompt_record_required_id():
    r = PromptRecord(
        prompt_id="wildchat_001",
        prompt="Hello",
        category=PromptCategory.benign,
        source="wildchat",
    )
    assert r.prompt_id == "wildchat_001"


def test_prompt_record_missing_id_raises():
    with pytest.raises(ValidationError):
        PromptRecord(prompt="Hello", category=PromptCategory.benign, source="test")


def test_prompt_record_defaults():
    r = PromptRecord(
        prompt_id="p1",
        prompt="Hello",
        category=PromptCategory.benign,
        source="test",
    )
    assert r.source_id is None
    assert r.created_at is None
    assert r.metadata == {}


def test_prompt_record_invalid_category():
    with pytest.raises(ValidationError):
        PromptRecord(
            prompt_id="p1",
            prompt="test",
            category="not_a_category",
            source="test",
        )


def test_prompt_record_metadata():
    r = PromptRecord(
        prompt_id="wc_042",
        prompt="How do I pick a lock?",
        category=PromptCategory.dual_use,
        source="wildchat",
        metadata={"split": "train", "language": "en"},
    )
    assert r.metadata["split"] == "train"


# ---------------------------------------------------------------------------
# ResponseRecord
# ---------------------------------------------------------------------------


def test_response_record_required_id():
    r = ResponseRecord(
        response_id="resp_001",
        prompt_id="p1",
        response="I can't help with that.",
        response_type=ResponseType.hard_refusal,
        model="meta-llama/Llama-3.1-8B-Instruct",
    )
    assert r.response_id == "resp_001"


def test_response_record_missing_id_raises():
    with pytest.raises(ValidationError):
        ResponseRecord(
            prompt_id="p1",
            response="...",
            response_type=ResponseType.hard_refusal,
            model="test-model",
        )


def test_response_record_defaults():
    r = ResponseRecord(
        response_id="r1",
        prompt_id="p1",
        response="I can't help with that.",
        response_type=ResponseType.hard_refusal,
        model="meta-llama/Llama-3.1-8B-Instruct",
    )
    assert r.temperature is None
    assert r.created_at is None
    assert r.metadata == {}


def test_response_record_roundtrip():
    r = ResponseRecord(
        response_id="r1",
        prompt_id="p1",
        response="Here is a safe answer.",
        response_type=ResponseType.safe_completion,
        model="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.7,
    )
    r2 = ResponseRecord.model_validate(r.model_dump())
    assert r == r2


def test_response_record_invalid_type():
    with pytest.raises(ValidationError):
        ResponseRecord(
            response_id="r1",
            prompt_id="p1",
            response="...",
            response_type="made_up_type",
            model="test-model",
        )


# ---------------------------------------------------------------------------
# ValidationRecord
# ---------------------------------------------------------------------------


def test_validation_record_required_id():
    v = ValidationRecord(
        validation_id="val_001",
        response_id="r1",
        prompt_id="p1",
        is_valid=True,
        validator="human",
    )
    assert v.validation_id == "val_001"


def test_validation_record_missing_id_raises():
    with pytest.raises(ValidationError):
        ValidationRecord(
            response_id="r1",
            prompt_id="p1",
            is_valid=True,
            validator="human",
        )


def test_validation_record_keeps_prompt_id():
    v = ValidationRecord(
        validation_id="val_001",
        response_id="r1",
        prompt_id="p1",
        is_valid=True,
        validator="human",
    )
    assert v.prompt_id == "p1"


def test_validation_record_with_issues():
    v = ValidationRecord(
        validation_id="val_002",
        response_id="r1",
        prompt_id="p1",
        is_valid=False,
        issues=["unsafe_compliance detected", "no refusal signal"],
        validator="rule-based",
    )
    assert len(v.issues) == 2


def test_validation_record_empty_issues_default():
    v = ValidationRecord(
        validation_id="val_003",
        response_id="r1",
        prompt_id="p1",
        is_valid=True,
        validator="human",
    )
    assert v.issues == []


# ---------------------------------------------------------------------------
# PreferencePair
# ---------------------------------------------------------------------------


def test_preference_pair_safe_completion():
    p = PreferencePair(
        pair_id="pair_001",
        prompt_id="p1",
        chosen_id="r_safe",
        rejected_id="r_refusal",
        pair_type=PairType.safe_completion,
    )
    assert p.pair_type == PairType.safe_completion


def test_preference_pair_baseline_refusal():
    p = PreferencePair(
        pair_id="pair_002",
        prompt_id="p1",
        chosen_id="r_refusal",
        rejected_id="r_unsafe",
        pair_type=PairType.baseline_refusal,
    )
    assert p.pair_type == PairType.baseline_refusal


def test_preference_pair_required_id():
    p = PreferencePair(
        pair_id="pair_003",
        prompt_id="p1",
        chosen_id="r1",
        rejected_id="r2",
        pair_type=PairType.safe_completion,
    )
    assert p.pair_id == "pair_003"


def test_preference_pair_missing_id_raises():
    with pytest.raises(ValidationError):
        PreferencePair(
            prompt_id="p1",
            chosen_id="r1",
            rejected_id="r2",
            pair_type=PairType.safe_completion,
        )


def test_preference_pair_invalid_type():
    with pytest.raises(ValidationError):
        PreferencePair(
            pair_id="pair_004",
            prompt_id="p1",
            chosen_id="r1",
            rejected_id="r2",
            pair_type="not_a_type",
        )
