"""Tests for safecomp_dpo.io."""

import pytest

from safecomp_dpo.io import (
    load_preference_pairs,
    load_prompt_records,
    load_response_records,
    load_validation_records,
    read_jsonl,
    write_jsonl,
)
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
# read_jsonl / write_jsonl — raw dicts
# ---------------------------------------------------------------------------


def test_write_read_raw_dicts(tmp_path):
    records = [{"a": 1}, {"b": "hello"}]
    path = tmp_path / "test.jsonl"
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert loaded == records


def test_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert read_jsonl(path) == []


def test_write_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "out.jsonl"
    write_jsonl([{"x": 1}], path)
    assert path.exists()


def test_one_record_per_line(tmp_path):
    records = [{"i": i} for i in range(5)]
    path = tmp_path / "multi.jsonl"
    write_jsonl(records, path)
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 5


# ---------------------------------------------------------------------------
# Typed loaders — PromptRecord
# ---------------------------------------------------------------------------


def test_write_load_prompt_records(tmp_path):
    prompts = [
        PromptRecord(
            prompt_id="p_001",
            prompt="Hello world",
            category=PromptCategory.benign,
            source="manual",
        ),
        PromptRecord(
            prompt_id="wc_001",
            prompt="How do I pick a lock?",
            category=PromptCategory.dual_use,
            source="wildchat",
            source_id="wc_001",
        ),
    ]
    path = tmp_path / "prompts.jsonl"
    write_jsonl(prompts, path)
    loaded = load_prompt_records(path)

    assert len(loaded) == 2
    assert loaded[0].prompt == "Hello world"
    assert loaded[0].category == PromptCategory.benign
    assert loaded[1].source_id == "wc_001"


def test_prompt_record_ids_preserved(tmp_path):
    p = PromptRecord(
        prompt_id="stable_id_abc",
        prompt="test",
        category=PromptCategory.benign,
        source="s",
    )
    path = tmp_path / "p.jsonl"
    write_jsonl([p], path)
    loaded = load_prompt_records(path)
    assert loaded[0].prompt_id == "stable_id_abc"


# ---------------------------------------------------------------------------
# Typed loaders — ResponseRecord
# ---------------------------------------------------------------------------


def test_write_load_response_records(tmp_path):
    responses = [
        ResponseRecord(
            response_id="r_001",
            prompt_id="p_001",
            response="I cannot help with that.",
            response_type=ResponseType.hard_refusal,
            model="meta-llama/Llama-3.1-8B-Instruct",
        )
    ]
    path = tmp_path / "responses.jsonl"
    write_jsonl(responses, path)
    loaded = load_response_records(path)
    assert loaded[0].response_type == ResponseType.hard_refusal
    assert loaded[0].prompt_id == "p_001"
    assert loaded[0].response_id == "r_001"


# ---------------------------------------------------------------------------
# Typed loaders — ValidationRecord
# ---------------------------------------------------------------------------


def test_write_load_validation_records(tmp_path):
    records = [
        ValidationRecord(
            validation_id="val_001",
            response_id="r_001",
            prompt_id="p_001",
            is_valid=False,
            issues=["unsafe content"],
            validator="human",
        )
    ]
    path = tmp_path / "validation.jsonl"
    write_jsonl(records, path)
    loaded = load_validation_records(path)
    assert loaded[0].is_valid is False
    assert loaded[0].issues == ["unsafe content"]
    assert loaded[0].prompt_id == "p_001"


# ---------------------------------------------------------------------------
# Typed loaders — PreferencePair
# ---------------------------------------------------------------------------


def test_write_load_preference_pairs(tmp_path):
    pairs = [
        PreferencePair(
            pair_id="pair_001",
            prompt_id="p_001",
            chosen_id="r_safe",
            rejected_id="r_refuse",
            pair_type=PairType.safe_completion,
        )
    ]
    path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs, path)
    loaded = load_preference_pairs(path)
    assert loaded[0].pair_type == PairType.safe_completion
    assert loaded[0].chosen_id == "r_safe"
    assert loaded[0].pair_id == "pair_001"
