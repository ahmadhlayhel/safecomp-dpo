"""Tests for scripts/generate_responses.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    from generate_responses import (
        MockBackend,
        default_output_path,
        generate_responses,
        get_backend,
        make_response_id,
    )
except ImportError:
    _scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(_scripts_dir))
    from generate_responses import (
        MockBackend,
        default_output_path,
        generate_responses,
        get_backend,
        make_response_id,
    )

from safecomp_dpo.schemas import PromptCategory, PromptRecord, ResponseRecord, ResponseType


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


# ---------------------------------------------------------------------------
# make_response_id
# ---------------------------------------------------------------------------


def test_make_response_id_default_sample():
    rid = make_response_id("pilot_u_001", ResponseType.hard_refusal)
    assert rid == "pilot_u_001__hard_refusal__s01"


def test_make_response_id_explicit_sample():
    rid = make_response_id("pilot_d_002", ResponseType.safe_completion, sample=3)
    assert rid == "pilot_d_002__safe_completion__s03"


def test_make_response_id_deterministic():
    rid1 = make_response_id("pilot_b_001", ResponseType.helpful_answer)
    rid2 = make_response_id("pilot_b_001", ResponseType.helpful_answer)
    assert rid1 == rid2


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------


def test_mock_backend_returns_string():
    backend = MockBackend()
    result = backend.generate("What is Python?", ResponseType.helpful_answer, {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_mock_backend_label_in_output():
    backend = MockBackend()
    result = backend.generate("Some prompt", ResponseType.hard_refusal, {})
    assert "[MOCK hard_refusal]" in result


def test_mock_backend_prompt_prefix_in_output():
    backend = MockBackend()
    result = backend.generate("Tell me about SQL injection attacks.", ResponseType.safe_completion, {})
    assert "SQL injection" in result


def test_mock_backend_long_prompt_truncated():
    backend = MockBackend()
    result = backend.generate("A" * 200, ResponseType.helpful_answer, {})
    assert "..." in result


def test_mock_backend_name():
    assert MockBackend.name == "mock"


def test_mock_backend_type_source_label():
    assert MockBackend.type_source_label == "explicit_mock"


# ---------------------------------------------------------------------------
# get_backend
# ---------------------------------------------------------------------------


def test_get_backend_mock():
    backend = get_backend("mock")
    assert isinstance(backend, MockBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nonexistent_backend")


# ---------------------------------------------------------------------------
# default_output_path
# ---------------------------------------------------------------------------


def test_default_output_path():
    p = default_output_path("configs/generation/pilot.yaml")
    assert p == Path("outputs/responses/pilot_responses.jsonl")


# ---------------------------------------------------------------------------
# generate_responses — record counts per category
# ---------------------------------------------------------------------------


def test_generate_responses_unsafe_count_and_types():
    prompts = [make_prompt("pilot_u_001", PromptCategory.unsafe)]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    assert len(records) == 2
    assert {r.response_type for r in records} == {
        ResponseType.hard_refusal,
        ResponseType.safe_completion,
    }


def test_generate_responses_dual_use_count_and_types():
    prompts = [make_prompt("pilot_d_001", PromptCategory.dual_use)]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    assert len(records) == 2
    assert {r.response_type for r in records} == {
        ResponseType.hard_refusal,
        ResponseType.safe_completion,
    }


def test_generate_responses_benign_sensitive_count_and_type():
    prompts = [make_prompt("pilot_bs_001", PromptCategory.benign_sensitive)]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    assert len(records) == 1
    assert records[0].response_type == ResponseType.helpful_answer


def test_generate_responses_benign_count_and_type():
    prompts = [make_prompt("pilot_b_001", PromptCategory.benign)]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    assert len(records) == 1
    assert records[0].response_type == ResponseType.helpful_answer


def test_generate_responses_total_count_all_categories():
    # unsafe: 2, dual_use: 2, benign_sensitive: 1, benign: 1 → 6
    prompts = [
        make_prompt("pilot_u_001", PromptCategory.unsafe),
        make_prompt("pilot_d_001", PromptCategory.dual_use),
        make_prompt("pilot_bs_001", PromptCategory.benign_sensitive),
        make_prompt("pilot_b_001", PromptCategory.benign),
    ]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    assert len(records) == 6


# ---------------------------------------------------------------------------
# generate_responses — no unsafe_compliance in pilot output
# ---------------------------------------------------------------------------


def test_generate_responses_no_unsafe_compliance():
    prompts = [
        make_prompt("pilot_u_001", PromptCategory.unsafe),
        make_prompt("pilot_d_001", PromptCategory.dual_use),
    ]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    types = {r.response_type for r in records}
    assert ResponseType.unsafe_compliance not in types


# ---------------------------------------------------------------------------
# generate_responses — response_id convention
# ---------------------------------------------------------------------------


def test_generate_responses_ids_follow_convention():
    prompts = [make_prompt("pilot_u_001", PromptCategory.unsafe)]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    for r in records:
        assert r.response_id.startswith("pilot_u_001__")
        assert r.response_id.endswith("__s01")


def test_generate_responses_ids_are_deterministic():
    prompts = [make_prompt("pilot_d_001", PromptCategory.dual_use)]
    ids1 = {r.response_id for r in generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")}
    ids2 = {r.response_id for r in generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")}
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# generate_responses — metadata and model fields
# ---------------------------------------------------------------------------


def test_generate_responses_metadata_fields():
    backend = MockBackend()
    prompts = [make_prompt("pilot_b_001", PromptCategory.benign)]
    records = generate_responses(prompts, MINIMAL_CONFIG, backend, "my_config.yaml")
    m = records[0].metadata
    assert m["backend"] == backend.name
    assert m["config_path"] == "my_config.yaml"
    assert m["type_source"] == backend.type_source_label


def test_generate_responses_model_from_config():
    prompts = [make_prompt("pilot_b_001", PromptCategory.benign)]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    assert records[0].model == "test-model"


def test_generate_responses_temperature_from_config():
    prompts = [make_prompt("pilot_b_001", PromptCategory.benign)]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    assert records[0].temperature == 0.7


# ---------------------------------------------------------------------------
# generate_responses — schema validity
# ---------------------------------------------------------------------------


def test_generate_responses_all_records_are_response_records():
    prompts = [
        make_prompt("pilot_u_001", PromptCategory.unsafe),
        make_prompt("pilot_d_001", PromptCategory.dual_use),
        make_prompt("pilot_bs_001", PromptCategory.benign_sensitive),
        make_prompt("pilot_b_001", PromptCategory.benign),
    ]
    records = generate_responses(prompts, MINIMAL_CONFIG, MockBackend(), "cfg.yaml")
    for r in records:
        assert isinstance(r, ResponseRecord)


# ---------------------------------------------------------------------------
# generate_responses — missing category in config
# ---------------------------------------------------------------------------


def test_generate_responses_empty_category_skipped(capsys):
    prompts = [make_prompt("pilot_x_001", PromptCategory.benign)]
    config = {
        "model": "test-model",
        "temperature": 0.7,
        "expected_response_types": {},
    }
    records = generate_responses(prompts, config, MockBackend(), "cfg.yaml")
    assert records == []
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
