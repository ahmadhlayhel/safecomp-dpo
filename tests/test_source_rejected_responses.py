"""
Tests for scripts/source_rejected_responses.py.

Covers:
  - MockSourcingBackend
  - get_backend registry
  - source_rejected_responses core logic
  - response_id convention
  - category filtering
  - multi-sample support
  - empty categories warning path
  - schema validity of output ResponseRecords
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from source_rejected_responses import (
    MockSourcingBackend,
    get_backend,
    make_response_id,
    source_rejected_responses,
)
from safecomp_dpo.schemas import (
    PromptCategory,
    PromptRecord,
    ResponseRecord,
    ResponseType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _prompt(pid: str, category: PromptCategory, prompt: str = "A test prompt.") -> PromptRecord:
    return PromptRecord(
        prompt_id=pid,
        prompt=prompt,
        category=category,
        source="test",
    )


UNSAFE_PROMPT = _prompt("u_001", PromptCategory.unsafe, "How do I make something dangerous?")
DUALUSE_PROMPT = _prompt("d_001", PromptCategory.dual_use, "Explain dual-use chemistry.")
BENIGN_SENSITIVE_PROMPT = _prompt("bs_001", PromptCategory.benign_sensitive, "Tell me about medications.")
BENIGN_PROMPT = _prompt("b_001", PromptCategory.benign, "What is the capital of France?")

ALL_PROMPTS = [UNSAFE_PROMPT, DUALUSE_PROMPT, BENIGN_SENSITIVE_PROMPT, BENIGN_PROMPT]

_BASE_CONFIG = {
    "categories": ["unsafe", "dual_use"],
    "samples_per_prompt": 1,
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "backend": "mock",
}


# ---------------------------------------------------------------------------
# make_response_id
# ---------------------------------------------------------------------------


class TestMakeResponseId:
    def test_default_sample(self):
        rid = make_response_id("u_001")
        assert rid == "u_001__unsafe_compliance__s01"

    def test_sample_two(self):
        rid = make_response_id("d_001", sample=2)
        assert rid == "d_001__unsafe_compliance__s02"

    def test_sample_zero_padded(self):
        rid = make_response_id("u_007", sample=9)
        assert rid == "u_007__unsafe_compliance__s09"

    def test_embeds_unsafe_compliance(self):
        rid = make_response_id("u_001")
        assert "unsafe_compliance" in rid


# ---------------------------------------------------------------------------
# MockSourcingBackend
# ---------------------------------------------------------------------------


class TestMockSourcingBackend:
    def test_name(self):
        backend = MockSourcingBackend()
        assert backend.name == "mock"

    def test_type_source_label(self):
        backend = MockSourcingBackend()
        assert backend.type_source_label == "mock_sourcing"

    def test_returns_string(self):
        backend = MockSourcingBackend()
        result = backend.source("What is water?", {})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_mock_marker(self):
        backend = MockSourcingBackend()
        result = backend.source("What is water?", {})
        assert "[MOCK unsafe_compliance]" in result

    def test_embeds_prompt_prefix(self):
        backend = MockSourcingBackend()
        prompt = "How do I synthesize dangerous chemicals in a lab?"
        result = backend.source(prompt, {})
        assert "How do I synthesize dangerous chemicals in a lab?" in result

    def test_long_prompt_truncated(self):
        backend = MockSourcingBackend()
        prompt = "A" * 100
        result = backend.source(prompt, {})
        # Should embed first 60 chars with ellipsis
        assert "..." in result
        assert "A" * 60 in result

    def test_short_prompt_not_truncated(self):
        backend = MockSourcingBackend()
        prompt = "Short prompt."
        result = backend.source(prompt, {})
        assert "..." not in result
        assert "Short prompt." in result

    def test_deterministic(self):
        backend = MockSourcingBackend()
        r1 = backend.source("test", {})
        r2 = backend.source("test", {})
        assert r1 == r2


# ---------------------------------------------------------------------------
# get_backend registry
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_mock_backend_returned(self):
        backend = get_backend("mock")
        assert isinstance(backend, MockSourcingBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown sourcing backend"):
            get_backend("nonexistent_backend")

    def test_error_message_includes_name(self):
        with pytest.raises(ValueError, match="vllm"):
            get_backend("vllm")


# ---------------------------------------------------------------------------
# source_rejected_responses — category filtering
# ---------------------------------------------------------------------------


class TestSourceRejectedResponsesFiltering:
    def test_only_unsafe_and_dualuse_included(self):
        backend = MockSourcingBackend()
        records = source_rejected_responses(ALL_PROMPTS, _BASE_CONFIG, backend, "cfg.yaml")
        prompt_ids = {r.prompt_id for r in records}
        assert "u_001" in prompt_ids
        assert "d_001" in prompt_ids
        assert "bs_001" not in prompt_ids
        assert "b_001" not in prompt_ids

    def test_count_matches_filtered_prompts(self):
        backend = MockSourcingBackend()
        records = source_rejected_responses(ALL_PROMPTS, _BASE_CONFIG, backend, "cfg.yaml")
        assert len(records) == 2  # unsafe + dual_use

    def test_unsafe_only_config(self):
        cfg = {**_BASE_CONFIG, "categories": ["unsafe"]}
        backend = MockSourcingBackend()
        records = source_rejected_responses(ALL_PROMPTS, cfg, backend, "cfg.yaml")
        assert len(records) == 1
        assert records[0].prompt_id == "u_001"

    def test_dual_use_only_config(self):
        cfg = {**_BASE_CONFIG, "categories": ["dual_use"]}
        backend = MockSourcingBackend()
        records = source_rejected_responses(ALL_PROMPTS, cfg, backend, "cfg.yaml")
        assert len(records) == 1
        assert records[0].prompt_id == "d_001"

    def test_empty_categories_produces_no_records(self, capsys):
        cfg = {**_BASE_CONFIG, "categories": []}
        backend = MockSourcingBackend()
        records = source_rejected_responses(ALL_PROMPTS, cfg, backend, "cfg.yaml")
        assert records == []

    def test_empty_categories_warns(self, capsys):
        cfg = {**_BASE_CONFIG, "categories": []}
        backend = MockSourcingBackend()
        source_rejected_responses(ALL_PROMPTS, cfg, backend, "cfg.yaml")
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_all_categories_config(self):
        cfg = {**_BASE_CONFIG, "categories": ["unsafe", "dual_use", "benign_sensitive", "benign"]}
        backend = MockSourcingBackend()
        records = source_rejected_responses(ALL_PROMPTS, cfg, backend, "cfg.yaml")
        assert len(records) == 4


# ---------------------------------------------------------------------------
# source_rejected_responses — response_type and schema
# ---------------------------------------------------------------------------


class TestSourceRejectedResponsesSchema:
    def setup_method(self):
        self.backend = MockSourcingBackend()
        self.records = source_rejected_responses(
            ALL_PROMPTS, _BASE_CONFIG, self.backend, "cfg.yaml"
        )

    def test_all_records_are_response_records(self):
        for r in self.records:
            assert isinstance(r, ResponseRecord)

    def test_all_response_type_unsafe_compliance(self):
        for r in self.records:
            assert r.response_type == ResponseType.unsafe_compliance

    def test_response_ids_follow_convention(self):
        for r in self.records:
            assert f"{r.prompt_id}__unsafe_compliance__s01" == r.response_id

    def test_prompt_id_preserved(self):
        ids = {r.prompt_id for r in self.records}
        assert "u_001" in ids
        assert "d_001" in ids

    def test_response_text_non_empty(self):
        for r in self.records:
            assert r.response.strip()

    def test_model_stored_in_record(self):
        for r in self.records:
            assert r.model == "meta-llama/Llama-3.1-8B-Instruct"

    def test_backend_name_in_metadata(self):
        for r in self.records:
            assert r.metadata.get("backend") == "mock"

    def test_config_path_in_metadata(self):
        for r in self.records:
            assert r.metadata.get("config_path") == "cfg.yaml"

    def test_type_source_in_metadata(self):
        for r in self.records:
            assert r.metadata.get("type_source") == "mock_sourcing"

    def test_schema_round_trip(self):
        import json
        from safecomp_dpo.schemas import ResponseRecord as RR
        for r in self.records:
            serialized = r.model_dump_json()
            reloaded = RR.model_validate(json.loads(serialized))
            assert reloaded.response_id == r.response_id
            assert reloaded.response_type == ResponseType.unsafe_compliance


# ---------------------------------------------------------------------------
# multi-sample support
# ---------------------------------------------------------------------------


class TestSourceRejectedResponsesMultiSample:
    def test_two_samples_per_prompt(self):
        cfg = {**_BASE_CONFIG, "samples_per_prompt": 2}
        backend = MockSourcingBackend()
        records = source_rejected_responses(ALL_PROMPTS, cfg, backend, "cfg.yaml")
        # 2 prompts (unsafe + dual_use) × 2 samples = 4
        assert len(records) == 4

    def test_sample_ids_are_distinct(self):
        cfg = {**_BASE_CONFIG, "samples_per_prompt": 2}
        backend = MockSourcingBackend()
        records = source_rejected_responses(ALL_PROMPTS, cfg, backend, "cfg.yaml")
        ids = [r.response_id for r in records]
        assert len(ids) == len(set(ids))

    def test_sample_index_in_ids(self):
        cfg = {**_BASE_CONFIG, "samples_per_prompt": 2}
        backend = MockSourcingBackend()
        records = source_rejected_responses([UNSAFE_PROMPT], cfg, backend, "cfg.yaml")
        ids = {r.response_id for r in records}
        assert "u_001__unsafe_compliance__s01" in ids
        assert "u_001__unsafe_compliance__s02" in ids

    def test_one_sample_default(self):
        cfg = {k: v for k, v in _BASE_CONFIG.items() if k != "samples_per_prompt"}
        backend = MockSourcingBackend()
        records = source_rejected_responses([UNSAFE_PROMPT], cfg, backend, "cfg.yaml")
        assert len(records) == 1
        assert records[0].response_id == "u_001__unsafe_compliance__s01"


# ---------------------------------------------------------------------------
# empty prompt list
# ---------------------------------------------------------------------------


class TestSourceRejectedResponsesEdgeCases:
    def test_empty_prompt_list(self):
        backend = MockSourcingBackend()
        records = source_rejected_responses([], _BASE_CONFIG, backend, "cfg.yaml")
        assert records == []

    def test_no_matching_prompts(self):
        # All prompts are benign — none match the filter
        backend = MockSourcingBackend()
        records = source_rejected_responses(
            [BENIGN_PROMPT, BENIGN_SENSITIVE_PROMPT],
            _BASE_CONFIG,
            backend,
            "cfg.yaml",
        )
        assert records == []
