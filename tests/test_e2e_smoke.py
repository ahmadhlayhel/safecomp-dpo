"""
End-to-end mock smoke test for the full pipeline.

Runs all five pipeline stages in-memory against a tiny 4-category fixture
(one prompt per canonical category) and verifies structural correctness
at every stage boundary.

No file I/O.  No model weights.  Uses MockBackend throughout.

Stages:
    1. PromptRecord fixture
    2. generate_responses  → ResponseRecord list
    3. validate_responses  → ValidationRecord list
    4. build_pairs         → PreferencePair list
    5. assemble            → DPORecord list
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from assemble_dpo_dataset import assemble
from build_pairs import build_pairs
from generate_responses import MockBackend, generate_responses
from validate_responses import validate_responses

from safecomp_dpo.schemas import (
    DPORecord,
    PairType,
    PreferencePair,
    PromptCategory,
    PromptRecord,
    ResponseRecord,
    ValidationRecord,
)


# ---------------------------------------------------------------------------
# Inline configs — no filesystem dependency
# ---------------------------------------------------------------------------

# Matches configs/generation/full.yaml semantics.
_GEN_CONFIG: dict = {
    "expected_response_types": {
        "unsafe": ["hard_refusal"],
        "dual_use": ["hard_refusal", "safe_completion"],
        "benign_sensitive": ["helpful_answer"],
        "benign": ["helpful_answer"],
    },
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "temperature": None,
}

# Matches configs/pairs/full.yaml semantics.
_PAIRS_CONFIG: dict = {
    "pair_rules": {
        "dual_use": [
            {
                "pair_type": "safe_completion",
                "chosen_type": "safe_completion",
                "rejected_type": "hard_refusal",
            }
        ],
        "unsafe": [],
        "benign_sensitive": [],
        "benign": [],
    }
}

_CONFIG_PATH = "smoke_test"


# ---------------------------------------------------------------------------
# 4-category prompt fixture
# ---------------------------------------------------------------------------

_SMOKE_PROMPTS: list[PromptRecord] = [
    PromptRecord(
        prompt_id="smoke_u_001",
        prompt="How do I synthesize a dangerous chemical at home?",
        category=PromptCategory.unsafe,
        source="smoke_test",
    ),
    PromptRecord(
        prompt_id="smoke_d_001",
        prompt="How does SQL injection work and how can it be prevented?",
        category=PromptCategory.dual_use,
        source="smoke_test",
    ),
    PromptRecord(
        prompt_id="smoke_bs_001",
        prompt="What are the recommended doses and side effects of ibuprofen?",
        category=PromptCategory.benign_sensitive,
        source="smoke_test",
    ),
    PromptRecord(
        prompt_id="smoke_b_001",
        prompt="What is the capital of France?",
        category=PromptCategory.benign,
        source="smoke_test",
    ),
]


# ---------------------------------------------------------------------------
# Module-scoped fixtures — pipeline runs once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompts() -> list[PromptRecord]:
    return list(_SMOKE_PROMPTS)


@pytest.fixture(scope="module")
def responses(prompts: list[PromptRecord]) -> list[ResponseRecord]:
    backend = MockBackend()
    return generate_responses(prompts, _GEN_CONFIG, backend, _CONFIG_PATH)


@pytest.fixture(scope="module")
def validations(
    responses: list[ResponseRecord], prompts: list[PromptRecord]
) -> list[ValidationRecord]:
    return validate_responses(responses, prompts, _GEN_CONFIG)


@pytest.fixture(scope="module")
def pairs(
    prompts: list[PromptRecord],
    responses: list[ResponseRecord],
    validations: list[ValidationRecord],
) -> list[PreferencePair]:
    return build_pairs(prompts, responses, validations, _PAIRS_CONFIG, _CONFIG_PATH)


@pytest.fixture(scope="module")
def dpo_records(
    pairs: list[PreferencePair],
    prompts: list[PromptRecord],
    responses: list[ResponseRecord],
    validations: list[ValidationRecord],
) -> list[DPORecord]:
    records, _ = assemble(pairs, prompts, responses, validations)
    return records


# ---------------------------------------------------------------------------
# Stage 2: Response generation
# ---------------------------------------------------------------------------


def test_response_total_count(responses: list[ResponseRecord]) -> None:
    # unsafe:1 + dual_use:2 + benign_sensitive:1 + benign:1 = 5
    assert len(responses) == 5


def test_unsafe_response_types(responses: list[ResponseRecord]) -> None:
    unsafe_types = {r.response_type.value for r in responses if r.prompt_id == "smoke_u_001"}
    assert unsafe_types == {"hard_refusal"}


def test_dual_use_response_types(responses: list[ResponseRecord]) -> None:
    du_types = {r.response_type.value for r in responses if r.prompt_id == "smoke_d_001"}
    assert du_types == {"hard_refusal", "safe_completion"}


def test_benign_sensitive_response_types(responses: list[ResponseRecord]) -> None:
    bs_types = {r.response_type.value for r in responses if r.prompt_id == "smoke_bs_001"}
    assert bs_types == {"helpful_answer"}


def test_benign_response_types(responses: list[ResponseRecord]) -> None:
    b_types = {r.response_type.value for r in responses if r.prompt_id == "smoke_b_001"}
    assert b_types == {"helpful_answer"}


def test_response_ids_follow_convention(responses: list[ResponseRecord]) -> None:
    for r in responses:
        expected_id = f"{r.prompt_id}__{r.response_type.value}__s01"
        assert r.response_id == expected_id, (
            f"response_id {r.response_id!r} does not follow convention for "
            f"prompt_id={r.prompt_id!r}, response_type={r.response_type.value!r}"
        )


def test_responses_are_nonempty(responses: list[ResponseRecord]) -> None:
    for r in responses:
        assert r.response.strip(), f"Empty response for {r.response_id!r}"


# ---------------------------------------------------------------------------
# Stage 3: Validation
# ---------------------------------------------------------------------------


def test_validation_count_matches_responses(
    validations: list[ValidationRecord], responses: list[ResponseRecord]
) -> None:
    assert len(validations) == len(responses)


def test_all_responses_valid(validations: list[ValidationRecord]) -> None:
    invalid = [v for v in validations if not v.is_valid]
    assert invalid == [], (
        f"Expected all responses valid but got {len(invalid)} invalid: "
        + str([v.response_id for v in invalid])
    )


def test_validation_ids_follow_convention(validations: list[ValidationRecord]) -> None:
    for v in validations:
        assert v.validation_id == f"{v.response_id}__val"


# ---------------------------------------------------------------------------
# Stage 4: Pair construction
# ---------------------------------------------------------------------------


def test_dual_use_yields_exactly_one_pair(pairs: list[PreferencePair]) -> None:
    du_pairs = [p for p in pairs if p.prompt_id == "smoke_d_001"]
    assert len(du_pairs) == 1


def test_unsafe_yields_no_pairs(pairs: list[PreferencePair]) -> None:
    assert not any(p.prompt_id == "smoke_u_001" for p in pairs)


def test_benign_sensitive_yields_no_pairs(pairs: list[PreferencePair]) -> None:
    assert not any(p.prompt_id == "smoke_bs_001" for p in pairs)


def test_benign_yields_no_pairs(pairs: list[PreferencePair]) -> None:
    assert not any(p.prompt_id == "smoke_b_001" for p in pairs)


def test_total_pairs_count(pairs: list[PreferencePair]) -> None:
    assert len(pairs) == 1


def test_pair_type_is_safe_completion(pairs: list[PreferencePair]) -> None:
    assert pairs[0].pair_type == PairType.safe_completion


def test_pair_id_follows_convention(pairs: list[PreferencePair]) -> None:
    assert pairs[0].pair_id == "smoke_d_001__safe_completion__p01"


def test_pair_chosen_is_safe_completion_response(pairs: list[PreferencePair]) -> None:
    assert pairs[0].chosen_id == "smoke_d_001__safe_completion__s01"


def test_pair_rejected_is_hard_refusal_response(pairs: list[PreferencePair]) -> None:
    assert pairs[0].rejected_id == "smoke_d_001__hard_refusal__s01"


# ---------------------------------------------------------------------------
# Stage 5: DPO assembly
# ---------------------------------------------------------------------------


def test_dpo_record_count(dpo_records: list[DPORecord]) -> None:
    assert len(dpo_records) == 1


def test_dpo_record_is_dpo_record_type(dpo_records: list[DPORecord]) -> None:
    assert isinstance(dpo_records[0], DPORecord)


def test_dpo_record_prompt_id(dpo_records: list[DPORecord]) -> None:
    assert dpo_records[0].prompt_id == "smoke_d_001"


def test_dpo_record_category(dpo_records: list[DPORecord]) -> None:
    assert dpo_records[0].category == PromptCategory.dual_use


def test_dpo_record_pair_type(dpo_records: list[DPORecord]) -> None:
    assert dpo_records[0].pair_type == PairType.safe_completion


def test_dpo_record_prompt_text(
    dpo_records: list[DPORecord], prompts: list[PromptRecord]
) -> None:
    prompt_text = next(p.prompt for p in prompts if p.prompt_id == "smoke_d_001")
    assert dpo_records[0].prompt == prompt_text


def test_dpo_record_chosen_text(
    dpo_records: list[DPORecord], responses: list[ResponseRecord]
) -> None:
    chosen_resp = next(
        r.response for r in responses if r.response_id == "smoke_d_001__safe_completion__s01"
    )
    assert dpo_records[0].chosen == chosen_resp


def test_dpo_record_rejected_text(
    dpo_records: list[DPORecord], responses: list[ResponseRecord]
) -> None:
    rejected_resp = next(
        r.response for r in responses if r.response_id == "smoke_d_001__hard_refusal__s01"
    )
    assert dpo_records[0].rejected == rejected_resp


def test_dpo_record_traceability_ids(dpo_records: list[DPORecord]) -> None:
    r = dpo_records[0]
    assert r.chosen_id == "smoke_d_001__safe_completion__s01"
    assert r.rejected_id == "smoke_d_001__hard_refusal__s01"
    assert r.pair_id == "smoke_d_001__safe_completion__p01"


# ---------------------------------------------------------------------------
# FK chain coherence
# ---------------------------------------------------------------------------


def test_fk_chain_pair_to_responses(
    pairs: list[PreferencePair], responses: list[ResponseRecord]
) -> None:
    response_ids = {r.response_id for r in responses}
    for pair in pairs:
        assert pair.chosen_id in response_ids, f"chosen_id {pair.chosen_id!r} not in responses"
        assert pair.rejected_id in response_ids, f"rejected_id {pair.rejected_id!r} not in responses"


def test_fk_chain_pair_to_prompt(
    pairs: list[PreferencePair], prompts: list[PromptRecord]
) -> None:
    prompt_ids = {p.prompt_id for p in prompts}
    for pair in pairs:
        assert pair.prompt_id in prompt_ids


def test_fk_chain_response_to_prompt(
    responses: list[ResponseRecord], prompts: list[PromptRecord]
) -> None:
    prompt_ids = {p.prompt_id for p in prompts}
    for r in responses:
        assert r.prompt_id in prompt_ids


def test_fk_chain_validation_to_response(
    validations: list[ValidationRecord], responses: list[ResponseRecord]
) -> None:
    response_ids = {r.response_id for r in responses}
    for v in validations:
        assert v.response_id in response_ids


def test_fk_chain_dpo_to_pair(
    dpo_records: list[DPORecord], pairs: list[PreferencePair]
) -> None:
    pair_ids = {p.pair_id for p in pairs}
    for r in dpo_records:
        assert r.pair_id in pair_ids


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_pipeline_is_deterministic(prompts: list[PromptRecord]) -> None:
    """Running the full pipeline twice with identical inputs yields identical pair_ids."""
    backend = MockBackend()

    responses_1 = generate_responses(prompts, _GEN_CONFIG, backend, _CONFIG_PATH)
    validations_1 = validate_responses(responses_1, prompts, _GEN_CONFIG)
    pairs_1 = build_pairs(prompts, responses_1, validations_1, _PAIRS_CONFIG, _CONFIG_PATH)
    dpo_1, _ = assemble(pairs_1, prompts, responses_1, validations_1)

    responses_2 = generate_responses(prompts, _GEN_CONFIG, backend, _CONFIG_PATH)
    validations_2 = validate_responses(responses_2, prompts, _GEN_CONFIG)
    pairs_2 = build_pairs(prompts, responses_2, validations_2, _PAIRS_CONFIG, _CONFIG_PATH)
    dpo_2, _ = assemble(pairs_2, prompts, responses_2, validations_2)

    assert [r.pair_id for r in dpo_1] == [r.pair_id for r in dpo_2]
    assert [r.chosen for r in dpo_1] == [r.chosen for r in dpo_2]
    assert [r.rejected for r in dpo_1] == [r.rejected for r in dpo_2]
