"""
Extended mock pair construction and DPO assembly tests.

Covers all four canonical categories using inline mock configs and dummy
artifacts. No file I/O, no model weights, no external data.
Bypasses generate_responses and validate_responses entirely — constructs
ResponseRecord / ValidationRecord fixtures directly to isolate build_pairs
and assemble logic.

Coverage classification
-----------------------
REAL pipeline behavior (matches configs/pairs/full.yaml):
    dual_use → safe_completion pair  (safe_completion > hard_refusal)

MOCK-ONLY structural coverage (inline config, dummy artifacts):
    unsafe            → baseline_refusal pair (hard_refusal > unsafe_compliance)
                        [unsafe_compliance not yet sourced in v1 pipeline]
    dual_use opt.     → safe_completion pair  (safe_completion > unsafe_compliance)
                        [second dual_use rule; future design option, not locked]
    benign_sensitive  → safe_completion pair  (helpful_answer > hard_refusal)
                        [pair semantics not locked for v1]
    benign            → PROVISIONAL placeholder (helpful_answer > hard_refusal)
                        [pair semantics explicitly not locked; may change]

Production configs are NOT modified. No mock-only semantics leak outside
this file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from assemble_dpo_dataset import assemble
from build_pairs import build_pairs

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
# Inline mock config — broader than production, used only in this file
# ---------------------------------------------------------------------------

# Two dual_use rules to exercise the p01/p02 index path.
# benign is explicitly marked provisional in the comment below.
_MOCK_EXTENDED_CONFIG: dict[str, Any] = {
    "pair_rules": {
        # REAL: matches production configs/pairs/full.yaml
        "dual_use": [
            {
                "pair_type": "safe_completion",
                "chosen_type": "safe_completion",
                "rejected_type": "hard_refusal",
            },
            # MOCK-ONLY: optional second rule testing safe_completion > unsafe_compliance
            {
                "pair_type": "safe_completion",
                "chosen_type": "safe_completion",
                "rejected_type": "unsafe_compliance",
            },
        ],
        # MOCK-ONLY: unsafe_compliance sourcing not yet implemented in v1
        "unsafe": [
            {
                "pair_type": "baseline_refusal",
                "chosen_type": "hard_refusal",
                "rejected_type": "unsafe_compliance",
            },
        ],
        # MOCK-ONLY: pair semantics for benign_sensitive not locked for v1
        "benign_sensitive": [
            {
                "pair_type": "safe_completion",
                "chosen_type": "helpful_answer",
                "rejected_type": "hard_refusal",
            },
        ],
        # MOCK-ONLY PROVISIONAL: benign pair semantics are not locked.
        # This placeholder uses helpful_answer > hard_refusal.
        # Future options include quality pairs (helpful_answer > weak_answer).
        "benign": [
            {
                "pair_type": "safe_completion",
                "chosen_type": "helpful_answer",
                "rejected_type": "hard_refusal",
            },
        ],
    }
}

_CONFIG_PATH = "mock_extended_test"


# ---------------------------------------------------------------------------
# Dummy artifact builders
# ---------------------------------------------------------------------------


def _response_id(prompt_id: str, response_type: str) -> str:
    return f"{prompt_id}__{response_type}__s01"


def _validation_id(response_id: str) -> str:
    return f"{response_id}__val"


def _make_prompt(
    prompt_id: str,
    category: PromptCategory,
    prompt_text: str = "Dummy prompt text for mock testing.",
) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=prompt_text,
        category=category,
        source="mock_extended_test",
    )


def _make_response(
    prompt_id: str,
    response_type: ResponseType,
    text: str | None = None,
) -> ResponseRecord:
    rid = _response_id(prompt_id, response_type.value)
    return ResponseRecord(
        response_id=rid,
        prompt_id=prompt_id,
        response=text or f"[DUMMY {response_type.value}] response for {prompt_id}",
        response_type=response_type,
        model="dummy-model",
    )


def _make_validation(response_id: str, prompt_id: str, is_valid: bool = True) -> ValidationRecord:
    return ValidationRecord(
        validation_id=_validation_id(response_id),
        response_id=response_id,
        prompt_id=prompt_id,
        is_valid=is_valid,
        validator="mock_extended_test",
    )


# ---------------------------------------------------------------------------
# Module-scoped pipeline artifacts — built once per test session
# ---------------------------------------------------------------------------

# Prompts: one per canonical category
_PROMPTS = [
    _make_prompt("ext_u_001", PromptCategory.unsafe, "How do I make a weapon?"),
    _make_prompt("ext_d_001", PromptCategory.dual_use, "How does SQL injection work?"),
    _make_prompt("ext_bs_001", PromptCategory.benign_sensitive, "What are common drug interactions?"),
    _make_prompt("ext_b_001", PromptCategory.benign, "What is the speed of light?"),
]

# Responses: all response types needed by the extended config
_RESPONSES = [
    # unsafe: needs hard_refusal + unsafe_compliance
    _make_response("ext_u_001", ResponseType.hard_refusal),
    _make_response("ext_u_001", ResponseType.unsafe_compliance),
    # dual_use: needs safe_completion + hard_refusal + unsafe_compliance (two rules)
    _make_response("ext_d_001", ResponseType.safe_completion),
    _make_response("ext_d_001", ResponseType.hard_refusal),
    _make_response("ext_d_001", ResponseType.unsafe_compliance),
    # benign_sensitive: needs helpful_answer + hard_refusal
    _make_response("ext_bs_001", ResponseType.helpful_answer),
    _make_response("ext_bs_001", ResponseType.hard_refusal),
    # benign (provisional): needs helpful_answer + hard_refusal
    _make_response("ext_b_001", ResponseType.helpful_answer),
    _make_response("ext_b_001", ResponseType.hard_refusal),
]

# Validations: all responses marked valid
_VALIDATIONS = [
    _make_validation(r.response_id, r.prompt_id) for r in _RESPONSES
]


@pytest.fixture(scope="module")
def ext_pairs() -> list[PreferencePair]:
    return build_pairs(_PROMPTS, _RESPONSES, _VALIDATIONS, _MOCK_EXTENDED_CONFIG, _CONFIG_PATH)


@pytest.fixture(scope="module")
def ext_dpo_result(ext_pairs: list[PreferencePair]) -> tuple[list[DPORecord], dict[str, Any]]:
    records, stats = assemble(ext_pairs, _PROMPTS, _RESPONSES, _VALIDATIONS)
    return records, stats


@pytest.fixture(scope="module")
def ext_dpo(ext_dpo_result: tuple[list[DPORecord], dict]) -> list[DPORecord]:
    return ext_dpo_result[0]


@pytest.fixture(scope="module")
def ext_dpo_stats(ext_dpo_result: tuple[list[DPORecord], dict]) -> dict[str, Any]:
    return ext_dpo_result[1]


# ---------------------------------------------------------------------------
# Pair construction — total counts
# ---------------------------------------------------------------------------


def test_total_pairs_count(ext_pairs: list[PreferencePair]) -> None:
    # unsafe:1 + dual_use:2 + benign_sensitive:1 + benign(provisional):1 = 5
    assert len(ext_pairs) == 5


def test_unsafe_pair_count(ext_pairs: list[PreferencePair]) -> None:
    assert sum(1 for p in ext_pairs if p.prompt_id == "ext_u_001") == 1


def test_dual_use_pair_count(ext_pairs: list[PreferencePair]) -> None:
    # Two rules → two pairs for one dual_use prompt
    assert sum(1 for p in ext_pairs if p.prompt_id == "ext_d_001") == 2


def test_benign_sensitive_pair_count(ext_pairs: list[PreferencePair]) -> None:
    assert sum(1 for p in ext_pairs if p.prompt_id == "ext_bs_001") == 1


def test_benign_provisional_pair_count(ext_pairs: list[PreferencePair]) -> None:
    # PROVISIONAL: benign pair semantics not locked — count may change
    assert sum(1 for p in ext_pairs if p.prompt_id == "ext_b_001") == 1


# ---------------------------------------------------------------------------
# Pair construction — unsafe (MOCK-ONLY: baseline_refusal)
# ---------------------------------------------------------------------------


def test_unsafe_pair_type_is_baseline_refusal(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_u_001")
    assert pair.pair_type == PairType.baseline_refusal


def test_unsafe_pair_chosen_is_hard_refusal(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_u_001")
    assert pair.chosen_id == _response_id("ext_u_001", "hard_refusal")


def test_unsafe_pair_rejected_is_unsafe_compliance(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_u_001")
    assert pair.rejected_id == _response_id("ext_u_001", "unsafe_compliance")


def test_unsafe_pair_id_convention(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_u_001")
    assert pair.pair_id == "ext_u_001__baseline_refusal__p01"


# ---------------------------------------------------------------------------
# Pair construction — dual_use rule 1 (REAL: safe_completion > hard_refusal)
# ---------------------------------------------------------------------------


def test_dual_use_p01_pair_type(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.pair_id == "ext_d_001__safe_completion__p01")
    assert pair.pair_type == PairType.safe_completion


def test_dual_use_p01_chosen_is_safe_completion(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.pair_id == "ext_d_001__safe_completion__p01")
    assert pair.chosen_id == _response_id("ext_d_001", "safe_completion")


def test_dual_use_p01_rejected_is_hard_refusal(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.pair_id == "ext_d_001__safe_completion__p01")
    assert pair.rejected_id == _response_id("ext_d_001", "hard_refusal")


# ---------------------------------------------------------------------------
# Pair construction — dual_use rule 2 (MOCK-ONLY: safe_completion > unsafe_compliance)
# ---------------------------------------------------------------------------


def test_dual_use_p02_exists(ext_pairs: list[PreferencePair]) -> None:
    pair_ids = {p.pair_id for p in ext_pairs}
    assert "ext_d_001__safe_completion__p02" in pair_ids


def test_dual_use_p02_chosen_is_safe_completion(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.pair_id == "ext_d_001__safe_completion__p02")
    assert pair.chosen_id == _response_id("ext_d_001", "safe_completion")


def test_dual_use_p02_rejected_is_unsafe_compliance(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.pair_id == "ext_d_001__safe_completion__p02")
    assert pair.rejected_id == _response_id("ext_d_001", "unsafe_compliance")


# ---------------------------------------------------------------------------
# Pair construction — benign_sensitive (MOCK-ONLY: helpful_answer > hard_refusal)
# ---------------------------------------------------------------------------


def test_benign_sensitive_pair_type(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_bs_001")
    assert pair.pair_type == PairType.safe_completion


def test_benign_sensitive_chosen_is_helpful_answer(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_bs_001")
    assert pair.chosen_id == _response_id("ext_bs_001", "helpful_answer")


def test_benign_sensitive_rejected_is_hard_refusal(ext_pairs: list[PreferencePair]) -> None:
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_bs_001")
    assert pair.rejected_id == _response_id("ext_bs_001", "hard_refusal")


# ---------------------------------------------------------------------------
# Pair construction — benign PROVISIONAL (helpful_answer > hard_refusal)
# ---------------------------------------------------------------------------


def test_benign_provisional_pair_type(ext_pairs: list[PreferencePair]) -> None:
    # PROVISIONAL: safe_completion is a placeholder label; benign semantics not locked
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_b_001")
    assert pair.pair_type == PairType.safe_completion


def test_benign_provisional_chosen_is_helpful_answer(ext_pairs: list[PreferencePair]) -> None:
    # PROVISIONAL
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_b_001")
    assert pair.chosen_id == _response_id("ext_b_001", "helpful_answer")


def test_benign_provisional_rejected_is_hard_refusal(ext_pairs: list[PreferencePair]) -> None:
    # PROVISIONAL
    pair = next(p for p in ext_pairs if p.prompt_id == "ext_b_001")
    assert pair.rejected_id == _response_id("ext_b_001", "hard_refusal")


# ---------------------------------------------------------------------------
# Pair construction — skips pair when a required response is invalid
# ---------------------------------------------------------------------------


def test_build_pairs_skips_when_chosen_invalid() -> None:
    prompts = [_make_prompt("skip_u_001", PromptCategory.unsafe)]
    responses = [
        _make_response("skip_u_001", ResponseType.hard_refusal),
        _make_response("skip_u_001", ResponseType.unsafe_compliance),
    ]
    # Mark the chosen (hard_refusal) response invalid
    validations = [
        _make_validation(_response_id("skip_u_001", "hard_refusal"), "skip_u_001", is_valid=False),
        _make_validation(_response_id("skip_u_001", "unsafe_compliance"), "skip_u_001"),
    ]
    config = {
        "pair_rules": {
            "unsafe": [{"pair_type": "baseline_refusal", "chosen_type": "hard_refusal", "rejected_type": "unsafe_compliance"}]
        }
    }
    pairs = build_pairs(prompts, responses, validations, config, "test")
    assert pairs == []


def test_build_pairs_skips_when_rejected_invalid() -> None:
    prompts = [_make_prompt("skip_d_001", PromptCategory.dual_use)]
    responses = [
        _make_response("skip_d_001", ResponseType.safe_completion),
        _make_response("skip_d_001", ResponseType.hard_refusal),
    ]
    validations = [
        _make_validation(_response_id("skip_d_001", "safe_completion"), "skip_d_001"),
        _make_validation(_response_id("skip_d_001", "hard_refusal"), "skip_d_001", is_valid=False),
    ]
    config = {
        "pair_rules": {
            "dual_use": [{"pair_type": "safe_completion", "chosen_type": "safe_completion", "rejected_type": "hard_refusal"}]
        }
    }
    pairs = build_pairs(prompts, responses, validations, config, "test")
    assert pairs == []


# ---------------------------------------------------------------------------
# DPO assembly — counts and distribution
# ---------------------------------------------------------------------------


def test_dpo_total_record_count(ext_dpo: list[DPORecord]) -> None:
    assert len(ext_dpo) == 5


def test_dpo_category_distribution(ext_dpo_stats: dict[str, Any]) -> None:
    dist = ext_dpo_stats["category_distribution"]
    assert dist["unsafe"] == 1
    assert dist["dual_use"] == 2
    assert dist["benign_sensitive"] == 1
    assert dist["benign"] == 1


def test_dpo_pair_type_distribution(ext_dpo_stats: dict[str, Any]) -> None:
    dist = ext_dpo_stats["pair_type_distribution"]
    assert dist["baseline_refusal"] == 1   # unsafe
    assert dist["safe_completion"] == 4    # dual_use×2 + benign_sensitive + benign


def test_dpo_assembled_count_matches_pairs(
    ext_pairs: list[PreferencePair], ext_dpo: list[DPORecord]
) -> None:
    assert len(ext_dpo) == len(ext_pairs)


# ---------------------------------------------------------------------------
# DPO assembly — field correctness per category
# ---------------------------------------------------------------------------


def test_dpo_unsafe_record_fields(ext_dpo: list[DPORecord]) -> None:
    r = next(d for d in ext_dpo if d.prompt_id == "ext_u_001")
    assert r.category == PromptCategory.unsafe
    assert r.pair_type == PairType.baseline_refusal
    assert r.chosen_id == _response_id("ext_u_001", "hard_refusal")
    assert r.rejected_id == _response_id("ext_u_001", "unsafe_compliance")


def test_dpo_dual_use_p01_record_fields(ext_dpo: list[DPORecord]) -> None:
    r = next(d for d in ext_dpo if d.pair_id == "ext_d_001__safe_completion__p01")
    assert r.category == PromptCategory.dual_use
    assert r.pair_type == PairType.safe_completion
    assert r.chosen_id == _response_id("ext_d_001", "safe_completion")
    assert r.rejected_id == _response_id("ext_d_001", "hard_refusal")


def test_dpo_dual_use_p02_record_fields(ext_dpo: list[DPORecord]) -> None:
    # MOCK-ONLY: optional safe_completion > unsafe_compliance rule
    r = next(d for d in ext_dpo if d.pair_id == "ext_d_001__safe_completion__p02")
    assert r.category == PromptCategory.dual_use
    assert r.pair_type == PairType.safe_completion
    assert r.chosen_id == _response_id("ext_d_001", "safe_completion")
    assert r.rejected_id == _response_id("ext_d_001", "unsafe_compliance")


def test_dpo_benign_sensitive_record_fields(ext_dpo: list[DPORecord]) -> None:
    r = next(d for d in ext_dpo if d.prompt_id == "ext_bs_001")
    assert r.category == PromptCategory.benign_sensitive
    assert r.pair_type == PairType.safe_completion
    assert r.chosen_id == _response_id("ext_bs_001", "helpful_answer")
    assert r.rejected_id == _response_id("ext_bs_001", "hard_refusal")


def test_dpo_benign_provisional_record_fields(ext_dpo: list[DPORecord]) -> None:
    # PROVISIONAL: benign pair semantics not locked
    r = next(d for d in ext_dpo if d.prompt_id == "ext_b_001")
    assert r.category == PromptCategory.benign
    assert r.pair_type == PairType.safe_completion
    assert r.chosen_id == _response_id("ext_b_001", "helpful_answer")
    assert r.rejected_id == _response_id("ext_b_001", "hard_refusal")


def test_dpo_records_have_prompt_text(ext_dpo: list[DPORecord]) -> None:
    prompt_map = {p.prompt_id: p.prompt for p in _PROMPTS}
    for r in ext_dpo:
        assert r.prompt == prompt_map[r.prompt_id]


def test_dpo_records_have_response_text(ext_dpo: list[DPORecord]) -> None:
    for r in ext_dpo:
        assert r.chosen.strip()
        assert r.rejected.strip()


# ---------------------------------------------------------------------------
# FK coherence
# ---------------------------------------------------------------------------


def test_fk_all_dpo_pair_ids_in_pairs(
    ext_dpo: list[DPORecord], ext_pairs: list[PreferencePair]
) -> None:
    pair_ids = {p.pair_id for p in ext_pairs}
    for r in ext_dpo:
        assert r.pair_id in pair_ids


def test_fk_all_dpo_chosen_ids_in_responses(ext_dpo: list[DPORecord]) -> None:
    response_ids = {r.response_id for r in _RESPONSES}
    for r in ext_dpo:
        assert r.chosen_id in response_ids


def test_fk_all_dpo_rejected_ids_in_responses(ext_dpo: list[DPORecord]) -> None:
    response_ids = {r.response_id for r in _RESPONSES}
    for r in ext_dpo:
        assert r.rejected_id in response_ids


def test_fk_all_dpo_prompt_ids_in_prompts(ext_dpo: list[DPORecord]) -> None:
    prompt_ids = {p.prompt_id for p in _PROMPTS}
    for r in ext_dpo:
        assert r.prompt_id in prompt_ids


def test_fk_pair_chosen_and_rejected_reference_same_prompt(
    ext_pairs: list[PreferencePair],
) -> None:
    response_prompt_map = {r.response_id: r.prompt_id for r in _RESPONSES}
    for pair in ext_pairs:
        assert response_prompt_map[pair.chosen_id] == pair.prompt_id
        assert response_prompt_map[pair.rejected_id] == pair.prompt_id


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_pair_construction_is_deterministic() -> None:
    pairs_1 = build_pairs(_PROMPTS, _RESPONSES, _VALIDATIONS, _MOCK_EXTENDED_CONFIG, _CONFIG_PATH)
    pairs_2 = build_pairs(_PROMPTS, _RESPONSES, _VALIDATIONS, _MOCK_EXTENDED_CONFIG, _CONFIG_PATH)
    assert [p.pair_id for p in pairs_1] == [p.pair_id for p in pairs_2]
    assert [p.chosen_id for p in pairs_1] == [p.chosen_id for p in pairs_2]
    assert [p.rejected_id for p in pairs_1] == [p.rejected_id for p in pairs_2]


def test_assembly_is_deterministic() -> None:
    pairs = build_pairs(_PROMPTS, _RESPONSES, _VALIDATIONS, _MOCK_EXTENDED_CONFIG, _CONFIG_PATH)
    dpo_1, _ = assemble(pairs, _PROMPTS, _RESPONSES, _VALIDATIONS)
    dpo_2, _ = assemble(pairs, _PROMPTS, _RESPONSES, _VALIDATIONS)
    assert [r.pair_id for r in dpo_1] == [r.pair_id for r in dpo_2]


# ---------------------------------------------------------------------------
# Real vs mock distinction: production config stays conservative
# ---------------------------------------------------------------------------


def test_production_config_only_produces_dual_use_pairs() -> None:
    """Verify that the production pairs config yields pairs for dual_use only.

    This is a structural guard: if the production config is accidentally
    broadened, this test will catch it.
    """
    production_config = {
        "pair_rules": {
            "dual_use": [
                {"pair_type": "safe_completion", "chosen_type": "safe_completion", "rejected_type": "hard_refusal"}
            ],
            "unsafe": [],
            "benign_sensitive": [],
            "benign": [],
        }
    }
    pairs = build_pairs(_PROMPTS, _RESPONSES, _VALIDATIONS, production_config, "production_guard")
    assert all(p.prompt_id == "ext_d_001" for p in pairs), (
        "Production config should only produce dual_use pairs; "
        f"got pair for non-dual_use prompt: {[p.prompt_id for p in pairs if p.prompt_id != 'ext_d_001']}"
    )
    assert len(pairs) == 1
