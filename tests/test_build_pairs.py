"""Tests for scripts/build_pairs.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    from build_pairs import (
        build_pairs,
        build_valid_response_index,
        default_output_path,
        make_pair_id,
    )
except ImportError:
    _scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(_scripts_dir))
    from build_pairs import (
        build_pairs,
        build_valid_response_index,
        default_output_path,
        make_pair_id,
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
    prompt_id: str,
    response_type: ResponseType,
    response_id: str | None = None,
) -> ResponseRecord:
    rid = response_id or f"{prompt_id}__{response_type.value}__s01"
    return ResponseRecord(
        response_id=rid,
        prompt_id=prompt_id,
        response="A response.",
        response_type=response_type,
        model="test-model",
    )


def make_validation(response_id: str, is_valid: bool = True) -> ValidationRecord:
    return ValidationRecord(
        validation_id=f"{response_id}__val",
        response_id=response_id,
        prompt_id=response_id.split("__")[0],
        is_valid=is_valid,
        validator="rule_based_pilot",
    )


DUAL_USE_CONFIG = {
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


# ---------------------------------------------------------------------------
# make_pair_id
# ---------------------------------------------------------------------------


def test_make_pair_id_default_index():
    pid = make_pair_id("pilot_d_001", PairType.safe_completion)
    assert pid == "pilot_d_001__safe_completion__p01"


def test_make_pair_id_explicit_index():
    pid = make_pair_id("pilot_d_001", PairType.baseline_refusal, index=2)
    assert pid == "pilot_d_001__baseline_refusal__p02"


def test_make_pair_id_deterministic():
    pid1 = make_pair_id("pilot_d_001", PairType.safe_completion)
    pid2 = make_pair_id("pilot_d_001", PairType.safe_completion)
    assert pid1 == pid2


# ---------------------------------------------------------------------------
# default_output_path
# ---------------------------------------------------------------------------


def test_default_output_path():
    p = default_output_path("configs/pairs/pilot.yaml")
    assert p == Path("outputs/pairs/pilot_pairs.jsonl")


# ---------------------------------------------------------------------------
# build_valid_response_index
# ---------------------------------------------------------------------------


def test_valid_response_index_includes_valid():
    r = make_response("pilot_d_001", ResponseType.safe_completion)
    v = make_validation(r.response_id, is_valid=True)
    index = build_valid_response_index([r], [v])
    assert index["pilot_d_001"]["safe_completion"] == r.response_id


def test_valid_response_index_excludes_invalid():
    r = make_response("pilot_d_001", ResponseType.safe_completion)
    v = make_validation(r.response_id, is_valid=False)
    index = build_valid_response_index([r], [v])
    assert "pilot_d_001" not in index


def test_valid_response_index_excludes_unvalidated():
    r = make_response("pilot_d_001", ResponseType.safe_completion)
    # No validation record for this response
    index = build_valid_response_index([r], [])
    assert "pilot_d_001" not in index


def test_valid_response_index_multiple_types():
    r1 = make_response("pilot_d_001", ResponseType.safe_completion)
    r2 = make_response("pilot_d_001", ResponseType.hard_refusal)
    v1 = make_validation(r1.response_id)
    v2 = make_validation(r2.response_id)
    index = build_valid_response_index([r1, r2], [v1, v2])
    assert "safe_completion" in index["pilot_d_001"]
    assert "hard_refusal" in index["pilot_d_001"]


def test_valid_response_index_multiple_prompts():
    r1 = make_response("pilot_d_001", ResponseType.safe_completion)
    r2 = make_response("pilot_d_002", ResponseType.safe_completion)
    index = build_valid_response_index(
        [r1, r2],
        [make_validation(r1.response_id), make_validation(r2.response_id)],
    )
    assert "pilot_d_001" in index
    assert "pilot_d_002" in index


# ---------------------------------------------------------------------------
# build_pairs — happy path
# ---------------------------------------------------------------------------


def test_build_pairs_dual_use_produces_pair():
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_sc.response_id), make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")

    assert len(pairs) == 1
    assert pairs[0].pair_type == PairType.safe_completion
    assert pairs[0].chosen_id == r_sc.response_id
    assert pairs[0].rejected_id == r_hr.response_id


def test_build_pairs_correct_prompt_id():
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_sc.response_id), make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert pairs[0].prompt_id == "pilot_d_001"


def test_build_pairs_two_dual_use_prompts():
    prompts = [
        make_prompt("pilot_d_001", PromptCategory.dual_use),
        make_prompt("pilot_d_002", PromptCategory.dual_use),
    ]
    responses = [
        make_response("pilot_d_001", ResponseType.safe_completion),
        make_response("pilot_d_001", ResponseType.hard_refusal),
        make_response("pilot_d_002", ResponseType.safe_completion),
        make_response("pilot_d_002", ResponseType.hard_refusal),
    ]
    validations = [make_validation(r.response_id) for r in responses]

    pairs = build_pairs(prompts, responses, validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert len(pairs) == 2


def test_build_pairs_returns_preference_pair_instances():
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_sc.response_id), make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert all(isinstance(p, PreferencePair) for p in pairs)


# ---------------------------------------------------------------------------
# build_pairs — pair_id convention
# ---------------------------------------------------------------------------


def test_build_pairs_pair_id_convention():
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_sc.response_id), make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert pairs[0].pair_id == "pilot_d_001__safe_completion__p01"


def test_build_pairs_pair_ids_deterministic():
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_sc.response_id), make_validation(r_hr.response_id)]

    ids1 = [p.pair_id for p in build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")]
    ids2 = [p.pair_id for p in build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")]
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# build_pairs — metadata
# ---------------------------------------------------------------------------


def test_build_pairs_metadata_fields():
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_sc.response_id), make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "my_config.yaml")
    m = pairs[0].metadata
    assert m["config_path"] == "my_config.yaml"
    assert m["chosen_type"] == "safe_completion"
    assert m["rejected_type"] == "hard_refusal"


# ---------------------------------------------------------------------------
# build_pairs — skipping when responses are missing or invalid
# ---------------------------------------------------------------------------


def test_build_pairs_missing_chosen_skipped(capsys):
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert pairs == []
    assert "WARNING" in capsys.readouterr().err


def test_build_pairs_missing_rejected_skipped(capsys):
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    validations = [make_validation(r_sc.response_id)]

    pairs = build_pairs([prompt], [r_sc], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert pairs == []
    assert "WARNING" in capsys.readouterr().err


def test_build_pairs_invalid_chosen_excluded(capsys):
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [
        make_validation(r_sc.response_id, is_valid=False),
        make_validation(r_hr.response_id, is_valid=True),
    ]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert pairs == []
    assert "WARNING" in capsys.readouterr().err


def test_build_pairs_invalid_rejected_excluded(capsys):
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [
        make_validation(r_sc.response_id, is_valid=True),
        make_validation(r_hr.response_id, is_valid=False),
    ]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert pairs == []
    assert "WARNING" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# build_pairs — empty and missing rules
# ---------------------------------------------------------------------------


def test_build_pairs_empty_rules_no_pairs():
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    config = {"pair_rules": {"dual_use": []}}
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_sc.response_id), make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, config, "cfg.yaml")
    assert pairs == []


def test_build_pairs_category_absent_from_config_no_pairs():
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    config = {"pair_rules": {}}
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_sc.response_id), make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, config, "cfg.yaml")
    assert pairs == []


def test_build_pairs_unsafe_no_pairs_with_pilot_config():
    """Unsafe prompts produce no pairs under the pilot config (empty rules)."""
    prompt = make_prompt("pilot_u_001", PromptCategory.unsafe)
    r_hr = make_response("pilot_u_001", ResponseType.hard_refusal)
    validations = [make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert pairs == []


def test_build_pairs_benign_no_pairs_with_pilot_config():
    prompt = make_prompt("pilot_b_001", PromptCategory.benign)
    r_ha = make_response("pilot_b_001", ResponseType.helpful_answer)
    validations = [make_validation(r_ha.response_id)]

    pairs = build_pairs([prompt], [r_ha], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert pairs == []


# ---------------------------------------------------------------------------
# build_pairs — one pair per prompt, not one per response
# ---------------------------------------------------------------------------


def test_build_pairs_one_pair_per_dual_use_prompt():
    prompt = make_prompt("pilot_d_001", PromptCategory.dual_use)
    r_sc = make_response("pilot_d_001", ResponseType.safe_completion)
    r_hr = make_response("pilot_d_001", ResponseType.hard_refusal)
    validations = [make_validation(r_sc.response_id), make_validation(r_hr.response_id)]

    pairs = build_pairs([prompt], [r_sc, r_hr], validations, DUAL_USE_CONFIG, "cfg.yaml")
    assert len(pairs) == 1
