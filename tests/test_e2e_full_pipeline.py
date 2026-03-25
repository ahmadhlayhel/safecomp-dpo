"""
End-to-end smoke test for the full v1 pipeline.

Covers the complete pipeline from PromptRecord through regime-specific training
datasets:

  1. generate_responses        → ResponseRecords (acceptable types)
  2. source_rejected_responses → ResponseRecords (unsafe_compliance)
  3. validate_responses        → ValidationRecords (merged response set)
  4. build_pairs               → PreferencePairs
  5. assemble_dpo_dataset      → DPO records (all pairs)
  6. select_for_regime         → baseline DPORecords
  7. select_for_regime         → safecomp DPORecords

All stages run in-memory with mock backends.  No disk I/O.
No GPU or model weights required.

Prompt fixture covers all 4 categories:
  unsafe          → generates hard_refusal; sources unsafe_compliance
  dual_use        → generates hard_refusal + safe_completion; sources unsafe_compliance
  benign_sensitive → generates helpful_answer + hard_refusal
  benign          → generates helpful_answer + hard_refusal

Locked v1 pair logic verified:
  unsafe:           baseline_refusal (hard_refusal > unsafe_compliance)        → 1 pair
  dual_use:         safe_completion p01 (safe > hard), p02 (safe > unsafe_c)   → 2 pairs
                    baseline_refusal (hard > unsafe_c)                          → 1 pair
  benign_sensitive: safe_completion (helpful > hard)                            → 1 pair
  benign:           safe_completion (helpful > hard)                            → 1 pair
  Total full DPO records: 6

Baseline regime: unsafe_br(1) + du_br(1) + bs_sc(1) + b_sc(1)  = 4
SafeComp regime: unsafe_br(1) + du_sc_p01+p02(2) + bs_sc(1) + b_sc(1) = 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_responses import MockBackend as MockGenBackend, generate_responses
from source_rejected_responses import MockSourcingBackend, source_rejected_responses
from validate_responses import validate_responses
from build_pairs import build_pairs
from assemble_dpo_dataset import assemble
from build_training_dataset import select_for_regime

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
# Inline configs (mirror full.yaml semantics)
# ---------------------------------------------------------------------------

_GEN_CONFIG = {
    "expected_response_types": {
        "unsafe": ["hard_refusal"],
        "dual_use": ["hard_refusal", "safe_completion"],
        "benign_sensitive": ["helpful_answer", "hard_refusal"],
        "benign": ["helpful_answer", "hard_refusal"],
    },
    "samples_per_type": 1,
    "model": "mock-model",
    "backend": "mock",
}

_SOURCING_CONFIG = {
    "categories": ["unsafe", "dual_use"],
    "samples_per_prompt": 1,
    "model": "mock-model",
    "backend": "mock",
}

# Validation config covers both generation AND sourcing response types.
_VAL_CONFIG = {
    "expected_response_types": {
        "unsafe": ["hard_refusal", "unsafe_compliance"],
        "dual_use": ["hard_refusal", "safe_completion", "unsafe_compliance"],
        "benign_sensitive": ["helpful_answer", "hard_refusal"],
        "benign": ["helpful_answer", "hard_refusal"],
    },
}

_PAIR_CONFIG = {
    "pair_rules": {
        "unsafe": [
            {"pair_type": "baseline_refusal", "chosen_type": "hard_refusal", "rejected_type": "unsafe_compliance"},
        ],
        "dual_use": [
            {"pair_type": "safe_completion", "chosen_type": "safe_completion", "rejected_type": "hard_refusal"},
            {"pair_type": "safe_completion", "chosen_type": "safe_completion", "rejected_type": "unsafe_compliance"},
            {"pair_type": "baseline_refusal", "chosen_type": "hard_refusal", "rejected_type": "unsafe_compliance"},
        ],
        "benign_sensitive": [
            {"pair_type": "safe_completion", "chosen_type": "helpful_answer", "rejected_type": "hard_refusal"},
        ],
        "benign": [
            {"pair_type": "safe_completion", "chosen_type": "helpful_answer", "rejected_type": "hard_refusal"},
        ],
    }
}

_BASELINE_RULES = [
    {"category": "unsafe", "pair_type": "baseline_refusal"},
    {"category": "dual_use", "pair_type": "baseline_refusal"},
    {"category": "benign_sensitive", "pair_type": "safe_completion"},
    {"category": "benign", "pair_type": "safe_completion"},
]

_SAFECOMP_RULES = [
    {"category": "unsafe", "pair_type": "baseline_refusal"},
    {"category": "dual_use", "pair_type": "safe_completion"},
    {"category": "benign_sensitive", "pair_type": "safe_completion"},
    {"category": "benign", "pair_type": "safe_completion"},
]


# ---------------------------------------------------------------------------
# Prompt fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prompts() -> list[PromptRecord]:
    return [
        PromptRecord(
            prompt_id="e2e_u_001",
            prompt="Describe a dangerous synthesis process.",
            category=PromptCategory.unsafe,
            source="test",
        ),
        PromptRecord(
            prompt_id="e2e_d_001",
            prompt="Explain dual-use chemistry concepts.",
            category=PromptCategory.dual_use,
            source="test",
        ),
        PromptRecord(
            prompt_id="e2e_bs_001",
            prompt="Describe common over-the-counter medications.",
            category=PromptCategory.benign_sensitive,
            source="test",
        ),
        PromptRecord(
            prompt_id="e2e_b_001",
            prompt="What is the capital of France?",
            category=PromptCategory.benign,
            source="test",
        ),
    ]


# ---------------------------------------------------------------------------
# Stage 1+2: Response generation and rejected sourcing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_responses(prompts) -> list[ResponseRecord]:
    gen_backend = MockGenBackend()
    accepted = generate_responses(prompts, _GEN_CONFIG, gen_backend, "gen_config")

    sourcing_backend = MockSourcingBackend()
    rejected = source_rejected_responses(prompts, _SOURCING_CONFIG, sourcing_backend, "src_config")

    return accepted + rejected


class TestResponseGeneration:
    def test_accepted_count(self, prompts):
        gen_backend = MockGenBackend()
        accepted = generate_responses(prompts, _GEN_CONFIG, gen_backend, "cfg")
        # unsafe(1) + dual_use(2) + benign_sensitive(2) + benign(2) = 7
        assert len(accepted) == 7

    def test_sourced_count(self, prompts):
        sourcing_backend = MockSourcingBackend()
        rejected = source_rejected_responses(prompts, _SOURCING_CONFIG, sourcing_backend, "cfg")
        # unsafe(1) + dual_use(1) = 2
        assert len(rejected) == 2

    def test_total_response_count(self, all_responses):
        assert len(all_responses) == 9

    def test_no_unsafe_compliance_in_accepted(self, prompts):
        gen_backend = MockGenBackend()
        accepted = generate_responses(prompts, _GEN_CONFIG, gen_backend, "cfg")
        types = {r.response_type for r in accepted}
        assert ResponseType.unsafe_compliance not in types

    def test_only_unsafe_compliance_in_sourced(self, prompts):
        sourcing_backend = MockSourcingBackend()
        rejected = source_rejected_responses(prompts, _SOURCING_CONFIG, sourcing_backend, "cfg")
        types = {r.response_type for r in rejected}
        assert types == {ResponseType.unsafe_compliance}

    def test_sourced_only_for_unsafe_and_dualuse(self, prompts):
        sourcing_backend = MockSourcingBackend()
        rejected = source_rejected_responses(prompts, _SOURCING_CONFIG, sourcing_backend, "cfg")
        categories_present = {r.prompt_id.split("_")[1] for r in rejected}
        # e2e_u_001 → u, e2e_d_001 → d
        for r in rejected:
            assert r.prompt_id in {"e2e_u_001", "e2e_d_001"}

    def test_all_response_ids_unique(self, all_responses):
        ids = [r.response_id for r in all_responses]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Stage 3: Validation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def validations(prompts, all_responses) -> list[ValidationRecord]:
    return validate_responses(all_responses, prompts, _VAL_CONFIG)


class TestValidation:
    def test_one_validation_per_response(self, all_responses, validations):
        assert len(validations) == len(all_responses)

    def test_all_valid(self, validations):
        invalid = [v for v in validations if not v.is_valid]
        assert invalid == [], f"Invalid: {[v.issues for v in invalid]}"


# ---------------------------------------------------------------------------
# Stage 4: Pair construction
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pairs(prompts, all_responses, validations) -> list[PreferencePair]:
    return build_pairs(prompts, all_responses, validations, _PAIR_CONFIG, "pair_config")


class TestPairConstruction:
    def test_total_pair_count(self, pairs):
        # unsafe(1) + dual_use(3) + benign_sensitive(1) + benign(1) = 6
        assert len(pairs) == 6

    def test_unsafe_has_one_pair(self, pairs):
        unsafe_pairs = [p for p in pairs if "e2e_u_001" in p.pair_id]
        assert len(unsafe_pairs) == 1
        assert unsafe_pairs[0].pair_type == PairType.baseline_refusal

    def test_dual_use_has_three_pairs(self, pairs):
        du_pairs = [p for p in pairs if "e2e_d_001" in p.pair_id]
        assert len(du_pairs) == 3

    def test_dual_use_pair_types(self, pairs):
        du_pairs = [p for p in pairs if "e2e_d_001" in p.pair_id]
        sc_pairs = [p for p in du_pairs if p.pair_type == PairType.safe_completion]
        br_pairs = [p for p in du_pairs if p.pair_type == PairType.baseline_refusal]
        assert len(sc_pairs) == 2
        assert len(br_pairs) == 1

    def test_benign_sensitive_has_one_pair(self, pairs):
        bs_pairs = [p for p in pairs if "e2e_bs_001" in p.pair_id]
        assert len(bs_pairs) == 1
        assert bs_pairs[0].pair_type == PairType.safe_completion

    def test_benign_has_one_pair(self, pairs):
        b_pairs = [p for p in pairs if "e2e_b_001" in p.pair_id]
        assert len(b_pairs) == 1
        assert b_pairs[0].pair_type == PairType.safe_completion


# ---------------------------------------------------------------------------
# Stage 5: Assemble full DPO dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_dpo(prompts, all_responses, validations, pairs) -> list[DPORecord]:
    records, stats = assemble(pairs, prompts, all_responses, validations)
    return records


class TestFullDpoAssembly:
    def test_total_record_count(self, full_dpo):
        assert len(full_dpo) == 6

    def test_sorted_by_pair_id(self, full_dpo):
        ids = [r.pair_id for r in full_dpo]
        assert ids == sorted(ids)

    def test_categories_present(self, full_dpo):
        cats = {r.category for r in full_dpo}
        assert PromptCategory.unsafe in cats
        assert PromptCategory.dual_use in cats
        assert PromptCategory.benign_sensitive in cats
        assert PromptCategory.benign in cats

    def test_all_records_have_non_empty_text(self, full_dpo):
        for r in full_dpo:
            assert r.prompt.strip()
            assert r.chosen.strip()
            assert r.rejected.strip()

    def test_unsafe_record_pair_type(self, full_dpo):
        unsafe_recs = [r for r in full_dpo if r.category == PromptCategory.unsafe]
        assert len(unsafe_recs) == 1
        assert unsafe_recs[0].pair_type == PairType.baseline_refusal

    def test_dual_use_records(self, full_dpo):
        du_recs = [r for r in full_dpo if r.category == PromptCategory.dual_use]
        assert len(du_recs) == 3
        sc = [r for r in du_recs if r.pair_type == PairType.safe_completion]
        br = [r for r in du_recs if r.pair_type == PairType.baseline_refusal]
        assert len(sc) == 2
        assert len(br) == 1

    def test_chosen_differs_from_rejected(self, full_dpo):
        for r in full_dpo:
            assert r.chosen != r.rejected


# ---------------------------------------------------------------------------
# Stage 6+7: Regime-specific dataset construction
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline_dpo(full_dpo) -> list[DPORecord]:
    selected, _ = select_for_regime(full_dpo, _BASELINE_RULES)
    return selected


@pytest.fixture(scope="module")
def safecomp_dpo(full_dpo) -> list[DPORecord]:
    selected, _ = select_for_regime(full_dpo, _SAFECOMP_RULES)
    return selected


class TestBaselineDataset:
    def test_count(self, baseline_dpo):
        assert len(baseline_dpo) == 4

    def test_includes_unsafe(self, baseline_dpo):
        cats = [r.category for r in baseline_dpo]
        assert PromptCategory.unsafe in cats

    def test_dual_use_is_baseline_refusal(self, baseline_dpo):
        du = [r for r in baseline_dpo if r.category == PromptCategory.dual_use]
        assert len(du) == 1
        assert du[0].pair_type == PairType.baseline_refusal

    def test_no_dual_use_safe_completion(self, baseline_dpo):
        du_sc = [
            r for r in baseline_dpo
            if r.category == PromptCategory.dual_use and r.pair_type == PairType.safe_completion
        ]
        assert du_sc == []

    def test_all_categories_represented(self, baseline_dpo):
        cats = {r.category for r in baseline_dpo}
        assert cats == {
            PromptCategory.unsafe,
            PromptCategory.dual_use,
            PromptCategory.benign_sensitive,
            PromptCategory.benign,
        }

    def test_sorted_by_pair_id(self, baseline_dpo):
        ids = [r.pair_id for r in baseline_dpo]
        assert ids == sorted(ids)


class TestSafecompDataset:
    def test_count(self, safecomp_dpo):
        assert len(safecomp_dpo) == 5

    def test_includes_unsafe(self, safecomp_dpo):
        cats = [r.category for r in safecomp_dpo]
        assert PromptCategory.unsafe in cats

    def test_dual_use_has_two_safe_completion_pairs(self, safecomp_dpo):
        du = [r for r in safecomp_dpo if r.category == PromptCategory.dual_use]
        assert len(du) == 2
        for r in du:
            assert r.pair_type == PairType.safe_completion

    def test_no_dual_use_baseline_refusal(self, safecomp_dpo):
        du_br = [
            r for r in safecomp_dpo
            if r.category == PromptCategory.dual_use and r.pair_type == PairType.baseline_refusal
        ]
        assert du_br == []

    def test_unsafe_is_baseline_refusal(self, safecomp_dpo):
        unsafe = [r for r in safecomp_dpo if r.category == PromptCategory.unsafe]
        assert len(unsafe) == 1
        assert unsafe[0].pair_type == PairType.baseline_refusal

    def test_all_categories_represented(self, safecomp_dpo):
        cats = {r.category for r in safecomp_dpo}
        assert cats == {
            PromptCategory.unsafe,
            PromptCategory.dual_use,
            PromptCategory.benign_sensitive,
            PromptCategory.benign,
        }


# ---------------------------------------------------------------------------
# Cross-regime properties
# ---------------------------------------------------------------------------


class TestRegimeCrossProperties:
    def test_baseline_and_safecomp_share_unsafe_pair(self, baseline_dpo, safecomp_dpo):
        baseline_ids = {r.pair_id for r in baseline_dpo}
        safecomp_ids = {r.pair_id for r in safecomp_dpo}
        unsafe_b = {
            r.pair_id for r in baseline_dpo if r.category == PromptCategory.unsafe
        }
        unsafe_s = {
            r.pair_id for r in safecomp_dpo if r.category == PromptCategory.unsafe
        }
        assert unsafe_b == unsafe_s

    def test_baseline_and_safecomp_share_benign_pairs(self, baseline_dpo, safecomp_dpo):
        def benign_ids(records):
            return {
                r.pair_id for r in records
                if r.category in (PromptCategory.benign, PromptCategory.benign_sensitive)
            }
        assert benign_ids(baseline_dpo) == benign_ids(safecomp_dpo)

    def test_dual_use_pairs_disjoint(self, baseline_dpo, safecomp_dpo):
        b_du = {r.pair_id for r in baseline_dpo if r.category == PromptCategory.dual_use}
        s_du = {r.pair_id for r in safecomp_dpo if r.category == PromptCategory.dual_use}
        assert b_du.isdisjoint(s_du)

    def test_safecomp_strictly_larger_than_baseline(self, baseline_dpo, safecomp_dpo):
        assert len(safecomp_dpo) > len(baseline_dpo)

    def test_safecomp_has_one_more_record(self, baseline_dpo, safecomp_dpo):
        # safecomp gets 2 dual_use sc pairs vs baseline 1 dual_use br pair
        assert len(safecomp_dpo) - len(baseline_dpo) == 1

    def test_pipeline_is_deterministic(self, prompts):
        """Running the full pipeline twice produces identical results."""
        def run():
            gen_backend = MockGenBackend()
            src_backend = MockSourcingBackend()
            accepted = generate_responses(prompts, _GEN_CONFIG, gen_backend, "cfg")
            rejected = source_rejected_responses(prompts, _SOURCING_CONFIG, src_backend, "cfg")
            all_resp = accepted + rejected
            val = validate_responses(all_resp, prompts, _VAL_CONFIG)
            pairs_list = build_pairs(prompts, all_resp, val, _PAIR_CONFIG, "cfg")
            records, _ = assemble(pairs_list, prompts, all_resp, val)
            safecomp, _ = select_for_regime(records, _SAFECOMP_RULES)
            return [r.pair_id for r in safecomp]

        assert run() == run()
