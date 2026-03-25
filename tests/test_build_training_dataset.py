"""
Tests for scripts/build_training_dataset.py.

Covers:
  - select_for_regime core logic
  - include_rules matching (category × pair_type)
  - baseline and safecomp regime selection
  - shared pairs (unsafe, benign_sensitive, benign are same across regimes)
  - dual_use divergence: baseline gets baseline_refusal, safecomp gets safe_completion
  - safecomp gets BOTH dual_use safe_completion pairs (p01 and p02)
  - empty include_rules warning
  - stats structure
  - actual baseline.yaml and safecomp.yaml config integration
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_training_dataset import select_for_regime
from safecomp_dpo.schemas import DPORecord, PairType, PromptCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _record(
    pair_id: str,
    prompt_id: str,
    category: PromptCategory,
    pair_type: PairType,
) -> DPORecord:
    return DPORecord(
        pair_id=pair_id,
        prompt_id=prompt_id,
        category=category,
        pair_type=pair_type,
        prompt="A test prompt.",
        chosen="Chosen response.",
        rejected="Rejected response.",
        chosen_id=f"{pair_id}__chosen",
        rejected_id=f"{pair_id}__rejected",
    )


# Locked v1 pair set — mirrors what build_pairs produces for 1 prompt per category
UNSAFE_BR = _record(
    "u_001__baseline_refusal__p01",
    "u_001",
    PromptCategory.unsafe,
    PairType.baseline_refusal,
)
DU_SC_P01 = _record(
    "d_001__safe_completion__p01",
    "d_001",
    PromptCategory.dual_use,
    PairType.safe_completion,
)
DU_SC_P02 = _record(
    "d_001__safe_completion__p02",
    "d_001",
    PromptCategory.dual_use,
    PairType.safe_completion,
)
DU_BR = _record(
    "d_001__baseline_refusal__p03",
    "d_001",
    PromptCategory.dual_use,
    PairType.baseline_refusal,
)
BS_SC = _record(
    "bs_001__safe_completion__p01",
    "bs_001",
    PromptCategory.benign_sensitive,
    PairType.safe_completion,
)
B_SC = _record(
    "b_001__safe_completion__p01",
    "b_001",
    PromptCategory.benign,
    PairType.safe_completion,
)

ALL_RECORDS = [UNSAFE_BR, DU_SC_P01, DU_SC_P02, DU_BR, BS_SC, B_SC]

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
# Baseline regime selection
# ---------------------------------------------------------------------------


class TestBaselineSelection:
    def setup_method(self):
        self.selected, self.stats = select_for_regime(ALL_RECORDS, _BASELINE_RULES)

    def test_correct_count(self):
        # unsafe(1) + dual_use baseline(1) + benign_sensitive(1) + benign(1) = 4
        assert len(self.selected) == 4

    def test_unsafe_included(self):
        ids = {r.pair_id for r in self.selected}
        assert UNSAFE_BR.pair_id in ids

    def test_dual_use_baseline_refusal_included(self):
        ids = {r.pair_id for r in self.selected}
        assert DU_BR.pair_id in ids

    def test_dual_use_safe_completion_excluded(self):
        ids = {r.pair_id for r in self.selected}
        assert DU_SC_P01.pair_id not in ids
        assert DU_SC_P02.pair_id not in ids

    def test_benign_sensitive_included(self):
        ids = {r.pair_id for r in self.selected}
        assert BS_SC.pair_id in ids

    def test_benign_included(self):
        ids = {r.pair_id for r in self.selected}
        assert B_SC.pair_id in ids

    def test_sorted_by_pair_id(self):
        ids = [r.pair_id for r in self.selected]
        assert ids == sorted(ids)

    def test_stats_total_input(self):
        assert self.stats["total_input_records"] == len(ALL_RECORDS)

    def test_stats_selected(self):
        assert self.stats["selected_records"] == 4

    def test_stats_excluded(self):
        # dual_use sc p01 + p02 excluded = 2
        assert self.stats["excluded_records"] == 2

    def test_stats_has_category_distribution(self):
        dist = self.stats["category_distribution"]
        assert dist["unsafe"] == 1
        assert dist["dual_use"] == 1
        assert dist["benign_sensitive"] == 1
        assert dist["benign"] == 1

    def test_stats_has_pair_type_distribution(self):
        dist = self.stats["pair_type_distribution"]
        assert dist["baseline_refusal"] == 2  # unsafe + dual_use
        assert dist["safe_completion"] == 2   # benign_sensitive + benign

    def test_stats_includes_rules(self):
        assert self.stats["include_rules"] == _BASELINE_RULES


# ---------------------------------------------------------------------------
# SafeComp regime selection
# ---------------------------------------------------------------------------


class TestSafeCompSelection:
    def setup_method(self):
        self.selected, self.stats = select_for_regime(ALL_RECORDS, _SAFECOMP_RULES)

    def test_correct_count(self):
        # unsafe(1) + dual_use sc p01+p02(2) + benign_sensitive(1) + benign(1) = 5
        assert len(self.selected) == 5

    def test_unsafe_included(self):
        ids = {r.pair_id for r in self.selected}
        assert UNSAFE_BR.pair_id in ids

    def test_both_dual_use_safe_completion_pairs_included(self):
        ids = {r.pair_id for r in self.selected}
        assert DU_SC_P01.pair_id in ids
        assert DU_SC_P02.pair_id in ids

    def test_dual_use_baseline_refusal_excluded(self):
        ids = {r.pair_id for r in self.selected}
        assert DU_BR.pair_id not in ids

    def test_benign_sensitive_included(self):
        ids = {r.pair_id for r in self.selected}
        assert BS_SC.pair_id in ids

    def test_benign_included(self):
        ids = {r.pair_id for r in self.selected}
        assert B_SC.pair_id in ids

    def test_sorted_by_pair_id(self):
        ids = [r.pair_id for r in self.selected]
        assert ids == sorted(ids)

    def test_stats_selected(self):
        assert self.stats["selected_records"] == 5

    def test_stats_excluded(self):
        # only dual_use baseline_refusal excluded
        assert self.stats["excluded_records"] == 1

    def test_stats_dual_use_count(self):
        dist = self.stats["category_distribution"]
        assert dist["dual_use"] == 2  # both safe_completion pairs

    def test_stats_pair_type_distribution(self):
        dist = self.stats["pair_type_distribution"]
        assert dist["baseline_refusal"] == 1  # only unsafe
        assert dist["safe_completion"] == 4   # du p01, p02, benign_sensitive, benign


# ---------------------------------------------------------------------------
# Regime divergence: baseline vs safecomp differ only on dual_use
# ---------------------------------------------------------------------------


class TestRegimeDivergence:
    def test_shared_pairs_same_in_both_regimes(self):
        baseline, _ = select_for_regime(ALL_RECORDS, _BASELINE_RULES)
        safecomp, _ = select_for_regime(ALL_RECORDS, _SAFECOMP_RULES)
        baseline_ids = {r.pair_id for r in baseline}
        safecomp_ids = {r.pair_id for r in safecomp}
        # unsafe and benign_sensitive and benign are in both
        assert UNSAFE_BR.pair_id in baseline_ids and UNSAFE_BR.pair_id in safecomp_ids
        assert BS_SC.pair_id in baseline_ids and BS_SC.pair_id in safecomp_ids
        assert B_SC.pair_id in baseline_ids and B_SC.pair_id in safecomp_ids

    def test_dual_use_pairs_are_disjoint(self):
        baseline, _ = select_for_regime(ALL_RECORDS, _BASELINE_RULES)
        safecomp, _ = select_for_regime(ALL_RECORDS, _SAFECOMP_RULES)
        baseline_du = {r.pair_id for r in baseline if r.category == PromptCategory.dual_use}
        safecomp_du = {r.pair_id for r in safecomp if r.category == PromptCategory.dual_use}
        assert baseline_du.isdisjoint(safecomp_du)

    def test_baseline_has_one_dual_use_pair(self):
        baseline, _ = select_for_regime(ALL_RECORDS, _BASELINE_RULES)
        du = [r for r in baseline if r.category == PromptCategory.dual_use]
        assert len(du) == 1
        assert du[0].pair_type == PairType.baseline_refusal

    def test_safecomp_has_two_dual_use_pairs(self):
        safecomp, _ = select_for_regime(ALL_RECORDS, _SAFECOMP_RULES)
        du = [r for r in safecomp if r.category == PromptCategory.dual_use]
        assert len(du) == 2
        for r in du:
            assert r.pair_type == PairType.safe_completion


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSelectForRegimeEdgeCases:
    def test_empty_records(self):
        selected, stats = select_for_regime([], _BASELINE_RULES)
        assert selected == []
        assert stats["total_input_records"] == 0
        assert stats["selected_records"] == 0

    def test_empty_rules_produces_no_records(self, capsys):
        selected, stats = select_for_regime(ALL_RECORDS, [])
        assert selected == []
        assert stats["selected_records"] == 0

    def test_empty_rules_warns(self, capsys):
        select_for_regime(ALL_RECORDS, [])
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_unknown_category_rule_matches_nothing(self):
        rules = [{"category": "nonexistent", "pair_type": "baseline_refusal"}]
        selected, stats = select_for_regime(ALL_RECORDS, rules)
        assert selected == []

    def test_all_rules_selects_all(self):
        all_rules = [
            {"category": "unsafe", "pair_type": "baseline_refusal"},
            {"category": "dual_use", "pair_type": "safe_completion"},
            {"category": "dual_use", "pair_type": "baseline_refusal"},
            {"category": "benign_sensitive", "pair_type": "safe_completion"},
            {"category": "benign", "pair_type": "safe_completion"},
        ]
        selected, stats = select_for_regime(ALL_RECORDS, all_rules)
        assert len(selected) == len(ALL_RECORDS)


# ---------------------------------------------------------------------------
# Config file integration
# ---------------------------------------------------------------------------


_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent / "configs" / "training-datasets"
)


class TestBaselineYamlConfig:
    def test_file_exists(self):
        assert (_CONFIG_DIR / "baseline.yaml").exists()

    def test_regime_is_baseline(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "baseline.yaml").read_text())
        assert cfg["regime"] == "baseline"

    def test_has_four_include_rules(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "baseline.yaml").read_text())
        assert len(cfg["include_rules"]) == 4

    def test_unsafe_baseline_refusal_rule(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "baseline.yaml").read_text())
        rules = cfg["include_rules"]
        assert {"category": "unsafe", "pair_type": "baseline_refusal"} in rules

    def test_dual_use_baseline_refusal_rule(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "baseline.yaml").read_text())
        rules = cfg["include_rules"]
        assert {"category": "dual_use", "pair_type": "baseline_refusal"} in rules

    def test_no_dual_use_safe_completion_rule(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "baseline.yaml").read_text())
        rules = cfg["include_rules"]
        du_sc_rules = [r for r in rules if r["category"] == "dual_use" and r["pair_type"] == "safe_completion"]
        assert du_sc_rules == []

    def test_benign_categories_safe_completion(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "baseline.yaml").read_text())
        rules = cfg["include_rules"]
        bs_rules = [r for r in rules if r["category"] == "benign_sensitive"]
        assert bs_rules and bs_rules[0]["pair_type"] == "safe_completion"
        b_rules = [r for r in rules if r["category"] == "benign"]
        assert b_rules and b_rules[0]["pair_type"] == "safe_completion"

    def test_baseline_config_selects_correctly(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "baseline.yaml").read_text())
        rules = cfg["include_rules"]
        selected, stats = select_for_regime(ALL_RECORDS, rules)
        assert stats["selected_records"] == 4
        # dual_use safe_completion pairs should be excluded
        du = [r for r in selected if r.category == PromptCategory.dual_use]
        assert len(du) == 1
        assert du[0].pair_type == PairType.baseline_refusal


class TestSafecompYamlConfig:
    def test_file_exists(self):
        assert (_CONFIG_DIR / "safecomp.yaml").exists()

    def test_regime_is_safecomp(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "safecomp.yaml").read_text())
        assert cfg["regime"] == "safecomp"

    def test_has_four_include_rules(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "safecomp.yaml").read_text())
        assert len(cfg["include_rules"]) == 4

    def test_dual_use_safe_completion_rule(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "safecomp.yaml").read_text())
        rules = cfg["include_rules"]
        assert {"category": "dual_use", "pair_type": "safe_completion"} in rules

    def test_no_dual_use_baseline_refusal_rule(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "safecomp.yaml").read_text())
        rules = cfg["include_rules"]
        du_br = [r for r in rules if r["category"] == "dual_use" and r["pair_type"] == "baseline_refusal"]
        assert du_br == []

    def test_safecomp_config_selects_both_du_pairs(self):
        cfg = yaml.safe_load((_CONFIG_DIR / "safecomp.yaml").read_text())
        rules = cfg["include_rules"]
        selected, stats = select_for_regime(ALL_RECORDS, rules)
        assert stats["selected_records"] == 5
        du = [r for r in selected if r.category == PromptCategory.dual_use]
        assert len(du) == 2
        for r in du:
            assert r.pair_type == PairType.safe_completion
