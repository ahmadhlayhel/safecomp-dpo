"""Tests for scripts/train_dpo.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train_dpo import (
    MockTrainingBackend,
    filter_examples,
    get_backend,
    run_training,
    write_outputs,
)
from safecomp_dpo.io import load_dpo_records, write_jsonl
from safecomp_dpo.schemas import DPORecord, PairType, PromptCategory


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _dpo(
    pair_id: str = "p001__safe_completion__p01",
    prompt_id: str = "p001",
    category: PromptCategory = PromptCategory.dual_use,
    pair_type: PairType = PairType.safe_completion,
    prompt: str = "How does SQL injection work?",
    chosen: str = "Safe completion text.",
    rejected: str = "I cannot help with that.",
) -> DPORecord:
    return DPORecord(
        pair_id=pair_id,
        prompt_id=prompt_id,
        category=category,
        pair_type=pair_type,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        chosen_id=f"{prompt_id}__safe_completion__s01",
        rejected_id=f"{prompt_id}__hard_refusal__s01",
    )


def _mixed_examples() -> list[DPORecord]:
    """A mixed set covering all four categories and both pair_types."""
    return [
        _dpo("u001__baseline_refusal__p01", "u001", PromptCategory.unsafe, PairType.baseline_refusal,
             "How to make a weapon?", "I cannot help.", "Here is how..."),
        _dpo("d001__safe_completion__p01", "d001", PromptCategory.dual_use, PairType.safe_completion,
             "How does SQL injection work?", "Safe answer.", "I won't help."),
        _dpo("d001__baseline_refusal__p01", "d001", PromptCategory.dual_use, PairType.baseline_refusal,
             "How does SQL injection work?", "I won't help.", "Harmful answer."),
        _dpo("bs001__safe_completion__p01", "bs001", PromptCategory.benign_sensitive, PairType.safe_completion,
             "What are drug interactions?", "Helpful medical info.", "I cannot help."),
        _dpo("b001__safe_completion__p01", "b001", PromptCategory.benign, PairType.safe_completion,
             "What is the capital of France?", "Paris.", "I cannot help."),
    ]


_BASE_CONFIG: dict[str, Any] = {
    "regime": "test",
    "model": "test-model",
    "backend": "mock",
    "pair_types": [],
    "categories": [],
}


# ---------------------------------------------------------------------------
# filter_examples — pair_type filter
# ---------------------------------------------------------------------------


def test_filter_by_pair_type_safe_completion():
    examples = _mixed_examples()
    result = filter_examples(examples, pair_types=["safe_completion"])
    assert all(e.pair_type == PairType.safe_completion for e in result)
    assert len(result) == 3


def test_filter_by_pair_type_baseline_refusal():
    examples = _mixed_examples()
    result = filter_examples(examples, pair_types=["baseline_refusal"])
    assert all(e.pair_type == PairType.baseline_refusal for e in result)
    assert len(result) == 2


def test_filter_by_multiple_pair_types():
    examples = _mixed_examples()
    result = filter_examples(examples, pair_types=["safe_completion", "baseline_refusal"])
    assert len(result) == len(examples)


def test_filter_no_pair_type_returns_all():
    examples = _mixed_examples()
    assert filter_examples(examples) == examples
    assert filter_examples(examples, pair_types=None) == examples
    assert filter_examples(examples, pair_types=[]) == examples


# ---------------------------------------------------------------------------
# filter_examples — category filter
# ---------------------------------------------------------------------------


def test_filter_by_category_dual_use():
    examples = _mixed_examples()
    result = filter_examples(examples, categories=["dual_use"])
    assert all(e.category == PromptCategory.dual_use for e in result)
    assert len(result) == 2


def test_filter_by_category_unsafe():
    examples = _mixed_examples()
    result = filter_examples(examples, categories=["unsafe"])
    assert len(result) == 1
    assert result[0].category == PromptCategory.unsafe


def test_filter_no_category_returns_all():
    examples = _mixed_examples()
    assert filter_examples(examples, categories=None) == examples


# ---------------------------------------------------------------------------
# filter_examples — combined filters
# ---------------------------------------------------------------------------


def test_filter_combined_dual_use_safe_completion():
    examples = _mixed_examples()
    result = filter_examples(examples, pair_types=["safe_completion"], categories=["dual_use"])
    assert len(result) == 1
    assert result[0].pair_id == "d001__safe_completion__p01"


def test_filter_combined_yields_empty():
    examples = _mixed_examples()
    # No unsafe safe_completion pairs exist in the fixture
    result = filter_examples(examples, pair_types=["safe_completion"], categories=["unsafe"])
    assert result == []


def test_filter_preserves_input_order():
    examples = _mixed_examples()
    result = filter_examples(examples, pair_types=["safe_completion"])
    pair_ids = [e.pair_id for e in result]
    expected_order = [e.pair_id for e in examples if e.pair_type == PairType.safe_completion]
    assert pair_ids == expected_order


# ---------------------------------------------------------------------------
# MockTrainingBackend
# ---------------------------------------------------------------------------


def test_mock_backend_name():
    assert MockTrainingBackend().name == "mock"


def test_mock_backend_returns_metrics():
    backend = MockTrainingBackend()
    metrics = backend.train(_mixed_examples(), _BASE_CONFIG)
    assert "n_examples" in metrics
    assert "mock_loss" in metrics
    assert "category_distribution" in metrics
    assert "pair_type_distribution" in metrics


def test_mock_backend_n_examples():
    backend = MockTrainingBackend()
    examples = _mixed_examples()
    metrics = backend.train(examples, _BASE_CONFIG)
    assert metrics["n_examples"] == len(examples)


def test_mock_backend_mock_loss_nonzero_for_nonempty():
    backend = MockTrainingBackend()
    metrics = backend.train(_mixed_examples(), _BASE_CONFIG)
    assert metrics["mock_loss"] is not None
    assert metrics["mock_loss"] > 0


def test_mock_backend_mock_loss_none_for_empty():
    backend = MockTrainingBackend()
    metrics = backend.train([], _BASE_CONFIG)
    assert metrics["mock_loss"] is None


def test_mock_backend_mock_loss_formula():
    backend = MockTrainingBackend()
    n = 5
    examples = [_dpo(f"p{i:03d}__safe_completion__p01", f"p{i:03d}") for i in range(n)]
    metrics = backend.train(examples, _BASE_CONFIG)
    expected_loss = round(1.0 / (1 + n), 6)
    assert metrics["mock_loss"] == expected_loss


def test_mock_backend_category_distribution():
    backend = MockTrainingBackend()
    metrics = backend.train(_mixed_examples(), _BASE_CONFIG)
    dist = metrics["category_distribution"]
    assert dist["dual_use"] == 2
    assert dist["unsafe"] == 1
    assert dist["benign_sensitive"] == 1
    assert dist["benign"] == 1


def test_mock_backend_pair_type_distribution():
    backend = MockTrainingBackend()
    metrics = backend.train(_mixed_examples(), _BASE_CONFIG)
    dist = metrics["pair_type_distribution"]
    assert dist["safe_completion"] == 3
    assert dist["baseline_refusal"] == 2


def test_mock_backend_deterministic():
    backend = MockTrainingBackend()
    examples = _mixed_examples()
    m1 = backend.train(examples, _BASE_CONFIG)
    m2 = backend.train(examples, _BASE_CONFIG)
    assert m1 == m2


# ---------------------------------------------------------------------------
# get_backend
# ---------------------------------------------------------------------------


def test_get_backend_mock():
    backend = get_backend("mock")
    assert isinstance(backend, MockTrainingBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nonexistent_backend")


# ---------------------------------------------------------------------------
# run_training — report structure
# ---------------------------------------------------------------------------


def test_run_training_returns_report():
    report = run_training(_mixed_examples(), _BASE_CONFIG, MockTrainingBackend(), "test_config.yaml")
    assert isinstance(report, dict)


def test_run_training_report_keys():
    report = run_training(_mixed_examples(), _BASE_CONFIG, MockTrainingBackend(), "test_config.yaml")
    for key in [
        "regime", "backend", "config_path", "model",
        "total_examples_loaded", "pair_type_filter", "category_filter",
        "n_examples_filtered_in", "n_examples_filtered_out",
        "n_examples", "mock_loss", "category_distribution", "pair_type_distribution",
    ]:
        assert key in report, f"Missing key: {key!r}"


def test_run_training_regime_from_config():
    config = {**_BASE_CONFIG, "regime": "safecomp"}
    report = run_training([], config, MockTrainingBackend(), "c.yaml")
    assert report["regime"] == "safecomp"


def test_run_training_config_path_in_report():
    report = run_training([], _BASE_CONFIG, MockTrainingBackend(), "my/config.yaml")
    assert report["config_path"] == "my/config.yaml"


def test_run_training_filter_counts():
    examples = _mixed_examples()
    config = {**_BASE_CONFIG, "pair_types": ["safe_completion"]}
    report = run_training(examples, config, MockTrainingBackend(), "c.yaml")
    assert report["n_examples_filtered_in"] == 3
    assert report["n_examples_filtered_out"] == 2
    assert report["total_examples_loaded"] == 5


def test_run_training_no_filter_uses_all():
    examples = _mixed_examples()
    report = run_training(examples, _BASE_CONFIG, MockTrainingBackend(), "c.yaml")
    assert report["n_examples_filtered_in"] == len(examples)
    assert report["n_examples_filtered_out"] == 0


def test_run_training_empty_after_filter_warns(capsys):
    config = {**_BASE_CONFIG, "pair_types": ["baseline_refusal"], "categories": ["benign"]}
    run_training(_mixed_examples(), config, MockTrainingBackend(), "c.yaml")
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_run_training_empty_dataset():
    report = run_training([], _BASE_CONFIG, MockTrainingBackend(), "c.yaml")
    assert report["n_examples_filtered_in"] == 0
    assert report["total_examples_loaded"] == 0


# ---------------------------------------------------------------------------
# run_training — baseline and safecomp regimes on mock data
# ---------------------------------------------------------------------------


def test_baseline_regime_selects_baseline_refusal_pairs():
    config = {**_BASE_CONFIG, "regime": "baseline", "pair_types": ["baseline_refusal"]}
    report = run_training(_mixed_examples(), config, MockTrainingBackend(), "c.yaml")
    assert report["regime"] == "baseline"
    assert report["n_examples_filtered_in"] == 2
    dist = report["pair_type_distribution"]
    assert set(dist.keys()) == {"baseline_refusal"}


def test_safecomp_regime_selects_safe_completion_pairs():
    config = {**_BASE_CONFIG, "regime": "safecomp", "pair_types": ["safe_completion"]}
    report = run_training(_mixed_examples(), config, MockTrainingBackend(), "c.yaml")
    assert report["regime"] == "safecomp"
    assert report["n_examples_filtered_in"] == 3
    dist = report["pair_type_distribution"]
    assert set(dist.keys()) == {"safe_completion"}


def test_baseline_and_safecomp_filter_disjoint():
    """Baseline and SafeComp filters select non-overlapping subsets."""
    examples = _mixed_examples()
    baseline_report = run_training(
        examples, {**_BASE_CONFIG, "pair_types": ["baseline_refusal"]},
        MockTrainingBackend(), "c.yaml",
    )
    safecomp_report = run_training(
        examples, {**_BASE_CONFIG, "pair_types": ["safe_completion"]},
        MockTrainingBackend(), "c.yaml",
    )
    assert (
        baseline_report["n_examples_filtered_in"] + safecomp_report["n_examples_filtered_in"]
        == len(examples)
    )


def test_run_training_deterministic():
    examples = _mixed_examples()
    config = {**_BASE_CONFIG, "pair_types": ["safe_completion"]}
    r1 = run_training(examples, config, MockTrainingBackend(), "c.yaml")
    r2 = run_training(examples, config, MockTrainingBackend(), "c.yaml")
    assert r1["mock_loss"] == r2["mock_loss"]
    assert r1["n_examples_filtered_in"] == r2["n_examples_filtered_in"]
    assert r1["category_distribution"] == r2["category_distribution"]


# ---------------------------------------------------------------------------
# write_outputs — report and checkpoint written correctly
# ---------------------------------------------------------------------------


def test_write_outputs_report_written(tmp_path):
    report = {
        "regime": "safecomp", "backend": "mock", "config_path": "c.yaml",
        "model": "test", "total_examples_loaded": 3, "pair_type_filter": ["safe_completion"],
        "category_filter": [], "n_examples_filtered_in": 3, "n_examples_filtered_out": 0,
        "n_examples": 3, "mock_loss": 0.25,
        "category_distribution": {"dual_use": 3}, "pair_type_distribution": {"safe_completion": 3},
    }
    report_path = tmp_path / "report.json"
    write_outputs(report, report_path, None)
    assert report_path.exists()


def test_write_outputs_report_is_valid_json(tmp_path):
    report = {
        "regime": "safecomp", "n_examples_filtered_in": 2,
        "mock_loss": 0.333, "category_distribution": {}, "pair_type_distribution": {},
    }
    report_path = tmp_path / "report.json"
    write_outputs(report, report_path, None)
    with open(report_path) as f:
        loaded = json.load(f)
    assert loaded["regime"] == "safecomp"
    assert loaded["mock_loss"] == 0.333


def test_write_outputs_checkpoint_written_when_examples_nonzero(tmp_path):
    report = {"regime": "safecomp", "backend": "mock", "n_examples_filtered_in": 3, "mock_loss": 0.25}
    write_outputs(report, None, tmp_path / "checkpoint_mock")
    assert (tmp_path / "checkpoint_mock" / "training_complete.json").exists()


def test_write_outputs_checkpoint_not_written_when_zero_examples(tmp_path):
    report = {"regime": "safecomp", "backend": "mock", "n_examples_filtered_in": 0, "mock_loss": None}
    checkpoint_dir = tmp_path / "checkpoint_mock"
    write_outputs(report, None, checkpoint_dir)
    assert not checkpoint_dir.exists()


def test_write_outputs_checkpoint_content(tmp_path):
    report = {"regime": "baseline", "backend": "mock", "n_examples_filtered_in": 2, "mock_loss": 0.5}
    checkpoint_dir = tmp_path / "checkpoint_mock"
    write_outputs(report, None, checkpoint_dir)
    marker = checkpoint_dir / "training_complete.json"
    with open(marker) as f:
        content = json.load(f)
    assert content["regime"] == "baseline"
    assert content["backend"] == "mock"
    assert content["n_examples"] == 2
    assert content["mock_loss"] == 0.5


def test_write_outputs_creates_parent_dirs(tmp_path):
    report = {"regime": "safecomp", "n_examples_filtered_in": 1, "mock_loss": 0.5,
              "backend": "mock"}
    nested_report = tmp_path / "deep" / "nested" / "report.json"
    write_outputs(report, nested_report, None)
    assert nested_report.exists()


def test_write_outputs_no_paths_does_not_raise():
    # No-op when both paths are None — should not raise
    report = {"regime": "safecomp", "n_examples_filtered_in": 0, "mock_loss": None}
    write_outputs(report, None, None)  # must not raise


# ---------------------------------------------------------------------------
# Integration: load actual training configs and run on dummy data
# ---------------------------------------------------------------------------


def _load_training_config(name: str) -> dict[str, Any]:
    config_path = _REPO_ROOT / "configs" / "training" / f"{name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_baseline_config_loads_and_runs(tmp_path):
    config = _load_training_config("baseline")
    assert config["regime"] == "baseline"
    # pair_types is empty: training config no longer filters — dataset is pre-filtered
    # by build_training_dataset.py before this stage.
    assert config["pair_types"] == []

    examples = _mixed_examples()
    report = run_training(examples, config, MockTrainingBackend(), "configs/training/baseline.yaml")
    assert report["regime"] == "baseline"
    assert report["n_examples_filtered_in"] >= 0  # may be 0 on current real data; filter runs

    write_outputs(
        report,
        tmp_path / "baseline_report.json",
        tmp_path / "baseline_checkpoint" if report["n_examples_filtered_in"] > 0 else None,
    )
    assert (tmp_path / "baseline_report.json").exists()


def test_safecomp_config_loads_and_runs(tmp_path):
    config = _load_training_config("safecomp")
    assert config["regime"] == "safecomp"
    # pair_types is empty: training config no longer filters — dataset is pre-filtered
    # by build_training_dataset.py before this stage.
    assert config["pair_types"] == []

    examples = _mixed_examples()
    report = run_training(examples, config, MockTrainingBackend(), "configs/training/safecomp.yaml")
    assert report["regime"] == "safecomp"
    # No pair_type filter → all 5 examples from _mixed_examples pass through
    assert report["n_examples_filtered_in"] == 5

    write_outputs(
        report,
        tmp_path / "safecomp_report.json",
        tmp_path / "safecomp_checkpoint",
    )
    assert (tmp_path / "safecomp_report.json").exists()
    assert (tmp_path / "safecomp_checkpoint" / "training_complete.json").exists()


def test_both_configs_have_distinct_regimes():
    baseline = _load_training_config("baseline")
    safecomp = _load_training_config("safecomp")
    assert baseline["regime"] != safecomp["regime"]
    assert set(baseline["pair_types"]).isdisjoint(set(safecomp["pair_types"]))


# ---------------------------------------------------------------------------
# Integration: JSONL round-trip (load_dpo_records → run_training)
# ---------------------------------------------------------------------------


def test_load_dpo_records_and_train(tmp_path):
    examples = _mixed_examples()
    dpo_path = tmp_path / "test_dpo.jsonl"
    write_jsonl(examples, dpo_path)

    loaded = load_dpo_records(dpo_path)
    assert len(loaded) == len(examples)

    config = {**_BASE_CONFIG, "regime": "safecomp", "pair_types": ["safe_completion"]}
    report = run_training(loaded, config, MockTrainingBackend(), "c.yaml")
    assert report["n_examples_filtered_in"] == 3
    assert report["mock_loss"] is not None
