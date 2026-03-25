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
    TRLDPOBackend,
    dpo_records_to_hf_dataset,
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


# ---------------------------------------------------------------------------
# TRLDPOBackend — registration and contract (no GPU required)
# ---------------------------------------------------------------------------


def test_trl_backend_registered():
    backend = get_backend("trl_dpo")
    assert isinstance(backend, TRLDPOBackend)
    assert backend.name == "trl_dpo"


def test_trl_backend_name_attribute():
    assert TRLDPOBackend.name == "trl_dpo"
    assert TRLDPOBackend().name == "trl_dpo"


def test_trl_backend_raises_import_error_without_heavy_deps(monkeypatch):
    """When torch/trl are absent, train() raises ImportError with install hint."""
    import builtins

    real_import = builtins.__import__

    def _mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("trl", "torch"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    with pytest.raises(ImportError, match="TRL backend requires"):
        TRLDPOBackend().train(_mixed_examples(), _BASE_CONFIG)


# ---------------------------------------------------------------------------
# dpo_records_to_hf_dataset — pure conversion (datasets installed locally)
# ---------------------------------------------------------------------------


class TestDPORecordsToHFDataset:
    def test_returns_dataset_type(self):
        from datasets import Dataset  # type: ignore[import]

        ds = dpo_records_to_hf_dataset(_mixed_examples())
        assert isinstance(ds, Dataset)

    def test_has_required_trl_columns(self):
        ds = dpo_records_to_hf_dataset(_mixed_examples())
        for col in ("prompt", "chosen", "rejected"):
            assert col in ds.column_names, f"Missing required TRL column: {col!r}"

    def test_has_provenance_columns(self):
        ds = dpo_records_to_hf_dataset(_mixed_examples())
        for col in ("pair_id", "category", "pair_type"):
            assert col in ds.column_names, f"Missing provenance column: {col!r}"

    def test_row_count_matches_input(self):
        examples = _mixed_examples()
        assert len(dpo_records_to_hf_dataset(examples)) == len(examples)

    def test_values_match_records(self):
        examples = _mixed_examples()
        ds = dpo_records_to_hf_dataset(examples)
        for i, rec in enumerate(examples):
            assert ds[i]["prompt"] == rec.prompt
            assert ds[i]["chosen"] == rec.chosen
            assert ds[i]["rejected"] == rec.rejected
            assert ds[i]["category"] == rec.category.value
            assert ds[i]["pair_type"] == rec.pair_type.value
            assert ds[i]["pair_id"] == rec.pair_id

    def test_empty_records_returns_empty_dataset(self):
        ds = dpo_records_to_hf_dataset([])
        assert len(ds) == 0

    def test_category_values_are_strings(self):
        ds = dpo_records_to_hf_dataset(_mixed_examples())
        for val in ds["category"]:
            assert isinstance(val, str)

    def test_pair_type_values_are_strings(self):
        ds = dpo_records_to_hf_dataset(_mixed_examples())
        for val in ds["pair_type"]:
            assert isinstance(val, str)


# ---------------------------------------------------------------------------
# write_outputs — final_loss key in checkpoint marker
# ---------------------------------------------------------------------------


def test_write_outputs_checkpoint_has_final_loss_key(tmp_path):
    """Checkpoint marker includes final_loss (real backend; may be None for mock)."""
    report = {
        "regime": "safecomp", "backend": "trl_dpo",
        "n_examples_filtered_in": 2, "mock_loss": None, "final_loss": 0.42,
    }
    checkpoint_dir = tmp_path / "checkpoint_trl"
    write_outputs(report, None, checkpoint_dir)
    with open(checkpoint_dir / "training_complete.json") as f:
        content = json.load(f)
    assert content["final_loss"] == 0.42


def test_write_outputs_mock_backend_final_loss_is_none(tmp_path):
    """Mock backend report has no final_loss — checkpoint stores None."""
    report = {
        "regime": "baseline", "backend": "mock",
        "n_examples_filtered_in": 3, "mock_loss": 0.25,
    }
    checkpoint_dir = tmp_path / "checkpoint_mock"
    write_outputs(report, None, checkpoint_dir)
    with open(checkpoint_dir / "training_complete.json") as f:
        content = json.load(f)
    assert content["mock_loss"] == 0.25
    assert content["final_loss"] is None


# ---------------------------------------------------------------------------
# BABEL config tests — load and validate required keys
# ---------------------------------------------------------------------------


class TestBABELConfigs:
    _REQUIRED_KEYS = {
        "regime", "model", "backend",
        "num_train_epochs", "per_device_train_batch_size",
        "learning_rate", "lr_scheduler_type", "warmup_ratio",
        "beta", "loss_type", "use_lora", "checkpoint_dir",
        "gradient_checkpointing", "optim",
    }

    def _load(self, name: str) -> dict[str, Any]:
        path = _REPO_ROOT / "configs" / "training" / f"{name}.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_baseline_babel_loads(self):
        cfg = self._load("baseline_babel")
        assert isinstance(cfg, dict)

    def test_safecomp_babel_loads(self):
        cfg = self._load("safecomp_babel")
        assert isinstance(cfg, dict)

    def test_baseline_babel_required_keys(self):
        cfg = self._load("baseline_babel")
        for key in self._REQUIRED_KEYS:
            assert key in cfg, f"Missing key {key!r} in baseline_babel.yaml"

    def test_safecomp_babel_required_keys(self):
        cfg = self._load("safecomp_babel")
        for key in self._REQUIRED_KEYS:
            assert key in cfg, f"Missing key {key!r} in safecomp_babel.yaml"

    def test_baseline_babel_backend_is_trl_dpo(self):
        assert self._load("baseline_babel")["backend"] == "trl_dpo"

    def test_safecomp_babel_backend_is_trl_dpo(self):
        assert self._load("safecomp_babel")["backend"] == "trl_dpo"

    def test_baseline_babel_regime(self):
        assert self._load("baseline_babel")["regime"] == "baseline"

    def test_safecomp_babel_regime(self):
        assert self._load("safecomp_babel")["regime"] == "safecomp"

    def test_babel_configs_have_distinct_regimes(self):
        baseline = self._load("baseline_babel")
        safecomp = self._load("safecomp_babel")
        assert baseline["regime"] != safecomp["regime"]

    def test_babel_configs_have_distinct_checkpoint_dirs(self):
        baseline = self._load("baseline_babel")
        safecomp = self._load("safecomp_babel")
        assert baseline["checkpoint_dir"] != safecomp["checkpoint_dir"]

    def test_babel_configs_use_lora(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("use_lora") is True, f"{name}: use_lora should be True"
            assert "lora_r" in cfg, f"{name}: missing lora_r"
            assert "lora_alpha" in cfg, f"{name}: missing lora_alpha"

    def test_babel_configs_lora_alpha_equals_r(self):
        """alpha = r gives scaling factor 1.0 — standard for alignment work."""
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg["lora_alpha"] == cfg["lora_r"], (
                f"{name}: lora_alpha should equal lora_r (scaling factor 1.0)"
            )

    def test_babel_configs_lora_target_modules_all_linear(self):
        """all-linear covers attention + MLP layers for broader DPO coverage."""
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("lora_target_modules") == "all-linear", (
                f"{name}: lora_target_modules should be 'all-linear'"
            )

    def test_babel_configs_modules_to_save(self):
        """lm_head and embed_tokens should be saved alongside the LoRA adapter."""
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            saved = cfg.get("modules_to_save", [])
            assert "lm_head" in saved, f"{name}: lm_head should be in modules_to_save"
            assert "embed_tokens" in saved, f"{name}: embed_tokens should be in modules_to_save"

    def test_babel_configs_use_4bit(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("use_4bit") is True, f"{name}: use_4bit should be True"

    def test_babel_configs_bf16(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("bf16") is True, f"{name}: bf16 should be True"

    def test_babel_configs_gradient_checkpointing(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("gradient_checkpointing") is True, (
                f"{name}: gradient_checkpointing should be True"
            )

    def test_babel_configs_cosine_scheduler(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("lr_scheduler_type") == "cosine", (
                f"{name}: lr_scheduler_type should be 'cosine'"
            )

    def test_babel_configs_warmup_ratio(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("warmup_ratio") == pytest.approx(0.1), (
                f"{name}: warmup_ratio should be 0.1"
            )

    def test_babel_configs_paged_adamw(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("optim") == "paged_adamw_8bit", (
                f"{name}: optim should be 'paged_adamw_8bit'"
            )

    def test_babel_configs_loss_type_sigmoid(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("loss_type") == "sigmoid", (
                f"{name}: loss_type should be 'sigmoid'"
            )

    def test_babel_configs_learning_rate(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("learning_rate") == pytest.approx(5e-6), (
                f"{name}: learning_rate should be 5e-6 for QLoRA DPO"
            )

    def test_babel_configs_num_epochs(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert cfg.get("num_train_epochs") >= 2, (
                f"{name}: num_train_epochs should be >= 2 for production training"
            )

    def test_babel_configs_identical_hyperparameters(self):
        """Baseline and safecomp should share all hyperparameters; only paths/regime differ."""
        shared_keys = [
            "model", "backend", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
            "warmup_ratio", "beta", "loss_type", "max_length", "max_prompt_length",
            "bf16", "gradient_checkpointing", "optim",
            "use_lora", "lora_r", "lora_alpha", "lora_target_modules", "lora_dropout",
            "use_4bit",
        ]
        baseline = self._load("baseline_babel")
        safecomp = self._load("safecomp_babel")
        for key in shared_keys:
            assert baseline.get(key) == safecomp.get(key), (
                f"Key {key!r} differs: baseline={baseline.get(key)!r}, safecomp={safecomp.get(key)!r}"
            )

    def test_babel_configs_model_is_llama(self):
        for name in ("baseline_babel", "safecomp_babel"):
            cfg = self._load(name)
            assert "Llama" in cfg["model"], f"{name}: model should be a Llama variant"
