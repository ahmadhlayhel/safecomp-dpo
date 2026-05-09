"""
Tests for the standard benchmark evaluation layer.

Covers:
  - BenchmarkRecord schema and round-trip
  - BenchmarkPrompt schema
  - make_record_id convention
  - MockBenchmarkModelBackend (all behaviors, all benchmark types)
  - MockScorerBackend (tag parsing, StrongREJECT scores, fallback)
  - get_model_backend / get_scorer_backend / get_adapter registries
  - Each adapter: mock prompts, compute_metrics, JSONL load fallback path
  - run_benchmark() core function (per-record correctness, XSTest split injection)
  - compute_metrics() per benchmark:
      HarmBench    → asr
      StrongREJECT → mean_score
      DoNotAnswer  → percent_harmful
      XSTest       → orr (safe) + asr (unsafe)
      ORBench      → orr
      FalseRejectTest → compliance_rate
  - "safe_model" behavior gives ideal metrics on all benchmarks
  - "overrefusing_model" behavior gives ASR=0 but ORR=1.0
  - "unsafe_model" behavior gives worst-case safety metrics
  - Config file structure for all 6 benchmarks
  - BenchmarkRecord JSONL round-trip (write + load_benchmark_records)
  - End-to-end mock run for all 6 benchmarks
  - Determinism
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from safecomp_dpo.benchmarks import (
    ADAPTERS,
    BenchmarkPrompt,
    DoNotAnswerAdapter,
    FalseRejectTestAdapter,
    HarmBenchAdapter,
    MockBenchmarkModelBackend,
    MockScorerBackend,
    ORBenchAdapter,
    StrongREJECTAdapter,
    XSTestAdapter,
    get_adapter,
    get_model_backend,
    get_scorer_backend,
    make_record_id,
    run_benchmark,
)
from safecomp_dpo.io import load_benchmark_records, write_jsonl
from safecomp_dpo.schemas import BenchmarkRecord

_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent / "configs" / "eval" / "benchmarks"
)

_ALL_BENCHMARKS = [
    "harmbench", "strongreject", "do_not_answer",
    "xstest", "or_bench", "falsereject_test",
]


# ---------------------------------------------------------------------------
# BenchmarkRecord schema
# ---------------------------------------------------------------------------

class TestBenchmarkRecordSchema:
    def test_minimal_construction(self):
        r = BenchmarkRecord(
            record_id="harmbench_001__run__bench",
            benchmark="harmbench",
            prompt="Harmful prompt",
            response="I cannot help.",
            is_compliant=False,
        )
        assert r.is_compliant is False
        assert r.score is None
        assert r.split is None
        assert r.metadata == {}

    def test_with_score(self):
        r = BenchmarkRecord(
            record_id="sr_001__run__bench", benchmark="strongreject",
            prompt="p", response="r", is_compliant=True, score=0.7,
        )
        assert r.score == 0.7

    def test_with_split(self):
        r = BenchmarkRecord(
            record_id="xstest_safe_001__run__bench", benchmark="xstest",
            split="safe", prompt="p", response="r", is_compliant=True,
        )
        assert r.split == "safe"

    def test_json_round_trip(self):
        r = BenchmarkRecord(
            record_id="harmbench_001__run__bench", benchmark="harmbench",
            prompt="p", response="r", is_compliant=False,
            metadata={"model_backend": "mock"},
        )
        reloaded = BenchmarkRecord.model_validate(json.loads(r.model_dump_json()))
        assert reloaded.record_id == r.record_id
        assert reloaded.is_compliant == r.is_compliant
        assert reloaded.metadata["model_backend"] == "mock"


# ---------------------------------------------------------------------------
# make_record_id
# ---------------------------------------------------------------------------

class TestMakeRecordId:
    def test_basic(self):
        assert make_record_id("harmbench_001", "mock_run") == "harmbench_001__mock_run__bench"

    def test_ends_with_bench(self):
        assert make_record_id("x", "y").endswith("__bench")

    def test_embeds_prompt_id(self):
        assert "harmbench_001" in make_record_id("harmbench_001", "r")

    def test_embeds_run_id(self):
        assert "mock_run" in make_record_id("x", "mock_run")


# ---------------------------------------------------------------------------
# MockBenchmarkModelBackend
# ---------------------------------------------------------------------------

class TestMockBenchmarkModelBackend:
    def setup_method(self):
        self.m = MockBenchmarkModelBackend()

    def test_name(self):
        assert self.m.name == "mock"

    def test_returns_string(self):
        r = self.m.generate("test", {})
        assert isinstance(r, str) and len(r) > 0

    def test_embeds_mock_bench_tag(self):
        r = self.m.generate("test", {})
        assert "[MOCK BENCH" in r

    # safe_model behavior
    def test_safe_model_safety_benchmark_gives_hard_refusal(self):
        for b in ["harmbench", "strongreject", "do_not_answer"]:
            r = self.m.generate("p", {"mock_behavior": "safe_model", "benchmark": b})
            assert "hard_refusal" in r, f"{b} should give hard_refusal"

    def test_safe_model_overrefusal_benchmark_gives_helpful_answer(self):
        for b in ["or_bench", "falsereject_test"]:
            r = self.m.generate("p", {"mock_behavior": "safe_model", "benchmark": b})
            assert "helpful_answer" in r, f"{b} should give helpful_answer"

    def test_safe_model_xstest_safe_gives_helpful_answer(self):
        r = self.m.generate("p", {"mock_behavior": "safe_model", "_mock_split": "safe"})
        assert "helpful_answer" in r

    def test_safe_model_xstest_unsafe_gives_hard_refusal(self):
        r = self.m.generate("p", {"mock_behavior": "safe_model", "_mock_split": "unsafe"})
        assert "hard_refusal" in r

    # overrefusing_model behavior
    def test_overrefusing_always_refuses(self):
        for b in ["harmbench", "or_bench", "falsereject_test"]:
            r = self.m.generate("p", {"mock_behavior": "overrefusing_model", "benchmark": b})
            assert "hard_refusal" in r

    # unsafe_model behavior
    def test_unsafe_model_always_complies(self):
        for b in ["harmbench", "or_bench"]:
            r = self.m.generate("p", {"mock_behavior": "unsafe_model", "benchmark": b})
            assert "unsafe_compliance" in r

    def test_default_behavior_is_safe_model(self):
        # No mock_behavior → defaults to safe_model logic (safety benchmarks refuse)
        r = self.m.generate("p", {"benchmark": "harmbench"})
        assert "hard_refusal" in r

    def test_embeds_prompt_prefix(self):
        r = self.m.generate("Capital of France?", {})
        assert "Capital of France?" in r

    def test_long_prompt_truncated(self):
        r = self.m.generate("A" * 100, {})
        assert "..." in r

    def test_deterministic(self):
        r1 = self.m.generate("p", {"mock_behavior": "safe_model", "benchmark": "harmbench"})
        r2 = self.m.generate("p", {"mock_behavior": "safe_model", "benchmark": "harmbench"})
        assert r1 == r2


# ---------------------------------------------------------------------------
# MockScorerBackend
# ---------------------------------------------------------------------------

class TestMockScorerBackend:
    def setup_method(self):
        self.s = MockScorerBackend()

    def test_name(self):
        assert self.s.name == "mock"

    def test_hard_refusal_tag_not_compliant(self):
        is_c, _ = self.s.score("p", "[MOCK BENCH hard_refusal] text", "harmbench", {})
        assert is_c is False

    def test_unsafe_compliance_tag_compliant(self):
        is_c, _ = self.s.score("p", "[MOCK BENCH unsafe_compliance] text", "harmbench", {})
        assert is_c is True

    def test_helpful_answer_tag_compliant(self):
        is_c, _ = self.s.score("p", "[MOCK BENCH helpful_answer] text", "or_bench", {})
        assert is_c is True

    def test_safe_completion_tag_compliant(self):
        is_c, _ = self.s.score("p", "[MOCK BENCH safe_completion] text", "xstest", {})
        assert is_c is True

    def test_no_tag_defaults_to_compliant(self):
        is_c, _ = self.s.score("p", "some response with no tag", "harmbench", {})
        assert is_c is True  # conservative fallback for safety metrics

    def test_score_none_for_non_strongreject(self):
        for b in ["harmbench", "do_not_answer", "xstest", "or_bench", "falsereject_test"]:
            _, sc = self.s.score("p", "[MOCK BENCH hard_refusal] text", b, {})
            assert sc is None, f"{b} should have score=None"

    def test_score_zero_for_strongreject_refusal(self):
        _, sc = self.s.score("p", "[MOCK BENCH hard_refusal] text", "strongreject", {})
        assert sc == 0.0

    def test_score_nonzero_for_strongreject_compliance(self):
        _, sc = self.s.score("p", "[MOCK BENCH unsafe_compliance] text", "strongreject", {})
        assert sc is not None and sc > 0.0

    def test_score_nonzero_for_strongreject_helpful(self):
        _, sc = self.s.score("p", "[MOCK BENCH helpful_answer] text", "strongreject", {})
        assert sc is not None and sc > 0.0

    def test_deterministic(self):
        r1 = self.s.score("p", "[MOCK BENCH hard_refusal] t", "harmbench", {})
        r2 = self.s.score("p", "[MOCK BENCH hard_refusal] t", "harmbench", {})
        assert r1 == r2


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

class TestRegistries:
    def test_get_model_backend_mock(self):
        assert isinstance(get_model_backend("mock"), MockBenchmarkModelBackend)

    def test_get_scorer_backend_mock(self):
        assert isinstance(get_scorer_backend("mock"), MockScorerBackend)

    def test_get_model_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark model backend"):
            get_model_backend("vllm")

    def test_get_scorer_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer backend"):
            get_scorer_backend("nonexistent_scorer")

    def test_get_adapter_all_benchmarks(self):
        for name in _ALL_BENCHMARKS:
            adapter = get_adapter(name)
            assert adapter.benchmark_name == name

    def test_get_adapter_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_adapter("mmlu")

    def test_all_six_adapters_in_registry(self):
        assert set(ADAPTERS) == set(_ALL_BENCHMARKS)


# ---------------------------------------------------------------------------
# Adapter mock prompts
# ---------------------------------------------------------------------------

class TestAdapterMockPrompts:
    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_mock_prompts_non_empty(self, name):
        adapter = get_adapter(name)
        prompts = adapter._mock_prompts()
        assert len(prompts) >= 5

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_mock_prompts_have_correct_benchmark(self, name):
        adapter = get_adapter(name)
        for p in adapter._mock_prompts():
            assert p.benchmark == name

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_mock_prompts_have_unique_ids(self, name):
        adapter = get_adapter(name)
        ids = [p.prompt_id for p in adapter._mock_prompts()]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_mock_prompts_have_non_empty_text(self, name):
        adapter = get_adapter(name)
        for p in adapter._mock_prompts():
            assert p.prompt.strip()

    def test_xstest_has_both_splits(self):
        adapter = XSTestAdapter()
        prompts = adapter._mock_prompts()
        splits = {p.split for p in prompts}
        assert splits == {"safe", "unsafe"}

    def test_xstest_safe_and_unsafe_counts(self):
        adapter = XSTestAdapter()
        prompts = adapter._mock_prompts()
        safe = [p for p in prompts if p.split == "safe"]
        unsafe = [p for p in prompts if p.split == "unsafe"]
        assert len(safe) >= 3
        assert len(unsafe) >= 3

    def test_non_xstest_split_is_none(self):
        for name in ["harmbench", "strongreject", "do_not_answer", "or_bench", "falsereject_test"]:
            for p in get_adapter(name)._mock_prompts():
                assert p.split is None, f"{name}: split should be None"


# ---------------------------------------------------------------------------
# Adapter load() — mock fallback
# ---------------------------------------------------------------------------

class TestAdapterLoad:
    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_load_without_dataset_path_returns_mock(self, name, capsys):
        adapter = get_adapter(name)
        prompts = adapter.load({})
        assert len(prompts) >= 5
        captured = capsys.readouterr()
        assert "NOTE" in captured.err or "WARNING" in captured.err

    def test_load_with_missing_path_falls_back_to_mock(self, capsys):
        adapter = HarmBenchAdapter()
        prompts = adapter.load({"dataset_path": "/nonexistent/path/harmbench.jsonl"})
        assert len(prompts) >= 5
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_load_from_real_jsonl(self, tmp_path):
        # Write a minimal JSONL file and verify the adapter loads it
        jsonl = tmp_path / "harmbench.jsonl"
        rows = [
            {"prompt_id": "hb_001", "prompt": "Harmful prompt 1"},
            {"prompt_id": "hb_002", "prompt": "Harmful prompt 2"},
        ]
        jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        adapter = HarmBenchAdapter()
        prompts = adapter.load({"dataset_path": str(jsonl)})
        assert len(prompts) == 2
        assert prompts[0].prompt_id == "hb_001"
        assert prompts[0].prompt == "Harmful prompt 1"
        assert prompts[0].benchmark == "harmbench"

    def test_load_jsonl_without_prompt_id_auto_generates(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text(json.dumps({"prompt": "Some prompt"}), encoding="utf-8")
        adapter = HarmBenchAdapter()
        prompts = adapter.load({"dataset_path": str(jsonl)})
        assert len(prompts) == 1
        assert "harmbench" in prompts[0].prompt_id

    def test_load_jsonl_preserves_split(self, tmp_path):
        jsonl = tmp_path / "xstest.jsonl"
        rows = [
            {"prompt_id": "xs_s_001", "prompt": "Safe prompt", "split": "safe"},
            {"prompt_id": "xs_u_001", "prompt": "Unsafe prompt", "split": "unsafe"},
        ]
        jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        adapter = XSTestAdapter()
        prompts = adapter.load({"dataset_path": str(jsonl)})
        splits = {p.split for p in prompts}
        assert splits == {"safe", "unsafe"}


# ---------------------------------------------------------------------------
# run_benchmark() core function
# ---------------------------------------------------------------------------

def _run(benchmark: str, behavior: str = "safe_model") -> list[BenchmarkRecord]:
    adapter = get_adapter(benchmark)
    prompts = adapter._mock_prompts()
    config = {
        "benchmark": benchmark,
        "run_id": "test_run",
        "mock_behavior": behavior,
    }
    return run_benchmark(
        prompts,
        MockBenchmarkModelBackend(),
        MockScorerBackend(),
        config,
        "test_config.yaml",
    )


class TestRunBenchmark:
    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_one_record_per_prompt(self, name):
        adapter = get_adapter(name)
        n_prompts = len(adapter._mock_prompts())
        records = _run(name)
        assert len(records) == n_prompts

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_all_are_benchmark_records(self, name):
        for r in _run(name):
            assert isinstance(r, BenchmarkRecord)

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_benchmark_field_set(self, name):
        for r in _run(name):
            assert r.benchmark == name

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_record_ids_follow_convention(self, name):
        for r in _run(name):
            assert r.record_id.endswith("__bench")

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_record_ids_unique(self, name):
        records = _run(name)
        ids = [r.record_id for r in records]
        assert len(ids) == len(set(ids))

    def test_xstest_split_preserved_in_records(self):
        records = _run("xstest")
        splits = {r.split for r in records}
        assert splits == {"safe", "unsafe"}

    def test_non_xstest_split_none(self):
        for name in ["harmbench", "or_bench", "falsereject_test"]:
            for r in _run(name):
                assert r.split is None

    def test_strongreject_score_non_none_for_refusal(self):
        # safe_model → hard_refusal → score=0.0 (not None)
        records = _run("strongreject")
        for r in records:
            assert r.score is not None

    def test_non_strongreject_score_is_none(self):
        for name in ["harmbench", "do_not_answer", "xstest", "or_bench", "falsereject_test"]:
            for r in _run(name):
                assert r.score is None

    def test_metadata_has_backends(self):
        for r in _run("harmbench"):
            assert r.metadata["model_backend"] == "mock"
            assert r.metadata["scorer_backend"] == "mock"

    def test_metadata_has_mock_behavior(self):
        for r in _run("harmbench", "overrefusing_model"):
            assert r.metadata["mock_behavior"] == "overrefusing_model"

    def test_empty_prompt_list(self):
        config = {"benchmark": "harmbench", "run_id": "r", "mock_behavior": "safe_model"}
        records = run_benchmark([], MockBenchmarkModelBackend(), MockScorerBackend(), config, "cfg.yaml")
        assert records == []


# ---------------------------------------------------------------------------
# compute_metrics() per benchmark
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    # ---- HarmBench ----
    def test_harmbench_asr_zero_for_safe_model(self):
        records = _run("harmbench", "safe_model")
        metrics = HarmBenchAdapter().compute_metrics(records)
        assert metrics["asr"] == 0.0

    def test_harmbench_asr_one_for_unsafe_model(self):
        records = _run("harmbench", "unsafe_model")
        metrics = HarmBenchAdapter().compute_metrics(records)
        assert metrics["asr"] == 1.0

    def test_harmbench_asr_none_for_empty(self):
        metrics = HarmBenchAdapter().compute_metrics([])
        assert metrics["asr"] is None
        assert metrics["n"] == 0

    def test_harmbench_benchmark_field(self):
        metrics = HarmBenchAdapter().compute_metrics(_run("harmbench"))
        assert metrics["benchmark"] == "harmbench"

    # ---- StrongREJECT ----
    def test_strongreject_mean_score_zero_for_safe_model(self):
        records = _run("strongreject", "safe_model")
        metrics = StrongREJECTAdapter().compute_metrics(records)
        assert metrics["mean_score"] == 0.0

    def test_strongreject_mean_score_nonzero_for_unsafe_model(self):
        records = _run("strongreject", "unsafe_model")
        metrics = StrongREJECTAdapter().compute_metrics(records)
        assert metrics["mean_score"] is not None and metrics["mean_score"] > 0.0

    def test_strongreject_mean_score_none_for_empty(self):
        metrics = StrongREJECTAdapter().compute_metrics([])
        assert metrics["mean_score"] is None

    # ---- Do-Not-Answer ----
    def test_dna_percent_harmful_zero_for_safe_model(self):
        records = _run("do_not_answer", "safe_model")
        metrics = DoNotAnswerAdapter().compute_metrics(records)
        assert metrics["percent_harmful"] == 0.0

    def test_dna_percent_harmful_100_for_unsafe_model(self):
        records = _run("do_not_answer", "unsafe_model")
        metrics = DoNotAnswerAdapter().compute_metrics(records)
        assert metrics["percent_harmful"] == 100.0

    def test_dna_percent_harmful_none_for_empty(self):
        metrics = DoNotAnswerAdapter().compute_metrics([])
        assert metrics["percent_harmful"] is None

    # ---- XSTest ----
    def test_xstest_orr_zero_for_safe_model(self):
        records = _run("xstest", "safe_model")
        metrics = XSTestAdapter().compute_metrics(records)
        assert metrics["safe"]["orr"] == 0.0

    def test_xstest_asr_zero_for_safe_model(self):
        records = _run("xstest", "safe_model")
        metrics = XSTestAdapter().compute_metrics(records)
        assert metrics["unsafe"]["asr"] == 0.0

    def test_xstest_orr_one_for_overrefusing_model(self):
        records = _run("xstest", "overrefusing_model")
        metrics = XSTestAdapter().compute_metrics(records)
        assert metrics["safe"]["orr"] == 1.0

    def test_xstest_asr_zero_for_overrefusing_model(self):
        records = _run("xstest", "overrefusing_model")
        metrics = XSTestAdapter().compute_metrics(records)
        assert metrics["unsafe"]["asr"] == 0.0

    def test_xstest_orr_zero_for_unsafe_model(self):
        records = _run("xstest", "unsafe_model")
        metrics = XSTestAdapter().compute_metrics(records)
        assert metrics["safe"]["orr"] == 0.0

    def test_xstest_asr_one_for_unsafe_model(self):
        records = _run("xstest", "unsafe_model")
        metrics = XSTestAdapter().compute_metrics(records)
        assert metrics["unsafe"]["asr"] == 1.0

    def test_xstest_no_records_gives_none(self):
        metrics = XSTestAdapter().compute_metrics([])
        assert metrics["safe"]["orr"] is None
        assert metrics["unsafe"]["asr"] is None

    # ---- OR-Bench ----
    def test_orbench_orr_zero_for_safe_model(self):
        records = _run("or_bench", "safe_model")
        metrics = ORBenchAdapter().compute_metrics(records)
        assert metrics["orr"] == 0.0

    def test_orbench_orr_one_for_overrefusing_model(self):
        records = _run("or_bench", "overrefusing_model")
        metrics = ORBenchAdapter().compute_metrics(records)
        assert metrics["orr"] == 1.0

    def test_orbench_orr_none_for_empty(self):
        metrics = ORBenchAdapter().compute_metrics([])
        assert metrics["orr"] is None

    # ---- FalseReject-Test ----
    def test_falsereject_compliance_one_for_safe_model(self):
        records = _run("falsereject_test", "safe_model")
        metrics = FalseRejectTestAdapter().compute_metrics(records)
        assert metrics["compliance_rate"] == 1.0

    def test_falsereject_compliance_zero_for_overrefusing_model(self):
        records = _run("falsereject_test", "overrefusing_model")
        metrics = FalseRejectTestAdapter().compute_metrics(records)
        assert metrics["compliance_rate"] == 0.0

    def test_falsereject_compliance_none_for_empty(self):
        metrics = FalseRejectTestAdapter().compute_metrics([])
        assert metrics["compliance_rate"] is None


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------

class TestBenchmarkConfigs:
    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_config_file_exists(self, name):
        cfg_file = _CONFIG_DIR / f"{name}.yaml"
        assert cfg_file.exists(), f"Missing: {cfg_file}"

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_config_has_benchmark_key(self, name):
        cfg = yaml.safe_load((_CONFIG_DIR / f"{name}.yaml").read_text())
        assert cfg.get("benchmark") == name

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_config_has_run_id(self, name):
        cfg = yaml.safe_load((_CONFIG_DIR / f"{name}.yaml").read_text())
        assert "run_id" in cfg

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_config_has_mock_backends(self, name):
        cfg = yaml.safe_load((_CONFIG_DIR / f"{name}.yaml").read_text())
        assert cfg.get("model_backend") == "mock"
        assert cfg.get("scorer_backend") == "mock"

    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_config_runnable_end_to_end(self, name):
        cfg = yaml.safe_load((_CONFIG_DIR / f"{name}.yaml").read_text())
        adapter = get_adapter(name)
        model = get_model_backend(cfg["model_backend"])
        scorer = get_scorer_backend(cfg["scorer_backend"])
        prompts = adapter.load(cfg)
        records = run_benchmark(prompts, model, scorer, cfg, f"{name}.yaml")
        assert len(records) == len(prompts)
        metrics = adapter.compute_metrics(records)
        assert "benchmark" in metrics


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------

class TestBenchmarkRecordIO:
    def test_write_and_load(self, tmp_path):
        records = _run("harmbench")
        path = tmp_path / "harmbench_results.jsonl"
        write_jsonl(records, path)
        reloaded = load_benchmark_records(path)
        assert len(reloaded) == len(records)
        for orig, reload in zip(records, reloaded):
            assert reload.record_id == orig.record_id
            assert reload.is_compliant == orig.is_compliant
            assert reload.benchmark == orig.benchmark

    def test_strongreject_scores_survive_round_trip(self, tmp_path):
        records = _run("strongreject", "unsafe_model")
        path = tmp_path / "sr_results.jsonl"
        write_jsonl(records, path)
        reloaded = load_benchmark_records(path)
        for orig, reload in zip(records, reloaded):
            assert reload.score == orig.score

    def test_xstest_splits_survive_round_trip(self, tmp_path):
        records = _run("xstest")
        path = tmp_path / "xstest_results.jsonl"
        write_jsonl(records, path)
        reloaded = load_benchmark_records(path)
        for orig, reload in zip(records, reloaded):
            assert reload.split == orig.split


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("name", _ALL_BENCHMARKS)
    def test_deterministic(self, name):
        r1 = _run(name)
        r2 = _run(name)
        assert [x.record_id for x in r1] == [x.record_id for x in r2]
        assert [x.is_compliant for x in r1] == [x.is_compliant for x in r2]


# ---------------------------------------------------------------------------
# End-to-end: ideal model gets perfect scores on all benchmarks
# ---------------------------------------------------------------------------

class TestIdealModelScores:
    """A "safe_model" should get perfect (best-case) scores on every benchmark."""

    def _metrics(self, name: str) -> dict:
        adapter = get_adapter(name)
        return adapter.compute_metrics(_run(name, "safe_model"))

    def test_harmbench_asr_is_zero(self):
        assert self._metrics("harmbench")["asr"] == 0.0

    def test_strongreject_score_is_zero(self):
        assert self._metrics("strongreject")["mean_score"] == 0.0

    def test_dna_percent_harmful_is_zero(self):
        assert self._metrics("do_not_answer")["percent_harmful"] == 0.0

    def test_xstest_safe_orr_is_zero(self):
        assert self._metrics("xstest")["safe"]["orr"] == 0.0

    def test_xstest_unsafe_asr_is_zero(self):
        assert self._metrics("xstest")["unsafe"]["asr"] == 0.0

    def test_orbench_orr_is_zero(self):
        assert self._metrics("or_bench")["orr"] == 0.0

    def test_falsereject_compliance_is_one(self):
        assert self._metrics("falsereject_test")["compliance_rate"] == 1.0


# ---------------------------------------------------------------------------
# End-to-end: over-refusing model gets bad ORR scores but good safety scores
# ---------------------------------------------------------------------------

class TestOverRefusingModelScores:
    def _metrics(self, name: str) -> dict:
        adapter = get_adapter(name)
        return adapter.compute_metrics(_run(name, "overrefusing_model"))

    def test_harmbench_asr_zero(self):
        # Refuses everything → ASR = 0 (safe on this metric)
        assert self._metrics("harmbench")["asr"] == 0.0

    def test_xstest_safe_orr_one(self):
        # Over-refuses safe prompts → ORR = 1.0 (worst case)
        assert self._metrics("xstest")["safe"]["orr"] == 1.0

    def test_orbench_orr_one(self):
        assert self._metrics("or_bench")["orr"] == 1.0

    def test_falsereject_compliance_zero(self):
        assert self._metrics("falsereject_test")["compliance_rate"] == 0.0
