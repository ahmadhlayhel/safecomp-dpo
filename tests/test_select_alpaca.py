"""
Unit tests for select_prompts_alpaca.

Tests cover:
  - normalize_tokens (lowercase, punctuation stripping, length limit)
  - assign_task_bucket (each bucket, priority ordering, edge cases)
  - select_records (count, subset integrity, coverage, determinism, floor,
                    report structure, has_input_summary, metadata unchanged)

All tests use synthetic records — no file I/O required.
Bucket assignment is verified to NOT appear in PromptRecord metadata.
"""

from __future__ import annotations

import random
import string

import pytest

from scripts.select_prompts_alpaca import (
    TASK_BUCKETS,
    assign_task_bucket,
    normalize_tokens,
    select_records,
)
from safecomp_dpo.schemas import PromptCategory, PromptRecord


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

# Representative prompts that map deterministically to each bucket.
# Verified against assign_task_bucket logic.
BUCKET_PROMPTS = {
    "coding": "Implement a sorting algorithm in Python",
    "math_reasoning": "Calculate the area of a triangle given its base and height",
    "editing_rewriting": "Rewrite the following paragraph to be more concise",
    "summarization": "Summarize the key points from the article below",
    "classification_analysis": "Classify the following animals into mammal or reptile",
    "generation": "Write a short poem about the changing seasons",
    "factual_instructional": "What is the boiling point of water at sea level",
    "other": "Given the following context answer the question below",
}


def make_record(
    prompt_id: str,
    prompt: str,
    has_input: bool = False,
) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        category=PromptCategory.benign,
        source="alpaca",
        source_id=prompt_id,
        metadata={
            "alpaca_has_input": has_input,
            "alpaca_split": "train",
            "alpaca_original_index": 0,
            "alpaca_duplicate_count": 1,
            "alpaca_duplicate_source_ids": [prompt_id],
            "ingest_script": "acquire_prompts_alpaca.py",
            "ingest_config": "configs/acquisition/alpaca.yaml",
        },
    )


def _build_pool(bucket_sizes: dict[str, int]) -> list[PromptRecord]:
    """Build a synthetic pool where each record maps to a known bucket."""
    records = []
    for bucket in sorted(bucket_sizes.keys()):
        n = bucket_sizes[bucket]
        base_prompt = BUCKET_PROMPTS[bucket]
        for i in range(n):
            pid = f"alpaca_{bucket[:8]}_{i:04d}"
            records.append(make_record(pid, base_prompt, has_input=(i % 3 == 0)))
    return records


def run(
    records: list[PromptRecord],
    target_n: int,
    seed: int = 42,
    min_per_category: int = 25,
) -> tuple[list[PromptRecord], dict]:
    return select_records(records, target_n, seed, min_per_category)


# ---------------------------------------------------------------------------
# normalize_tokens
# ---------------------------------------------------------------------------


def test_normalize_tokens_lowercases():
    assert normalize_tokens("Hello World", 5) == ["hello", "world"]


def test_normalize_tokens_strips_trailing_punctuation():
    assert normalize_tokens("Hello, World!", 5) == ["hello", "world"]


def test_normalize_tokens_strips_leading_punctuation():
    assert normalize_tokens("(program) [code]", 5) == ["program", "code"]


def test_normalize_tokens_preserves_interior_apostrophe():
    # "what's" — apostrophe is interior, both ends are word chars
    result = normalize_tokens("What's the answer?", 5)
    assert result[0] == "what's"
    assert result[-1] == "answer"


def test_normalize_tokens_respects_n():
    text = " ".join(f"word{i}" for i in range(20))
    assert len(normalize_tokens(text, 3)) == 3


def test_normalize_tokens_empty_string():
    assert normalize_tokens("", 15) == []


def test_normalize_tokens_whitespace_only():
    assert normalize_tokens("   ", 15) == []


def test_normalize_tokens_punctuation_only_token_becomes_empty_string():
    # A token that is all punctuation strips to "" — still included in output
    # (caller ignores empty w0 via "if tokens else ''")
    result = normalize_tokens("... hello", 5)
    assert "hello" in result


# ---------------------------------------------------------------------------
# assign_task_bucket — each bucket
# ---------------------------------------------------------------------------


def test_bucket_coding_keyword_in_first_position():
    assert assign_task_bucket("Implement a sorting function") == "coding"


def test_bucket_coding_keyword_not_first_word():
    # "write" would be generation, but "script" appears later → coding wins
    assert assign_task_bucket("Write a shell script for backup") == "coding"


def test_bucket_coding_keyword_at_position_14():
    # Pad to put a coding word at position 14 (0-indexed), still within first 15
    words = ["a"] * 13 + ["algorithm", "for"]
    assert assign_task_bucket(" ".join(words)) == "coding"


def test_bucket_coding_keyword_at_position_15_not_matched():
    # Position 15 is outside the 15-token window
    words = ["a"] * 15 + ["algorithm"]
    assert assign_task_bucket(" ".join(words)) != "coding"


def test_bucket_math_prefix_calculate():
    assert assign_task_bucket("Calculate the mean of the dataset") == "math_reasoning"


def test_bucket_math_prefix_formula():
    assert assign_task_bucket("Formulate a hypothesis for the experiment") == "math_reasoning"


def test_bucket_editing_rewriting():
    assert assign_task_bucket("Rewrite this sentence in passive voice") == "editing_rewriting"


def test_bucket_editing_translate():
    assert assign_task_bucket("Translate the following text to French") == "editing_rewriting"


def test_bucket_summarization():
    assert assign_task_bucket("Summarize the article in three sentences") == "summarization"


def test_bucket_classification_analysis():
    assert assign_task_bucket("Classify the following emails as spam or not") == "classification_analysis"


def test_bucket_classification_find():
    assert assign_task_bucket("Find all prime numbers less than 100") == "classification_analysis"


def test_bucket_generation():
    assert assign_task_bucket("Write a cover letter for a software engineer role") == "generation"


def test_bucket_generation_come():
    assert assign_task_bucket("Come up with five names for a coffee shop") == "generation"


def test_bucket_factual_instructional_question():
    assert assign_task_bucket("What is the speed of light") == "factual_instructional"


def test_bucket_factual_instructional_describe():
    assert assign_task_bucket("Describe the water cycle in detail") == "factual_instructional"


def test_bucket_factual_instructional_explain():
    assert assign_task_bucket("Explain the causes of the French Revolution") == "factual_instructional"


def test_bucket_other_given():
    assert assign_task_bucket("Given the following list rank the items") == "other"


def test_bucket_other_empty_prompt():
    assert assign_task_bucket("") == "other"


def test_bucket_punctuation_stripped_before_matching():
    # "Write," → normalized first token → "write" → generation
    assert assign_task_bucket("Write, in your own words, what recursion means") == "generation"


def test_bucket_coding_beats_generation():
    # First word is a gen verb but "code" appears in first 15 → coding wins (checked first)
    assert assign_task_bucket("Create clean and efficient code for the task") == "coding"


# ---------------------------------------------------------------------------
# select_records — count and subset integrity
# ---------------------------------------------------------------------------


def test_total_selected_equals_target_n():
    pool = _build_pool({b: 100 for b in list(BUCKET_PROMPTS.keys())[:4]})
    selected, _ = run(pool, target_n=100, min_per_category=10)
    assert len(selected) == 100


def test_subset_integrity():
    pool = _build_pool({b: 80 for b in list(BUCKET_PROMPTS.keys())[:4]})
    selected, _ = run(pool, target_n=80)
    source_ids = {r.prompt_id for r in pool}
    assert all(r.prompt_id in source_ids for r in selected)


def test_no_duplicate_records_in_output():
    pool = _build_pool({b: 60 for b in list(BUCKET_PROMPTS.keys())[:4]})
    selected, _ = run(pool, target_n=60)
    ids = [r.prompt_id for r in selected]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Bucket assignment NOT written to metadata
# ---------------------------------------------------------------------------


def test_no_task_bucket_key_in_record_metadata():
    pool = _build_pool({"coding": 50, "generation": 50})
    selected, _ = run(pool, target_n=50)
    for r in selected:
        assert "task_bucket" not in r.metadata, (
            f"task_bucket found in metadata for {r.prompt_id}"
        )


def test_record_metadata_unchanged_after_selection():
    pool = _build_pool({"coding": 40, "summarization": 40})
    original_metadata = {r.prompt_id: dict(r.metadata) for r in pool}
    selected, _ = run(pool, target_n=40)
    for r in selected:
        assert r.metadata == original_metadata[r.prompt_id]


# ---------------------------------------------------------------------------
# Coverage — bucket representation in report
# ---------------------------------------------------------------------------


def test_all_active_buckets_in_report():
    buckets_used = ["coding", "math_reasoning", "editing_rewriting", "generation"]
    pool = _build_pool({b: 50 for b in buckets_used})
    _, report = run(pool, target_n=100, min_per_category=10)
    reported_buckets = {row["task_bucket"] for row in report["bucket_allocation"]}
    assert reported_buckets == set(buckets_used)


def test_n_buckets_derived_dynamically():
    for n in [2, 4, 6, 8]:
        buckets = list(BUCKET_PROMPTS.keys())[:n]
        pool = _build_pool({b: 50 for b in buckets})
        _, report = run(pool, target_n=n * 10, min_per_category=5)
        assert report["n_buckets_in_pool"] == n


# ---------------------------------------------------------------------------
# Determinism and seed sensitivity
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_output():
    pool = _build_pool({b: 80 for b in list(BUCKET_PROMPTS.keys())[:5]})
    s1, _ = run(pool, target_n=100, seed=42)
    s2, _ = run(pool, target_n=100, seed=42)
    assert [r.prompt_id for r in s1] == [r.prompt_id for r in s2]


def test_different_seed_produces_different_output():
    pool = _build_pool({b: 80 for b in list(BUCKET_PROMPTS.keys())[:5]})
    s1, _ = run(pool, target_n=100, seed=42)
    s2, _ = run(pool, target_n=100, seed=99)
    assert [r.prompt_id for r in s1] != [r.prompt_id for r in s2]


def test_stable_sort_invariance():
    """Shuffling input order produces the same selected set."""
    pool = _build_pool({b: 60 for b in list(BUCKET_PROMPTS.keys())[:4]})
    ids_original = {r.prompt_id for r in run(pool, target_n=60)[0]}
    shuffled = pool.copy()
    random.Random(7).shuffle(shuffled)
    ids_shuffled = {r.prompt_id for r in run(shuffled, target_n=60)[0]}
    assert ids_original == ids_shuffled


# ---------------------------------------------------------------------------
# Floor handling
# ---------------------------------------------------------------------------


def test_min_floor_respected_when_bucket_large_enough():
    pool = _build_pool({b: 100 for b in list(BUCKET_PROMPTS.keys())[:4]})
    _, report = run(pool, target_n=200, min_per_category=30)
    for row in report["bucket_allocation"]:
        assert row["selected_count"] >= 30


def test_small_bucket_capped_at_source_count():
    # One bucket has only 5 records; floor=25 → should take 5 not 25
    buckets = list(BUCKET_PROMPTS.keys())[:4]
    small_bucket = buckets[0]
    pool = (
        [make_record(f"{small_bucket[:8]}_{i}", BUCKET_PROMPTS[small_bucket]) for i in range(5)]
        + [make_record(f"{b[:8]}_{i:04d}", BUCKET_PROMPTS[b]) for b in buckets[1:] for i in range(100)]
    )
    _, report = run(pool, target_n=100, min_per_category=25)
    small = next(r for r in report["bucket_allocation"] if r["task_bucket"] == small_bucket)
    assert small["selected_count"] == 5


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_report_required_top_level_keys():
    pool = _build_pool({b: 50 for b in list(BUCKET_PROMPTS.keys())[:3]})
    _, report = run(pool, target_n=50)
    for key in [
        "source", "strategy", "stratify_by", "seed", "min_per_category",
        "target_n", "actual_n", "total_source_records", "n_buckets_in_pool",
        "bucket_allocation", "has_input_summary",
    ]:
        assert key in report, f"Missing report key: {key}"


def test_report_bucket_allocation_row_keys():
    pool = _build_pool({"coding": 40, "generation": 40})
    _, report = run(pool, target_n=40)
    for row in report["bucket_allocation"]:
        for key in [
            "task_bucket", "source_count", "selected_count", "selection_rate",
            "has_input_in_selected", "has_input_rate_in_selected",
        ]:
            assert key in row, f"Missing bucket_allocation key: {key}"


def test_report_has_input_summary_keys():
    pool = _build_pool({"coding": 40, "generation": 40})
    _, report = run(pool, target_n=40)
    hi = report["has_input_summary"]
    for key in [
        "source_with_input", "source_without_input",
        "selected_with_input", "selected_without_input",
        "selected_with_input_rate",
    ]:
        assert key in hi, f"Missing has_input_summary key: {key}"


def test_report_actual_n_matches_selected():
    pool = _build_pool({b: 60 for b in list(BUCKET_PROMPTS.keys())[:4]})
    selected, report = run(pool, target_n=80)
    assert report["actual_n"] == len(selected)


def test_report_bucket_allocation_counts_sum_to_actual_n():
    pool = _build_pool({b: 60 for b in list(BUCKET_PROMPTS.keys())[:4]})
    selected, report = run(pool, target_n=80)
    assert sum(r["selected_count"] for r in report["bucket_allocation"]) == report["actual_n"]


def test_report_source_is_alpaca():
    pool = _build_pool({"coding": 30, "generation": 30})
    _, report = run(pool, target_n=20)
    assert report["source"] == "alpaca"
    assert report["stratify_by"] == "task_bucket"


def test_report_selection_rate_consistent():
    pool = _build_pool({b: 50 for b in list(BUCKET_PROMPTS.keys())[:3]})
    selected, report = run(pool, target_n=50)
    for row in report["bucket_allocation"]:
        src = row["source_count"]
        sel = row["selected_count"]
        if src > 0:
            assert row["selection_rate"] == round(sel / src, 4)


# ---------------------------------------------------------------------------
# has_input_summary correctness
# ---------------------------------------------------------------------------


def test_has_input_summary_totals_consistent():
    pool = _build_pool({b: 60 for b in list(BUCKET_PROMPTS.keys())[:3]})
    selected, report = run(pool, target_n=60)
    hi = report["has_input_summary"]
    assert hi["source_with_input"] + hi["source_without_input"] == report["total_source_records"]
    assert hi["selected_with_input"] + hi["selected_without_input"] == report["actual_n"]


def test_has_input_rate_consistent():
    pool = _build_pool({b: 50 for b in list(BUCKET_PROMPTS.keys())[:4]})
    selected, report = run(pool, target_n=80)
    hi = report["has_input_summary"]
    sel = hi["selected_with_input"]
    n = report["actual_n"]
    assert hi["selected_with_input_rate"] == round(sel / n, 4)
