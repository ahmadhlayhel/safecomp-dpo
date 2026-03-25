"""
Unit tests for select_prompts_falsereject._allocate and select_records.

All tests use synthetic records — no file I/O required.
Category counts are always derived dynamically; no hardcoded category totals.
"""

from __future__ import annotations

import random

import pytest

from scripts.select_prompts_falsereject import _allocate, select_records
from safecomp_dpo.schemas import PromptCategory, PromptRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRATIFY_BY = "fr_category_id"


def make_record(prompt_id: str, category_id: int, category_text: str = "") -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=f"A realistic benign-sensitive prompt for testing purposes: {prompt_id}",
        category=PromptCategory.benign_sensitive,
        source="falsereject",
        source_id=prompt_id,
        metadata={
            "fr_category_id": category_id,
            "fr_category_text": category_text or f"category_{category_id}",
            "fr_split": "train",
            "fr_original_index": 0,
            "fr_duplicate_count": 1,
            "fr_duplicate_source_ids": [prompt_id],
            "ingest_script": "acquire_prompts_falsereject.py",
            "ingest_config": "configs/acquisition/falsereject.yaml",
        },
    )


def make_pool(category_sizes: dict[int, int]) -> list[PromptRecord]:
    """Build a synthetic pool with given per-category sizes."""
    records = []
    for cat_id, n in sorted(category_sizes.items()):
        for i in range(n):
            pid = f"fr_train_{cat_id:03d}_{i:04d}"
            records.append(make_record(pid, cat_id))
    return records


def run(
    records: list[PromptRecord],
    target_n: int,
    seed: int = 42,
    min_per_category: int = 10,
) -> tuple[list[PromptRecord], dict]:
    return select_records(records, target_n, seed, min_per_category, STRATIFY_BY)


# ---------------------------------------------------------------------------
# _allocate — unit tests
# ---------------------------------------------------------------------------


def test_allocate_sum_equals_target():
    counts = {1: 200, 2: 100, 3: 50}
    quota = _allocate(counts, 100, 10)
    assert sum(quota.values()) == 100


def test_allocate_floor_applied():
    # Category 3 has 5 records — smaller than min_per_category=10
    counts = {1: 200, 2: 100, 3: 5}
    quota = _allocate(counts, 100, 10)
    assert quota[3] == 5  # can't exceed source count


def test_allocate_large_category_gets_more():
    counts = {1: 500, 2: 50}
    quota = _allocate(counts, 200, 10)
    assert quota[1] > quota[2]


def test_allocate_all_same_size_equal_distribution():
    counts = {1: 100, 2: 100, 3: 100, 4: 100}
    quota = _allocate(counts, 40, 5)
    # All same size → equal share
    assert len(set(quota.values())) == 1
    assert sum(quota.values()) == 40


def test_allocate_base_total_equals_target_skips_phase2():
    # If base allocations exactly meet target, phase 2 is skipped
    counts = {1: 10, 2: 10, 3: 10}
    quota = _allocate(counts, 30, 10)
    assert quota == {1: 10, 2: 10, 3: 10}
    assert sum(quota.values()) == 30


def test_allocate_no_remaining_capacity_stops_at_base():
    # All categories have exactly min_per_category records
    counts = {1: 10, 2: 10}
    quota = _allocate(counts, 15, 10)
    # base_total = 20 >= 15, remaining_budget <= 0 → return base
    assert sum(quota.values()) == 20  # edge case: base_total > target_n


def test_allocate_single_category_gets_all():
    counts = {1: 500}
    quota = _allocate(counts, 100, 10)
    assert quota[1] == 100
    assert sum(quota.values()) == 100


def test_allocate_shortfall_category_capped_at_count():
    # Category 2 has 3 records — floor would want 10 but can only take 3
    counts = {1: 500, 2: 3}
    quota = _allocate(counts, 100, 10)
    assert quota[2] == 3
    assert sum(quota.values()) == 100


def test_allocate_deterministic():
    counts = {1: 200, 2: 150, 3: 80, 4: 40, 5: 15}
    q1 = _allocate(counts, 200, 10)
    q2 = _allocate(counts, 200, 10)
    assert q1 == q2


def test_allocate_larger_remainder_wins():
    # Controlled example: verify largest-remainder assignment
    counts = {1: 100, 2: 100, 3: 100}
    # base = {1:10, 2:10, 3:10}, remaining_budget = 7
    # remaining_cap = {1:90, 2:90, 3:90}, total_cap = 270
    # exact each = 7 * 90 / 270 = 2.333...
    # floored each = 2, leftover = 1
    # all same fraction → tie-break by key asc → key 1 gets the extra
    quota = _allocate(counts, 37, 10)
    assert sum(quota.values()) == 37
    # One category gets one more than the others
    values = sorted(quota.values())
    assert values[-1] == values[-2] + 1 or values[-1] == values[0]  # one extra somewhere


# ---------------------------------------------------------------------------
# select_records — count and subset integrity
# ---------------------------------------------------------------------------


def test_total_selected_equals_target_n():
    records = make_pool({1: 500, 2: 300, 3: 200, 4: 100})
    selected, _ = run(records, target_n=200)
    assert len(selected) == 200


def test_subset_integrity():
    records = make_pool({1: 100, 2: 100, 3: 100})
    selected, _ = run(records, target_n=50)
    input_ids = {r.prompt_id for r in records}
    assert all(r.prompt_id in input_ids for r in selected)


def test_no_duplicate_records_in_output():
    records = make_pool({1: 100, 2: 100})
    selected, _ = run(records, target_n=50)
    ids = [r.prompt_id for r in selected]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Category coverage — always derived dynamically
# ---------------------------------------------------------------------------


def test_all_categories_covered():
    category_sizes = {1: 50, 2: 30, 3: 15, 4: 8}
    records = make_pool(category_sizes)
    selected, report = run(records, target_n=50, min_per_category=5)
    categories_in_input = set(category_sizes.keys())
    categories_in_output = {r.metadata[STRATIFY_BY] for r in selected}
    assert categories_in_output == categories_in_input


def test_category_coverage_is_dynamic_not_hardcoded():
    # Works with 2 categories, 7 categories — any count
    for n_cats in [2, 5, 7, 12]:
        category_sizes = {i: 50 for i in range(n_cats)}
        records = make_pool(category_sizes)
        selected, report = run(records, target_n=min(n_cats * 10, n_cats * 50), min_per_category=5)
        assert report["n_categories_in_pool"] == n_cats
        cats_out = {r.metadata[STRATIFY_BY] for r in selected}
        assert cats_out == set(range(n_cats))


def test_small_category_gets_its_full_count_not_floor():
    # Category 2 has 3 records; floor is 10 — should get 3 (not 10)
    records = make_pool({1: 500, 2: 3})
    selected, _ = run(records, target_n=100, min_per_category=10)
    cat2_selected = [r for r in selected if r.metadata[STRATIFY_BY] == 2]
    assert len(cat2_selected) == 3


def test_larger_category_gets_proportionally_more():
    records = make_pool({1: 400, 2: 100})
    selected, report = run(records, target_n=100, min_per_category=5)
    counts = {cat["fr_category_id"]: cat["selected_count"] for cat in report["categories"]}
    assert counts[1] > counts[2]


# ---------------------------------------------------------------------------
# Determinism and seed sensitivity
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_output():
    records = make_pool({1: 200, 2: 150, 3: 80})
    s1, _ = run(records, target_n=100, seed=42)
    s2, _ = run(records, target_n=100, seed=42)
    assert [r.prompt_id for r in s1] == [r.prompt_id for r in s2]


def test_different_seed_produces_different_output():
    records = make_pool({1: 200, 2: 200, 3: 200})
    s1, _ = run(records, target_n=100, seed=42)
    s2, _ = run(records, target_n=100, seed=99)
    # With large pools and different seeds, selections will differ
    assert [r.prompt_id for r in s1] != [r.prompt_id for r in s2]


def test_stable_sort_invariance():
    """Shuffling input records should produce the same selected set."""
    records = make_pool({1: 100, 2: 100, 3: 100})
    selected_ids_original = {r.prompt_id for r in run(records, target_n=60)[0]}

    shuffled = records.copy()
    random.Random(7).shuffle(shuffled)
    selected_ids_shuffled = {r.prompt_id for r in run(shuffled, target_n=60)[0]}

    assert selected_ids_original == selected_ids_shuffled


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_report_required_keys():
    records = make_pool({1: 100, 2: 100})
    _, report = run(records, target_n=50)
    for key in [
        "source", "strategy", "stratify_by", "seed", "min_per_category",
        "target_n", "actual_n", "total_source_records", "n_categories_in_pool",
        "categories",
    ]:
        assert key in report, f"Missing report key: {key}"


def test_report_actual_n_matches_selected():
    records = make_pool({1: 200, 2: 100, 3: 50})
    selected, report = run(records, target_n=100)
    assert report["actual_n"] == len(selected)


def test_report_n_categories_derived_dynamically():
    for n in [2, 4, 8]:
        records = make_pool({i: 50 for i in range(n)})
        _, report = run(records, target_n=n * 10, min_per_category=5)
        assert report["n_categories_in_pool"] == n


def test_report_category_selected_counts_sum_to_actual_n():
    records = make_pool({1: 300, 2: 200, 3: 100, 4: 50})
    selected, report = run(records, target_n=150)
    assert sum(c["selected_count"] for c in report["categories"]) == report["actual_n"]


def test_report_total_source_records():
    records = make_pool({1: 100, 2: 200})
    _, report = run(records, target_n=50)
    assert report["total_source_records"] == 300


def test_report_category_entries_match_input_categories():
    category_sizes = {5: 100, 12: 80, 23: 40}
    records = make_pool(category_sizes)
    _, report = run(records, target_n=60, min_per_category=5)
    report_cat_ids = {cat[STRATIFY_BY] for cat in report["categories"]}
    assert report_cat_ids == set(category_sizes.keys())


def test_report_selection_rate_consistent():
    records = make_pool({1: 100, 2: 50})
    selected, report = run(records, target_n=30, min_per_category=5)
    for cat in report["categories"]:
        expected_rate = round(cat["selected_count"] / cat["source_count"], 4)
        assert cat["selection_rate"] == expected_rate


# ---------------------------------------------------------------------------
# Min-per-category floor
# ---------------------------------------------------------------------------


def test_min_per_category_floor_respected():
    # All categories have >= min_per_category records; each should get at least that
    records = make_pool({1: 500, 2: 500, 3: 500})
    selected, report = run(records, target_n=200, min_per_category=15)
    for cat in report["categories"]:
        assert cat["selected_count"] >= 15


def test_min_floor_does_not_exceed_source_count():
    # Category with 5 records and floor=10 → gets 5, not 10
    records = make_pool({1: 500, 2: 5})
    selected, report = run(records, target_n=100, min_per_category=10)
    cat2 = next(c for c in report["categories"] if c[STRATIFY_BY] == 2)
    assert cat2["selected_count"] == 5
