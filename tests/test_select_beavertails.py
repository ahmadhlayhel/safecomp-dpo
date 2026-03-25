"""
Unit tests for select_prompts_beavertails.

Tests cover:
  - compute_pool_freq
  - assign_primary_labels (least_frequent_active, tie-break alphabetical)
  - select_records (count, subset integrity, coverage, determinism, floor)

All tests use synthetic records — no file I/O required.
Category sets are always derived dynamically; no counts are hardcoded.
bt_active_categories entries are treated as atomic strings throughout.
"""

from __future__ import annotations

import random

import pytest

from scripts.select_prompts_beavertails import (
    assign_primary_labels,
    compute_pool_freq,
    select_records,
)
from safecomp_dpo.schemas import PromptCategory, PromptRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_LABELS = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]


def make_record(prompt_id: str, active_categories: list[str]) -> PromptRecord:
    cats_full = {k: (k in active_categories) for k in ALL_LABELS}
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=f"A realistic unsafe prompt for testing: {prompt_id}",
        category=PromptCategory.unsafe,
        source="beavertails",
        source_id=prompt_id,
        metadata={
            "bt_categories": cats_full,
            "bt_active_categories": sorted(active_categories),
            "bt_split": "30k_train",
            "bt_original_index": 0,
            "bt_duplicate_count": 1,
            "bt_duplicate_source_ids": [prompt_id],
            "ingest_script": "acquire_prompts_beavertails.py",
            "ingest_config": "configs/acquisition/beavertails.yaml",
        },
    )


def run(
    records: list[PromptRecord],
    target_n: int,
    seed: int = 42,
    min_per_category: int = 10,
) -> tuple[list[PromptRecord], dict]:
    return select_records(records, target_n, seed, min_per_category)


# ---------------------------------------------------------------------------
# compute_pool_freq
# ---------------------------------------------------------------------------


def test_pool_freq_counts_each_label():
    records = [
        make_record("r1", ["animal_abuse", "violence,aiding_and_abetting,incitement"]),
        make_record("r2", ["violence,aiding_and_abetting,incitement"]),
    ]
    freq = compute_pool_freq(records)
    assert freq["animal_abuse"] == 1
    assert freq["violence,aiding_and_abetting,incitement"] == 2


def test_pool_freq_treats_labels_as_atomic():
    # Labels containing commas must NOT be split
    records = [make_record("r1", ["drug_abuse,weapons,banned_substance"])]
    freq = compute_pool_freq(records)
    assert "drug_abuse,weapons,banned_substance" in freq
    assert "drug_abuse" not in freq
    assert "weapons" not in freq


def test_pool_freq_only_active_labels_counted():
    records = [make_record("r1", ["self_harm"])]
    freq = compute_pool_freq(records)
    assert set(freq.keys()) == {"self_harm"}


def test_pool_freq_empty_records():
    assert compute_pool_freq([]) == {}


# ---------------------------------------------------------------------------
# assign_primary_labels — least_frequent_active
# ---------------------------------------------------------------------------


def test_primary_label_single_active():
    records = [make_record("r1", ["child_abuse"])]
    freq = compute_pool_freq(records)
    labels = assign_primary_labels(records, freq)
    assert labels["r1"] == "child_abuse"


def test_primary_label_assigns_least_frequent():
    # r1 has two labels; child_abuse is rarer → assigned to child_abuse
    records = [
        make_record("r1", ["violence,aiding_and_abetting,incitement"]) for _ in range(100)
    ] + [
        make_record(f"child_{i}", ["child_abuse"]) for i in range(10)
    ] + [
        make_record("r_multi", ["child_abuse", "violence,aiding_and_abetting,incitement"]),
    ]
    freq = compute_pool_freq(records)
    labels = assign_primary_labels(records, freq)
    assert labels["r_multi"] == "child_abuse"


def test_primary_label_tie_break_alphabetical():
    # Both labels have identical pool frequency → alphabetical first wins
    records = [
        make_record("r1", ["animal_abuse"]),
        make_record("r2", ["self_harm"]),
        make_record("r_tie", ["animal_abuse", "self_harm"]),
    ]
    freq = compute_pool_freq(records)
    # animal_abuse and self_harm both appear twice; alphabetically animal_abuse < self_harm
    labels = assign_primary_labels(records, freq)
    assert labels["r_tie"] == "animal_abuse"


def test_primary_label_deterministic():
    records = (
        [make_record("r1", ["child_abuse", "violence,aiding_and_abetting,incitement"])]
        + [make_record(f"r2_{i}", ["violence,aiding_and_abetting,incitement"]) for i in range(50)]
    )
    freq = compute_pool_freq(records)
    l1 = assign_primary_labels(records, freq)
    l2 = assign_primary_labels(records, freq)
    assert l1 == l2


def test_every_record_gets_exactly_one_primary_label():
    records = [
        make_record("r1", ["animal_abuse"]),
        make_record("r2", ["animal_abuse", "child_abuse"]),
        make_record("r3", ["violence,aiding_and_abetting,incitement"]),
    ]
    freq = compute_pool_freq(records)
    labels = assign_primary_labels(records, freq)
    assert set(labels.keys()) == {"r1", "r2", "r3"}
    for pid, label in labels.items():
        assert isinstance(label, str) and len(label) > 0


# ---------------------------------------------------------------------------
# select_records — count and subset integrity
# ---------------------------------------------------------------------------


def _build_pool(label_sizes: dict[str, int]) -> list[PromptRecord]:
    records = []
    for label, n in sorted(label_sizes.items()):
        for i in range(n):
            records.append(make_record(f"{label[:8]}_{i:04d}", [label]))
    return records


def test_total_selected_equals_target_n():
    pool = _build_pool({l: 100 for l in ALL_LABELS[:4]})
    selected, _ = run(pool, target_n=100, min_per_category=5)
    assert len(selected) == 100


def test_subset_integrity():
    pool = _build_pool({l: 80 for l in ALL_LABELS[:4]})
    selected, _ = run(pool, target_n=80)
    source_ids = {r.prompt_id for r in pool}
    assert all(r.prompt_id in source_ids for r in selected)


def test_no_duplicate_records_in_output():
    pool = _build_pool({l: 60 for l in ALL_LABELS[:4]})
    selected, _ = run(pool, target_n=60)
    ids = [r.prompt_id for r in selected]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Category coverage — dynamic, never hardcoded
# ---------------------------------------------------------------------------


def test_all_primary_buckets_covered():
    label_sizes = {l: 50 for l in ALL_LABELS[:6]}
    pool = _build_pool(label_sizes)
    selected, report = run(pool, target_n=100, min_per_category=5)
    primary_cats_in_output = {r["bt_primary_category"] for r in report["primary_label_allocation"]}
    assert primary_cats_in_output == set(label_sizes.keys())


def test_n_categories_derived_dynamically():
    for n in [2, 5, 8, 14]:
        labels = ALL_LABELS[:n]
        pool = _build_pool({l: 50 for l in labels})
        _, report = run(pool, target_n=n * 10, min_per_category=5)
        assert report["n_categories_in_pool"] == n


def test_active_category_coverage_includes_all_pool_labels():
    pool = _build_pool({l: 30 for l in ALL_LABELS[:4]})
    _, report = run(pool, target_n=40, min_per_category=5)
    coverage_labels = {row["bt_category"] for row in report["active_category_coverage"]}
    pool_labels = {l for l in ALL_LABELS[:4]}
    assert coverage_labels == pool_labels


def test_active_category_coverage_counts_multilabel():
    # Record with two active labels should contribute to both coverage counts
    violence = "violence,aiding_and_abetting,incitement"
    child = "child_abuse"
    records = (
        [make_record(f"v_{i}", [violence]) for i in range(50)]
        + [make_record(f"c_{i}", [child]) for i in range(20)]
        + [make_record(f"multi_{i}", [child, violence]) for i in range(5)]
    )
    _, report = run(records, target_n=40, min_per_category=5)
    cov = {row["bt_category"]: row for row in report["active_category_coverage"]}
    # Multi-label records contribute to both categories' coverage
    assert cov[violence]["records_with_category_in_selected"] > 0
    assert cov[child]["records_with_category_in_selected"] > 0


# ---------------------------------------------------------------------------
# Determinism and seed sensitivity
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_output():
    pool = _build_pool({l: 80 for l in ALL_LABELS[:5]})
    s1, _ = run(pool, target_n=100, seed=42)
    s2, _ = run(pool, target_n=100, seed=42)
    assert [r.prompt_id for r in s1] == [r.prompt_id for r in s2]


def test_different_seed_produces_different_output():
    pool = _build_pool({l: 80 for l in ALL_LABELS[:5]})
    s1, _ = run(pool, target_n=100, seed=42)
    s2, _ = run(pool, target_n=100, seed=99)
    assert [r.prompt_id for r in s1] != [r.prompt_id for r in s2]


def test_stable_sort_invariance():
    """Shuffling input records produces the same selected set."""
    pool = _build_pool({l: 60 for l in ALL_LABELS[:4]})
    ids_original = {r.prompt_id for r in run(pool, target_n=60)[0]}
    shuffled = pool.copy()
    random.Random(7).shuffle(shuffled)
    ids_shuffled = {r.prompt_id for r in run(shuffled, target_n=60)[0]}
    assert ids_original == ids_shuffled


# ---------------------------------------------------------------------------
# Floor handling
# ---------------------------------------------------------------------------


def test_min_floor_respected_when_bucket_large_enough():
    pool = _build_pool({l: 100 for l in ALL_LABELS[:4]})
    _, report = run(pool, target_n=100, min_per_category=15)
    for row in report["primary_label_allocation"]:
        assert row["selected_count"] >= 15


def test_small_bucket_capped_at_source_count():
    # One bucket has only 5 records; floor=10 → should take 5 not 10
    labels = ALL_LABELS[:4]
    pool = (
        [make_record(f"{labels[0]}_{i}", [labels[0]]) for i in range(5)]
        + [make_record(f"{l}_{i}", [l]) for l in labels[1:] for i in range(100)]
    )
    _, report = run(pool, target_n=100, min_per_category=10)
    small = next(r for r in report["primary_label_allocation"] if r["bt_primary_category"] == labels[0])
    assert small["selected_count"] == 5


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_report_required_keys():
    pool = _build_pool({l: 50 for l in ALL_LABELS[:3]})
    _, report = run(pool, target_n=50)
    for key in [
        "source", "strategy", "primary_label_rule", "seed", "min_per_category",
        "target_n", "actual_n", "total_source_records", "n_categories_in_pool",
        "primary_label_allocation", "active_category_coverage",
    ]:
        assert key in report, f"Missing report key: {key}"


def test_report_actual_n_matches_selected():
    pool = _build_pool({l: 60 for l in ALL_LABELS[:4]})
    selected, report = run(pool, target_n=80)
    assert report["actual_n"] == len(selected)


def test_report_primary_alloc_counts_sum_to_actual_n():
    pool = _build_pool({l: 60 for l in ALL_LABELS[:4]})
    selected, report = run(pool, target_n=80)
    assert sum(r["selected_count"] for r in report["primary_label_allocation"]) == report["actual_n"]


def test_report_source_is_beavertails():
    pool = _build_pool({"animal_abuse": 30, "child_abuse": 30})
    _, report = run(pool, target_n=20, min_per_category=5)
    assert report["source"] == "beavertails"
    assert report["primary_label_rule"] == "least_frequent_active"


def test_report_coverage_rate_consistent():
    pool = _build_pool({l: 50 for l in ALL_LABELS[:3]})
    selected, report = run(pool, target_n=50)
    for row in report["active_category_coverage"]:
        src = row["records_with_category_in_source"]
        sel = row["records_with_category_in_selected"]
        if src > 0:
            assert row["coverage_rate"] == round(sel / src, 4)
