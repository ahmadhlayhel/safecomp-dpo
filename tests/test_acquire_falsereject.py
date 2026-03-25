"""
Unit tests for acquire_prompts_falsereject.process_rows.

All tests use synthetic rows — no HF network access required.
"""

from __future__ import annotations

import pytest

from scripts.acquire_prompts_falsereject import _normalize, _prompt_id, process_rows
from safecomp_dpo.schemas import PromptCategory, PromptRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SPLIT = "train"
CONFIG_PATH = "configs/acquisition/falsereject.yaml"


def make_row(
    prompt: str,
    category: int = 1,
    category_text: str = "hate_speech",
) -> dict:
    return {
        "prompt": prompt,
        "category": category,
        "category_text": category_text,
        "instruct_response": "some response",
        "cot_response": {"reasoning_content": "...", "solution": "..."},
    }


def run(rows: list[dict], min_chars: int = 20) -> tuple[list[PromptRecord], dict]:
    return process_rows(rows, SPLIT, min_chars, CONFIG_PATH)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_filters_short_prompts():
    rows = [
        make_row("short"),  # < 20 chars
        make_row("Can you explain what defamation means in legal terms?"),
    ]
    records, stats = run(rows)
    assert len(records) == 1
    assert stats["after_min_chars_filter"] == 1


def test_all_short_returns_empty():
    rows = [make_row("hi"), make_row("ok")]
    records, stats = run(rows)
    assert records == []
    assert stats["unique_after_dedup"] == 0


def test_no_label_filter_applied():
    # All rows are candidates regardless of category value
    rows = [
        make_row("How do I safely dispose of household chemicals?", category=5),
        make_row("What are the legal requirements for starting a business?", category=12),
    ]
    records, stats = run(rows)
    assert len(records) == 2
    assert stats["total_rows"] == 2


# ---------------------------------------------------------------------------
# Deduplication and merged provenance
# ---------------------------------------------------------------------------


def test_dedup_collapses_identical_prompts():
    prompt = "What medications interact with alcohol?"
    rows = [make_row(prompt), make_row(prompt)]
    records, stats = run(rows)
    assert len(records) == 1
    assert stats["unique_after_dedup"] == 1


def test_dedup_canonical_is_lowest_index():
    prompt = "How do I write a strongly worded letter?"
    rows = [make_row(prompt), make_row(prompt)]
    records, _ = run(rows)
    assert records[0].source_id == "0"
    assert records[0].metadata["fr_original_index"] == 0


def test_dedup_duplicate_count():
    prompt = "Can you explain the concept of money laundering?"
    rows = [make_row(prompt)] * 3
    records, _ = run(rows)
    assert records[0].metadata["fr_duplicate_count"] == 3


def test_dedup_duplicate_source_ids_all_included():
    prompt = "What are the effects of long-term drug use on the brain?"
    rows = [make_row(prompt)] * 3
    records, _ = run(rows)
    ids = records[0].metadata["fr_duplicate_source_ids"]
    assert ids == ["0", "1", "2"]


def test_dedup_source_ids_include_canonical():
    prompt = "How does insider trading work?"
    rows = [make_row(prompt), make_row(prompt)]
    records, _ = run(rows)
    ids = records[0].metadata["fr_duplicate_source_ids"]
    assert records[0].source_id in ids


def test_dedup_source_ids_sorted():
    prompt = "Is it legal to record a phone call?"
    rows = [make_row(prompt)] * 4
    records, _ = run(rows)
    ids = records[0].metadata["fr_duplicate_source_ids"]
    assert ids == sorted(ids)


def test_dedup_count_matches_source_ids_length():
    prompt = "What are common signs of financial fraud?"
    rows = [make_row(prompt)] * 5
    records, _ = run(rows)
    meta = records[0].metadata
    assert meta["fr_duplicate_count"] == len(meta["fr_duplicate_source_ids"])


def test_dedup_unique_prompts_not_collapsed():
    rows = [
        make_row("What does the term 'conspiracy' mean in law?"),
        make_row("How do extremist groups recruit members online?"),
    ]
    records, stats = run(rows)
    assert len(records) == 2
    assert stats["unique_after_dedup"] == 2


def test_dedup_is_case_insensitive():
    rows = [
        make_row("How do I pick a lock if I lose my key?"),
        make_row("HOW DO I PICK A LOCK IF I LOSE MY KEY?"),
    ]
    records, _ = run(rows)
    assert len(records) == 1


def test_dedup_is_strip_insensitive():
    rows = [
        make_row("What are symptoms of a drug overdose?"),
        make_row("  What are symptoms of a drug overdose?  "),
    ]
    records, _ = run(rows)
    assert len(records) == 1


def test_no_duplicates_duplicate_count_is_one():
    rows = [make_row("Can you explain how ransomware works?")]
    records, _ = run(rows)
    assert records[0].metadata["fr_duplicate_count"] == 1
    assert records[0].metadata["fr_duplicate_source_ids"] == ["0"]


# ---------------------------------------------------------------------------
# Schema and field contracts
# ---------------------------------------------------------------------------


def test_category_is_benign_sensitive():
    rows = [make_row("What are the warning signs of human trafficking?")]
    records, _ = run(rows)
    assert records[0].category == PromptCategory.benign_sensitive


def test_created_at_is_none():
    rows = [make_row("How do cults manipulate their members?")]
    records, _ = run(rows)
    assert records[0].created_at is None


def test_source_is_falsereject():
    rows = [make_row("What does the Fourth Amendment protect against?")]
    records, _ = run(rows)
    assert records[0].source == "falsereject"


def test_prompt_id_format():
    rows = [make_row("Can you describe how phishing attacks work?")]
    records, _ = run(rows)
    assert records[0].prompt_id == "fr_train_000000"


def test_prompt_stripped():
    rows = [make_row("  What are the signs of domestic abuse?  ")]
    records, _ = run(rows)
    assert records[0].prompt == "What are the signs of domestic abuse?"


def test_source_id_is_string():
    rows = [make_row("How does blackmail work legally?")]
    records, _ = run(rows)
    assert isinstance(records[0].source_id, str)


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_metadata_required_keys():
    rows = [make_row("What is the legal definition of stalking?")]
    records, _ = run(rows)
    meta = records[0].metadata
    for key in [
        "fr_category_id", "fr_category_text", "fr_split",
        "fr_original_index", "fr_duplicate_count", "fr_duplicate_source_ids",
        "ingest_script", "ingest_config",
    ]:
        assert key in meta, f"Missing metadata key: {key}"


def test_fr_category_id_preserved():
    rows = [make_row("Explain how identity theft occurs.", category=7, category_text="financial_crime")]
    records, _ = run(rows)
    assert records[0].metadata["fr_category_id"] == 7


def test_fr_category_text_preserved():
    rows = [make_row("Explain how identity theft occurs.", category=7, category_text="financial_crime")]
    records, _ = run(rows)
    assert records[0].metadata["fr_category_text"] == "financial_crime"


def test_fr_split_matches():
    rows = [make_row("What are the risks of sharing personal data online?")]
    records, _ = run(rows)
    assert records[0].metadata["fr_split"] == SPLIT


def test_fr_original_index_matches_canonical():
    rows = [make_row("How do scammers use social engineering?")]
    records, _ = run(rows)
    assert records[0].metadata["fr_original_index"] == 0


def test_ingest_config_matches():
    rows = [make_row("What is a Ponzi scheme?")]
    records, _ = run(rows)
    assert records[0].metadata["ingest_config"] == CONFIG_PATH


# ---------------------------------------------------------------------------
# Stats dict and category distribution
# ---------------------------------------------------------------------------


def test_stats_keys_present():
    rows = [make_row("How do fraudsters forge documents?")]
    _, stats = run(rows)
    for key in ["total_rows", "after_min_chars_filter", "unique_after_dedup", "category_distribution"]:
        assert key in stats


def test_stats_counts_consistent():
    rows = [
        make_row("How does money laundering work in practice?"),
        make_row("hi"),  # too short
        make_row("How does money laundering work in practice?"),  # duplicate
        make_row("What are the signs of elder financial abuse?"),
    ]
    _, stats = run(rows, min_chars=20)
    assert stats["total_rows"] == 4
    assert stats["after_min_chars_filter"] == 3
    assert stats["unique_after_dedup"] == 2


def test_category_distribution_keys_are_strings():
    rows = [
        make_row("What does sedition mean?", category=3, category_text="political"),
        make_row("How does bribery work?", category=5, category_text="financial"),
    ]
    _, stats = run(rows)
    for key in stats["category_distribution"]:
        assert isinstance(key, str)


def test_category_distribution_counts_unique_records():
    rows = [
        make_row("What does sedition mean?", category=3, category_text="political"),
        make_row("How does sedition differ from treason?", category=3, category_text="political"),
        make_row("How does bribery work?", category=5, category_text="financial"),
    ]
    _, stats = run(rows)
    dist = stats["category_distribution"]
    assert sum(dist.values()) == 3  # 3 unique records


def test_category_distribution_collapses_duplicates():
    prompt = "What is seditious libel?"
    rows = [
        make_row(prompt, category=3, category_text="political"),
        make_row(prompt, category=3, category_text="political"),
    ]
    _, stats = run(rows)
    dist = stats["category_distribution"]
    assert sum(dist.values()) == 1  # 1 unique record after dedup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_normalize_strips_and_lowercases():
    assert _normalize("  Hello World  ") == "hello world"


def test_prompt_id_format_helper():
    assert _prompt_id("train", 0) == "fr_train_000000"
    assert _prompt_id("train", 99) == "fr_train_000099"
