"""
Unit tests for acquire_prompts_alpaca.process_rows.

All tests use synthetic rows — no HF network access required.
"""

from __future__ import annotations

from scripts.acquire_prompts_alpaca import _build_prompt, _normalize, _prompt_id, process_rows
from safecomp_dpo.schemas import PromptCategory, PromptRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SPLIT = "train"
CONFIG_PATH = "configs/acquisition/alpaca.yaml"


def make_row(
    instruction: str = "Explain the concept of supply and demand.",
    input_text: str = "",
    output: str = "Supply and demand is...",
) -> dict:
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": f"{instruction}\n\n{input_text}\n\n{output}",
    }


def run(rows: list[dict], min_chars: int = 20) -> tuple[list[PromptRecord], dict]:
    return process_rows(rows, SPLIT, min_chars, CONFIG_PATH)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_prompt_no_input():
    assert _build_prompt("What is photosynthesis?", "") == "What is photosynthesis?"


def test_build_prompt_with_input():
    result = _build_prompt("Summarize the following.", "The sky is blue.")
    assert result == "Summarize the following.\n\nThe sky is blue."


def test_build_prompt_strips_both():
    result = _build_prompt("  Tell me a joke.  ", "  Why did the chicken?  ")
    assert result == "Tell me a joke.\n\nWhy did the chicken?"


def test_build_prompt_whitespace_only_input_treated_as_empty():
    result = _build_prompt("What is gravity?", "   ")
    assert result == "What is gravity?"


# ---------------------------------------------------------------------------
# Hygiene filtering
# ---------------------------------------------------------------------------


def test_filters_empty_instruction():
    rows = [
        make_row(instruction=""),
        make_row(instruction="What are the benefits of exercise?"),
    ]
    records, stats = run(rows)
    assert len(records) == 1
    assert stats["after_empty_instruction_filter"] == 1


def test_filters_whitespace_only_instruction():
    rows = [make_row(instruction="   ")]
    records, stats = run(rows)
    assert records == []
    assert stats["after_empty_instruction_filter"] == 0


def test_filters_short_constructed_prompt():
    rows = [
        make_row(instruction="Hi"),  # too short
        make_row(instruction="What is the capital of France?"),
    ]
    records, stats = run(rows, min_chars=20)
    assert len(records) == 1
    assert stats["after_min_prompt_chars_filter"] == 1


def test_empty_instruction_counted_separately_from_short():
    rows = [
        make_row(instruction=""),       # empty instruction
        make_row(instruction="Hi"),     # short prompt
        make_row(instruction="What are the main causes of climate change?"),
    ]
    _, stats = run(rows, min_chars=20)
    assert stats["total_rows"] == 3
    assert stats["after_empty_instruction_filter"] == 2  # 1 empty removed
    assert stats["after_min_prompt_chars_filter"] == 1   # 1 short also removed
    assert stats["unique_after_dedup"] == 1


def test_prompt_with_input_length_accounts_for_both_parts():
    # instruction alone < 20 chars, but instruction + input >= 20
    rows = [make_row(instruction="Fix this:", input_text="print('hello world')")]
    records, stats = run(rows, min_chars=20)
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Deduplication and merged provenance
# ---------------------------------------------------------------------------


def test_dedup_collapses_identical_prompts():
    prompt = "List three benefits of daily meditation."
    rows = [make_row(instruction=prompt), make_row(instruction=prompt)]
    records, stats = run(rows)
    assert len(records) == 1
    assert stats["unique_after_dedup"] == 1


def test_dedup_canonical_is_lowest_index():
    prompt = "Describe the water cycle."
    rows = [make_row(instruction=prompt), make_row(instruction=prompt)]
    records, _ = run(rows)
    assert records[0].source_id == "0"
    assert records[0].metadata["alpaca_original_index"] == 0


def test_dedup_duplicate_count():
    prompt = "What is the Pythagorean theorem?"
    rows = [make_row(instruction=prompt)] * 3
    records, _ = run(rows)
    assert records[0].metadata["alpaca_duplicate_count"] == 3


def test_dedup_duplicate_source_ids_all_included():
    prompt = "Explain Newton's first law of motion."
    rows = [make_row(instruction=prompt)] * 3
    records, _ = run(rows)
    assert records[0].metadata["alpaca_duplicate_source_ids"] == ["0", "1", "2"]


def test_dedup_source_ids_sorted():
    prompt = "What is the speed of light?"
    rows = [make_row(instruction=prompt)] * 4
    records, _ = run(rows)
    ids = records[0].metadata["alpaca_duplicate_source_ids"]
    assert ids == sorted(ids)


def test_dedup_count_matches_source_ids_length():
    prompt = "How does photosynthesis work?"
    rows = [make_row(instruction=prompt)] * 5
    records, _ = run(rows)
    meta = records[0].metadata
    assert meta["alpaca_duplicate_count"] == len(meta["alpaca_duplicate_source_ids"])


def test_dedup_source_ids_include_canonical():
    prompt = "What is the difference between a virus and a bacterium?"
    rows = [make_row(instruction=prompt), make_row(instruction=prompt)]
    records, _ = run(rows)
    assert records[0].source_id in records[0].metadata["alpaca_duplicate_source_ids"]


def test_dedup_is_case_insensitive():
    rows = [
        make_row(instruction="Describe the process of osmosis."),
        make_row(instruction="DESCRIBE THE PROCESS OF OSMOSIS."),
    ]
    records, _ = run(rows)
    assert len(records) == 1


def test_dedup_is_strip_insensitive():
    rows = [
        make_row(instruction="What is the boiling point of water?"),
        make_row(instruction="  What is the boiling point of water?  "),
    ]
    records, _ = run(rows)
    assert len(records) == 1


def test_no_duplicates_count_is_one():
    rows = [make_row(instruction="Explain the concept of entropy.")]
    records, _ = run(rows)
    assert records[0].metadata["alpaca_duplicate_count"] == 1
    assert records[0].metadata["alpaca_duplicate_source_ids"] == ["0"]


def test_dedup_with_input_considered_in_key():
    # Same instruction, different input → different prompts → not collapsed
    rows = [
        make_row(instruction="Translate the following.", input_text="Hello world."),
        make_row(instruction="Translate the following.", input_text="Goodbye world."),
    ]
    records, _ = run(rows)
    assert len(records) == 2


# ---------------------------------------------------------------------------
# Schema and field contracts
# ---------------------------------------------------------------------------


def test_category_is_benign():
    rows = [make_row()]
    records, _ = run(rows)
    assert records[0].category == PromptCategory.benign


def test_created_at_is_none():
    rows = [make_row()]
    records, _ = run(rows)
    assert records[0].created_at is None


def test_source_is_alpaca():
    rows = [make_row()]
    records, _ = run(rows)
    assert records[0].source == "alpaca"


def test_prompt_id_format_no_input():
    rows = [make_row(instruction="Explain the greenhouse effect.")]
    records, _ = run(rows)
    assert records[0].prompt_id == "alpaca_train_000000"


def test_prompt_contains_instruction_and_input():
    rows = [make_row(instruction="Summarize this.", input_text="The cat sat on the mat.")]
    records, _ = run(rows)
    assert records[0].prompt == "Summarize this.\n\nThe cat sat on the mat."


def test_prompt_instruction_only_when_no_input():
    rows = [make_row(instruction="What is the speed of sound?", input_text="")]
    records, _ = run(rows)
    assert records[0].prompt == "What is the speed of sound?"


def test_source_id_is_string():
    rows = [make_row()]
    records, _ = run(rows)
    assert isinstance(records[0].source_id, str)


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_metadata_required_keys():
    rows = [make_row()]
    records, _ = run(rows)
    meta = records[0].metadata
    for key in [
        "alpaca_has_input", "alpaca_split", "alpaca_original_index",
        "alpaca_duplicate_count", "alpaca_duplicate_source_ids",
        "ingest_script", "ingest_config",
    ]:
        assert key in meta, f"Missing metadata key: {key}"


def test_metadata_no_instruction_text():
    # Full instruction text must NOT be stored in metadata
    rows = [make_row(instruction="What is quantum entanglement?")]
    records, _ = run(rows)
    assert "alpaca_instruction" not in records[0].metadata


def test_metadata_no_input_text():
    rows = [make_row(input_text="Some context here.")]
    records, _ = run(rows)
    assert "alpaca_input" not in records[0].metadata


def test_alpaca_has_input_true():
    rows = [make_row(instruction="Correct this sentence.", input_text="the cat sit on mat")]
    records, _ = run(rows)
    assert records[0].metadata["alpaca_has_input"] is True


def test_alpaca_has_input_false():
    rows = [make_row(instruction="What is the speed of light?", input_text="")]
    records, _ = run(rows)
    assert records[0].metadata["alpaca_has_input"] is False


def test_alpaca_has_input_false_for_whitespace_input():
    rows = [make_row(instruction="Define the concept of entropy.", input_text="   ")]
    records, _ = run(rows)
    assert records[0].metadata["alpaca_has_input"] is False


def test_alpaca_split_matches():
    rows = [make_row()]
    records, _ = run(rows)
    assert records[0].metadata["alpaca_split"] == SPLIT


def test_ingest_config_matches():
    rows = [make_row()]
    records, _ = run(rows)
    assert records[0].metadata["ingest_config"] == CONFIG_PATH


# ---------------------------------------------------------------------------
# Stats dict
# ---------------------------------------------------------------------------


def test_stats_keys_present():
    rows = [make_row()]
    _, stats = run(rows)
    for key in [
        "total_rows", "after_empty_instruction_filter",
        "after_min_prompt_chars_filter", "unique_after_dedup",
        "unique_with_input", "unique_without_input",
    ]:
        assert key in stats


def test_stats_with_without_input_sum_to_total_unique():
    rows = [
        make_row(instruction="What is gravity?"),
        make_row(instruction="Translate this.", input_text="Hello"),
    ]
    _, stats = run(rows)
    assert stats["unique_with_input"] + stats["unique_without_input"] == stats["unique_after_dedup"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_normalize_strips_and_lowercases():
    assert _normalize("  Hello World  ") == "hello world"


def test_prompt_id_format_helper():
    assert _prompt_id("train", 0) == "alpaca_train_000000"
    assert _prompt_id("train", 999) == "alpaca_train_000999"
