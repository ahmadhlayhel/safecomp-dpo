"""Tests for scripts/acquire_prompts_dualuse.py."""

import json
import sys
from pathlib import Path

import pytest

# Allow running tests without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from acquire_prompts_dualuse import (
    _discover_files,
    _is_valid_raw_id,
    _load_jsonl_files,
    _normalize,
    _prompt_id,
    process_rows,
)
from safecomp_dpo.schemas import PromptCategory, PromptRecord


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = "configs/acquisition/dualuse.yaml"


def _make_keep(
    raw_id: str = "alice_harm_001",
    prompt: str = "What household chemicals should never be mixed together and why?",
    domain: str = "harm_prevention",
    author: str = "alice",
    model: str = "gpt-4o",
    template_version: str = "v1",
    verdict: str = "KEEP",
    reason: str = "Clear dual-use: safety knowledge with misuse potential.",
) -> dict:
    return {
        "raw_id": raw_id,
        "prompt": prompt,
        "domain": domain,
        "author": author,
        "generation_model": model,
        "generator_template_version": template_version,
        "judge_verdict": verdict,
        "judge_reason": reason,
    }


def _make_drop(
    raw_id: str = "alice_harm_drop_001",
    prompt: str = "Step-by-step guide to making chlorine gas at home.",
    domain: str = "harm_prevention",
    author: str = "alice",
    verdict: str = "DROP",
    reason: str = "Unambiguously harmful; no dual-use framing possible.",
) -> dict:
    return {
        "raw_id": raw_id,
        "prompt": prompt,
        "domain": domain,
        "author": author,
        "generation_model": "gpt-4o",
        "generator_template_version": "v1",
        "judge_verdict": verdict,
        "judge_reason": reason,
    }


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_strips_and_lowercases():
    assert _normalize("  Hello World  ") == "hello world"


def test_normalize_empty():
    assert _normalize("") == ""


# ---------------------------------------------------------------------------
# _prompt_id
# ---------------------------------------------------------------------------


def test_prompt_id_basic():
    assert _prompt_id("alice_harm_001") == "du_alice_harm_001"


def test_prompt_id_no_double_underscore_by_default():
    pid = _prompt_id("foo_bar_001")
    assert "__" not in pid


# ---------------------------------------------------------------------------
# _is_valid_raw_id
# ---------------------------------------------------------------------------


def test_is_valid_raw_id_alphanumeric():
    assert _is_valid_raw_id("alice001") is True


def test_is_valid_raw_id_with_single_underscores():
    assert _is_valid_raw_id("alice_harm_001") is True


def test_is_valid_raw_id_rejects_double_underscore():
    assert _is_valid_raw_id("foo__bar") is False


def test_is_valid_raw_id_rejects_leading_double_underscore():
    assert _is_valid_raw_id("__foo") is False


def test_is_valid_raw_id_rejects_trailing_double_underscore():
    assert _is_valid_raw_id("foo__") is False


def test_is_valid_raw_id_rejects_hyphen():
    assert _is_valid_raw_id("alice-001") is False


def test_is_valid_raw_id_rejects_space():
    assert _is_valid_raw_id("alice 001") is False


def test_is_valid_raw_id_rejects_dot():
    assert _is_valid_raw_id("alice.001") is False


def test_is_valid_raw_id_rejects_empty():
    assert _is_valid_raw_id("") is False


# ---------------------------------------------------------------------------
# process_rows — happy path
# ---------------------------------------------------------------------------


def test_process_rows_happy_path():
    keep = [_make_keep()]
    drop = [_make_drop()]
    records, drops, stats = process_rows(keep, drop, 20, CONFIG_PATH)

    assert len(records) == 1
    assert len(drops) == 1
    assert stats["unique_keep_records"] == 1
    assert stats["total_keep_rows"] == 1
    assert stats["total_drop_rows"] == 1


def test_process_rows_returns_prompt_records():
    keep = [_make_keep()]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert isinstance(records[0], PromptRecord)


def test_process_rows_category_is_dual_use():
    keep = [_make_keep()]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert records[0].category == PromptCategory.dual_use


def test_process_rows_source_is_dualuse_team_v1():
    keep = [_make_keep()]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert records[0].source == "dualuse_team_v1"


def test_process_rows_source_id_is_raw_id():
    keep = [_make_keep(raw_id="alice_harm_001")]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert records[0].source_id == "alice_harm_001"


def test_process_rows_prompt_id_convention():
    keep = [_make_keep(raw_id="alice_harm_001")]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert records[0].prompt_id == "du_alice_harm_001"


def test_process_rows_created_at_is_none():
    keep = [_make_keep()]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert records[0].created_at is None


# ---------------------------------------------------------------------------
# process_rows — metadata fields
# ---------------------------------------------------------------------------


def test_process_rows_metadata_fields_present():
    keep = [_make_keep(
        raw_id="alice_harm_001",
        domain="harm_prevention",
        author="alice",
        model="gpt-4o",
        template_version="v1",
        verdict="KEEP",
        reason="Test reason.",
    )]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    md = records[0].metadata

    assert md["du_domain"] == "harm_prevention"
    assert md["du_author"] == "alice"
    assert md["du_generation_model"] == "gpt-4o"
    assert md["du_generator_template_version"] == "v1"
    assert md["du_judge_verdict"] == "KEEP"
    assert md["du_judge_reason"] == "Test reason."
    assert md["du_raw_id"] == "alice_harm_001"
    assert md["ingest_script"] == "acquire_prompts_dualuse.py"
    assert md["ingest_config"] == CONFIG_PATH


def test_process_rows_metadata_duplicate_count_single():
    keep = [_make_keep()]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert records[0].metadata["du_duplicate_count"] == 1


def test_process_rows_metadata_duplicate_source_ids_single():
    keep = [_make_keep(raw_id="alice_harm_001")]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert records[0].metadata["du_duplicate_source_ids"] == ["alice_harm_001"]


# ---------------------------------------------------------------------------
# process_rows — min_prompt_chars filter
# ---------------------------------------------------------------------------


def test_process_rows_short_prompt_filtered():
    keep = [_make_keep(prompt="Hi")]  # 2 chars < 20
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 0
    assert stats["keep_short_prompt_skipped"] == 1
    assert stats["unique_keep_records"] == 0


def test_process_rows_exactly_min_chars_passes():
    prompt = "A" * 20
    keep = [_make_keep(prompt=prompt)]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 1


def test_process_rows_strip_before_length_check():
    # Prompt is spaces + short text = short after strip
    keep = [_make_keep(prompt="   Hi   ")]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 0
    assert stats["keep_short_prompt_skipped"] == 1


# ---------------------------------------------------------------------------
# process_rows — deduplication
# ---------------------------------------------------------------------------


def test_process_rows_exact_duplicate_deduplicated():
    prompt = "What household chemicals should never be mixed together and why?"
    keep = [
        _make_keep(raw_id="alice_001", prompt=prompt),
        _make_keep(raw_id="bob_001", prompt=prompt),
    ]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 1
    assert stats["keep_duplicates_removed"] == 1


def test_process_rows_case_insensitive_dedup():
    keep = [
        _make_keep(raw_id="alice_001", prompt="What is the boiling point of water?"),
        _make_keep(raw_id="bob_001", prompt="WHAT IS THE BOILING POINT OF WATER?"),
    ]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 1
    assert stats["keep_duplicates_removed"] == 1


def test_process_rows_strip_insensitive_dedup():
    keep = [
        _make_keep(raw_id="alice_001", prompt="  How do explosives work?  "),
        _make_keep(raw_id="bob_001", prompt="How do explosives work?"),
    ]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 1


def test_process_rows_canonical_is_first_by_raw_id_alphabetically():
    keep = [
        _make_keep(raw_id="zzz_999", prompt="How do explosives work?", author="zoe"),
        _make_keep(raw_id="aaa_001", prompt="How do explosives work?", author="alice"),
    ]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert records[0].source_id == "aaa_001"
    assert records[0].metadata["du_author"] == "alice"


def test_process_rows_duplicate_source_ids_lists_all():
    prompt = "How do lock picks work?"
    keep = [
        _make_keep(raw_id="alice_001", prompt=prompt),
        _make_keep(raw_id="bob_001", prompt=prompt),
        _make_keep(raw_id="charlie_001", prompt=prompt),
    ]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    assert sorted(records[0].metadata["du_duplicate_source_ids"]) == [
        "alice_001", "bob_001", "charlie_001"
    ]
    assert records[0].metadata["du_duplicate_count"] == 3


def test_process_rows_distinct_prompts_not_deduplicated():
    keep = [
        _make_keep(raw_id="alice_001", prompt="What household chemicals should never be mixed together and why?"),
        _make_keep(raw_id="alice_002", prompt="How do security researchers find software vulnerabilities?"),
    ]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 2
    assert stats["keep_duplicates_removed"] == 0


# ---------------------------------------------------------------------------
# process_rows — drop artifact handling
# ---------------------------------------------------------------------------


def test_process_rows_drop_rows_become_artifacts():
    drop = [_make_drop(verdict="DROP"), _make_drop(raw_id="alice_002", verdict="BORDERLINE")]
    _, drops, stats = process_rows([], drop, 20, CONFIG_PATH)
    assert len(drops) == 2
    assert stats["total_drop_rows"] == 2


def test_process_rows_drop_artifact_has_ingest_config():
    drop = [_make_drop()]
    _, drops, _ = process_rows([], drop, 20, CONFIG_PATH)
    assert drops[0]["ingest_config"] == CONFIG_PATH


def test_process_rows_drop_artifact_preserves_original_fields():
    drop = [_make_drop(raw_id="alice_drop_001", verdict="DROP", reason="Harmful.")]
    _, drops, _ = process_rows([], drop, 20, CONFIG_PATH)
    assert drops[0]["raw_id"] == "alice_drop_001"
    assert drops[0]["judge_verdict"] == "DROP"
    assert drops[0]["judge_reason"] == "Harmful."


def test_process_rows_borderline_in_drop_artifacts():
    drop = [_make_drop(verdict="BORDERLINE")]
    _, drops, stats = process_rows([], drop, 20, CONFIG_PATH)
    assert len(drops) == 1
    assert stats["drop_verdict_distribution"].get("BORDERLINE", 0) == 1


def test_process_rows_drop_rows_not_in_records():
    drop = [_make_drop()]
    records, _, _ = process_rows([], drop, 20, CONFIG_PATH)
    assert len(records) == 0


# ---------------------------------------------------------------------------
# process_rows — wrong verdict in keep_rows
# ---------------------------------------------------------------------------


def test_process_rows_keep_row_wrong_verdict_skipped(capsys):
    keep = [_make_keep(verdict="DROP")]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 0
    assert stats["keep_wrong_verdict_skipped"] == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_process_rows_keep_row_borderline_verdict_skipped():
    keep = [_make_keep(verdict="BORDERLINE")]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 0
    assert stats["keep_wrong_verdict_skipped"] == 1


# ---------------------------------------------------------------------------
# process_rows — missing fields
# ---------------------------------------------------------------------------


def test_process_rows_keep_row_missing_field_skipped(capsys):
    row = _make_keep()
    del row["judge_verdict"]
    records, _, stats = process_rows([row], [], 20, CONFIG_PATH)
    assert len(records) == 0
    assert stats["keep_missing_fields_skipped"] == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_process_rows_drop_row_missing_field_included_with_warning(capsys):
    row = _make_drop()
    del row["judge_reason"]
    _, drops, stats = process_rows([], [row], 20, CONFIG_PATH)
    # Drop artifacts include even malformed rows, just with a warning
    assert len(drops) == 1
    assert stats["drop_missing_fields"] == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


# ---------------------------------------------------------------------------
# process_rows — invalid raw_id hard rejection
# ---------------------------------------------------------------------------


def test_process_rows_double_underscore_raw_id_rejected(capsys):
    keep = [_make_keep(raw_id="foo__bar")]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 0
    assert stats["keep_invalid_raw_id_skipped"] == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


def test_process_rows_hyphen_raw_id_rejected(capsys):
    keep = [_make_keep(raw_id="alice-001")]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 0
    assert stats["keep_invalid_raw_id_skipped"] == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


def test_process_rows_valid_raw_ids_not_rejected():
    keep = [
        _make_keep(raw_id="alice_001"),
        _make_keep(raw_id="bob002", prompt="How does social engineering work in practice?"),
    ]
    records, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert len(records) == 2
    assert stats["keep_invalid_raw_id_skipped"] == 0


def test_stats_keep_invalid_raw_id_skipped_present_in_stats():
    _, _, stats = process_rows([], [], 20, CONFIG_PATH)
    assert "keep_invalid_raw_id_skipped" in stats


# ---------------------------------------------------------------------------
# process_rows — stats consistency
# ---------------------------------------------------------------------------


def test_stats_total_keep_rows_counts_all_inputs():
    keep = [_make_keep(raw_id=f"alice_{i:03d}") for i in range(5)]
    _, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert stats["total_keep_rows"] == 5


def test_stats_unique_plus_duplicates_equals_after_filter():
    prompt = "How does social engineering work?"
    keep = [
        _make_keep(raw_id="alice_001", prompt=prompt),
        _make_keep(raw_id="bob_001", prompt=prompt),
        _make_keep(raw_id="charlie_001", prompt="How do people pick locks?"),
    ]
    _, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert stats["unique_keep_records"] + stats["keep_duplicates_removed"] == stats["keep_after_filter"]


def test_stats_domain_distribution():
    keep = [
        _make_keep(raw_id="alice_001", domain="harm_prevention"),
        _make_keep(raw_id="alice_002", prompt="How do lockpicks work in practice?", domain="physical_security"),
    ]
    _, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert stats["domain_distribution"]["harm_prevention"] == 1
    assert stats["domain_distribution"]["physical_security"] == 1


def test_stats_author_distribution():
    keep = [
        _make_keep(raw_id="alice_001", author="alice"),
        _make_keep(raw_id="alice_002", prompt="How do lockpicks work in practice?", author="alice"),
        _make_keep(raw_id="bob_001", prompt="How does malware propagate through networks?", author="bob"),
    ]
    _, _, stats = process_rows(keep, [], 20, CONFIG_PATH)
    assert stats["author_distribution"]["alice"] == 2
    assert stats["author_distribution"]["bob"] == 1


# ---------------------------------------------------------------------------
# process_rows — schema roundtrip
# ---------------------------------------------------------------------------


def test_process_rows_schema_roundtrip():
    """PromptRecord survives JSON serialization and reparsing."""
    keep = [_make_keep()]
    records, _, _ = process_rows(keep, [], 20, CONFIG_PATH)
    serialized = records[0].model_dump(mode="json")
    reparsed = PromptRecord.model_validate(serialized)
    assert reparsed.prompt_id == records[0].prompt_id
    assert reparsed.category == PromptCategory.dual_use


# ---------------------------------------------------------------------------
# _load_jsonl_files and _discover_files
# ---------------------------------------------------------------------------


def test_load_jsonl_files_basic(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    rows = _load_jsonl_files([f])
    assert rows == [{"a": 1}, {"b": 2}]


def test_load_jsonl_files_missing_file_warns(capsys, tmp_path):
    rows = _load_jsonl_files([tmp_path / "nonexistent.jsonl"])
    assert rows == []
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_load_jsonl_files_empty_lines_skipped(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    rows = _load_jsonl_files([f])
    assert len(rows) == 2


def test_discover_files_finds_prompts_and_drops(tmp_path):
    (tmp_path / "alice_harm_prompts.jsonl").write_text("")
    (tmp_path / "alice_harm_drops.jsonl").write_text("")
    (tmp_path / "bob_safety_prompts.jsonl").write_text("")
    (tmp_path / "bob_safety_drops.jsonl").write_text("")
    (tmp_path / "unrelated.jsonl").write_text("")

    keep_files, drop_files = _discover_files(tmp_path)
    assert len(keep_files) == 2
    assert len(drop_files) == 2


def test_discover_files_sorted(tmp_path):
    (tmp_path / "zzz_prompts.jsonl").write_text("")
    (tmp_path / "aaa_prompts.jsonl").write_text("")
    keep_files, _ = _discover_files(tmp_path)
    names = [f.name for f in keep_files]
    assert names == sorted(names)
