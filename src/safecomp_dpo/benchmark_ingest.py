"""
Benchmark dataset normalization functions.

Each function takes a raw row dict (from HuggingFace or a CSV) and returns a
dict conforming to the BenchmarkAdapter JSONL contract:

    Required: "prompt" (str)
    Optional: "prompt_id" (str), "split" (str), "metadata" (dict)

These functions are pure (no I/O) and are called by the ingest script.
They are also the unit under test in tests/test_benchmark_ingest.py.

Dataset sources
---------------
harmbench       cais/harmbench              (HF, subset: standard_behaviors)
strongreject    alexandrasouly/strongreject (HF)
do_not_answer   LibrAI/do-not-answer        (HF)
xstest          rottger/xstest              (HF)
or_bench        bench-llm/or-bench          (HF, multiple hardness subsets)
falsereject_test AmazonScience/FalseReject  (HF, split: test)

Raw column names documented per function. If a real download reveals different
field names, update only the normalization function — the adapter and tests are
unaffected.
"""

from __future__ import annotations

from typing import Any, Callable


# ---------------------------------------------------------------------------
# HarmBench
# ---------------------------------------------------------------------------
# Source: cais/harmbench, subset="standard_behaviors"
# Key raw columns: BehaviorID, Behavior, SemanticCategory, FunctionalCategory,
#                  Tags, ContextString
# ---------------------------------------------------------------------------


def normalize_harmbench(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a HarmBench standard-behaviors row.

    prompt_id: harmbench_{BehaviorID}
    prompt:    Behavior
    category:  SemanticCategory
    """
    behavior_id = str(row.get("BehaviorID", row.get("behavior_id", ""))).strip()
    prompt_text = str(row.get("Behavior", row.get("behavior", ""))).strip()

    if not prompt_text:
        raise ValueError(f"HarmBench row missing Behavior field: {row}")

    prompt_id = f"harmbench_{behavior_id}" if behavior_id else None

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "category": str(row.get("SemanticCategory", row.get("semantic_category", ""))).strip(),
        "metadata": {
            "functional_category": str(row.get("FunctionalCategory", row.get("functional_category", ""))).strip(),
            "tags": str(row.get("Tags", row.get("tags", ""))).strip(),
            "source": "harmbench",
        },
    }


# ---------------------------------------------------------------------------
# StrongREJECT
# ---------------------------------------------------------------------------
# Source: alexandrasouly/strongreject
# Key raw columns: forbidden_prompt  (primary), category (optional)
# Fallback columns tried in order: "forbidden_prompt", "prompt", "question"
# ---------------------------------------------------------------------------


def normalize_strongreject(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a StrongREJECT forbidden-prompts row.

    prompt_id: strongreject_{index} (auto-numbered by caller if no id column)
    prompt:    forbidden_prompt (or prompt / question fallback)
    """
    prompt_text = (
        row.get("forbidden_prompt")
        or row.get("prompt")
        or row.get("question")
        or ""
    )
    prompt_text = str(prompt_text).strip()

    if not prompt_text:
        raise ValueError(f"StrongREJECT row missing prompt field: {row}")

    raw_id = row.get("id", row.get("index", row.get("idx", "")))
    prompt_id = f"strongreject_{raw_id}" if raw_id != "" and raw_id is not None else None

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "metadata": {
            "category": str(row.get("category", row.get("type", ""))).strip(),
            "source": "strongreject",
        },
    }


# ---------------------------------------------------------------------------
# Do-Not-Answer
# ---------------------------------------------------------------------------
# Source: LibrAI/do-not-answer
# Key raw columns: id, question, types_of_harm, specific_harms, risk_area
# ---------------------------------------------------------------------------


def normalize_do_not_answer(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Do-Not-Answer row.

    prompt_id: dna_{id}
    prompt:    question
    category:  types_of_harm
    """
    raw_id = row.get("id", "")
    prompt_text = str(row.get("question", row.get("prompt", ""))).strip()

    if not prompt_text:
        raise ValueError(f"Do-Not-Answer row missing question field: {row}")

    prompt_id = f"dna_{raw_id}" if raw_id else None

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "category": str(row.get("types_of_harm", row.get("harm_type", ""))).strip(),
        "metadata": {
            "risk_area": str(row.get("risk_area", "")).strip(),
            "specific_harms": str(row.get("specific_harms", "")).strip(),
            "source": "do_not_answer",
        },
    }


# ---------------------------------------------------------------------------
# XSTest
# ---------------------------------------------------------------------------
# Source: GitHub CSV  https://github.com/paul-rottger/xstest
# Key raw columns: id, type, prompt, label, focus, note
#
# split derivation (in priority order):
#   1. "label" column ("safe" / "unsafe") — authoritative when present
#   2. "type" prefix fallback:
#      type starts with "contrast" → "unsafe"
#      anything else               → "safe"
#
# The GitHub CSV has 250 safe rows and 200 unsafe rows (450 total).
# Safe types include: safe_contexts, safe_targets, definitions, homonyms,
#   figurative_language, historical_events, privacy_*, nons_group_*, etc.
# Unsafe types all start with "contrast_".
# Using type-prefix alone misclassifies the non-safe_* safe types — always
# prefer the label column.
# ---------------------------------------------------------------------------


def normalize_xstest(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize an XSTest row.

    prompt_id: xstest_{id}
    prompt:    prompt
    split:     "safe" or "unsafe" — from "label" column (authoritative),
               falling back to "contrast_" type-prefix detection.
    """
    raw_id = row.get("id_v2", row.get("id", ""))
    prompt_text = str(row.get("prompt", "")).strip()
    type_str = str(row.get("type", "")).strip().lower()

    if not prompt_text:
        raise ValueError(f"XSTest row missing prompt field: {row}")

    # Prefer explicit label column; fall back to type-prefix heuristic
    label = str(row.get("label", "")).strip().lower()
    if label in ("safe", "unsafe"):
        split = label
    elif type_str.startswith("contrast"):
        split = "unsafe"
    else:
        split = "safe"

    prompt_id = f"xstest_{raw_id}" if raw_id else None

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "split": split,
        "metadata": {
            "type": type_str,
            "focus": str(row.get("focus", "")).strip(),
            "note": str(row.get("note", "")).strip(),
            "source": "xstest",
        },
    }


# ---------------------------------------------------------------------------
# OR-Bench
# ---------------------------------------------------------------------------
# Source: bench-llm/or-bench (multiple hardness-level subsets)
# Key raw columns: prompt (or question), id (optional), hardness / difficulty
#
# All OR-Bench prompts are benign. "Hardness" refers to how often models
# over-refuse them — higher hardness = more likely to trigger over-refusal.
# The ingest script downloads all rows; callers may filter by hardness.
# ---------------------------------------------------------------------------


def normalize_or_bench(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize an OR-Bench row.

    prompt_id: orbench_{id} (or orbench_{index})
    prompt:    prompt (or question fallback)
    """
    raw_id = row.get("id", row.get("idx", row.get("index", "")))
    prompt_text = (
        row.get("prompt")
        or row.get("question")
        or row.get("text")
        or ""
    )
    prompt_text = str(prompt_text).strip()

    if not prompt_text:
        raise ValueError(f"OR-Bench row missing prompt field: {row}")

    prompt_id = f"orbench_{raw_id}" if raw_id != "" and raw_id is not None else None

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "metadata": {
            "hardness": str(row.get("hardness", row.get("difficulty", ""))).strip(),
            "source": "or_bench",
        },
    }


# ---------------------------------------------------------------------------
# FalseReject-Test
# ---------------------------------------------------------------------------
# Source: AmazonScience/FalseReject, split="test"
# Key raw columns: prompt, category, category_text
# (same dataset as acquisition but different split)
# ---------------------------------------------------------------------------


def normalize_falsereject_test(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a FalseReject test-split row.

    prompt_id: falsereject_test_{index} (auto-numbered; no stable ID column)
    prompt:    prompt
    category:  category_text (human-readable category name)
    """
    prompt_text = str(row.get("prompt", "")).strip()

    if not prompt_text:
        raise ValueError(f"FalseReject row missing prompt field: {row}")

    # FalseReject has no stable row-level ID; caller provides index
    raw_id = row.get("id", row.get("idx", row.get("index", "")))
    prompt_id = f"falsereject_test_{raw_id}" if raw_id != "" and raw_id is not None else None

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "category": str(row.get("category_text", row.get("category", ""))).strip(),
        "metadata": {
            "category_id": str(row.get("category", "")).strip(),
            "source": "falsereject_test",
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NORMALIZERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "harmbench":        normalize_harmbench,
    "strongreject":     normalize_strongreject,
    "do_not_answer":    normalize_do_not_answer,
    "xstest":           normalize_xstest,
    "or_bench":         normalize_or_bench,
    "falsereject_test": normalize_falsereject_test,
}

ALL_BENCHMARKS: list[str] = list(NORMALIZERS)


def normalize_row(benchmark: str, row: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to the appropriate normalization function.

    Args:
        benchmark: benchmark name (must be in NORMALIZERS)
        row:       raw row dict from the source dataset

    Returns:
        Normalized dict conforming to BenchmarkAdapter JSONL contract.

    Raises:
        ValueError: if benchmark is unknown or required fields are missing.
    """
    if benchmark not in NORMALIZERS:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}. "
            f"Available: {sorted(NORMALIZERS)}"
        )
    return NORMALIZERS[benchmark](row)
