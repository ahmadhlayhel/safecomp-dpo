"""
JSONL read/write utilities for SafeComp-DPO pipeline records.

All pipeline stages persist state as JSONL files.
Each line is one JSON object corresponding to one record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safecomp_dpo.schemas import (
    PreferencePair,
    PromptRecord,
    ResponseRecord,
    ValidationRecord,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of raw dicts."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[Any], path: str | Path) -> None:
    """Write records to a JSONL file.

    Accepts a list of Pydantic model instances or plain dicts.
    Creates parent directories if they do not exist.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "model_dump_json"):
                line = record.model_dump_json()
            else:
                line = json.dumps(record)
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Typed loaders
# ---------------------------------------------------------------------------


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    return [PromptRecord.model_validate(r) for r in read_jsonl(path)]


def load_response_records(path: str | Path) -> list[ResponseRecord]:
    return [ResponseRecord.model_validate(r) for r in read_jsonl(path)]


def load_validation_records(path: str | Path) -> list[ValidationRecord]:
    return [ValidationRecord.model_validate(r) for r in read_jsonl(path)]


def load_preference_pairs(path: str | Path) -> list[PreferencePair]:
    return [PreferencePair.model_validate(r) for r in read_jsonl(path)]
