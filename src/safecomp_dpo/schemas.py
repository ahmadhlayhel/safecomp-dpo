"""
Core Pydantic schemas for the SafeComp-DPO pipeline.

Record hierarchy:
    PromptRecord       — a single prompt with its category and provenance
    ResponseRecord     — a single model response to a prompt
    ValidationRecord   — human or automated judgment on a response
    PreferencePair     — (chosen, rejected) pair derived from responses

ID policy:
    All primary-key fields are required and caller-supplied.
    IDs must be stable across pipeline rebuilds — use upstream dataset IDs
    or deterministic hashes, not random UUIDs.  Unstable IDs break FK chains
    (ResponseRecord.prompt_id, ValidationRecord.response_id, etc.) silently.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Canonical enums — do not rename these values
# ---------------------------------------------------------------------------


class PromptCategory(str, Enum):
    unsafe = "unsafe"
    dual_use = "dual_use"
    benign_sensitive = "benign_sensitive"
    benign = "benign"


class ResponseType(str, Enum):
    hard_refusal = "hard_refusal"
    safe_completion = "safe_completion"
    unsafe_compliance = "unsafe_compliance"
    helpful_answer = "helpful_answer"


class PairType(str, Enum):
    safe_completion = "safe_completion"
    baseline_refusal = "baseline_refusal"
    strict_refusal = "strict_refusal"


# ---------------------------------------------------------------------------
# Core record types
# ---------------------------------------------------------------------------


class PromptRecord(BaseModel):
    """A single prompt with its category and source provenance."""

    prompt_id: str
    prompt: str
    category: PromptCategory
    source: str
    source_id: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResponseRecord(BaseModel):
    """A single model response to a prompt."""

    response_id: str
    prompt_id: str
    response: str
    response_type: ResponseType
    model: str
    temperature: float | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationRecord(BaseModel):
    """Human or automated judgment on a single response.

    prompt_id is denormalized here (also reachable via response_id →
    ResponseRecord) to allow standalone validation files to be joined
    and audited without loading ResponseRecord.
    """

    validation_id: str
    response_id: str
    prompt_id: str
    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    validator: str
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferencePair(BaseModel):
    """A (chosen, rejected) response pair for DPO training."""

    pair_id: str
    prompt_id: str
    chosen_id: str
    rejected_id: str
    pair_type: PairType
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DPORecord(BaseModel):
    """Flat record assembled from pipeline artifacts, ready for DPO training."""

    pair_id: str
    prompt_id: str
    category: PromptCategory
    pair_type: PairType
    prompt: str
    chosen: str
    rejected: str
    chosen_id: str
    rejected_id: str


class BenchmarkRecord(BaseModel):
    """Per-example result from a standard benchmark evaluation run.

    is_compliant: True if the model followed the prompt's request.
    Interpretation depends on benchmark type:
      safety benchmarks   (HarmBench, StrongREJECT, DoNotAnswer):
          is_compliant=True  means harmful content was produced  (BAD)
      over-refusal benchmarks (XSTest safe, OR-Bench, FalseRejectTest):
          is_compliant=True  means model gave a helpful response  (GOOD)

    score: continuous score used by StrongREJECT (0.0–1.0); None for all
           other benchmarks that only need binary is_compliant.

    record_id convention: {prompt_id}__{run_id}__bench
    Example:              harmbench_001__mock_run__bench
    """

    record_id: str
    benchmark: str
    split: str | None = None          # "safe"/"unsafe" for XSTest; None otherwise
    prompt: str
    response: str
    is_compliant: bool
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRecord(BaseModel):
    """Per-example evaluation result for a single model run.

    Captures the model response and judge label for one (prompt, run) pair.
    Used to compute category-level metrics via aggregate_metrics().

    eval_id convention: {prompt_id}__{run_id}__eval
    Example:            u_001__mock_safecomp__eval

    run_id should be a clean alphanumeric slug (no __ separators).
    """

    eval_id: str
    prompt_id: str
    run_id: str
    category: PromptCategory
    prompt: str
    response: str
    judge_label: ResponseType
    metadata: dict[str, Any] = Field(default_factory=dict)
