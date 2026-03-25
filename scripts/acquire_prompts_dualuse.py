"""
Acquire dual_use prompts from contributor JSONLs and write PromptRecord JSONL.

Contract:
    Input:  Contributor bundle files (two files per submission bundle):
              {author}_{domain_code}_prompts.jsonl  — KEEP prompts
              {author}_{domain_code}_drops.jsonl    — DROP and BORDERLINE prompts
            Each record must contain:
              raw_id, prompt, domain, author, generation_model,
              generator_template_version, judge_verdict, judge_reason

    Output: PromptRecord JSONL — one record per unique KEEP prompt
            Drop artifact JSONL — all DROP/BORDERLINE rows unchanged (+ ingest_config)
            JSON acquisition report — stats and provenance

Repo-side mapping:
    category   = dual_use
    source     = dualuse_team_v1
    source_id  = raw_id
    created_at = None
    prompt_id  = "du_" + raw_id   (raw_id must match [a-zA-Z0-9_]+ with no "__")
    metadata:
        du_domain, du_author, du_generation_model,
        du_generator_template_version, du_judge_verdict, du_judge_reason,
        du_raw_id, du_duplicate_count, du_duplicate_source_ids,
        ingest_script, ingest_config

Usage:
    python scripts/acquire_prompts_dualuse.py --config configs/acquisition/dualuse.yaml \\
        --input-dir hf_data/prompts/dual_use/submissions/
    python scripts/acquire_prompts_dualuse.py --config configs/acquisition/dualuse.yaml \\
        --prompts-files alice_safety_prompts.jsonl \\
        --drops-files alice_safety_drops.jsonl \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from safecomp_dpo.io import write_jsonl
from safecomp_dpo.schemas import PromptCategory, PromptRecord


_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "raw_id", "prompt", "domain", "author",
    "generation_model", "generator_template_version",
    "judge_verdict", "judge_reason",
})

_KEEP_VERDICT = "KEEP"

# raw_id safety: alphanumeric and single underscores only.
# Double underscores are forbidden because downstream IDs use "__" as a
# separator: {prompt_id}__{response_type}__s{sample:02d}.
# Any other non-alphanumeric character is also rejected.
_RAW_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Canonical key for deduplication: strip + lowercase."""
    return text.strip().lower()


def _is_valid_raw_id(raw_id: str) -> bool:
    """Return True if raw_id is safe for use in downstream IDs.

    Rules:
      - Must match [a-zA-Z0-9_]+ (alphanumeric and single underscores only)
      - Must not contain "__" (reserved separator in response_id / pair_id)
    """
    return bool(_RAW_ID_PATTERN.fullmatch(raw_id)) and "__" not in raw_id


def _prompt_id(raw_id: str) -> str:
    """Build a deterministic prompt ID.

    Convention: "du_" + raw_id
    Caller must have validated raw_id with _is_valid_raw_id() first.
    """
    return f"du_{raw_id}"


# ---------------------------------------------------------------------------
# Core transformation (pure — no file I/O, fully testable)
# ---------------------------------------------------------------------------


def process_rows(
    keep_rows: list[dict[str, Any]],
    drop_rows: list[dict[str, Any]],
    min_prompt_chars: int,
    config_path: str,
) -> tuple[list[PromptRecord], list[dict[str, Any]], dict[str, Any]]:
    """Filter, deduplicate, and build PromptRecords from contributor rows.

    Args:
        keep_rows:        rows from *_prompts.jsonl files (expected verdict: KEEP)
        drop_rows:        rows from *_drops.jsonl files (expected verdict: DROP/BORDERLINE)
        min_prompt_chars: minimum character length after stripping
        config_path:      stored in metadata for provenance

    Returns:
        (records, drop_artifacts, stats)
        records:        PromptRecord list — unique KEEP prompts
        drop_artifacts: raw dicts for DROP/BORDERLINE rows (+ ingest_config field)
        stats:          acquisition statistics dict
    """
    # ── Stage 1: validate and filter keep_rows ──────────────────────────────
    n_total_keep = len(keep_rows)
    n_wrong_verdict = 0
    n_missing_fields = 0
    n_invalid_raw_id = 0
    n_short = 0
    eligible: list[dict[str, Any]] = []

    for row in keep_rows:
        missing = [f for f in _REQUIRED_FIELDS if f not in row]
        if missing:
            n_missing_fields += 1
            print(
                f"  WARNING: keep row missing fields {missing!r} — skipping",
                file=sys.stderr,
            )
            continue
        if not _is_valid_raw_id(row["raw_id"]):
            n_invalid_raw_id += 1
            print(
                f"  ERROR: keep row raw_id={row['raw_id']!r} is invalid. "
                "raw_id must match [a-zA-Z0-9_]+ with no '__'. "
                "Row rejected — fix the contributor file before re-ingesting.",
                file=sys.stderr,
            )
            continue
        if row["judge_verdict"] != _KEEP_VERDICT:
            n_wrong_verdict += 1
            print(
                f"  WARNING: keep row raw_id={row['raw_id']!r} has "
                f"verdict={row['judge_verdict']!r} (expected KEEP) — skipping",
                file=sys.stderr,
            )
            continue
        if len(row["prompt"].strip()) < min_prompt_chars:
            n_short += 1
            continue
        eligible.append(row)

    after_filter = len(eligible)

    # ── Stage 2: group by normalized prompt ─────────────────────────────────
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        key = _normalize(row["prompt"])
        groups[key].append(row)

    # ── Stage 3: build one PromptRecord per group ────────────────────────────
    records: list[PromptRecord] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["raw_id"])
        canonical = group[0]
        all_raw_ids = [r["raw_id"] for r in group]

        record = PromptRecord(
            prompt_id=_prompt_id(canonical["raw_id"]),
            prompt=canonical["prompt"].strip(),
            category=PromptCategory.dual_use,
            source="dualuse_team_v1",
            source_id=canonical["raw_id"],
            created_at=None,
            metadata={
                "du_domain": canonical["domain"],
                "du_author": canonical["author"],
                "du_generation_model": canonical["generation_model"],
                "du_generator_template_version": canonical["generator_template_version"],
                "du_judge_verdict": canonical["judge_verdict"],
                "du_judge_reason": canonical["judge_reason"],
                "du_raw_id": canonical["raw_id"],
                "du_duplicate_count": len(group),
                "du_duplicate_source_ids": all_raw_ids,
                "ingest_script": "acquire_prompts_dualuse.py",
                "ingest_config": config_path,
            },
        )
        records.append(record)

    # ── Stage 4: pass through drop_rows as audit artifacts ──────────────────
    drop_artifacts: list[dict[str, Any]] = []
    n_drop_missing_fields = 0

    for row in drop_rows:
        missing = [f for f in _REQUIRED_FIELDS if f not in row]
        if missing:
            n_drop_missing_fields += 1
            print(
                f"  WARNING: drop row missing fields {missing!r} — including anyway",
                file=sys.stderr,
            )
        artifact = dict(row)
        artifact["ingest_config"] = config_path
        drop_artifacts.append(artifact)

    # ── Stats ────────────────────────────────────────────────────────────────
    drop_verdict_dist: dict[str, int] = defaultdict(int)
    for row in drop_rows:
        drop_verdict_dist[row.get("judge_verdict", "UNKNOWN")] += 1

    domain_dist: dict[str, int] = defaultdict(int)
    author_dist: dict[str, int] = defaultdict(int)
    for r in records:
        domain_dist[r.metadata["du_domain"]] += 1
        author_dist[r.metadata["du_author"]] += 1

    stats: dict[str, Any] = {
        "total_keep_rows": n_total_keep,
        "total_drop_rows": len(drop_rows),
        "keep_wrong_verdict_skipped": n_wrong_verdict,
        "keep_missing_fields_skipped": n_missing_fields,
        "keep_invalid_raw_id_skipped": n_invalid_raw_id,
        "keep_short_prompt_skipped": n_short,
        "keep_after_filter": after_filter,
        "keep_duplicates_removed": after_filter - len(records),
        "unique_keep_records": len(records),
        "drop_missing_fields": n_drop_missing_fields,
        "drop_verdict_distribution": dict(drop_verdict_dist),
        "domain_distribution": dict(domain_dist),
        "author_distribution": dict(author_dist),
    }
    return records, drop_artifacts, stats


# ---------------------------------------------------------------------------
# File loading and discovery
# ---------------------------------------------------------------------------


def _load_jsonl_files(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load and concatenate rows from a list of JSONL file paths."""
    rows: list[dict[str, Any]] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  WARNING: file not found: {p}", file=sys.stderr)
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover_files(input_dir: str | Path) -> tuple[list[Path], list[Path]]:
    """Glob for contributor bundle files in a directory.

    Returns (keep_files, drop_files) sorted for deterministic ordering.
    """
    d = Path(input_dir)
    keep_files = sorted(d.glob("*_prompts.jsonl"))
    drop_files = sorted(d.glob("*_drops.jsonl"))
    return keep_files, drop_files


def acquire(
    keep_files: list[str | Path],
    drop_files: list[str | Path],
    config: dict[str, Any],
    config_path: str,
) -> tuple[list[PromptRecord], list[dict[str, Any]], dict[str, Any]]:
    """Load contributor files and run process_rows."""
    keep_rows = _load_jsonl_files(keep_files)
    drop_rows = _load_jsonl_files(drop_files)
    return process_rows(
        keep_rows,
        drop_rows,
        config.get("min_prompt_chars", 20),
        config_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire dual_use prompts from contributor JSONLs",
    )
    parser.add_argument("--config", required=True, help="Path to dualuse.yaml config")
    parser.add_argument(
        "--input-dir",
        help="Directory with contributor bundles (*_prompts.jsonl, *_drops.jsonl)",
    )
    parser.add_argument(
        "--prompts-files",
        nargs="+",
        metavar="FILE",
        help="Explicit KEEP prompts JSONL files (alternative to --input-dir)",
    )
    parser.add_argument(
        "--drops-files",
        nargs="+",
        metavar="FILE",
        default=[],
        help="Explicit DROP/BORDERLINE JSONL files (alternative to --input-dir)",
    )
    parser.add_argument("--output", help="Override output_path from config")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print stats only; do not write"
    )
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve input files
    if args.input_dir:
        keep_files, drop_files = _discover_files(args.input_dir)
        print(
            f"Discovered {len(keep_files)} prompts file(s) and "
            f"{len(drop_files)} drops file(s) in {args.input_dir}"
        )
    elif args.prompts_files:
        keep_files = args.prompts_files
        drop_files = args.drops_files
    elif "input_dir" in config:
        keep_files, drop_files = _discover_files(config["input_dir"])
        print(
            f"Discovered {len(keep_files)} prompts file(s) and "
            f"{len(drop_files)} drops file(s) in {config['input_dir']}"
        )
    else:
        print(
            "ERROR: specify --input-dir, --prompts-files, or input_dir in config.",
            file=sys.stderr,
        )
        sys.exit(1)

    records, drop_artifacts, stats = acquire(keep_files, drop_files, config, config_path)

    print(f"Total KEEP rows:          {stats['total_keep_rows']}")
    print(f"Total DROP/BORDERLINE:    {stats['total_drop_rows']}")
    print(f"After filter (min chars): {stats['keep_after_filter']}")
    print(f"Unique after dedup:       {stats['unique_keep_records']}")
    if stats["drop_verdict_distribution"]:
        print(f"Drop verdict dist:        {stats['drop_verdict_distribution']}")
    if stats["domain_distribution"]:
        print(f"Domain distribution:      {stats['domain_distribution']}")
    if stats["author_distribution"]:
        print(f"Author distribution:      {stats['author_distribution']}")

    if args.dry_run:
        print("Dry run — no output written.")
        return

    output_path = args.output or config["output_path"]
    write_jsonl(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")

    drops_output_path = config.get("drops_output_path")
    if drops_output_path and drop_artifacts:
        write_jsonl(drop_artifacts, drops_output_path)
        print(f"Wrote {len(drop_artifacts)} drop artifacts to {drops_output_path}")

    sample_path = config.get("sample_path")
    sample_n = int(config.get("sample_n", 5))
    if sample_path and records:
        write_jsonl(records[:sample_n], sample_path)
        print(f"Wrote {sample_n} sample records to {sample_path}")

    report_path = config.get("report_path")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
