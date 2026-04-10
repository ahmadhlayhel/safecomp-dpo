"""
One-time consolidation of uploaded dual-use prompt bundles.

Reads the contributor zip (or a pre-extracted directory), fixes encoding
errors, drops DROP-verdict records, and writes:

  hf_data/prompts/dual_use/dualuse_prompts.jsonl
      Merged PromptRecord JSONL — KEEP + BORDERLINE, ready for generation.
      This is the primary deliverable and the file downstream stages consume.

  hf_data/prompts/dual_use/submissions/{author}_{domain}_prompts.jsonl
      Per-(author, domain) KEEP records in raw contributor-bundle format.
      Accepted as-is by acquire_prompts_dualuse.py if you later want to
      re-run the formal acquisition stage.

  hf_data/prompts/dual_use/submissions/{author}_{domain}_drops.jsonl
      Per-(author, domain) DROP + BORDERLINE records for the audit trail.
      (BORDERLINE records are kept in the main pool but also recorded here
      so the drops file is a complete audit artifact.)

  outputs/acquisition_reports/dualuse_pool_report.json
      Counts by verdict, author, domain.

Usage:
    python scripts/consolidate_dualuse_prompts.py \\
        --zip "path/to/dual-use prompts.zip"

    python scripts/consolidate_dualuse_prompts.py \\
        --input-dir "path/to/Prompts/"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
import tempfile
import pathlib
from collections import Counter, defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETAIN_VERDICTS = {"KEEP", "BORDERLINE"}
DROP_VERDICTS = {"DROP"}

REQUIRED_FIELDS = frozenset({
    "raw_id", "prompt", "domain", "author",
    "generation_model", "generator_template_version",
    "judge_verdict", "judge_reason",
})

# Domain-name typo corrections (lowercase source -> corrected value)
DOMAIN_CORRECTIONS: dict[str, str] = {
    "offensive_cysecurity": "offensive_cybersecurity",
}

INGEST_SCRIPT = "consolidate_dualuse_prompts.py"
INGEST_CONFIG = "configs/acquisition/dualuse.yaml"

# ---------------------------------------------------------------------------
# JSON repair — handles unescaped double-quotes inside the "prompt" value.
# The fix captures everything between `"prompt":"` and `","domain":` using
# a non-greedy DOTALL match, then re-encodes the captured string correctly.
# ---------------------------------------------------------------------------

_PROMPT_PAT = re.compile(r'"prompt":"(.*?)","domain":', re.DOTALL)


def _repair_line(line: str) -> dict[str, Any] | None:
    """Try to parse a JSONL line; if it fails, attempt prompt-field repair."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass

    m = _PROMPT_PAT.search(line)
    if not m:
        return None
    raw_prompt = m.group(1)
    escaped = json.dumps(raw_prompt)[1:-1]          # proper JSON escaping
    fixed = line[: m.start(1)] + escaped + line[m.end(1):]
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Load from directory of JSONL files
# ---------------------------------------------------------------------------


def load_all_records(base: pathlib.Path) -> tuple[list[dict], int, int]:
    """Walk a Prompts/ directory tree and load all records.

    Returns (records, n_repaired, n_unrecoverable).
    """
    records: list[dict] = []
    n_repaired = 0
    n_unrecoverable = 0

    for person_dir in sorted(base.iterdir()):
        if not person_dir.is_dir():
            continue
        for f in sorted(person_dir.glob("*.jsonl")):
            for lineno, raw_line in enumerate(
                f.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError:
                    rec = _repair_line(line)
                    if rec is not None:
                        records.append(rec)
                        n_repaired += 1
                    else:
                        n_unrecoverable += 1
                        print(
                            f"  UNRECOVERABLE: {f.name} line {lineno}: {line[:80]}",
                            file=sys.stderr,
                        )
    return records, n_repaired, n_unrecoverable


# ---------------------------------------------------------------------------
# Normalize and validate
# ---------------------------------------------------------------------------


def normalize_verdict(v: str) -> str:
    """Upper-case and strip; map minor variants."""
    v = v.strip().upper()
    aliases = {
        "KEEP": "KEEP",
        "BORDERLINE": "BORDERLINE",
        "DROP": "DROP",
        "REMOVE": "DROP",
        "DISCARD": "DROP",
        "REJECT": "DROP",
    }
    return aliases.get(v, v)


def fix_domain(domain: str) -> str:
    return DOMAIN_CORRECTIONS.get(domain.strip().lower(), domain.strip())


# ---------------------------------------------------------------------------
# Build PromptRecord dict (matches the schema in src/safecomp_dpo/schemas.py)
# ---------------------------------------------------------------------------


def to_prompt_record(rec: dict[str, Any]) -> dict[str, Any]:
    raw_id = rec["raw_id"]
    return {
        "prompt_id": f"du_{raw_id}",
        "prompt": rec["prompt"].strip(),
        "category": "dual_use",
        "source": "dualuse_team_v1",
        "source_id": raw_id,
        "created_at": None,
        "metadata": {
            "du_domain": fix_domain(rec["domain"]),
            "du_author": rec["author"],
            "du_generation_model": rec["generation_model"],
            "du_generator_template_version": rec["generator_template_version"],
            "du_judge_verdict": rec["judge_verdict"],
            "du_judge_reason": rec["judge_reason"],
            "du_raw_id": raw_id,
            "du_duplicate_count": 1,
            "du_duplicate_source_ids": [raw_id],
            "ingest_script": INGEST_SCRIPT,
            "ingest_config": INGEST_CONFIG,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Consolidate dual-use contributor prompts into repo pool."
    )
    ap.add_argument("--zip", help="Path to dual-use prompts zip file")
    ap.add_argument("--input-dir", help="Pre-extracted Prompts/ directory")
    ap.add_argument(
        "--output",
        default="hf_data/prompts/dual_use/dualuse_prompts.jsonl",
        help="Output path for merged PromptRecord JSONL",
    )
    ap.add_argument(
        "--submissions-dir",
        default="hf_data/prompts/dual_use/submissions",
        help="Output directory for contributor bundle files",
    )
    ap.add_argument(
        "--report",
        default="outputs/acquisition_reports/dualuse_pool_report.json",
        help="Output path for acquisition stats JSON",
    )
    args = ap.parse_args()

    # ── Resolve input ──────────────────────────────────────────────────────
    if args.zip:
        tmpdir = tempfile.mkdtemp(prefix="dualuse_extract_")
        print(f"Extracting {args.zip} to {tmpdir}")
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(tmpdir)
        # Find the Prompts/ subdirectory
        candidates = list(pathlib.Path(tmpdir).rglob("Prompts"))
        if not candidates:
            print("ERROR: No 'Prompts' directory found in zip.", file=sys.stderr)
            sys.exit(1)
        base = candidates[0]
    elif args.input_dir:
        base = pathlib.Path(args.input_dir)
        if not base.exists():
            print(f"ERROR: --input-dir does not exist: {base}", file=sys.stderr)
            sys.exit(1)
    else:
        print("ERROR: specify --zip or --input-dir.", file=sys.stderr)
        sys.exit(1)

    # ── Load ───────────────────────────────────────────────────────────────
    print(f"Loading records from {base}")
    raw_records, n_repaired, n_unrecoverable = load_all_records(base)
    print(f"  Loaded:       {len(raw_records)} raw records")
    if n_repaired:
        print(f"  JSON-repaired: {n_repaired}")
    if n_unrecoverable:
        print(f"  Unrecoverable: {n_unrecoverable}  ← check stderr")

    # ── Validate required fields ───────────────────────────────────────────
    valid_records: list[dict] = []
    n_missing_fields = 0
    for rec in raw_records:
        missing = REQUIRED_FIELDS - set(rec.keys())
        if missing:
            n_missing_fields += 1
            print(
                f"  WARNING: record missing fields {sorted(missing)!r} "
                f"(raw_id={rec.get('raw_id', '?')!r}) — skipping",
                file=sys.stderr,
            )
            continue
        valid_records.append(rec)

    # ── Normalize verdicts and fix domains ────────────────────────────────
    for rec in valid_records:
        rec["judge_verdict"] = normalize_verdict(rec["judge_verdict"])
        rec["domain"] = fix_domain(rec["domain"])

    # ── Split by verdict ───────────────────────────────────────────────────
    retain: list[dict] = []
    drop: list[dict] = []
    unknown_verdict: list[dict] = []

    for rec in valid_records:
        v = rec["judge_verdict"]
        if v in RETAIN_VERDICTS:
            retain.append(rec)
        elif v in DROP_VERDICTS:
            drop.append(rec)
        else:
            unknown_verdict.append(rec)
            print(
                f"  WARNING: unknown verdict {v!r} for raw_id={rec['raw_id']!r} "
                "— treating as DROP",
                file=sys.stderr,
            )
            drop.append(rec)

    # ── Stats ──────────────────────────────────────────────────────────────
    verdict_dist = Counter(r["judge_verdict"] for r in retain)
    author_dist = Counter(r["author"] for r in retain)
    domain_dist = Counter(r["domain"] for r in retain)
    drop_verdict_dist = Counter(r["judge_verdict"] for r in drop)

    print()
    print("=" * 60)
    print(f"Total raw records loaded:     {len(raw_records)}")
    print(f"Valid (all required fields):  {len(valid_records)}")
    print(f"  Retained (KEEP+BORDERLINE): {len(retain)}")
    print(f"  Dropped:                    {len(drop)}")
    print()
    print("Retained by verdict:")
    for v, n in sorted(verdict_dist.items()):
        print(f"  {v}: {n}")
    print()
    print("Retained by author:")
    for a, n in sorted(author_dist.items()):
        print(f"  {a}: {n}")
    print()
    print("Retained by domain:")
    for d, n in sorted(domain_dist.items()):
        print(f"  {d}: {n}")
    print()
    print("Dropped by verdict:")
    for v, n in sorted(drop_verdict_dist.items()):
        print(f"  {v}: {n}")
    print("=" * 60)

    # ── Write merged PromptRecord JSONL ────────────────────────────────────
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in retain:
            pr = to_prompt_record(rec)
            f.write(json.dumps(pr) + "\n")
    print(f"\nWrote {len(retain)} PromptRecords -> {out_path}")

    # ── Write contributor bundle files ─────────────────────────────────────
    sub_dir = pathlib.Path(args.submissions_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Group retained KEEP records -> *_prompts.jsonl
    # Group DROP + BORDERLINE records -> *_drops.jsonl
    # (BORDERLINE is kept in pool but also logged in drops for audit trail)
    keeps_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    drops_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for rec in retain:
        key = (rec["author"], rec["domain"])
        if rec["judge_verdict"] == "KEEP":
            keeps_by_key[key].append(rec)
        else:  # BORDERLINE
            drops_by_key[key].append(rec)

    for rec in drop:
        key = (rec["author"], rec["domain"])
        drops_by_key[key].append(rec)

    n_bundle_files = 0
    for key in sorted(set(list(keeps_by_key.keys()) + list(drops_by_key.keys()))):
        author, domain = key
        stem = f"{author}_{domain}"

        if keeps_by_key[key]:
            p = sub_dir / f"{stem}_prompts.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for rec in keeps_by_key[key]:
                    f.write(json.dumps(rec) + "\n")
            n_bundle_files += 1

        if drops_by_key[key]:
            p = sub_dir / f"{stem}_drops.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for rec in drops_by_key[key]:
                    f.write(json.dumps(rec) + "\n")
            n_bundle_files += 1

    print(f"Wrote {n_bundle_files} contributor bundle files -> {sub_dir}")

    # ── Write report ───────────────────────────────────────────────────────
    report_path = pathlib.Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_raw_records": len(raw_records),
        "n_json_repaired": n_repaired,
        "n_unrecoverable": n_unrecoverable,
        "n_missing_fields_skipped": n_missing_fields,
        "total_valid": len(valid_records),
        "total_retained": len(retain),
        "total_dropped": len(drop),
        "verdict_distribution_retained": dict(sorted(verdict_dist.items())),
        "author_distribution_retained": dict(sorted(author_dist.items())),
        "domain_distribution_retained": dict(sorted(domain_dist.items())),
        "verdict_distribution_dropped": dict(sorted(drop_verdict_dist.items())),
        "output_path": str(out_path),
        "submissions_dir": str(sub_dir),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report -> {report_path}")


if __name__ == "__main__":
    main()
