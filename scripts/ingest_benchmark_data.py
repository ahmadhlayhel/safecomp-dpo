"""
Benchmark dataset ingestion: download and normalize to JSONL.

Downloads each benchmark dataset from its canonical source (HuggingFace or
GitHub CSV), normalizes each row via benchmark_ingest.py, and writes a JSONL
file to hf_data/benchmarks/.

This script is run ONCE per benchmark to prepare local data. It is NOT run
during training or evaluation — the benchmark runner reads the pre-normalized
JSONL directly.

Usage
-----
    # Ingest one benchmark:
    python scripts/ingest_benchmark_data.py --benchmark harmbench
    python scripts/ingest_benchmark_data.py --benchmark xstest

    # Ingest all benchmarks:
    python scripts/ingest_benchmark_data.py --benchmark all

    # Preview without writing:
    python scripts/ingest_benchmark_data.py --benchmark do_not_answer --dry-run

    # Limit rows (useful for smoke-testing):
    python scripts/ingest_benchmark_data.py --benchmark harmbench --limit 20

Requirements
------------
    pip install datasets        (for HF-sourced benchmarks)
    urllib (stdlib)             (for GitHub CSV-sourced benchmarks)

Output
------
    hf_data/benchmarks/{benchmark_name}.jsonl
    (one record per line, format: {"prompt_id", "prompt", "split"?, "category"?, "metadata"})

Dataset sources
---------------
    harmbench        GitHub CSV  (centerforaisafety/HarmBench)
    strongreject     GitHub CSV  (alexandrasouly/strongreject)
    do_not_answer    HuggingFace LibrAI/do-not-answer
    xstest           GitHub CSV  (paul-rottger/xstest)
    or_bench         HuggingFace bench-llm/or-bench (hard-1k + 80k subsets)
    falsereject_test HuggingFace AmazonScience/FalseReject (split: test)

Notes on individual datasets
-----------------------------
HarmBench:
    Downloads harmbench_behaviors_text_all.csv from the HarmBench GitHub repo.
    Contains 400 standard harmful behaviors used as the main evaluation set.

StrongREJECT:
    Downloads strongreject_dataset.csv from the StrongREJECT GitHub repo.
    Contains ~313 jailbreak-style forbidden prompts (no stable row ID).

Do-Not-Answer:
    HuggingFace dataset LibrAI/do-not-answer (train split, 939 rows).
    Covers 5 risk areas of harmful questions.

XSTest:
    Downloads xstest_prompts.csv from the XSTest GitHub repo.
    250 safe + 200 unsafe prompts. Split is read from the "label" column
    ("safe" / "unsafe") which is authoritative.

OR-Bench:
    HuggingFace bench-llm/or-bench.
    Available subsets: or-bench-hard-1k (1 000 hardest benign prompts) and
    or-bench-80k (80 000 benign prompts).
    This script downloads or-bench-hard-1k only (most discriminative for ORR
    measurement). Use --or-bench-full to include or-bench-80k as well.

FalseReject-Test:
    HuggingFace AmazonScience/FalseReject, split="test" (1187 rows).
    The train split is used for prompt acquisition; the test split is reserved
    for evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterator

try:
    from safecomp_dpo.benchmark_ingest import ALL_BENCHMARKS, normalize_row
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from safecomp_dpo.benchmark_ingest import ALL_BENCHMARKS, normalize_row


# ---------------------------------------------------------------------------
# Source configs
# ---------------------------------------------------------------------------

# GitHub CSV sources: benchmark → raw CSV URL
_GITHUB_CSV_SOURCES: dict[str, str] = {
    "harmbench": (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
        "/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    ),
    "strongreject": (
        "https://raw.githubusercontent.com/alexandrasouly/strongreject"
        "/main/strongreject_dataset/strongreject_dataset.csv"
    ),
    "xstest": (
        "https://raw.githubusercontent.com/paul-rottger/xstest"
        "/main/xstest_prompts.csv"
    ),
}

# HuggingFace sources: benchmark → (hf_path, hf_config_or_None, split)
_HF_SOURCES: dict[str, tuple[str, str | None, str]] = {
    "do_not_answer":    ("LibrAI/do-not-answer",       None,  "train"),
    "falsereject_test": ("AmazonScience/FalseReject",  None,  "test"),
}

# OR-Bench is handled separately — multiple HF subsets concatenated.
# Verified subset names from bench-llm/or-bench dataset card.
_OR_BENCH_HF_PATH = "bench-llm/or-bench"
_OR_BENCH_SUBSET_HARD = "or-bench-hard-1k"   # ~1 000 hardest benign prompts
_OR_BENCH_SUBSET_FULL = "or-bench-80k"        # ~80 000 benign prompts

DEFAULT_OUTPUT_DIR = Path("hf_data/benchmarks")

# HTTP timeout for GitHub downloads (seconds)
_HTTP_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _require_datasets() -> Any:
    """Import the datasets library or exit with a clear error."""
    try:
        import datasets  # type: ignore[import]
        return datasets
    except ImportError:
        print(
            "ERROR: 'datasets' package not installed.\n"
            "Run: pip install datasets\n"
            "(Do not add to pyproject.toml — data-prep dependency only.)",
            file=sys.stderr,
        )
        sys.exit(1)


def _download_github_csv(url: str) -> list[dict[str, str]]:
    """Download a CSV from GitHub and return as list of dicts."""
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as r:
            text = r.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def _iter_github_csv_rows(benchmark: str, limit: int | None) -> list[dict[str, str]]:
    """Download GitHub CSV and return rows (as list, not generator)."""
    url = _GITHUB_CSV_SOURCES[benchmark]
    rows = _download_github_csv(url)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _iter_hf_rows(
    hf_path: str,
    hf_name: str | None,
    hf_split: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Load a HuggingFace dataset and return rows as list."""
    ds_lib = _require_datasets()
    kwargs: dict[str, Any] = {"split": hf_split, "trust_remote_code": False}
    if hf_name:
        ds = ds_lib.load_dataset(hf_path, hf_name, **kwargs)
    else:
        ds = ds_lib.load_dataset(hf_path, **kwargs)
    rows = [dict(row) for row in ds]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _iter_or_bench_rows(limit: int | None, include_full: bool = False) -> list[dict[str, Any]]:
    """Load OR-Bench hard-1k (and optionally 80k) subset."""
    ds_lib = _require_datasets()
    rows: list[dict[str, Any]] = []
    subsets = [_OR_BENCH_SUBSET_HARD]
    if include_full:
        subsets.append(_OR_BENCH_SUBSET_FULL)

    loaded_any = False
    for subset in subsets:
        try:
            ds = ds_lib.load_dataset(
                _OR_BENCH_HF_PATH,
                subset,
                split="train",
                trust_remote_code=False,
            )
            subset_rows = [dict(row) for row in ds]
            rows.extend(subset_rows)
            print(f"  Loaded OR-Bench subset {subset!r}: {len(subset_rows)} rows")
            loaded_any = True
        except Exception as e:
            raise RuntimeError(
                f"Could not load OR-Bench subset {subset!r} from {_OR_BENCH_HF_PATH!r}: {e}\n"
                "Manual fallback: download CSVs from https://github.com/exunion/or-bench\n"
                "and normalize manually to hf_data/benchmarks/or_bench.jsonl"
            ) from e

    if not loaded_any:
        raise RuntimeError("No OR-Bench subsets loaded.")

    if limit is not None:
        rows = rows[:limit]
    return rows


# ---------------------------------------------------------------------------
# Per-benchmark ingestion
# ---------------------------------------------------------------------------


def ingest_one(
    benchmark: str,
    output_dir: Path,
    limit: int | None = None,
    dry_run: bool = False,
    or_bench_full: bool = False,
) -> int:
    """Ingest one benchmark dataset.

    Downloads, normalizes each row, writes JSONL.
    Returns number of rows written (or that would be written on dry-run).
    Returns 0 on failure (error printed to stderr).
    """
    print(f"\n[{benchmark}] Starting ingestion...")

    # --- Fetch raw rows ---
    try:
        if benchmark in _GITHUB_CSV_SOURCES:
            print(f"  Source: GitHub CSV ({_GITHUB_CSV_SOURCES[benchmark].split('github.com/')[1].split('/raw')[0] if 'raw' not in _GITHUB_CSV_SOURCES[benchmark] else _GITHUB_CSV_SOURCES[benchmark].split('githubusercontent.com/')[1].split('/main')[0]})")
            raw_rows = _iter_github_csv_rows(benchmark, limit)
        elif benchmark == "or_bench":
            print(f"  Source: HuggingFace {_OR_BENCH_HF_PATH} / {_OR_BENCH_SUBSET_HARD}")
            raw_rows = _iter_or_bench_rows(limit, include_full=or_bench_full)
        else:
            hf_path, hf_name, hf_split = _HF_SOURCES[benchmark]
            print(f"  Source: HuggingFace {hf_path}"
                  + (f" / {hf_name}" if hf_name else "")
                  + f" (split={hf_split})")
            raw_rows = _iter_hf_rows(hf_path, hf_name, hf_split, limit)
    except RuntimeError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"  ERROR fetching {benchmark!r}: {e}", file=sys.stderr)
        return 0

    print(f"  Downloaded {len(raw_rows)} raw rows")

    # --- Normalize ---
    records: list[dict[str, Any]] = []
    skipped = 0

    for i, row in enumerate(raw_rows):
        try:
            normalized = normalize_row(benchmark, row)
            # Fill in auto-numbered prompt_id if normalization returned None
            if not normalized.get("prompt_id"):
                normalized["prompt_id"] = f"{benchmark}_{i + 1:05d}"
            records.append(normalized)
        except (ValueError, KeyError) as e:
            print(f"  WARNING: skipping row {i}: {e}", file=sys.stderr)
            skipped += 1

    print(f"  Normalized: {len(records)} rows ({skipped} skipped)")

    if dry_run:
        output_path = output_dir / f"{benchmark}.jsonl"
        print(f"  Dry run — would write to {output_path}")
        if records:
            print(f"  First record: {json.dumps(records[0], ensure_ascii=False)[:200]}")
        return len(records)

    # --- Write ---
    output_path = output_dir / f"{benchmark}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(records)} records to {output_path}")
    return len(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest standard benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--benchmark",
        required=True,
        choices=ALL_BENCHMARKS + ["all"],
        help="Which benchmark to ingest (or 'all' to ingest all six).",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows per benchmark (for smoke-testing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Normalize rows but do not write output files.",
    )
    p.add_argument(
        "--or-bench-full",
        action="store_true",
        help="For OR-Bench: include the full 80k subset in addition to hard-1k.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    benchmarks = ALL_BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    print(f"Output dir:  {output_dir}")
    print(f"Benchmarks:  {', '.join(benchmarks)}")
    if args.limit:
        print(f"Row limit:   {args.limit} per benchmark")
    if args.dry_run:
        print("Mode:        dry-run (no files written)")

    results: dict[str, int] = {}
    for bm in benchmarks:
        n = ingest_one(
            bm,
            output_dir,
            limit=args.limit,
            dry_run=args.dry_run,
            or_bench_full=getattr(args, "or_bench_full", False),
        )
        results[bm] = n

    print("\nSummary:")
    for bm, n in results.items():
        status = f"{n} rows" if n > 0 else "FAILED (0 rows — see errors above)"
        print(f"  {bm:<20} {status}")


if __name__ == "__main__":
    main()
