"""
DPO training entrypoint.

Contract
--------
Input:  DPORecord JSONL (assembled by assemble_dpo_dataset.py)
Config: Training YAML — regime, backend, example filter, output paths
Output: Training report JSON + checkpoint marker (if examples > 0)

Training regimes
----------------
baseline  — Refusal-DPO: filters to baseline_refusal pairs
            (hard_refusal preferred over unsafe_compliance)
safecomp  — SafeComp-DPO: filters to safe_completion pairs
            (safe_completion preferred over hard_refusal)

Both regimes are defined by their pair_type filter in the config.
The same backend and training routine executes either regime.

Example filter
--------------
  pair_types:  list of PairType values to include  (empty = no filter)
  categories:  list of PromptCategory values to include (empty = no filter)

Backend
-------
  mock  — deterministic mock; fully runnable without GPU or model weights
  (real backend interface provided; real implementation deferred to BABEL)

Outputs
-------
  report_path:    JSON report with regime, filter stats, training metrics
  checkpoint_dir: directory written when n_examples > 0, containing
                  training_complete.json (structured checkpoint marker)

Usage
-----
    python scripts/train_dpo.py --config configs/training/safecomp.yaml \\
        --dataset hf_data/dpo/full_dpo_dataset.jsonl
    python scripts/train_dpo.py --config configs/training/baseline.yaml \\
        --dataset hf_data/dpo/full_dpo_dataset.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

import yaml

from safecomp_dpo.io import load_dpo_records
from safecomp_dpo.schemas import DPORecord


# ---------------------------------------------------------------------------
# Backend protocol — seam for future real BABEL backend
# ---------------------------------------------------------------------------


class TrainingBackend(Protocol):
    """Interface all training backends must satisfy."""

    name: str

    def train(
        self,
        examples: list[DPORecord],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Run training on filtered DPO examples; return metrics dict."""
        ...


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockTrainingBackend:
    """Deterministic mock backend. No GPU or model weights required.

    Simulates DPO training by consuming all examples in order,
    computing a fake loss, and returning structured metrics.

    mock_loss formula: round(1.0 / (1 + n), 6)
    — decreases as more training examples are available, deterministic
      given the same filtered example count.
    """

    name: str = "mock"

    def train(
        self,
        examples: list[DPORecord],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        n = len(examples)
        mock_loss: float | None = round(1.0 / (1 + n), 6) if n > 0 else None

        category_dist: dict[str, int] = defaultdict(int)
        pair_type_dist: dict[str, int] = defaultdict(int)
        for ex in examples:
            category_dist[ex.category.value] += 1
            pair_type_dist[ex.pair_type.value] += 1

        return {
            "backend": self.name,
            "n_examples": n,
            "mock_loss": mock_loss,
            "category_distribution": dict(category_dist),
            "pair_type_distribution": dict(pair_type_dist),
        }


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


_BACKENDS: dict[str, type] = {
    "mock": MockTrainingBackend,
}


def get_backend(name: str) -> TrainingBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]()


# ---------------------------------------------------------------------------
# Filter (pure — no I/O)
# ---------------------------------------------------------------------------


def filter_examples(
    examples: list[DPORecord],
    pair_types: list[str] | None = None,
    categories: list[str] | None = None,
) -> list[DPORecord]:
    """Filter DPO examples by pair_type and/or category.

    Args:
        examples:   full DPO record list
        pair_types: if non-empty, keep only examples with matching pair_type
        categories: if non-empty, keep only examples with matching category

    Returns the filtered list preserving input order.
    Passing None or an empty list for either filter means no filtering on
    that dimension.
    """
    result = examples
    if pair_types:
        allowed = set(pair_types)
        result = [e for e in result if e.pair_type.value in allowed]
    if categories:
        allowed_cats = set(categories)
        result = [e for e in result if e.category.value in allowed_cats]
    return result


# ---------------------------------------------------------------------------
# Core training logic (pure — no I/O)
# ---------------------------------------------------------------------------


def run_training(
    examples: list[DPORecord],
    config: dict[str, Any],
    backend: TrainingBackend,
    config_path: str,
) -> dict[str, Any]:
    """Filter examples and run training. Returns a report dict.

    Args:
        examples:    full DPO record list (pre-loaded)
        config:      training config dict
        backend:     TrainingBackend instance
        config_path: stored in report for provenance

    Returns:
        report dict — regime, filter stats, training metrics
    """
    n_total = len(examples)

    pair_types: list[str] = config.get("pair_types") or []
    categories: list[str] = config.get("categories") or []

    filtered = filter_examples(
        examples,
        pair_types if pair_types else None,
        categories if categories else None,
    )
    n_filtered = len(filtered)

    if n_filtered == 0:
        print(
            f"  WARNING: no training examples remain after filtering "
            f"(pair_types={pair_types!r}, categories={categories!r}). "
            "Training will proceed on an empty set.",
            file=sys.stderr,
        )

    metrics = backend.train(filtered, config)

    report: dict[str, Any] = {
        "regime": config.get("regime", "unknown"),
        "backend": backend.name,
        "config_path": config_path,
        "model": config.get("model", "unknown"),
        "total_examples_loaded": n_total,
        "pair_type_filter": pair_types,
        "category_filter": categories,
        "n_examples_filtered_in": n_filtered,
        "n_examples_filtered_out": n_total - n_filtered,
        **metrics,
    }
    return report


# ---------------------------------------------------------------------------
# Output writing (extracted for testability)
# ---------------------------------------------------------------------------


def write_outputs(
    report: dict[str, Any],
    report_path: str | Path | None,
    checkpoint_dir: str | Path | None,
) -> None:
    """Write training report JSON and checkpoint marker.

    Checkpoint marker is only written when n_examples_filtered_in > 0.
    Both outputs are skipped silently if their paths are None.
    """
    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if checkpoint_dir and report["n_examples_filtered_in"] > 0:
        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        marker = ckpt / "training_complete.json"
        with marker.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "regime": report["regime"],
                    "backend": report["backend"],
                    "n_examples": report["n_examples_filtered_in"],
                    "mock_loss": report.get("mock_loss"),
                },
                f,
                indent=2,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--dataset", help="Override dataset_path from config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and filter only; do not train or write outputs",
    )
    args = parser.parse_args()

    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_path = args.dataset or config.get("dataset_path")
    if not dataset_path:
        print(
            "ERROR: specify --dataset or dataset_path in config.",
            file=sys.stderr,
        )
        sys.exit(1)

    examples = load_dpo_records(dataset_path)
    print(f"Loaded {len(examples)} DPO records from {dataset_path}")

    pair_types: list[str] = config.get("pair_types") or []
    categories: list[str] = config.get("categories") or []
    filtered = filter_examples(
        examples,
        pair_types if pair_types else None,
        categories if categories else None,
    )
    print(
        f"After filter: {len(filtered)} examples "
        f"(pair_types={pair_types!r}, categories={categories!r})"
    )

    if args.dry_run:
        print("Dry run — no training performed.")
        return

    backend_name = config.get("backend", "mock")
    backend = get_backend(backend_name)
    print(f"Backend: {backend.name}")

    print(f"Running training (regime: {config.get('regime', 'unknown')})...")
    report = run_training(examples, config, backend, config_path)

    print(f"  Examples trained on: {report['n_examples_filtered_in']}")
    if report.get("mock_loss") is not None:
        print(f"  Mock loss:           {report['mock_loss']}")
    if report.get("category_distribution"):
        print(f"  Category dist:       {report['category_distribution']}")
    if report.get("pair_type_distribution"):
        print(f"  Pair type dist:      {report['pair_type_distribution']}")

    write_outputs(
        report,
        config.get("report_path"),
        config.get("checkpoint_dir"),
    )

    if config.get("report_path"):
        print(f"Wrote report to {config['report_path']}")
    if config.get("checkpoint_dir") and report["n_examples_filtered_in"] > 0:
        marker = Path(config["checkpoint_dir"]) / "training_complete.json"
        print(f"Wrote checkpoint marker to {marker}")

    print("Done.")


if __name__ == "__main__":
    main()
