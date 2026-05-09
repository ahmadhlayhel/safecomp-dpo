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

Backends
--------
  mock     — deterministic mock; fully runnable without GPU or model weights
  trl_dpo  — real TRL DPOTrainer backend; requires GPU + BABEL environment
             (pip install trl transformers accelerate torch; optionally peft bitsandbytes)

Outputs
-------
  report_path:    JSON report with regime, filter stats, training metrics
  checkpoint_dir: directory written when n_examples > 0, containing
                  training_complete.json (structured checkpoint marker)

Usage
-----
    # Mock backend (local dev):
    python scripts/train_dpo.py --config configs/training/safecomp.yaml \\
        --dataset hf_data/dpo/full_dpo_dataset.jsonl

    # Real TRL backend (BABEL):
    python scripts/train_dpo.py --config configs/training/safecomp_babel.yaml \\
        --dataset hf_data/dpo/safecomp_dpo_dataset.jsonl

    # Dry-run (filter only, no training):
    python scripts/train_dpo.py --config configs/training/baseline.yaml \\
        --dataset hf_data/dpo/full_dpo_dataset.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
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
# Dataset conversion helper (exported — testable without TRL)
# ---------------------------------------------------------------------------


def dpo_records_to_hf_dataset(records: list[DPORecord]) -> Any:
    """Convert DPORecord list to a HuggingFace Dataset for TRL DPOTrainer.

    Columns required by TRL: ``prompt``, ``chosen``, ``rejected``.
    Provenance columns included: ``pair_id``, ``category``, ``pair_type``.

    Args:
        records: DPO training examples (may be empty).

    Returns:
        ``datasets.Dataset`` with the columns above.

    Requires:
        ``pip install datasets``
    """
    from datasets import Dataset  # type: ignore[import]

    return Dataset.from_dict(
        {
            "prompt":    [r.prompt for r in records],
            "chosen":    [r.chosen for r in records],
            "rejected":  [r.rejected for r in records],
            "pair_id":   [r.pair_id for r in records],
            "category":  [r.category.value for r in records],
            "pair_type": [r.pair_type.value for r in records],
        }
    )


# ---------------------------------------------------------------------------
# Real TRL backend (BABEL only)
# ---------------------------------------------------------------------------


class TRLDPOBackend:
    """Real DPO training backend using TRL's DPOTrainer.

    Designed for GPU execution on BABEL. Heavy dependencies (trl, transformers,
    torch, accelerate) are imported lazily inside ``train()`` so that the module
    loads without them on dev machines. Optional: peft (LoRA) and bitsandbytes
    (4-bit quantization).

    Config keys consumed (beyond the shared keys handled by run_training):
        model                        HuggingFace model path or ID
        checkpoint_dir               where to save the trained checkpoint
        num_train_epochs             default 1  (BABEL configs use 3)
        per_device_train_batch_size  default 4
        gradient_accumulation_steps  default 4  (effective batch = 16)
        learning_rate                default 5e-6  (appropriate for QLoRA + LoRA)
        lr_scheduler_type            default "cosine"
        warmup_ratio                 default 0.1
        beta                         DPO temperature (default 0.1)
        loss_type                    DPO loss variant (default "sigmoid" = vanilla DPO)
        max_length                   default 1024
        max_prompt_length            default 512
        logging_steps                default 10
        save_strategy                default "epoch"
        bf16                         default True
        gradient_checkpointing       default True  (required for memory on 8B + 4-bit)
        optim                        default "paged_adamw_8bit"  (QLoRA paper recommendation)
        use_4bit                     enable bitsandbytes 4-bit QLoRA (default False)
        use_lora                     enable PEFT LoRA (default False)
        lora_r                       LoRA rank (default 16)
        lora_alpha                   LoRA alpha (default 16; alpha=r gives scaling factor 1.0)
        lora_target_modules          module names or "all-linear" (default "all-linear")
        lora_dropout                 default 0.05
        modules_to_save              list of full-weight modules to save with adapter
                                     (default None; recommended: ["lm_head","embed_tokens"])
    """

    name: str = "trl_dpo"

    def train(
        self,
        examples: list[DPORecord],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        # --- Import heavy deps (BABEL-only) ---
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import DPOConfig, DPOTrainer
        except ImportError as e:
            raise ImportError(
                "TRL backend requires: trl transformers torch accelerate\n"
                "  On BABEL: pip install trl transformers accelerate torch\n"
                "  Optional: pip install peft bitsandbytes\n"
                f"  Missing: {e}"
            ) from e

        import inspect

        import torch

        model_name: str = config.get("model", "meta-llama/Llama-3.1-8B-Instruct")
        checkpoint_dir: str = config.get("checkpoint_dir", "outputs/training/trl_dpo")

        # --- Optional: 4-bit QLoRA via bitsandbytes ---
        bnb_config = None
        if config.get("use_4bit", False):
            try:
                from transformers import BitsAndBytesConfig  # type: ignore[import]

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                print(
                    "WARNING: bitsandbytes not available; "
                    "loading in full precision (use_4bit ignored).",
                    file=sys.stderr,
                )

        # --- Load tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Load model ---
        model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            # Multi-GPU/DDP: pin each rank to its own device so the 4-bit
            # weights aren't sharded across all visible GPUs (which would
            # OOM under DDP). Single-GPU runs (no LOCAL_RANK) keep the
            # prior "auto" behavior.
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if local_rank >= 0:
                model_kwargs["device_map"] = {"": local_rank}
            else:
                model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # --- Optional: LoRA via PEFT ---
        if config.get("use_lora", False):
            try:
                from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]

                # modules_to_save preserves full-weight copies of lm_head and
                # embed_tokens alongside the LoRA adapter (important for QLoRA).
                modules_to_save = config.get("modules_to_save", None)

                lora_cfg = LoraConfig(
                    r=config.get("lora_r", 16),
                    lora_alpha=config.get("lora_alpha", 16),  # alpha=r → scaling factor 1.0
                    target_modules=config.get("lora_target_modules", "all-linear"),
                    lora_dropout=config.get("lora_dropout", 0.05),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    modules_to_save=modules_to_save,
                )
                model = get_peft_model(model, lora_cfg)
                model.print_trainable_parameters()
            except ImportError:
                print(
                    "WARNING: peft not available; LoRA disabled (use_lora ignored).",
                    file=sys.stderr,
                )

        # --- Gradient checkpointing (use_reentrant=False required for 4-bit compat) ---
        if config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # --- Build HF Dataset ---
        dataset = dpo_records_to_hf_dataset(examples)

        # --- DPO training arguments ---
        # Build the kwarg set, then filter to fields actually accepted by the
        # installed TRL version. TRL 1.x dropped `max_prompt_length` from
        # DPOConfig (consolidated into max_length); older TRL accepts it.
        dpo_kwargs: dict[str, Any] = dict(
            output_dir=checkpoint_dir,
            num_train_epochs=int(config.get("num_train_epochs", 1)),
            per_device_train_batch_size=int(config.get("per_device_train_batch_size", 4)),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
            learning_rate=float(config.get("learning_rate", 5e-6)),
            lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
            warmup_ratio=float(config.get("warmup_ratio", 0.1)),
            beta=float(config.get("beta", 0.1)),
            loss_type=config.get("loss_type", "sigmoid"),
            max_length=int(config.get("max_length", 1024)),
            max_prompt_length=int(config.get("max_prompt_length", 512)),
            logging_steps=int(config.get("logging_steps", 10)),
            save_strategy=config.get("save_strategy", "epoch"),
            bf16=bool(config.get("bf16", True)),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
            optim=config.get("optim", "paged_adamw_8bit"),
            remove_unused_columns=False,
        )
        _dpo_params = inspect.signature(DPOConfig.__init__).parameters
        dropped = sorted(set(dpo_kwargs) - set(_dpo_params))
        if dropped:
            print(
                f"[train_dpo] DPOConfig does not accept {dropped}; dropping for "
                "this TRL version.",
                file=sys.stderr,
            )
        dpo_args = DPOConfig(**{k: v for k, v in dpo_kwargs.items() if k in _dpo_params})

        # --- TRL version-compatible trainer construction ---
        # TRL >= 0.9 renamed `tokenizer` → `processing_class`; support both.
        trainer_params = inspect.signature(DPOTrainer.__init__).parameters
        tok_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"

        trainer = DPOTrainer(
            model=model,
            args=dpo_args,
            train_dataset=dataset,
            **{tok_kwarg: tokenizer},
        )

        # --- Train ---
        train_result = trainer.train()
        trainer.save_model(checkpoint_dir)

        metrics = getattr(train_result, "metrics", {}) or {}
        return {
            "backend": self.name,
            "n_examples": len(examples),
            "final_loss": getattr(train_result, "training_loss", None),
            "train_runtime": metrics.get("train_runtime"),
            "train_samples_per_second": metrics.get("train_samples_per_second"),
        }


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


_BACKENDS: dict[str, type] = {
    "mock": MockTrainingBackend,
    "trl_dpo": TRLDPOBackend,
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
                    "final_loss": report.get("final_loss"),
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
