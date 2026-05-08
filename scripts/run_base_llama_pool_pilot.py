"""
Base-Llama unsafe_compliance pool feasibility pilot — generation runner.

Loads meta-llama/Llama-3.1-8B (the non-Instruct base model), formats each
sampled prompt as a plain Question/Answer continuation, and writes a
ResponseRecord-shaped JSONL.

Why a dedicated runner (instead of scripts/generate_responses.py):
- That script's VLLMBackend is server-mode and assumes a chat-templated
  Instruct model. The base model has no chat template, and we want to
  avoid the RLHF refusal priors that live in the Instruct variant.
- This script is intentionally small and self-contained.

Privacy:
- Only aggregate counts/timing are printed to stdout. Generated text is
  never echoed.
- Output path defaults to a gitignored directory.

Backend selection:
- Try vLLM offline (fast batch). Fall back to HF transformers if vllm is
  not installed in the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_prompt(template: str, user_text: str) -> str:
    return template.format(prompt=user_text)


def have_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


def generate_with_vllm(model_id: str, prompts: list[str], gen_cfg: dict):
    from vllm import LLM, SamplingParams
    print(f"[backend] vLLM offline: loading {model_id}", flush=True)
    t0 = time.time()
    llm = LLM(model=model_id, dtype="bfloat16")
    print(f"[backend] vLLM ready in {time.time()-t0:.1f}s", flush=True)

    sp = SamplingParams(
        temperature=float(gen_cfg["temperature"]),
        top_p=float(gen_cfg["top_p"]),
        max_tokens=int(gen_cfg["max_new_tokens"]),
        n=1,
    )
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    texts: list[str] = []
    n_in: list[int] = []
    n_out: list[int] = []
    for o in outs:
        texts.append(o.outputs[0].text)
        n_in.append(len(o.prompt_token_ids))
        n_out.append(len(o.outputs[0].token_ids))
    return texts, n_in, n_out, elapsed


def generate_with_hf(model_id: str, prompts: list[str], gen_cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[backend] HF transformers: loading {model_id}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"[backend] HF ready in {time.time()-t0:.1f}s", flush=True)

    texts: list[str] = []
    n_in: list[int] = []
    n_out: list[int] = []
    t0 = time.time()
    for i, p in enumerate(prompts, 1):
        enc = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=int(gen_cfg["max_new_tokens"]),
                do_sample=bool(gen_cfg.get("do_sample", True)),
                temperature=float(gen_cfg["temperature"]),
                top_p=float(gen_cfg["top_p"]),
                pad_token_id=tok.eos_token_id,
            )
        new_ids = out[0, enc["input_ids"].shape[1]:]
        texts.append(tok.decode(new_ids, skip_special_tokens=True))
        n_in.append(int(enc["input_ids"].shape[1]))
        n_out.append(int(new_ids.shape[0]))
        if i % 10 == 0 or i == len(prompts):
            print(f"[gen] {i}/{len(prompts)} done", flush=True)
    elapsed = time.time() - t0
    return texts, n_in, n_out, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--prompts", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prompts_path = args.prompts or Path(cfg["prompts_path"])
    output_path = args.output or Path(cfg["output_path"])
    report_path = Path(cfg.get("report_path", str(output_path) + ".report.json"))
    model_id = cfg["model"]
    gen_cfg = cfg["generation"]
    template = cfg.get("prompt_format", "Question: {prompt}\nAnswer:")
    response_type = cfg.get("response_type", "unsafe_compliance")
    run_id = cfg.get("run_id", "base_llama_pool_pilot_v1")

    records = load_jsonl(prompts_path)
    print(f"[main] loaded {len(records)} prompts from {prompts_path}", flush=True)

    formatted = [format_prompt(template, r["prompt"]) for r in records]

    if args.backend == "vllm" or (args.backend == "auto" and have_vllm()):
        try:
            texts, n_in, n_out, elapsed = generate_with_vllm(model_id, formatted, gen_cfg)
            backend_used = "vllm"
        except Exception as e:
            print(f"[backend] vLLM failed ({e!r}); falling back to HF", flush=True)
            texts, n_in, n_out, elapsed = generate_with_hf(model_id, formatted, gen_cfg)
            backend_used = "hf"
    else:
        texts, n_in, n_out, elapsed = generate_with_hf(model_id, formatted, gen_cfg)
        backend_used = "hf"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with output_path.open("w", encoding="utf-8") as fh:
        for r, txt, ni, no_ in zip(records, texts, n_in, n_out):
            md_in = r.get("metadata") or {}
            rec = {
                "response_id": f"{r['prompt_id']}__{response_type}__base_llama_pilot__s01",
                "prompt_id": r["prompt_id"],
                "response": txt,
                "response_type": response_type,
                "model": model_id,
                "temperature": float(gen_cfg["temperature"]),
                "created_at": now,
                "metadata": {
                    "du_author": md_in.get("du_author"),
                    "du_domain": md_in.get("du_domain"),
                    "n_input_tokens": ni,
                    "n_new_tokens": no_,
                    "prompt_format": template,
                    "top_p": float(gen_cfg["top_p"]),
                    "max_new_tokens": int(gen_cfg["max_new_tokens"]),
                    "backend": backend_used,
                    "run_id": run_id,
                    "source": "base_llama_pool_pilot",
                },
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    avg_in = sum(n_in) / max(len(n_in), 1)
    avg_out = sum(n_out) / max(len(n_out), 1)
    report = {
        "run_id": run_id,
        "model": model_id,
        "backend": backend_used,
        "n_prompts": len(records),
        "elapsed_seconds": round(elapsed, 2),
        "avg_input_tokens": round(avg_in, 1),
        "avg_new_tokens": round(avg_out, 1),
        "wrote": str(output_path),
        "created_at": now,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"[main] wrote {len(records)} records -> {output_path}", flush=True)
    print(f"[main] elapsed={elapsed:.1f}s  avg_in={avg_in:.1f}  avg_new={avg_out:.1f}", flush=True)
    print(f"[main] report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
