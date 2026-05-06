"""
Adapter inference smoke for SafeComp-DPO.

Loads base Llama-3.1-8B-Instruct in 4-bit, attaches a LoRA adapter
(modules_to_save=[lm_head, embed_tokens] are auto-restored by PEFT), and
generates greedy responses for a small mixed-category prompt set.

This is an *in-pool* inference smoke, not a held-out evaluation. It verifies:
  - base + adapter loading
  - generation pipeline
  - that baseline vs safecomp adapters produce different outputs
    (especially on dual_use)

Usage
-----
    # Run for both regimes, append to one file
    python scripts/eval_adapter_smoke.py \\
        --regime baseline \\
        --adapter outputs/training/baseline_babel/checkpoint \\
        --prompts hf_data/eval/smoke_prompts.jsonl \\
        --output outputs/eval/smoke/adapter_smoke_outputs.jsonl

    python scripts/eval_adapter_smoke.py \\
        --regime safecomp \\
        --adapter outputs/training/safecomp_babel/checkpoint \\
        --prompts hf_data/eval/smoke_prompts.jsonl \\
        --output outputs/eval/smoke/adapter_smoke_outputs.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def load_model_with_adapter(adapter_path: str):
    # Inference uses bf16 base (not 4-bit). Avoids the slow PEFT path that
    # reconciles bf16 modules_to_save (lm_head/embed_tokens) against a
    # 4-bit-quantized base. Memory: ~16 GB for the base + ~1 GB for the
    # restored modules_to_save — fits on a 40 GB A100.
    t0 = time.time()
    print(f"[smoke] loading base {BASE_MODEL} (bf16)…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"[smoke] base loaded in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"[smoke] attaching adapter from {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    print(f"[smoke] adapter attached in {time.time()-t1:.1f}s", flush=True)
    model.eval()
    return model


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_prompts(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=["baseline", "safecomp"])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--mode",
        choices=["raw", "chat"],
        default="raw",
        help="raw: feed prompt as plain text (matches DPO training format). "
        "chat: wrap with Llama chat template before generation.",
    )
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    print(f"[smoke] {len(prompts)} prompts loaded", flush=True)

    tok = load_tokenizer()
    model = load_model_with_adapter(args.adapter)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem_start = torch.cuda.max_memory_allocated() / 1e9
        print(f"[smoke] cuda peak mem after load: {peak_mem_start:.2f} GB", flush=True)

    with open(out_path, "a") as outf:
        for i, p in enumerate(prompts):
            user_text = p["prompt"]
            if args.mode == "chat":
                input_ids = tok.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
                attn = torch.ones_like(input_ids)
            else:
                enc = tok(user_text, return_tensors="pt")
                input_ids = enc["input_ids"].to(model.device)
                attn = enc["attention_mask"].to(model.device)

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            dt = time.time() - t0
            new_tokens = out[0, input_ids.shape[1]:]
            response = tok.decode(new_tokens, skip_special_tokens=True)

            rec = {
                "prompt_id": p["prompt_id"],
                "category": p["category"],
                "regime": args.regime,
                "mode": args.mode,
                "prompt": user_text,
                "response": response,
                "n_input_tokens": int(input_ids.shape[1]),
                "n_new_tokens": int(new_tokens.shape[0]),
                "gen_seconds": round(dt, 2),
                "adapter_path": args.adapter,
                "base_model": BASE_MODEL,
            }
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outf.flush()
            print(
                f"[smoke] [{args.regime}] {i+1}/{len(prompts)} "
                f"cat={p['category']} id={p['prompt_id']} "
                f"new_toks={int(new_tokens.shape[0])} t={dt:.1f}s",
                flush=True,
            )

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[smoke] cuda peak mem: {peak_mem:.2f} GB", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
