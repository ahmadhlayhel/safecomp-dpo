"""
Format ablation — raw vs chat-template generation for {base, baseline, safecomp}.

Loads each model variant once, generates greedy responses for the same prompt
set under both prompt formats, writes one combined JSONL. Used to test whether
the missing dual_use effect is partly an eval-format artifact: training and
eval both use raw prompts, but Llama-3.1-8B-Instruct's RLHF priors live in
the chat-template format.

Output schema (one record per generation):
    {variant, mode, prompt_id, category, prompt, response, n_input_tokens,
     n_new_tokens, gen_seconds, adapter_path|null}

Usage (BABEL):
    sbatch scripts/babel_eval_format_ablation.sbatch
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

VARIANTS = [
    {"name": "base",     "adapter": None},
    {"name": "baseline", "adapter": "outputs/training/baseline_babel/checkpoint"},
    {"name": "safecomp", "adapter": "outputs/training/safecomp_babel/checkpoint"},
]

MODES = ["raw", "chat"]


def load_prompts(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_model(variant: dict):
    print(f"\n[load] variant={variant['name']} adapter={variant['adapter']}", flush=True)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if variant["adapter"]:
        model = PeftModel.from_pretrained(base, variant["adapter"], is_trainable=False)
    else:
        model = base
    model.eval()
    print(f"[load] {variant['name']} ready in {time.time()-t0:.1f}s", flush=True)
    return model


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_one(model, tok, user_text: str, mode: str, max_new_tokens: int) -> tuple[str, int, int, float]:
    if mode == "chat":
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    else:
        enc = tok(user_text, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_ids = out[0, input_ids.shape[1]:]
    response = tok.decode(new_ids, skip_special_tokens=True)
    return response, int(input_ids.shape[1]), int(new_ids.shape[0]), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    print(f"[main] {len(prompts)} prompts loaded from {args.prompts}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[main] writing to {args.output}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out_fh = args.output.open("w", encoding="utf-8")
    try:
        for variant in VARIANTS:
            model = load_model(variant)
            for mode in MODES:
                print(f"\n[gen] variant={variant['name']} mode={mode}", flush=True)
                for i, p in enumerate(prompts, 1):
                    response, n_in, n_new, dt = generate_one(
                        model, tok, p["prompt"], mode, args.max_new_tokens
                    )
                    rec = {
                        "variant": variant["name"],
                        "mode": mode,
                        "prompt_id": p["prompt_id"],
                        "category": p["category"],
                        "prompt": p["prompt"],
                        "response": response,
                        "n_input_tokens": n_in,
                        "n_new_tokens": n_new,
                        "gen_seconds": round(dt, 2),
                        "adapter_path": variant["adapter"],
                        "base_model": BASE_MODEL,
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    if i % 10 == 0 or i == len(prompts):
                        print(f"  {variant['name']}/{mode} {i}/{len(prompts)}", flush=True)
            free(model)
    finally:
        out_fh.close()

    print(f"\n[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
