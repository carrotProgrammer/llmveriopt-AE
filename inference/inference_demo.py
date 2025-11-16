#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple inference script for artifact reproduction.

- Base model path: /artifact/model_latency_3b/base_model
- Adapter path (LoRA/PEFT): /artifact/model_latency_3b/adapter/model_latency_1epoch
- Dataset (HF datasets on disk): /artifact/model_latency_3b/test_data
  * Must contain a string column named "prompt"

Usage (defaults already set):
  python inference_demo.py \
    --base_model /baseline_model/qwen2.5-3b-instruct-local \
    --adapter /artifact/model_latency_3b/adapter/model_latency_1epoch \
    --dataset /artifact/model_latency_3b/test_data \
    --output /artifact/model_latency_3b/output \
    --batch_size 4 \
    --max_new_tokens 2048 \
    --limit 10

Note: fp32, no quantization
"""

import argparse
import os
import csv    
from typing import List

import torch
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@torch.no_grad()
def batched_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    eos_token_id: int = None,
    device: str = "cuda",
):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(tokenizer, "model_max_length", 2048),
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    outputs = model.generate(**enc, **gen_kwargs)
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    input_texts = tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)

    cleaned = []
    for full, inp in zip(texts, input_texts):
        if full.startswith(inp):
            cleaned.append(full[len(inp):].lstrip())
        else:
            cleaned.append(full)
    return cleaned

def main():
    parser = argparse.ArgumentParser()

    # ====== CLI 参数（有默认值，之后会被 YAML 覆盖）======
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=str,
                        default="artifact/model_latency_3b/adapter/model_latency_1epoch")
    parser.add_argument("--dataset", type=str,
                        default="artifact/model_latency_3b/test_data")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--output", type=str,
                        default="artifact/model_latency_3b/output")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # ====== 读取 YAML 配置，覆盖 args（如果存在）======
    config_path = os.path.join(SCRIPT_DIR, "inference_config.yaml") 
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        args.base_model = cfg.get("base_model", args.base_model)
        args.adapter = cfg.get("adapter", args.adapter)
        args.dataset = cfg.get("dataset", args.dataset)
        args.output = cfg.get("output_dir", args.output)
        args.prompt_column = cfg.get("prompt_column", args.prompt_column)

        args.batch_size = cfg.get("batch_size", args.batch_size)
        args.max_new_tokens = cfg.get("max_new_tokens", args.max_new_tokens)
        args.limit = cfg.get("limit", args.limit)

        args.temperature = cfg.get("temperature", args.temperature)
        args.top_p = cfg.get("top_p", args.top_p)
        args.do_sample = cfg.get("do_sample", args.do_sample)
        args.trust_remote_code = cfg.get("trust_remote_code", args.trust_remote_code)

        print(f"[Info] Loaded config from {config_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Demanding fp32

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "results.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "output"])

    print(f"[Info] Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Info] Loading base model (fp32) from: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=None,   # Demanding Single GPU
        trust_remote_code=args.trust_remote_code,
    )

    print(f"[Info] Loading adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        torch_dtype=dtype,
    )

    try:
        print("[Info] Merging LoRA adapter into base model...")
        model = model.merge_and_unload()
    except Exception as e:
        print(f"[Warn] merge_and_unload failed: {e}")

    model.eval()
    model.to(device)

    print(f"[Info] Loading dataset from: {args.dataset}")
    ds = load_from_disk(args.dataset)

    if args.prompt_column not in ds.column_names:
        raise ValueError(f"Dataset does not contain column '{args.prompt_column}'")

    total_samples = len(ds)
    if args.limit is not None:
        total_samples = min(total_samples, args.limit)

    print(f"[Info] Running {total_samples} samples from dataset (prompt column: '{args.prompt_column}')")

    prompts_buffer: List[str] = []

    def flush_batch():
        if not prompts_buffer:
            return

        gens = batched_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts_buffer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
        )

        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for p, g in zip(prompts_buffer, gens):
                text = g
                lower = text.lower()
                if "<answer>" in lower and "</answer>" in lower:
                    try:
                        start = lower.index("<answer>") + len("<answer>")
                        end = lower.index("</answer>")
                        extracted = text[start:end].strip()
                        text = extracted
                    except Exception:
                        pass

                writer.writerow([p, text])

        prompts_buffer.clear()

    for i in range(total_samples):
        p = ds[i][args.prompt_column]
        prompts_buffer.append(str(p))
        if len(prompts_buffer) >= args.batch_size:
            flush_batch()
    flush_batch()

    print(f"[Done] Saved streaming CSV results to: {csv_path}")



if __name__ == "__main__":
    main()
