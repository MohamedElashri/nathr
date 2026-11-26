"""Benchmark the eager FP32 CPU reference using the Phase 0 fixed workload."""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()
    load_ms = 1000 * (time.perf_counter() - start)
    prompt_length, decode_length, context = 128, 32, 512
    warmups, repetitions = 2, 7
    ids = torch.tensor([[17 + (i * 7919 + 12345) % (49152 - 17)
                         for i in range(prompt_length + decode_length)]])
    prompt = ids[:, :prompt_length]
    decode_inputs = ids[:, prompt_length:].split(1, dim=1)
    times = []
    for repetition in range(-warmups, repetitions):
        cache = DynamicCache(config=model.config)
        start = time.perf_counter()
        model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        prefetched = time.perf_counter()
        for token in decode_inputs:
            model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        finished = time.perf_counter()
        if repetition >= 0:
            times.append([1000 * (prefetched - start), 1000 * (finished - prefetched)])
    print(json.dumps({
        "prompt_tokens": prompt_length, "decode_tokens": decode_length, "context": context,
        "threads": args.threads, "warmups": warmups, "load_ms": load_ms,
        "prefill_and_decode_ms": times, "torch": torch.__version__,
        "cache_policy": "dynamic FP32 cache, active to 160 positions",
    }))


if __name__ == "__main__":
    main()
