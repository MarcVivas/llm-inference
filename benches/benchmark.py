#!/usr/bin/env python3
"""
benches/benchmark.py
Full benchmark capturing both:
  1. Prefill (Time to First Token / Compute-bound)
  2. Decode (Tokens/second / Bandwidth-bound)
Across:
  - PyTorch Eager
  - PyTorch Compiled (reduce-overhead)
  - Custom C++ Engine (HIP)
"""

import argparse
import json
import subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BENCHES_DIR = Path(__file__).resolve().parent
DEFAULT_CPP_JSON = BENCHES_DIR / "cpp_bench.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Suite")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--weights", type=str, default="weights.bin", help="Path to weights.bin")
    parser.add_argument("--max-tokens", type=int, default=128, help="Tokens to generate")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain GPU memory bandwidth and latency in high performance computing.",
        help="Benchmark prompt text",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default="./build/linux/x86_64/release/benchmark_engine",
        help="Path to compiled C++ binary",
    )
    return parser.parse_args()


def format_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


# Measure prefill latency and decode throughput separately
def measure_full_generation(model, inputs, max_new_tokens: int) -> dict:
    prompt_len = inputs.input_ids.shape[1]

    # Measure Prefill (TTFT) via 1-token generation ---
    ev_1_start = torch.cuda.Event(enable_timing=True)
    ev_1_end = torch.cuda.Event(enable_timing=True)

    ev_1_start.record()
    with torch.no_grad():
        out_1 = model.generate(
            **inputs,
            max_new_tokens=1,
            use_cache=True,
            do_sample=False,
        )
    ev_1_end.record()
    torch.cuda.synchronize()

    t_1_ms = ev_1_start.elapsed_time(ev_1_end)
    prefill_tok_per_sec = (prompt_len / t_1_ms) * 1000.0

    # Measure Full Run (Prefill + N tokens) ---
    ev_n_start = torch.cuda.Event(enable_timing=True)
    ev_n_end = torch.cuda.Event(enable_timing=True)

    ev_n_start.record()
    with torch.no_grad():
        out_n = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
        )
    ev_n_end.record()
    torch.cuda.synchronize()

    t_n_ms = ev_n_start.elapsed_time(ev_n_end)
    generated_count = out_n.shape[1] - prompt_len

    # Isolate Decode Time (T_N - T_1) ---
    # Decoding generated_count - 1 tokens takes (T_N - T_1) ms
    decode_ms = max(t_n_ms - t_1_ms, 1.0)
    decode_tokens = max(generated_count - 1, 1)
    decode_tok_per_sec = (decode_tokens / decode_ms) * 1000.0

    return {
        "prompt_tokens": prompt_len,
        "generated_tokens": generated_count,
        "prefill_ms": t_1_ms,
        "prefill_tok_s": prefill_tok_per_sec,
        "decode_ms": decode_ms,
        "decode_tok_s": decode_tok_per_sec,
    }

def benchmark_pytorch_eager(model, inputs, max_new_tokens: int) -> dict:
    print("\n[PyTorch Eager] Running warmup...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    torch.cuda.synchronize()

    print("[PyTorch Eager] Benchmarking...")
    return measure_full_generation(model, inputs, max_new_tokens)


def benchmark_pytorch_compiled(model, inputs, max_new_tokens: int) -> dict:
    prompt_len = inputs.input_ids.shape[1]
    total_max_len = prompt_len + max_new_tokens

    print("\n[torch.compile] Configuring StaticCache + reduce-overhead...")
    model.generation_config.cache_implementation = "static"
    model.generation_config.max_length = total_max_len

    compiled_model = torch.compile(model, mode="reduce-overhead")

    print(f"[torch.compile] Running full JIT warmup ({max_new_tokens} tokens)...")
    with torch.no_grad():
        _ = compiled_model.generate(**inputs, max_new_tokens=1, use_cache=True)
        _ = compiled_model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    torch.cuda.synchronize()

    print("[torch.compile] Benchmarking...")
    return measure_full_generation(compiled_model, inputs, max_new_tokens)


def run_cpp_benchmark(binary_path: str, weights_path: str, max_new_tokens: int) -> dict:
    print(f"\n[Custom C++ HIP] Invoking binary: {binary_path}...")
    cmd = [
        binary_path,
        weights_path,
        str(max_new_tokens),
        str(DEFAULT_CPP_JSON),
    ]
    subprocess.run(cmd, check=True)

    with open(DEFAULT_CPP_JSON, "r") as f:
        return json.load(f)


def print_comparison_table(eager: dict, compiled: dict, cpp: dict):
    cpp_decode_tok_s = cpp.get("decode_tokens_per_sec", 0.0)
    cpp_prefill_ms = cpp.get("prefill_time_ms", 0.0)
    cpp_prefill_tok_s = cpp.get("prefill_tokens_per_sec", 0.0)

    print("\n" + "=" * 80)
    print("      FULL INFERENCE BENCHMARK: PREFILL vs DECODE (AMD Radeon RX 6800 XT)     ")
    print("=" * 80)
    print(f"{'Metric':<30} | {'PyTorch Eager':<14} | {'torch.compile':<14} | {'Custom C++ HIP':<14}")
    print("-" * 80)

    # Prefill Latency
    print(f"{'Prefill Latency (ms)':<30} | {eager['prefill_ms']:<14.2f} | {compiled['prefill_ms']:<14.2f} | {cpp_prefill_ms:<14.2f}")

    # Prefill Throughput
    print(f"{'Prefill Speed (tok/s)':<30} | {eager['prefill_tok_s']:<14.2f} | {compiled['prefill_tok_s']:<14.2f} | {cpp_prefill_tok_s:<14.2f}")

    # Decode Throughput
    print(f"{'Decode Speed (tok/s)':<30} | {eager['decode_tok_s']:<14.2f} | {compiled['decode_tok_s']:<14.2f} | {cpp_decode_tok_s:<14.2f}")

    print("-" * 80)
    if compiled["decode_tok_s"] > 0:
        speedup = cpp_decode_tok_s / compiled["decode_tok_s"]
        print(f"🚀 Decode Speedup vs torch.compile: {speedup:.2f}x")
    print("=" * 80 + "\n")


def main():
    args = parse_arguments()

    print(f"[Setup] Loading {args.model} onto GPU...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    formatted_prompt = format_chat_prompt(tokenizer, args.prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    eager_res = benchmark_pytorch_eager(model, inputs, args.max_tokens)
    compiled_res = benchmark_pytorch_compiled(model, inputs, args.max_tokens)

    print("\n[Cleanup] Freeing PyTorch GPU memory for C++ engine...")
    del model
    del inputs
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    cpp_res = run_cpp_benchmark(args.binary, args.weights, args.max_tokens)

    print_comparison_table(eager_res, compiled_res, cpp_res)


if __name__ == "__main__":
    main()