#!/usr/bin/env python3
"""Compare identical greedy workloads, with wall-clock timing and warmed medians.

A fixed number of tokens is generated even after EOS to keep workloads equal.
PyTorch uses a KV cache; the current C++ engine recomputes the growing context.
"""
import argparse
import gc
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_CPP_JSON = Path(__file__).resolve().parent / "cpp_bench.json"


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", default="weights.bin")
    parser.add_argument("--max-tokens", type=positive_int, default=64)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--warmups", type=positive_int, default=1)
    parser.add_argument("--prompt", default="Explain GPU memory bandwidth.")
    parser.add_argument("--binary", default="./build/linux/x86_64/release/benchmark_engine")
    return parser.parse_args()


def format_chat_prompt(tokenizer, user_text):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


@torch.inference_mode()
def generate(forward, inputs, count):
    # Synchronization plus .item() ensures each interval includes GPU completion
    # and the host token transfer, as in the C++ engine.
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = forward(**inputs, use_cache=True, return_dict=True)
    next_id = output.logits[:, -1, :].argmax(dim=-1).item()
    prefill_ms = (time.perf_counter() - start) * 1000
    generated = [next_id]
    cache = output.past_key_values
    del output
    mask = inputs.attention_mask
    start = time.perf_counter()

    for _ in range(1, count):
        token = torch.tensor([[next_id]], device=inputs.input_ids.device)
        mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
        output = forward(input_ids=token, attention_mask=mask,
                         past_key_values=cache, use_cache=True, return_dict=True)
        next_id = output.logits[:, -1, :].argmax(dim=-1).item()
        generated.append(next_id)
        cache = output.past_key_values
        del output

    generation_ms = (time.perf_counter() - start) * 1000

    return prefill_ms, generation_ms, generated


def benchmark(forward, inputs, args, label):
    print(f"[{label}] Warming up {args.warmups} full generation runs...")

    for _ in range(args.warmups):
        generate(forward, inputs, args.max_tokens)

    print(f"[{label}] Benchmarking...")
    runs = [generate(forward, inputs, args.max_tokens) for _ in range(args.repeats)]
    prefill_ms = statistics.median(run[0] for run in runs)
    generation_ms = statistics.median(run[1] for run in runs)

    return {
        "prompt_tokens": inputs.input_ids.shape[1],
        "generated_tokens": len(runs[-1][2]), "generated_token_ids": runs[-1][2],
        "prefill_time_ms": prefill_ms,
        "prefill_tokens_per_sec": inputs.input_ids.shape[1] * 1000 / prefill_ms,
        "generation_after_first_tokens_per_sec": (
            (args.max_tokens - 1) * 1000 / generation_ms if args.max_tokens > 1 else 0
        ),
    }


def run_cpp_benchmark(args, token_ids):
    with tempfile.TemporaryDirectory(prefix="llm-benchmark-") as directory:
        tokens_file = Path(directory) / "tokens.json"
        output_file = Path(directory) / "results.json"
        tokens_file.write_text(json.dumps(token_ids))
        subprocess.run([args.binary, args.weights, str(args.max_tokens),
                        str(output_file), str(tokens_file), str(args.repeats),
                        str(args.warmups)], check=True)
        result = json.loads(output_file.read_text())

    if result["input_token_ids"] != token_ids:
        raise RuntimeError("C++ benchmark did not use the same input tokens")

    if result["generated_tokens"] != args.max_tokens:
        raise RuntimeError("C++ benchmark generated an unexpected token count")

    DEFAULT_CPP_JSON.write_text(json.dumps(result, indent=4))

    return result


def print_comparison_table(eager, compiled, cpp):
    if len({result["prompt_tokens"] for result in (eager, compiled, cpp)}) != 1:
        raise RuntimeError("Prompt lengths differ")

    print(f"\nIdentical prompt: {eager['prompt_tokens']} tokens; generation: {eager['generated_tokens']} tokens")
    print("Fixed token count; EOS stopping disabled. Times are warmed wall-clock medians.")
    print(f"{'Metric':<34} | {'PyTorch Eager':<14} | {'torch.compile':<14} | {'Custom C++ HIP':<14}")
    print("-" * 90)
    print(f"{'KV cache':<34} | {'yes':<14} | {'yes':<14} | {'no':<14}")

    for label, key in [
        ("Time to first token (ms)", "prefill_time_ms"),
        ("Prefill speed (input tok/s)", "prefill_tokens_per_sec"),
        ("Generation after first (tok/s)", "generation_after_first_tokens_per_sec"),
    ]:
        print(f"{label:<34} | {eager[key]:<14.2f} | {compiled[key]:<14.2f} | {cpp[key]:<14.2f}")

    if eager["generated_token_ids"] != compiled["generated_token_ids"]:
        print("NOTE: compiled and eager token sequences differ.")

    if eager["generated_token_ids"] != cpp["generated_token_ids"]:
        print("NOTE: C++ and PyTorch token sequences differ; performance does not establish correctness.")


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    ).eval()

    prompt = format_chat_prompt(tokenizer, args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        inputs["logits_to_keep"] = 1  # Match C++'s last-token-only LM head.

    token_ids = inputs.input_ids[0].tolist()
    if len(token_ids) + args.max_tokens - 1 > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generation exceeds the context window")

    eager = benchmark(model.forward, inputs, args, "PyTorch Eager")

    # Compile the callable actually used by the timed loop, rather than wrapping
    # the model and calling an inherited generate() method.
    # Dynamic KV tensors survive between calls. Disable graph replay, which can
    # overwrite their storage; Inductor compilation still optimizes forward().
    compiled_forward = torch.compile(
        model.forward, dynamic=True, options={"triton.cudagraphs": False},
    )
    compiled = benchmark(compiled_forward, inputs, args, "torch.compile")
    del compiled_forward, model, inputs

    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    cpp = run_cpp_benchmark(args, token_ids)

    print_comparison_table(eager, compiled, cpp)


if __name__ == "__main__":
    main()
