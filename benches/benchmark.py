#!/usr/bin/env python3
"""Compare identical greedy workloads, with wall-clock timing and warmed medians.

A fixed number of tokens is generated even after EOS to keep workloads equal.
All engines use a KV cache. Optional llama.cpp timings include HTTP streaming.
"""
import argparse
import gc
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

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
    parser.add_argument("--llama-url", help="Optional local llama-server URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--llama-gguf", help="FP16 GGUF; launch llama.cpp after other engines release VRAM")
    parser.add_argument("--llama-server", default="./build/llama-hip/bin/llama-server")
    parser.add_argument("--llama-port", type=positive_int, default=8080)
    parser.add_argument("--pytorch-output", help=argparse.SUPPRESS)
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


def run_llama_benchmark(args, token_ids):
    def run():
        request = urllib.request.Request(
            args.llama_url.rstrip("/") + "/completion",
            data=json.dumps({"prompt": token_ids, "n_predict": args.max_tokens,
                             "temperature": 0, "repeat_penalty": 1.0,
                             "ignore_eos": True, "cache_prompt": False,
                             "return_tokens": True, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        first = None
        generated = []
        final = None
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                event = json.loads(line[5:])
                if "error" in event:
                    raise RuntimeError(event["error"])
                tokens = event.get("tokens", [])
                if tokens and first is None:
                    first = time.perf_counter()
                generated.extend(tokens)
                if event.get("stop"):
                    final = event
                    break
        end = time.perf_counter()
        if first is None or final is None or len(generated) != args.max_tokens:
            raise RuntimeError("llama.cpp did not return the requested token count")
        if final["timings"]["prompt_n"] != len(token_ids):
            raise RuntimeError("llama.cpp prompt length differs")
        return (first - start) * 1000, (end - first) * 1000, generated

    print("[llama.cpp] Warming up and benchmarking local HTTP streaming...")
    for _ in range(args.warmups):
        run()
    runs = [run() for _ in range(args.repeats)]
    prefill = statistics.median(r[0] for r in runs)
    decode = statistics.median(r[1] for r in runs)
    result = {"prompt_tokens": len(token_ids), "input_token_ids": token_ids,
              "generated_tokens": len(runs[-1][2]), "generated_token_ids": runs[-1][2],
              "prefill_time_ms": prefill, "prefill_tokens_per_sec": len(token_ids) * 1000 / prefill,
              "generation_after_first_tokens_per_sec": (args.max_tokens - 1) * 1000 / decode,
              "timing_method": "HTTP streaming wall clock", "uses_kv_cache": True}
    DEFAULT_CPP_JSON.with_name("llama_bench.json").write_text(json.dumps(result, indent=4))
    return result


def print_comparison_table(eager, compiled, cpp, llama=None):
    entries = [("PyTorch Eager", eager), ("torch.compile", compiled), ("Custom C++ HIP", cpp)]
    if llama is not None:
        entries.append(("llama.cpp HTTP", llama))
    if len({result["prompt_tokens"] for _, result in entries}) != 1:
        raise RuntimeError("Prompt lengths differ")

    print(f"\nIdentical prompt: {eager['prompt_tokens']} tokens; generation: {eager['generated_tokens']} tokens")
    print("Fixed token count; EOS stopping disabled. Times are warmed wall-clock medians.")
    print(f"{'Metric':<34} | " + " | ".join(f"{name:<14}" for name, _ in entries))
    print("-" * (37 + 17 * len(entries)))
    print(f"{'KV cache':<34} | " + " | ".join(f"{'yes':<14}" for _ in entries))

    for label, key in [
        ("Time to first token (ms)", "prefill_time_ms"),
        ("Prefill speed (input tok/s)", "prefill_tokens_per_sec"),
        ("Generation after first (tok/s)", "generation_after_first_tokens_per_sec"),
    ]:
        print(f"{label:<34} | " + " | ".join(f"{result[key]:<14.2f}" for _, result in entries))

    if llama is not None:
        print("NOTE: llama.cpp includes HTTP streaming overhead; use an FP16 GGUF with all layers on GPU.")
        if eager["generated_token_ids"] != llama["generated_token_ids"]:
            print("NOTE: llama.cpp and PyTorch token sequences differ.")

    if eager["generated_token_ids"] != compiled["generated_token_ids"]:
        print("NOTE: compiled and eager token sequences differ.")

    if eager["generated_token_ids"] != cpp["generated_token_ids"]:
        print("NOTE: C++ and PyTorch token sequences differ; performance does not establish correctness.")


def run_pytorch_benchmarks(args):
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

    return eager, compiled, token_ids


def main():
    args = parse_arguments()
    if args.llama_gguf and args.llama_url:
        raise ValueError("Use either --llama-gguf or --llama-url")
    if args.pytorch_output:
        eager, compiled, token_ids = run_pytorch_benchmarks(args)
        Path(args.pytorch_output).write_text(json.dumps({
            "eager": eager, "compiled": compiled, "input_token_ids": token_ids,
        }))
        return

    # Process exit releases all PyTorch/Inductor device allocations before
    # benchmarking the independent engines, including allocations retained
    # beyond empty_cache() and compiler.reset().
    with tempfile.TemporaryDirectory(prefix="llm-pytorch-benchmark-") as directory:
        result_path = Path(directory) / "results.json"
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                        *sys.argv[1:], "--pytorch-output", str(result_path)], check=True)
        results = json.loads(result_path.read_text())
    eager, compiled, token_ids = results["eager"], results["compiled"], results["input_token_ids"]

    cpp = run_cpp_benchmark(args, token_ids)

    llama = None
    if args.llama_gguf:
        args.llama_url = f"http://127.0.0.1:{args.llama_port}"
        log_path = DEFAULT_CPP_JSON.with_name("llama_server.log")
        with log_path.open("w") as log:
            server = subprocess.Popen([
                args.llama_server, "-m", args.llama_gguf, "-ngl", "999",
                "-lv", "4",
                "-c", str(max(512, len(token_ids) + args.max_tokens)),
                "--host", "127.0.0.1", "--port", str(args.llama_port),
                "--parallel", "1", "-ctk", "f16", "-ctv", "f16",
            ], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 180
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"llama-server exited; see {log_path}")
                    try:
                        with urllib.request.urlopen(args.llama_url + "/health", timeout=1):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f"llama-server startup timed out; see {log_path}")
                        time.sleep(0.2)
                llama = run_llama_benchmark(args, token_ids)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    elif args.llama_url:
        llama = run_llama_benchmark(args, token_ids)
    print_comparison_table(eager, compiled, cpp, llama)
    DEFAULT_CPP_JSON.with_name("comparison.json").write_text(json.dumps({
        "model": args.model, "repeats": args.repeats, "warmups": args.warmups,
        "eager": eager, "compiled": compiled, "cpp": cpp, "llama": llama,
        "llama_gguf": args.llama_gguf,
    }, indent=4))


if __name__ == "__main__":
    main()
