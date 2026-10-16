#!/usr/bin/env python3
"""Compare identical greedy workloads, with wall-clock timing and warmed medians.

A fixed number of tokens is generated even after EOS to keep workloads equal.
All engines use a KV cache. Optional llama.cpp timings include HTTP streaming.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
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
DEFAULT_PYTORCH_CACHE = Path(__file__).resolve().parent / "pytorch_cache.json"
CACHE_VERSION = 1


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
    parser.add_argument("--pytorch-cache", default=str(DEFAULT_PYTORCH_CACHE),
                        help="JSON cache for PyTorch eager/compile results")
    parser.add_argument("--refresh-pytorch-cache", action="store_true",
                        help="rerun PyTorch even when a matching cached result exists")
    parser.add_argument("--no-pytorch-cache", action="store_true",
                        help="do not read or write cached PyTorch results")
    return parser.parse_args()


def format_chat_prompt(tokenizer, user_text):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def prepare_prompt(tokenizer, prompt_text):
    prompt = format_chat_prompt(tokenizer, prompt_text)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return inputs.input_ids[0].tolist()


def model_cache_identity(model_name, tokenizer):
    model_path = Path(model_name).expanduser()
    if model_path.is_dir():
        # For local models, file metadata invalidates the cache when weights or
        # tokenizer/config files are replaced without reading multi-GB weights.
        files = []
        for path in model_path.rglob("*"):
            if path.is_file():
                stat = path.stat()
                files.append((str(path.relative_to(model_path)), stat.st_size, stat.st_mtime_ns))
        revision = hashlib.sha256(json.dumps(sorted(files), separators=(",", ":")).encode()).hexdigest()
        return {"path": str(model_path.resolve()), "revision": revision}

    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    return {
        "name": model_name,
        "revision": init_kwargs.get("_commit_hash"),
    }


def pytorch_cache_metadata(args, token_ids, tokenizer):
    return {
        "cache_version": CACHE_VERSION,
        "model": model_cache_identity(args.model, tokenizer),
        "prompt_token_ids": token_ids,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "dtype": "float16",
        "compile": {"dynamic": True, "triton_cudagraphs": False},
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        # Include device-selection overrides without initializing a GPU context
        # in the parent, which must release all accelerator memory before HIP runs.
        "device_selection": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
    }


def read_pytorch_cache(cache_path):
    try:
        cache = json.loads(cache_path.read_text())
        if cache.get("version") == CACHE_VERSION and isinstance(cache.get("entries"), dict):
            return cache
    except (OSError, ValueError, AttributeError):
        pass
    return {"version": CACHE_VERSION, "entries": {}}


def write_pytorch_cache(cache_path, cache):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_path.parent,
                                         prefix=cache_path.name + ".", suffix=".tmp",
                                         delete=False) as output:
            temp_path = Path(output.name)
            json.dump(cache, output, indent=2)
            output.write("\n")
        os.replace(temp_path, cache_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


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
    del compiled_forward
    del model, inputs

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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_ids = prepare_prompt(tokenizer, args.prompt)
    metadata = pytorch_cache_metadata(args, token_ids, tokenizer)
    cache_key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_path = Path(args.pytorch_cache).expanduser()
    cache = read_pytorch_cache(cache_path) if not args.no_pytorch_cache else None

    cached = None
    if cache is not None and not args.refresh_pytorch_cache:
        entry = cache["entries"].get(cache_key)
        if isinstance(entry, dict) and entry.get("metadata") == metadata:
            cached = entry.get("results")

    if not isinstance(cached, dict) or not isinstance(cached.get("eager"), dict) or not isinstance(cached.get("compiled"), dict):
        cached = None

    # Process exit releases all PyTorch/Inductor device allocations before
    # benchmarking the independent engines, including allocations retained
    # beyond empty_cache() and compiler.reset().
    if cached is not None:
        print(f"[PyTorch cache] Reusing eager/compile results ({cache_path})")
        eager, compiled = cached["eager"], cached["compiled"]
    else:
        print("[PyTorch cache] No matching entry; running PyTorch eager and compile")
        with tempfile.TemporaryDirectory(prefix="llm-pytorch-benchmark-") as directory:
            result_path = Path(directory) / "results.json"
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            *sys.argv[1:], "--pytorch-output", str(result_path)], check=True)
            results = json.loads(result_path.read_text())
        eager, compiled = results["eager"], results["compiled"]
        if results["input_token_ids"] != token_ids:
            raise RuntimeError("PyTorch subprocess tokenized a different prompt")

        if cache is not None:
            cache["entries"][cache_key] = {"metadata": metadata,
                                           "results": {"eager": eager, "compiled": compiled}}
            write_pytorch_cache(cache_path, cache)

    for label, result in (("PyTorch eager", eager), ("torch.compile", compiled)):
        if result.get("prompt_tokens") != len(token_ids) or result.get("generated_tokens") != args.max_tokens:
            raise RuntimeError(f"Cached {label} result does not match the requested workload")

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
