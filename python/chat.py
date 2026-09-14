#!/usr/bin/env python3
"""
chat.py
Interactive CLI chat with the model using PyTorch and Hugging Face transformers.
Supports streaming decode output and multi-turn conversation history.
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer



def main():
    args = parse_args()

    # In ROCm PyTorch builds, AMD GPUs are accessed through the "cuda" device interface
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else "cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()

    # Stream tokens to stdout as they are generated
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."}
    ]

    print("\nModel ready. Type 'exit' or 'quit' to end the session.")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting chat.")
                break

            messages.append({"role": "user", "content": user_input})

            # Format the conversation using the model's tokenizer chat template
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            print("Assistant: ", end="", flush=True)

            gen_kwargs = {
                "streamer": streamer,
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": args.temperature > 0.0,
            }
           
            if gen_kwargs["do_sample"]:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p
           
            with torch.no_grad():
                # Unpack **inputs so attention_mask is preserved
                generated_ids = model.generate(**inputs, **gen_kwargs)


            # Slice out only the new tokens for conversation history tracking
            new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
            reply_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": reply_text})

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat using PyTorch reference.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face repo ID or local path")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens to generate per turn")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()