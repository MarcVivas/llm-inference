#!/usr/bin/env python3
"""
reference.py
Runs a reference forward pass and dumps layer activations to binary files 
for C++/HIP unit testing.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from tokenizer import get_tokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def dump_tensor(tensor: torch.Tensor, output_path: Path, name: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().to(torch.float16).contiguous().cpu().numpy()
    arr.tofile(output_path)
    print(f"  [SAVED ACTIVATION] {name: <30} shape: {list(tensor.shape)} -> {output_path.name}")


def tokenize_input_prompt(tokenizer, prompt):
    """ Transform the input text prompt into token ids """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"Prompt: '{prompt}' ({seq_len} tokens)")
    return input_ids, seq_len

def save_input_tokens(fixtures_dir, input_ids):
    """ Save input token ids into a .bin file as int32_t"""
    token_ids_path = fixtures_dir / "input_tokens.bin"
    input_ids.to(torch.int32).numpy().tofile(token_ids_path)
    print(f"Saved token IDs -> {token_ids_path.name}")


def save_activations(model, tokenizer, fixtures_dir, seq_len, input_ids):
    """ Perform a forward pass and save the activations of each layer into a .bin file as f16 """
    layer_0 = model.model.layers[0]

    with torch.no_grad():
        # 1. Embedding lookup
        hidden_states = model.model.embed_tokens(input_ids)
        dump_tensor(hidden_states, fixtures_dir / "layer0_input_embeddings.bin", "Input Embeddings")
 
        # 2. Input RMSNorm
        normed_states = layer_0.input_layernorm(hidden_states)
        dump_tensor(normed_states, fixtures_dir / "layer0_rms_norm_out.bin", "RMSNorm Output")
 
        # 3. QKV Projections
        q = layer_0.self_attn.q_proj(normed_states)
        k = layer_0.self_attn.k_proj(normed_states)
        v = layer_0.self_attn.v_proj(normed_states)
        dump_tensor(q, fixtures_dir / "layer0_q_proj_out.bin", "Q Projection")
        dump_tensor(k, fixtures_dir / "layer0_k_proj_out.bin", "K Projection")
        dump_tensor(v, fixtures_dir / "layer0_v_proj_out.bin", "V Projection")

        # 4. RoPE (Rotary Position Embeddings)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        rotary_emb = model.model.rotary_emb
        cos, sin = rotary_emb(v, position_ids)
        
        num_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // num_heads)
        
        q_reshaped = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        k_reshaped = k.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        q_rot, k_rot = apply_rotary_pos_emb(q_reshaped, k_reshaped, cos, sin)
        
        # Dump rotated Q and K
        dump_tensor(q_rot.transpose(1, 2).contiguous().reshape(1, seq_len, -1), fixtures_dir / "layer0_q_rope.bin", "Rotated Q")
        dump_tensor(k_rot.transpose(1, 2).contiguous().reshape(1, seq_len, -1), fixtures_dir / "layer0_k_rope.bin", "Rotated K")

        # 5. Causal Self-Attention (with GQA repeat)
        num_queries_per_kv = num_heads // num_kv_heads
        v_reshaped = v.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads for GQA in PyTorch SDPA
        k_sdpa = k_rot.repeat_interleave(num_queries_per_kv, dim=1)
        v_sdpa = v_reshaped.repeat_interleave(num_queries_per_kv, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q_rot, 
            k_sdpa, 
            v_sdpa, 
            is_causal=True
        )
        
        # Reshape to [1, seq_len, hidden_size] and dump
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(1, seq_len, -1)
        dump_tensor(attn_out, fixtures_dir / "layer0_attn_out.bin", "Attention Output")

        # 6. Attention Output Projection (o_proj)
        attn_proj = layer_0.self_attn.o_proj(attn_out)
        dump_tensor(attn_proj, fixtures_dir / "layer0_o_proj_out.bin", "Attention Out Projection")

        # 7. First Residual Connection: x = x + attn_proj
        hidden_states = hidden_states + attn_proj

        # 8. Post-Attention RMSNorm (Now receives the CORRECT input!)
        post_attn_norm = layer_0.post_attention_layernorm(hidden_states)
        dump_tensor(post_attn_norm, fixtures_dir / "layer0_post_attn_norm_out.bin", "Post-Attn Norm")

        # 9. SwiGLU MLP
        gate = layer_0.mlp.gate_proj(post_attn_norm)
        up = layer_0.mlp.up_proj(post_attn_norm)
        swiglu_intermediate = F.silu(gate) * up
        dump_tensor(swiglu_intermediate, fixtures_dir / "layer0_swiglu_intermediate.bin", "SwiGLU Act Output")
 
        mlp_out = layer_0.mlp.down_proj(swiglu_intermediate)
        dump_tensor(mlp_out, fixtures_dir / "layer0_mlp_out.bin", "MLP Down Projection")

        # 10. Second Residual Connection: x = x + mlp_out
        hidden_states = hidden_states + mlp_out
 
        # 11. End-to-End Full Forward Pass -> Final Logits
        print("\nRunning full model pass for final logits...")
        full_outputs = model(input_ids)
        final_logits = full_outputs.logits
        dump_tensor(final_logits, fixtures_dir / "final_logits.bin", "Final Logits")
 
        next_token_id = torch.argmax(final_logits[:, -1, :], dim=-1).item()
        decoded_str = tokenizer.decode([next_token_id])
        print(f"\nReference Greedy Token: ID={next_token_id} ('{decoded_str}')")
    
def run_reference(model_id: str, prompt: str, out_dir: str):
    fixtures_dir = Path(out_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer & model: {model_id}")
    tokenizer = get_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    # Tokenize input prompt
    input_ids, seq_len = tokenize_input_prompt(tokenizer, prompt)

    # Save prompt token IDs as int32
    save_input_tokens(fixtures_dir, input_ids)

    # Perform a forward pass and save the activations of each layer
    save_activations(model, tokenizer, fixtures_dir, seq_len, input_ids)

    print(f"\nAll reference fixtures exported successfully to: {fixtures_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PyTorch activation ground truth for test_kernels.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID or path")
    parser.add_argument("--prompt", type=str, default="Inference engineering is", help="Reference test prompt")
    parser.add_argument("--output-dir", type=str, default="./tests/reference", help="Directory to store .bin dumps")
    args = parser.parse_args()

    run_reference(args.model, args.prompt, args.output_dir)