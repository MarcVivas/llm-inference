#!/usr/bin/env python3
"""
export_weights.py
Extracts and converts Qwen/Qwen-style safetensors weights to FP16 flat binary format
compatible with the C++/HIP inference runner.
"""

import argparse
import struct
from pathlib import Path
import numpy as np
from model_config import ModelConfig
import torch
from transformers import AutoConfig, AutoModelForCausalLM




def export_model_weights(model_config, model_id_or_path, out_file):
    """
    This function exports a weights.bin file containing the model weights and config
    Header: 4 bytes ('QWEN') 8 uint32_t (28 bytes) and 2 f32 values (8 bytes)
    Total bytes header: 44 bytes
    Then you have the model weights as f16 in the order of the network layers.
    """
    print("Loading model tensors into host memory (casting to FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    state_dict = model.state_dict()

    # Binary file header format:
    # Magic Number: 'Q','W','E','N' (4 bytes)
    # Header format: 8 uint32 integers, 2 float32 values
    magic = b"QWEN"
    header_bytes = struct.pack(
        "<4sIIIIIIIIff",
        magic,
        model_config.vocab_size,
        model_config.hidden_size,
        model_config.num_layers,
        model_config.num_heads,
        model_config.num_kv_heads,
        model_config.head_dim,
        model_config.intermediate_size,
        model_config.max_seq_len,
        model_config.rms_norm_eps,
        model_config.rope_theta,
    )

    def write_tensor(file_handle, tensor: torch.Tensor, name: str):
        # Force FP16, contiguous memory layout
        t = tensor.detach().to(torch.float16).contiguous().numpy()
        t.tofile(file_handle)
        print(f"  [WROTE] {name} - shape: {list(tensor.shape)}, size: {t.nbytes / (1024**2):.2f} MB")

    print(f"Writing packed weights to: {out_file}...")
    with open(out_file, "wb") as f:
        f.write(header_bytes)

        # 1. Token Embeddings
        embed_name = "model.embed_tokens.weight"
        write_tensor(f, state_dict[embed_name], embed_name)

        # 2. Transformer Blocks
        for i in range(model_config.num_layers):
            pfx = f"model.layers.{i}."
            print(f"\n--- Exporting Layer {i} ---")

            # Input RMSNorm
            write_tensor(f, state_dict[f"{pfx}input_layernorm.weight"], f"{pfx}input_layernorm.weight")

            # Q, K, V Projections
            write_tensor(f, state_dict[f"{pfx}self_attn.q_proj.weight"], f"{pfx}self_attn.q_proj.weight")
            write_tensor(f, state_dict[f"{pfx}self_attn.k_proj.weight"], f"{pfx}self_attn.k_proj.weight")
            write_tensor(f, state_dict[f"{pfx}self_attn.v_proj.weight"], f"{pfx}self_attn.v_proj.weight")

            # QK-Norm (Present in Qwen2 / Qwen3 architectures)
            if f"{pfx}self_attn.q_norm.weight" in state_dict:
                write_tensor(f, state_dict[f"{pfx}self_attn.q_norm.weight"], f"{pfx}self_attn.q_norm.weight")
                write_tensor(f, state_dict[f"{pfx}self_attn.k_norm.weight"], f"{pfx}self_attn.k_norm.weight")

            # Attention Output Projection
            write_tensor(f, state_dict[f"{pfx}self_attn.o_proj.weight"], f"{pfx}self_attn.o_proj.weight")

            # Post-Attention RMSNorm
            write_tensor(f, state_dict[f"{pfx}post_attention_layernorm.weight"], f"{pfx}post_attention_layernorm.weight")

            # SwiGLU MLP: Gate, Up, Down Projections
            write_tensor(f, state_dict[f"{pfx}mlp.gate_proj.weight"], f"{pfx}mlp.gate_proj.weight")
            write_tensor(f, state_dict[f"{pfx}mlp.up_proj.weight"], f"{pfx}mlp.up_proj.weight")
            write_tensor(f, state_dict[f"{pfx}mlp.down_proj.weight"], f"{pfx}mlp.down_proj.weight")

        # 3. Final RMSNorm
        print("\n--- Exporting Final Head ---")
        write_tensor(f, state_dict["model.norm.weight"], "model.norm.weight")

        # 4. LM Head (Unembedding)
        lm_head_key = "lm_head.weight"
        if lm_head_key in state_dict:
            write_tensor(f, state_dict[lm_head_key], lm_head_key)
        else:
            # Fallback if weights are tied
            write_tensor(f, state_dict[embed_name], "lm_head.weight (tied)")

    total_mb = out_file.stat().st_size / (1024**2)
    print(f"\nCompleted weight export successfully. Total file size: {total_mb:.2f} MB")

def export_model(model_id_or_path: str, output_path: str):
    print(f"Loading configuration for: {model_id_or_path}")
    config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
    
    # Extract structural hyperparameters
    model_config = ModelConfig(config)

    model_config.print_architecture()
   
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Export human-readable config JSON alongside binary
    model_config.export_json(out_file)

    # Export weights
    export_model_weights(model_config, model_id_or_path, out_file)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Hugging Face weights to raw FP16 binary.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face repo ID or local directory")
    parser.add_argument("--output", type=str, default="../weights.bin", help="Output destination binary path")
    args = parser.parse_args()

    export_model(args.model, args.output)