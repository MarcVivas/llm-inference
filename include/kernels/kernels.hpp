#pragma once
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstddef>

void launch_embedding_lookup(
    const int32_t* d_input_tokens, 
    const __half* d_embed_table, 
    __half* d_output_embeddings, 
    size_t seq_len, 
    size_t hidden_size
);


void launch_rms_norm(
    const __half* d_input, 
    __half* d_out, 
    const __half* d_gamma,
    const size_t seq_len, 
    const size_t hidden_size,
    const float eps,
    const uint warp_size
);


void launch_qkv_gemm(
    const __half* d_input_norm,
    __half* d_qkv_out,
    const __half* d_qkv_weights,
    const size_t seq_len,
    const size_t hidden_size, 
    const size_t total_qkv_dim
);

void launch_rope(
    __half* d_q,                 // [seq_len, num_heads, head_dim]
    __half* d_k,                 // [seq_len, num_kv_heads, head_dim]
    const __half* d_cos,         // [max_seq_len, head_dim / 2]
    const __half* d_sin,         // [max_seq_len, head_dim / 2]
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t start_pos = 0         // Position offset (0 for prefill)
);


void launch_causal_attention(
    const __half* d_q,
    const __half* d_k,
    const __half* d_v,
    __half* d_out,
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim
);



void launch_attention_out_projection(
    const __half* d_in, //[seq_len, num_heads, head_dim]
    __half* d_out,  //[seq_len, hidden_size]
    const __half* d_o_proj_weights, // [hidden_size, num_heads * head_dim] (Pytorch [out, in])
    size_t seq_len,
    size_t hidden_size,
    size_t num_heads,
    size_t head_dim,
    float beta = 0.f    // 0 for unit test, 1 for fused residual add
);