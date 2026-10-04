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
    size_t q_stride,
    size_t k_stride,
    size_t start_pos = 0         // Position offset (0 for prefill)
);


void launch_rope_and_store_kv(
    __half* d_q,                 // [seq_len, num_heads, head_dim]
    __half* d_k,                 // [seq_len, num_kv_heads, head_dim]
    __half* d_v,                 // [seq_len, num_kv_heads, head_dim]
    __half* d_k_cache,           // [max_seq_len, num_kv_heads, head_dim]
    __half* d_v_cache,           // [max_seq_len, num_kv_heads, head_dim]
    const __half* d_cos,         // [max_seq_len, head_dim / 2]
    const __half* d_sin,         // [max_seq_len, head_dim / 2]
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t q_stride,
    size_t k_stride,
    size_t start_pos,        // Position offset (0 for prefill)
    bool apply_rotation
);

void launch_causal_attention_prefill(
    const __half* d_q,
    const __half* d_k,
    const __half* d_v,
    __half* d_out,
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t q_stride = 0,  // Elements between tokens; 0 means packed Q.
    size_t kv_stride = 0  // Elements between tokens; 0 means packed K/V.
);


void launch_causal_attention_decode(
    const __half* d_q,
    const __half* d_k,
    const __half* d_v,
    __half* d_out,
    size_t query_len, // 1 during decode
    size_t kv_cache_len, // All valid cached tokens
    size_t start_pos, // Absolute position of the first query. 
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t q_stride = 0,  // Elements between tokens; 0 means packed Q.
    size_t kv_stride = 0  // Elements between tokens; 0 means packed K/V.
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

void launch_gate_up_gemm(
    const __half* d_input_norm,      // Input:  [seq_len, hidden_size]
    __half* d_gate_up_out,     // Output: [seq_len, 2 * intermediate_size]
    const __half* d_weight_gate_up,       // Weight: [2 * intermediate_size, hidden_size]
    size_t seq_len,
    size_t hidden_size,
    size_t intermediate_size
);


void launch_swiglu(
    const __half* d_gate_up,   // [seq_len, 2 * intermediate_size]
    __half* d_out,          // [seq_len, intermediate_size]
    size_t seq_len, 
    size_t total_elements,   // seq_len * intermediate_size
    size_t intermediate_size
);


void launch_down_proj_gemm(
    const __half* d_in, // [seq_len, intermediate_size]
    __half* d_out,  // [seq_len, hidden_size]
    const __half* d_weights_down, // [intermediate_size, hidden_size]
    size_t seq_len,
    size_t hidden_size,
    size_t intermediate_size,
    float beta = 0.0f    
);

void launch_lm_head_gemm(
    const __half* d_final_norm_out, // [seq_len, hidden_size]
    __half*       d_logits_out, // [seq_len, vocab_size]
    const __half* d_w_lm_head, // Pytorch: [vocab_size, hidden_size] Actual: [hidden_size, vocab_size]
    size_t seq_len,
    size_t hidden_size,
    size_t vocab_size
);

void launch_argmax(const __half *d_logits, int* d_best_token_id, size_t vocab_size);
