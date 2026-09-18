#pragma once
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstddef>
#include <rocblas/rocblas.h>

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
    const size_t total_qkv_dim,
    rocblas_handle rocblas_handle
);