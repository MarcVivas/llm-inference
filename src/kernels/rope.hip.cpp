#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "kernels/kernels.hpp"

#define BLOCK_SIZE 64

// Rotate vectors
__global__ void rope(
    __half* __restrict__ d_q,                 // [seq_len, num_heads, head_dim]
    __half* __restrict__ d_k,                 // [seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ d_cos,         // [max_seq_len, head_dim / 2]
    const __half* __restrict__ d_sin,         // [max_seq_len, head_dim / 2]
    const uint total_q_heads,
    const uint num_heads,
    const uint head_dim, 
    const uint half_head_dim,
    const uint num_kv_heads,
    const uint start_pos
){
    const uint global_head_id = blockIdx.x;
    
    
    uint token_idx;
    __half* __restrict__ head_ptr = nullptr;
    
    if(global_head_id < total_q_heads){
        // Processing Q
        token_idx = global_head_id / num_heads; 
        head_ptr = d_q + (global_head_id * head_dim);
    }
    else {
        // Processing K
        const uint k_head_id = global_head_id - total_q_heads;
        token_idx = k_head_id / num_kv_heads;
        head_ptr = d_k + (k_head_id * head_dim);
    }

    const uint cos_sin_idx = (start_pos + token_idx) * half_head_dim + threadIdx.x; 
    
    const float cosine = __half2float(d_cos[cos_sin_idx]);
    const float sine = __half2float(d_sin[cos_sin_idx]);
    
    const float x0 = __half2float(head_ptr[threadIdx.x]);
    const float x1 = __half2float(head_ptr[threadIdx.x + half_head_dim]);

    const __half out0 = __float2half((x0 * cosine) - (x1 * sine));
    const __half out1 = __float2half((x1 * cosine) + (x0 * sine));

    head_ptr[threadIdx.x] = out0;
    head_ptr[threadIdx.x + half_head_dim] = out1; 
}

void launch_rope(
    __half* d_q,                 // [seq_len, num_heads, head_dim]
    __half* d_k,                 // [seq_len, num_kv_heads, head_dim]
    const __half* d_cos,         // [max_seq_len, head_dim / 2]
    const __half* d_sin,         // [max_seq_len, head_dim / 2]
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t start_pos         // Position offset (0 for prefill)
){
    if(seq_len == 0) return;

    
    const size_t total_q_heads  = seq_len * num_heads;
    const size_t total_k_heads  = seq_len * num_kv_heads;
    const size_t total_heads = total_q_heads + total_k_heads;
    const size_t half_head_dim = head_dim / 2; 
    const dim3 grid(total_heads);
    constexpr dim3 block_dim(BLOCK_SIZE);
    rope<<<grid, block_dim>>>(d_q, d_k, d_cos, d_sin, total_q_heads, num_heads, head_dim, half_head_dim, num_kv_heads, start_pos);
}