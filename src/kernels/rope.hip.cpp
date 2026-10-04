#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "kernels/kernels.hpp"

#define BLOCK_SIZE 64

__global__ void copy_to_kv_cache(
    const __half* __restrict__ d_k,                 // [seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ d_v,                 // [seq_len, num_kv_heads, head_dim]
    __half* __restrict__ d_k_cache,           // [max_seq_len, num_kv_heads, head_dim]
    __half* __restrict__ d_v_cache,           // [max_seq_len, num_kv_heads, head_dim]
    uint seq_len,
    uint kv_dim,
    uint source_stride,
    uint start_pos,
    uint total_k_elements, 
    uint total_kv_elements
){
    const uint global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_id >= total_kv_elements) return;

    const bool is_v = global_id >= total_k_elements;
    const uint element = is_v ? global_id - total_k_elements : global_id;
    
    // Local indexes
    const uint token = element / kv_dim;
    const uint col = element % kv_dim;

    // VRAM indexes
    const uint src = token * source_stride + col;
    const uint dst = (start_pos + token) * kv_dim + col;

    if(is_v){
        d_v_cache[dst] = d_v[src];        
        return; 
    }
    
    d_k_cache[dst] = d_k[src];
}

__device__ __forceinline__ void rotate_pair(
    __half* __restrict__ head_ptr,
    float cosine,
    float sine,
    uint lane,
    uint half_dim,
    __half& out0,
    __half& out1
) {
    const float x0 = __half2float(head_ptr[lane]);
    const float x1 = __half2float(head_ptr[lane + half_dim]);

    out0 = __float2half(x0 * cosine - x1 * sine);
    out1 = __float2half(x1 * cosine + x0 * sine);
}

// Rotate vectors
__global__ void rope_and_store_kv(
    __half* __restrict__ d_q,                 // [seq_len, num_heads, head_dim]
    __half* __restrict__ d_k,                 // [seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ d_v,                 // [seq_len, num_kv_heads, head_dim]
    __half* __restrict__ d_k_cache,           // [max_seq_len, num_kv_heads, head_dim]
    __half* __restrict__ d_v_cache,           // [max_seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ d_cos,         // [max_seq_len, head_dim / 2]
    const __half* __restrict__ d_sin,         // [max_seq_len, head_dim / 2]
    const uint total_q_heads,
    const uint num_heads,
    const uint head_dim,
    const uint half_head_dim,
    const uint num_kv_heads,
    const uint q_stride,
    const uint k_stride,
    const uint start_pos
){
    const uint global_head_id = blockIdx.x;
    const uint32_t lane_id = threadIdx.x; // 0 to half_head_dim - 1
    if (lane_id >= half_head_dim) return;


   
    const bool is_q = global_head_id < total_q_heads; 
    
    if(is_q){
        // Processing Q
        const uint token_idx = global_head_id / num_heads;
        const uint head_in_token = global_head_id % num_heads;
        __half* __restrict__ q = d_q + (token_idx * q_stride) + (head_in_token * head_dim);

        const uint cos_sin_idx = (start_pos + token_idx) * half_head_dim + lane_id;
    
        const float cosine = __half2float(d_cos[cos_sin_idx]);
        const float sine = __half2float(d_sin[cos_sin_idx]);

        __half out0, out1; 
        rotate_pair(q, cosine, sine, lane_id, half_head_dim, out0, out1);
        
        q[lane_id] = out0;
        q[lane_id + half_head_dim] = out1; 
    }
    else {
        // Processing K and V
        const uint k_head_id = global_head_id - total_q_heads;
        const uint token_idx = k_head_id / num_kv_heads;
        const uint head_in_token = k_head_id % num_kv_heads;
        const uint head_offset = head_in_token * head_dim; 
        const uint src = token_idx * k_stride + head_offset;
        const uint dst = (start_pos + token_idx) * num_kv_heads * head_dim + head_offset;

        const uint cos_sin_idx = (start_pos + token_idx) * half_head_dim + lane_id;
    
        const float cosine = __half2float(d_cos[cos_sin_idx]);
        const float sine = __half2float(d_sin[cos_sin_idx]);

        __half out0, out1; 
        rotate_pair(d_k + src, cosine, sine, lane_id, half_head_dim, out0, out1);

        // Preserve the standalone in-place RoPE API used by existing tests.
        if (d_k_cache == nullptr) {
            d_k[src + lane_id] = out0;
            d_k[src + lane_id + half_head_dim] = out1;
            return;
        }
        
        d_k_cache[dst + lane_id] = out0;
        d_k_cache[dst + lane_id + half_head_dim] = out1;

        // Write unchanged V to the cache.
        d_v_cache[dst + lane_id] = d_v[src + lane_id];
        d_v_cache[dst + lane_id + half_head_dim] = d_v[src + lane_id + half_head_dim];
    }
}



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
){
    if(seq_len == 0) return;


    const size_t total_q_heads  = seq_len * num_heads;
    const size_t total_k_heads  = seq_len * num_kv_heads;
    const size_t total_heads = total_q_heads + total_k_heads;
    const size_t half_head_dim = head_dim / 2;
    dim3 grid(total_heads);
    constexpr dim3 block_dim(BLOCK_SIZE);

    if(apply_rotation){
        rope_and_store_kv<<<grid, block_dim>>>(d_q, d_k, d_v, d_k_cache, d_v_cache, d_cos, d_sin,
            static_cast<uint>(total_q_heads),
            static_cast<uint>(num_heads),
            static_cast<uint>(head_dim),
            static_cast<uint>(half_head_dim),
            static_cast<uint>(num_kv_heads),
            static_cast<uint>(q_stride),
            static_cast<uint>(k_stride),
            static_cast<uint>(start_pos)
        );

        return;
    }

    // Launch threads only for k and v.

    const size_t total_v_heads = total_k_heads;
    const size_t total_k_elements = total_k_heads * head_dim; 
    const size_t total_v_elements = total_v_heads * head_dim;
    const size_t total_kv_elements = total_k_elements + total_v_elements;
    grid = (total_kv_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t kv_dim = num_kv_heads * head_dim;
    
    copy_to_kv_cache<<<grid, block_dim>>>(d_k, d_v, d_k_cache, d_v_cache, seq_len, kv_dim, k_stride, start_pos, total_k_elements, total_kv_elements);

}

void launch_rope(
    __half* d_q, __half* d_k, const __half* d_cos, const __half* d_sin,
    size_t seq_len, size_t num_heads, size_t num_kv_heads, size_t head_dim,
    size_t q_stride, size_t k_stride, size_t start_pos
) {
    launch_rope_and_store_kv(d_q, d_k, nullptr, nullptr, nullptr, d_cos, d_sin,
        seq_len, num_heads, num_kv_heads, head_dim, q_stride, k_stride, start_pos, true);
}
