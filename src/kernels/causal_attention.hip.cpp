#include <__clang_hip_math.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "kernels/kernels.hpp"
#include <cmath>

/// Implementation of GROUPED QUERY ATTENTION

// Grouped-query attention: each query head computes standard scaled
// dot-product attention, but a group of query heads shares the same K/V head.
// For query head h, kv_head = h / (num_query_heads / num_kv_heads):
//   scores = Q[h] * transpose(K[kv_head]) / sqrt(head_dim) + causal_mask
//   weights = softmax(scores)
//   output[h] = weights * V[kv_head]

// Softmax converts a row of attention scores into weights that sum to 1.
// Subtract the row maximum first to avoid overflow:
//   max_score = max(scores)
//   weights[i] = exp(scores[i] - max_score)
//   weights[i] /= sum_j exp(scores[j] - max_score)
// Apply the causal mask before softmax so future positions get weight 0.

#define BLOCK_SIZE 16

// Query tile: [query_tile_size, head_dim]
#define QUERY_TILE_SIZE 4

// Key value tile: [KV_TILE_SIZE, head_dim]
#define KV_TILE_SIZE 4

#define NUM_TILE_SCORES QUERY_TILE_SIZE * KV_TILE_SIZE 

static_assert((QUERY_TILE_SIZE % 2) == 0,
              "QUERY_TILE_SIZE must be even for the shared-memory layout");

__device__ inline void init_shared_mem(
    const __half* __restrict__ d_q,                 
    __half* __restrict__ q_tile,
    float* __restrict__  output_accumulator,
    float* __restrict__  max_score,
    float* __restrict__ weight_sum,
    const uint q_tile_elements,
    const uint head_dim, 
    const uint query_row_stride,
    const uint query_start_id,
    const uint query_row,
    const uint seq_len
){
    // Load q tile to shared memory. 
    for(uint i = threadIdx.x; i < q_tile_elements; i+=blockDim.x){

        // Get the local 2D coords the thread will be processing
        const uint tile_row = i / head_dim;
        const uint tile_col = i % head_dim;

        const uint token = query_row + tile_row; 
        if(token < seq_len){
            const uint global_id = query_start_id + tile_row * query_row_stride + tile_col;
            q_tile[i] = d_q[global_id];            
        }
        else {
            q_tile[i] = __float2half(0.f);
        }
        output_accumulator[i] = 0.f; 

    }

    __syncthreads();



    // Initialize arrays
    for(uint i = threadIdx.x; i < QUERY_TILE_SIZE; i+=blockDim.x){
        max_score[i] = -INFINITY;
        weight_sum[i] = 0.f; 
    }
    __syncthreads();
}

__device__ inline void load_kv_tile(
    const uint kv_tile_elements,
    const uint head_dim,
    const uint kv_row_stride,
    const uint seq_len,
    const uint kv_start_row,
    const uint kv_row_start_idx,
    const __half* __restrict__ d_k,
    const __half* __restrict__ d_v,
    __half* __restrict__ k_tile,
    __half* __restrict__ v_tile
){
    for(uint e = threadIdx.x; e < kv_tile_elements; e+=blockDim.x){

        const uint local_row = e  / head_dim;
        const uint local_col = e  % head_dim;
        const uint token = local_row + kv_start_row;

        if(token < seq_len){
            const uint global_id = kv_row_start_idx + local_row * kv_row_stride + local_col;
            k_tile[e] = d_k[global_id];
            v_tile[e] = d_v[global_id];
        }
        else {
            k_tile[e] = __float2half(0.f);
            v_tile[e] = __float2half(0.f);
        }
    }
}

__device__ inline void compute_score_tile(
    const uint head_dim,
    const uint query_row,
    const uint seq_len,
    const uint kv_start_row,
    const float rsqrt_head_dim,
    __half* __restrict__ q_tile,
    __half* __restrict__ k_tile,
    float* __restrict__ score_tile
){
    // One thread reads one q row and one k row and computes the dot product between them.        
    for(uint score_id = threadIdx.x; score_id < NUM_TILE_SCORES; score_id+=blockDim.x){
        const uint local_q_row = score_id / KV_TILE_SIZE;
        const uint local_kv_row = score_id % KV_TILE_SIZE; 

        float score = 0.f;

        for(uint col = 0; col < head_dim; col++){
            score += __half2float(q_tile[local_q_row * head_dim + col]) * __half2float(k_tile[local_kv_row * head_dim + col]);
        }

        score *= rsqrt_head_dim;

        // Apply causal mask so that future tokens receive 0 attention
        const uint query_token = query_row + local_q_row;
        const uint key_token = kv_start_row + local_kv_row; 
        if(query_token >= seq_len || key_token >= seq_len  || key_token > query_token){
            score = -INFINITY;
        }
        
        score_tile[score_id] = score; 
    }
}

__device__ inline void compute_tile_max_scores(
    const uint query_row,
    const uint seq_len,
    float* __restrict__ tile_max_score,
    float* __restrict__ score_tile
){
    for(uint q_row = threadIdx.x; q_row < QUERY_TILE_SIZE; q_row+=blockDim.x){
        const uint query_token = query_row + q_row; 

        if(query_token < seq_len){
            float tile_max = -INFINITY; 

            for(uint kv_col = 0; kv_col < KV_TILE_SIZE; kv_col++){
                const float score = score_tile[q_row * KV_TILE_SIZE + kv_col];

                tile_max = fmaxf(tile_max, score);
            }

           tile_max_score[q_row] = tile_max; 
        }
    }
}

__device__ inline void update_softmax_state(
    float* __restrict__ tile_max_score,
    float* __restrict__ score_tile,
    float* __restrict__ max_score,
    float* __restrict__ old_scales,
    float* __restrict__ weights,
    float* __restrict__ weight_sum
){
    for(uint q_row = threadIdx.x; q_row < QUERY_TILE_SIZE; q_row+=blockDim.x){
        const float old_max = max_score[q_row];
        const float new_max = fmaxf(old_max, tile_max_score[q_row]);

        if(new_max == -INFINITY){
            old_scales[q_row] = 1.0f; 
            for(uint k = 0; k < KV_TILE_SIZE; k++){
                weights[q_row*KV_TILE_SIZE+k] = 0.f; 
            }
            continue;
        } 
        
        const float old_scale = old_max == -INFINITY ? 0.f : expf(old_max - new_max);

        float tile_weight_sum = 0.f; 
        
        for(uint kv_col = 0; kv_col < KV_TILE_SIZE; kv_col++){
            const float score = score_tile[q_row * KV_TILE_SIZE + kv_col];
            const float weight = expf(score - new_max);
            weights[q_row * KV_TILE_SIZE + kv_col] = weight;
            tile_weight_sum += weight; 

        }

        old_scales[q_row] = old_scale;
        weight_sum[q_row] = old_scale * weight_sum[q_row] + tile_weight_sum;
        max_score[q_row] = new_max; 
        
    }
}

__device__ inline void update_output_accumulator(
    const uint q_tile_elements,
    const uint head_dim, 
    float* __restrict__ output_accumulator,
    float* __restrict__ old_scales,
    float* __restrict__ weights,
    __half* __restrict__ v_tile
){
    for(uint out_id = threadIdx.x; out_id < q_tile_elements; out_id+=blockDim.x){
        const uint q_row = out_id / head_dim;
        const uint col = out_id % head_dim;

        float tile_value_sum = 0.f;
        for(uint kv_col = 0; kv_col < KV_TILE_SIZE; kv_col++){
            tile_value_sum += weights[q_row * KV_TILE_SIZE + kv_col] * __half2float(v_tile[kv_col * head_dim + col]);
        }

        output_accumulator[out_id] = old_scales[q_row] * output_accumulator[out_id] + tile_value_sum; 
    }
}


__global__ void causal_attention(
    const __half* __restrict__ d_q,                 // [seq_len, num_heads, head_dim]
    const __half* __restrict__ d_k,                 // [seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ d_v,                 // [seq_len, num_kv_heads, head_dim]
    __half* __restrict__ d_out,                     // [seq_len, num_heads, head_dim]
    const uint num_heads,
    const uint head_dim, 
    const uint query_group_size,
    const uint seq_len,
    const uint q_tile_elements,
    const uint query_row_stride,
    const uint num_kv_tiles,
    const uint kv_tile_elements,
    const uint kv_row_stride,
    const uint num_kv_heads,
    const float rsqrt_head_dim
){
    const uint query_tile_id = blockIdx.x; 
    const uint query_head_id = blockIdx.y;
    
    const uint kv_head_id = query_head_id / query_group_size; 

    const uint query_row = query_tile_id * QUERY_TILE_SIZE;
    const uint query_start_id = (query_row * query_row_stride) + (query_head_id * head_dim);


    // Declaration of shared memory buffers
    extern __shared__ __half shared_mem[];

    __half* __restrict__ q_tile = shared_mem; 
    
    // weighted sum of V accumulated so far
    float* __restrict__ output_accumulator = reinterpret_cast<float*>(q_tile + QUERY_TILE_SIZE * head_dim);

    __half* __restrict__ k_tile = reinterpret_cast<__half*>(output_accumulator + QUERY_TILE_SIZE * head_dim);
    __half* __restrict__ v_tile = k_tile + KV_TILE_SIZE * head_dim;
    
    // Each row has a max score
    // Stores the largest attention score seen from earlier KV tiles.
    __shared__ float max_score[QUERY_TILE_SIZE];

    // Largest score in the current KV tile 
    __shared__ float tile_max_score[QUERY_TILE_SIZE];

    // For each row, softmax denominator accumulated so far
    __shared__ float weight_sum[QUERY_TILE_SIZE];

    __shared__ float score_tile[QUERY_TILE_SIZE * KV_TILE_SIZE];

    __shared__ float weights[QUERY_TILE_SIZE * KV_TILE_SIZE];
    __shared__ float old_scales[QUERY_TILE_SIZE];
    
    init_shared_mem(d_q, q_tile, output_accumulator, max_score, weight_sum, q_tile_elements, head_dim, query_row_stride, query_start_id, query_row, seq_len);


    // Q tile fixed
    // Iterate each K, V tile ROWS
    for(uint i = 0; i < num_kv_tiles; i++){

        // Each row is processing KV_TILE tokens
        const uint kv_start_row = i * KV_TILE_SIZE;
        
        // Get the starting index of this row.
        const uint kv_row_start_idx = kv_start_row * (num_kv_heads * head_dim) + (kv_head_id * head_dim);
        
        // Load tiles into shared memory
        load_kv_tile(kv_tile_elements, head_dim, kv_row_stride, seq_len, kv_start_row, kv_row_start_idx, d_k, d_v, k_tile, v_tile);
        __syncthreads();

        // Compute attention scores S = Q x K_Transposed
        compute_score_tile(head_dim, query_row, seq_len, kv_start_row, rsqrt_head_dim, q_tile, k_tile, score_tile);
        __syncthreads();

        // Compute the max scores
        // One maximum score per query row
        compute_tile_max_scores(query_row, seq_len, tile_max_score, score_tile);
        __syncthreads();

        // Compute weights and update scores
        update_softmax_state(tile_max_score, score_tile, max_score, old_scales, weights, weight_sum);
        __syncthreads();

        // weights * V
        update_output_accumulator(q_tile_elements, head_dim, output_accumulator, old_scales, weights, v_tile);
        __syncthreads();
    }

    // Store to VRAM
    for(uint out_id = threadIdx.x; out_id < q_tile_elements; out_id+=blockDim.x){
        const uint q_row = out_id / head_dim;
        const uint col = out_id % head_dim;
        const uint query_token = query_row + q_row; 

        if(query_token >= seq_len) continue;

        const float result = output_accumulator[out_id] / weight_sum[q_row]; 

        const size_t global_id = query_token * query_row_stride + query_head_id * head_dim + col;
        d_out[global_id] = __float2half(result);
    }
    
}

void launch_causal_attention(
    const __half* d_q,
    const __half* d_k,
    const __half* d_v,
    __half* d_out,
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim
){
    if (seq_len == 0 || num_kv_heads == 0) {
        return;
    }
    
    if (num_heads % num_kv_heads != 0) {
        // Invalid grouped-query-attention configuration.
        return;
    }

    
    // Each Query head shares a kv head!
    // Compute the size of each query group
    const size_t query_group_size = num_heads / num_kv_heads;

    const size_t q_tile_elements = QUERY_TILE_SIZE * head_dim;
    const size_t kv_tile_elements = KV_TILE_SIZE * head_dim;
    
    const size_t num_q_tiles = (seq_len + QUERY_TILE_SIZE - 1) / QUERY_TILE_SIZE;
    const size_t num_kv_tiles = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

    const size_t query_row_stride = num_heads * head_dim; 
    const size_t kv_row_stride = num_kv_heads * head_dim;

    const float rsqrt_head_dim = static_cast<float>(rsqrtf(head_dim));
    
    // Each block will process 1 query tile
    const dim3 grid(
        num_q_tiles,                                        // query tile
        num_heads,                                          // query head
        1
    );
    constexpr dim3 block_dim(BLOCK_SIZE);

    const size_t shared_bytes = 
        QUERY_TILE_SIZE * head_dim * sizeof(__half) +   // Input q tile 
        QUERY_TILE_SIZE * head_dim * sizeof(float) + // Output q tile accumulator
        KV_TILE_SIZE * head_dim * sizeof(__half) +  // K tile
        KV_TILE_SIZE * head_dim * sizeof(__half)    // V tile
    ;

    causal_attention<<<grid, block_dim, shared_bytes>>>(
        d_q, 
        d_k,
        d_v,
        d_out,
        num_heads,
        head_dim,
        query_group_size,
        seq_len,
        q_tile_elements,
        query_row_stride,
        num_kv_tiles,
        kv_tile_elements,
        kv_row_stride,
        num_kv_heads,
        rsqrt_head_dim
    );
}