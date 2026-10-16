#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_warp_functions.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "gpu_utils.hpp"
#include "kernels/kernels.hpp"
#include <cmath>
#include <algorithm>

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

#define BLOCK_SIZE 128

// Query tile: [query_tile_size, head_dim]
#define QUERY_TILE_SIZE 8

// Key value tile: [KV_TILE_SIZE, head_dim]
#define KV_TILE_SIZE 8

#define NUM_TILE_SCORES QUERY_TILE_SIZE * KV_TILE_SIZE

static_assert((QUERY_TILE_SIZE % 2) == 0, "QUERY_TILE_SIZE must be even for the shared-memory layout");

// Performs a block reduction
__device__ inline float block_reduction(float val, float* warp_sums){

    const uint warp_id = threadIdx.x / warpSize;
    const uint lane_id = threadIdx.x % warpSize;


    // Warp reductions
    for(uint offset = warpSize >> 1; offset > 0; offset>>=1){
        val += __shfl_down(val, offset);
    }

    // Write to shared memory the warp sum
    if(lane_id == 0){
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Block reduction
    if(warp_id == 0){
        float block_sum = lane_id < (BLOCK_SIZE / warpSize) ? warp_sums[lane_id]: 0.0f;

        for(uint offset = warpSize >> 1; offset > 0; offset >>= 1){
            block_sum += __shfl_down(block_sum, offset);
        }

        // Share the result
        if(lane_id == 0){
            warp_sums[0] = block_sum;
        }
    }
    __syncthreads();

    float block_sum = warp_sums[0];
    return block_sum;
}
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
    const uint query_len
){
    // Load q tile to shared memory.
    for(uint i = threadIdx.x; i < q_tile_elements; i+=blockDim.x){

        // Get the local 2D coords the thread will be processing
        const uint tile_row = i / head_dim;
        const uint tile_col = i % head_dim;

        const uint token = query_row + tile_row;
        if(token < query_len){
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
    const uint kv_cache_len,
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

        if(token < kv_cache_len){
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
    const uint query_len, // 1 during decode
    const uint kv_cache_len, // All valid cached tokens
    const uint start_pos, // Absolute position of the first query.
    const uint kv_start_row,
    const float rsqrt_head_dim,
    __half* __restrict__ q_tile,
    __half* __restrict__ k_tile,
    float* __restrict__ score_tile
){
    // One thread reads one q row and one k row and computes the dot product between them.
    // TODO: cooperative q * k multiplication + block reduction
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
        if(query_token >= query_len || key_token >= kv_cache_len  || key_token > start_pos + query_token){
            score = -INFINITY;
        }

        score_tile[score_id] = score;
    }
}

__device__ inline void compute_tile_max_scores(
    const uint query_row,
    const uint query_len,
    float* __restrict__ tile_max_score,
    float* __restrict__ score_tile
){
    for(uint q_row = threadIdx.x; q_row < QUERY_TILE_SIZE; q_row+=blockDim.x){
        const uint query_token = query_row + q_row;

        if(query_token < query_len){
            float tile_max = -INFINITY;

            for(uint kv_col = 0; kv_col < KV_TILE_SIZE; kv_col++){
                const float score = score_tile[q_row * KV_TILE_SIZE + kv_col];

                tile_max = fmaxf(tile_max, score);
            }

           tile_max_score[q_row] = tile_max;
        } else {
            tile_max_score[q_row] = -INFINITY;
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


__global__ void causal_attention_prefill(
    const __half* __restrict__ d_q,                 // [seq_len, num_heads, head_dim]
    const __half* __restrict__ d_k,                 // [seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ d_v,                 // [seq_len, num_kv_heads, head_dim]
    __half* __restrict__ d_out,                     // [seq_len, num_heads, head_dim]
    const uint num_heads,
    const uint head_dim,
    const uint query_group_size,
    const uint query_len, // 1 during decode
    const uint kv_cache_len, // All valid cached tokens
    const uint start_pos, // Absolute position of the first query.
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

    init_shared_mem(d_q, q_tile, output_accumulator, max_score, weight_sum, q_tile_elements, head_dim, query_row_stride, query_start_id, query_row, query_len);


    // Q tile fixed
    // Iterate each K, V tile ROWS
    for(uint i = 0; i < num_kv_tiles; i++){

        // Each row is processing KV_TILE tokens
        const uint kv_start_row = i * KV_TILE_SIZE;

        // Get the starting index of this row.
        const uint kv_row_start_idx = kv_start_row * kv_row_stride + (kv_head_id * head_dim);

        // Load tiles into shared memory
        load_kv_tile(kv_tile_elements, head_dim, kv_row_stride, kv_cache_len, kv_start_row, kv_row_start_idx, d_k, d_v, k_tile, v_tile);
        __syncthreads();

        // Compute attention scores S = Q x K_Transposed
        compute_score_tile(head_dim, query_row, query_len, kv_cache_len, start_pos, kv_start_row, rsqrt_head_dim, q_tile, k_tile, score_tile);
        __syncthreads();

        // Compute the max scores
        // One maximum score per query row
        compute_tile_max_scores(query_row, query_len, tile_max_score, score_tile);
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

        if(query_token >= query_len) continue;

        const float result = output_accumulator[out_id] / weight_sum[q_row];

        const size_t global_id = query_token * (num_heads * head_dim) + query_head_id * head_dim + col;
        d_out[global_id] = __float2half(result);
    }

}

void launch_causal_attention_prefill(
    const __half* d_q,
    const __half* d_k_cache,
    const __half* d_v_cache,
    __half* d_out,
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t q_stride,
    size_t kv_stride
){
    const size_t query_len = seq_len;
    const size_t kv_cache_len = seq_len;
    const size_t start_pos = 0;

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

    const size_t query_row_stride = q_stride ? q_stride : num_heads * head_dim;
    const size_t kv_row_stride = kv_stride ? kv_stride : num_kv_heads * head_dim;

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

    causal_attention_prefill<<<grid, block_dim, shared_bytes>>>(
        d_q,
        d_k_cache,
        d_v_cache,
        d_out,
        num_heads,
        head_dim,
        query_group_size,
        query_len,
        kv_cache_len,
        start_pos,
        q_tile_elements,
        query_row_stride,
        num_kv_tiles,
        kv_tile_elements,
        kv_row_stride,
        num_kv_heads,
        rsqrt_head_dim
    );
}


constexpr uint VECTOR_SIZE = sizeof(float4) / sizeof(__half); 


__device__ inline float block_max(float val, float* shared_mem){

    const uint warp_id = threadIdx.x / warpSize;
    const uint lane_id = threadIdx.x % warpSize;


    // Warp max
    for(uint i = warpSize >> 1; i > 0;  i >>=1){
        val = fmaxf(val, __shfl_down(val, i));
    }

    // The leader of the warp Stores warp max
    if(lane_id == 0){
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    // Block reduction
    if(warp_id == 0){

        float maximum = lane_id < (BLOCK_SIZE/warpSize) ? shared_mem[lane_id] : -INFINITY;

        for(uint i = warpSize >> 1; i > 0; i >>= 1){
            maximum = fmaxf(maximum, __shfl_down(maximum, i));
        }

        // Broadcast result to other blocks
        if(lane_id == 0){
            shared_mem[0] = maximum;
        }
    }
    __syncthreads();

    return shared_mem[0];
}

__global__ void causal_attention_decode(
    const __half* __restrict__ d_q,                 // [seq_len=1, num_heads, head_dim]
    const __half* __restrict__ d_k,                 // [seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ d_v,                 // [seq_len, num_kv_heads, head_dim]
    float* __restrict__ d_partial_output,          // [num_heads, num_kv_chunks, head_dim]
    float* __restrict__ d_partial_max,
    float* __restrict__ d_partial_weight_sum,
    const uint query_group_size,
    const uint head_dim,
    const uint kv_cache_len,
    const uint kv_stride,
    const float rsqrt_head_dim,
    const uint num_kv_chunks
){
    const uint head_id = blockIdx.x;
    const uint kv_chunk_id = blockIdx.y;

    const uint kv_head_id = head_id / query_group_size;

    // Get the token range
    const uint kv_start_token_id = kv_chunk_id * KV_CHUNK;
    const uint kv_end_token_id = kv_start_token_id + KV_CHUNK < kv_cache_len ? kv_start_token_id + KV_CHUNK : kv_cache_len;


    __shared__ float warp_sums[BLOCK_SIZE / 32];

    __shared__ float chunk_scores[KV_CHUNK];

    // Compute attention score of each token
    
    const uint warp_id = threadIdx.x / warpSize;  // 0..3
    const uint lane_id = threadIdx.x % warpSize;  // 0..31
    
    for (uint token_base = kv_start_token_id; token_base < kv_end_token_id; token_base += 4)
    {
        const uint token_id = token_base + warp_id;
        float sum = 0.f;
    
        // This warp handles one Q·K dot product.
        for (uint d = lane_id; d < head_dim; d += warpSize) {
            const float q = __half2float(d_q[head_id * head_dim + d]);
            const float k = __half2float(d_k[kv_head_id * head_dim + token_id * kv_stride + d]);
            sum += q * k;
        }
    
        // Reduce only within this warp; no block-wide barrier here.
        for (uint offset = warpSize >> 1; offset > 0; offset>>=1){
            sum += __shfl_down(sum, offset);            
        }
    
        if (lane_id == 0){
            chunk_scores[token_id - kv_start_token_id] = sum * rsqrt_head_dim;
        }
    }
    __syncthreads(); // all chunk scores are ready

    // Compute the maximum computed score of the block
    float local_max = -INFINITY;
    const uint valid_tokens = kv_end_token_id - kv_start_token_id;  
    for(uint i = threadIdx.x; i < valid_tokens; i+=blockDim.x){
        local_max = fmaxf(chunk_scores[i], local_max);
    }

    const float max_score = block_max(local_max, warp_sums);

    // Compute the weights of each score
    for (uint i = threadIdx.x; i < valid_tokens; i += blockDim.x) {
        chunk_scores[i] = expf(chunk_scores[i] - max_score);
    }
    __syncthreads();

    // Compute the softmax denominator
    float local_weight_sum = 0.f; 

    // Sum the local weights
    for(uint i = threadIdx.x; i < valid_tokens; i+= blockDim.x){
        local_weight_sum += chunk_scores[i];
    }

    const float weight_sum = block_reduction(local_weight_sum, warp_sums);

    // Multiply each token V vector with its computed weight!
    for(uint i = threadIdx.x; i < head_dim; i+=blockDim.x){

        float partial_output = 0.f; 

        for(uint t = kv_start_token_id; t < kv_end_token_id; t++){
            const uint global_id = t * kv_stride + kv_head_id * head_dim + i; 
            partial_output += chunk_scores[t - kv_start_token_id] * __half2float(d_v[global_id]);
        }

        // Store the partial output to VRAM
        d_partial_output[head_id * num_kv_chunks * head_dim + kv_chunk_id * head_dim + i] = partial_output;
    }

    // Store to Vram the max score and the weight sum of the chunk
    if (threadIdx.x == 0) {
        const size_t partial_id = head_id * num_kv_chunks + kv_chunk_id;
    
        d_partial_max[partial_id] = max_score;
        d_partial_weight_sum[partial_id] = weight_sum;
    }
}

// Read eight adjacent FP16 values with one aligned 128-bit load. Each uint
// contains the bit patterns of one __half2 pair; arithmetic remains FP32.
__device__ inline float dot_qk_half8(
    const __half* __restrict__ q,
    const __half* __restrict__ k
){
    const uint4 q_bits = *reinterpret_cast<const uint4*>(q);
    const uint4 k_bits = *reinterpret_cast<const uint4*>(k);
    const __half2* q_pairs = reinterpret_cast<const __half2*>(&q_bits);
    const __half2* k_pairs = reinterpret_cast<const __half2*>(&k_bits);

    float sum = 0.f;
    #pragma unroll
    for (uint pair = 0; pair < 4; ++pair) {
        const float2 qf = __half22float2(q_pairs[pair]);
        const float2 kf = __half22float2(k_pairs[pair]);
        sum += qf.x * kf.x + qf.y * kf.y;
    }
    return sum;
}

// each active lane loads eight adjacent half values through a 128-bit load.
__global__ void causal_attention_decode_vectorized(
    const __half* __restrict__ d_q,
    const __half* __restrict__ d_k,
    const __half* __restrict__ d_v,
    float* __restrict__ d_partial_output,
    float* __restrict__ d_partial_max,
    float* __restrict__ d_partial_weight_sum,
    const uint query_group_size,
    const uint head_dim,
    const uint kv_cache_len,
    const uint kv_stride,
    const float rsqrt_head_dim,
    const uint num_kv_chunks
){
    const uint head_id = blockIdx.x;
    const uint kv_chunk_id = blockIdx.y;
    const uint kv_head_id = head_id / query_group_size;
    const uint kv_start_token_id = kv_chunk_id * KV_CHUNK;
    const uint kv_end_token_id = min(kv_start_token_id + KV_CHUNK, kv_cache_len);

    __shared__ float warp_sums[BLOCK_SIZE / 32];
    __shared__ float chunk_scores[KV_CHUNK];

    const uint warp_id = threadIdx.x / 32;
    const uint lane_id = threadIdx.x % 32;
    const __half* q_head = d_q + head_id * head_dim;
   

    // Four warps compute four token scores concurrently. The last chunk can
    // be partial, so each warp checks its token before reading K.
    for (uint token_base = kv_start_token_id; token_base < kv_end_token_id; token_base += 4) {

        const uint token_id = token_base + warp_id;

        if (token_id < kv_end_token_id) {
            const __half* k_token = d_k + token_id * kv_stride + kv_head_id * head_dim;
            float sum = 0.f;

            for (uint vec = lane_id; vec < head_dim / 8; vec += 32) {
                sum += dot_qk_half8(q_head + vec * 8, k_token + vec * 8);
            }
        
            for (uint offset = warpSize >> 1; offset > 0; offset >>= 1){
                sum += __shfl_down(sum, offset);
            }

            if (lane_id == 0){
                chunk_scores[token_id - kv_start_token_id] = sum * rsqrt_head_dim;
            }
        }
    }
    __syncthreads();

    const uint valid_tokens = kv_end_token_id - kv_start_token_id;
    float local_max = -INFINITY;
    for (uint i = threadIdx.x; i < valid_tokens; i += blockDim.x){
        local_max = fmaxf(chunk_scores[i], local_max);
    }
    const float max_score = block_max(local_max, warp_sums);

    for (uint i = threadIdx.x; i < valid_tokens; i += blockDim.x){
        chunk_scores[i] = expf(chunk_scores[i] - max_score);
    }
    __syncthreads();

    float local_weight_sum = 0.f;
    for (uint i = threadIdx.x; i < valid_tokens; i += blockDim.x){
        local_weight_sum += chunk_scores[i];        
    }
    const float weight_sum = block_reduction(local_weight_sum, warp_sums);

    for (uint i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float partial_output = 0.f;
        for (uint t = kv_start_token_id; t < kv_end_token_id; ++t) {
            const uint global_id = t * kv_stride + kv_head_id * head_dim + i;
            partial_output += chunk_scores[t - kv_start_token_id] * __half2float(d_v[global_id]);
        }
        d_partial_output[head_id * num_kv_chunks * head_dim + kv_chunk_id * head_dim + i] = partial_output;
    }

    if (threadIdx.x == 0) {
        const size_t partial_id = head_id * num_kv_chunks + kv_chunk_id;
        d_partial_max[partial_id] = max_score;
        d_partial_weight_sum[partial_id] = weight_sum;
    }
}

// Each thread scans some chunks, then block_max combines the results
__device__ inline float get_global_max_score(
    const float* __restrict__ d_partial_max, // [num_heads, num_kv_chunks]
    float* __restrict__ warp_scratch,
    const uint head_id,
    const uint num_kv_chunks
){

    // Each thread in the block will store a local max in between a number of kv_chunks
    float local_max_score = -INFINITY; 

    // For each kv_chunk [num_heads, num_kv_chunks]
    for(uint i = threadIdx.x; i < num_kv_chunks; i+=blockDim.x){
        const float kv_chunk_max_score = d_partial_max[head_id * num_kv_chunks + i];
        local_max_score = fmaxf(local_max_score, kv_chunk_max_score);
    }

    const float max_score = block_max(local_max_score, warp_scratch);

    return max_score; 
}

// Each thread scans some chunks, then block_reduction combines the results
__device__ inline float get_global_weight_sum(
    const float* __restrict__ d_partial_weight_sum, // [num_heads, num_kv_chunks]
    const float* __restrict__ d_partial_max, // [num_heads, num_kv_chunks]
    float* __restrict__ warp_scratch,
    const uint head_id,
    const uint num_kv_chunks,
    const float global_max_score
){

    // Each thread in the block will store a local max in between a number of kv_chunks
    float local_weight_sum = 0.f; 

    // For each kv_chunk [num_heads, num_kv_chunks]
    for(uint i = threadIdx.x; i < num_kv_chunks; i+=blockDim.x){
        const uint global_id = head_id * num_kv_chunks + i;
        
        const float kv_chunk_weight_sum = d_partial_weight_sum[global_id];
        const float kv_chunk_max_score = d_partial_max[global_id];

        const float scale = expf(kv_chunk_max_score - global_max_score);
        
        local_weight_sum += scale * kv_chunk_weight_sum;
    }

    const float global_weight_sum = block_reduction(local_weight_sum, warp_scratch);

    return global_weight_sum; 
}


// Merge partial output vectors of each chunk of each head into one vector [head_dim]
__device__ void merge_partial_out_vectors(
    const float* __restrict__ d_partial_output, // [num_heads, num_kv_chunks, head_dim]
    const float* __restrict__ d_partial_max, // [num_heads, num_kv_chunks]
    __half* __restrict__ d_out, // [1, num_heads, head_dim]
    const float global_weight_sum,
    const float global_max_score,
    const uint head_id,
    const uint head_dim, 
    const uint num_kv_chunks
){
    
    for(uint i = threadIdx.x; i < head_dim; i+=blockDim.x){

        float sum0 = 0.f;
        float sum1 = 0.f;
        float sum2 = 0.f;
        float sum3 = 0.f;
        
        uint chunk = 0;
        for (; chunk + 3 < num_kv_chunks; chunk += 4) {
            const uint state0 = head_id * num_kv_chunks + chunk;
            const uint state1 = state0 + 1;
            const uint state2 = state0 + 2;
            const uint state3 = state0 + 3;
        
            const float scale0 = expf(d_partial_max[state0] - global_max_score);
            const float scale1 = expf(d_partial_max[state1] - global_max_score);
            const float scale2 = expf(d_partial_max[state2] - global_max_score);
            const float scale3 = expf(d_partial_max[state3] - global_max_score);
        
            sum0 += scale0 * d_partial_output[state0 * head_dim + i];
            sum1 += scale1 * d_partial_output[state1 * head_dim + i];
            sum2 += scale2 * d_partial_output[state2 * head_dim + i];
            sum3 += scale3 * d_partial_output[state3 * head_dim + i];
        }
        
        float output_sum = sum0 + sum1 + sum2 + sum3;
        
        // Handle chunks left over when num_kv_chunks isn't divisible by four.
        for (; chunk < num_kv_chunks; ++chunk) {
            const uint state = head_id * num_kv_chunks + chunk;
            const float scale = expf(d_partial_max[state] - global_max_score);
            output_sum += scale * d_partial_output[state * head_dim + i];
        }
        
        d_out[head_id * head_dim + i] = __float2half(output_sum / global_weight_sum);
    }
}

__global__ void merge_attention_decode(
    const float* __restrict__ d_partial_output, // [num_heads, num_kv_chunks, head_dim]
    const float* __restrict__ d_partial_max, // [num_heads, num_kv_chunks]
    const float* __restrict__ d_partial_weight_sum, // [num_heads, num_kv_chunks]
    __half* __restrict__ d_out, // [1, num_heads, head_dim]
    const uint num_kv_chunks,
    const uint head_dim
){
    const uint head_id = blockIdx.x;

    __shared__ float warp_scratch[BLOCK_SIZE / 32];

    const float global_max_score = get_global_max_score(d_partial_max, warp_scratch, head_id, num_kv_chunks); 
    __syncthreads();
    const float global_weight_sum = get_global_weight_sum(d_partial_weight_sum, d_partial_max, warp_scratch, head_id, num_kv_chunks, global_max_score);

    merge_partial_out_vectors(d_partial_output, d_partial_max, d_out, global_weight_sum, global_max_score, head_id, head_dim, num_kv_chunks);
}

void launch_causal_attention_decode(
    const __half* d_q,              // [1, num_heads, head_dim]
    const __half* d_k_cache,        // [kv_cache_len, num_kv_heads, head_dim]
    const __half* d_v_cache,        // [kv_cache_len, num_kv_heads, head_dim]
    __half* d_out,                  // [1, num_heads, head_dim]
    float* d_partial_out,           // [num_heads, num_kv_chunks, head_dim]
    float* d_partial_max,           // [num_heads, num_kv_chunks]
    float* d_partial_weight_sum,    // [num_heads, num_kv_chunks]
    size_t query_len,               // Must be 1 for this decode implementation.
    size_t kv_cache_len,            // Number of valid cached tokens.
    size_t start_pos,               // Absolute position of the query token.
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t q_stride,                // Elements between Q tokens; 0 means packed.
    size_t kv_stride, // Elements between K/V tokens; 0 means packed.
    hipStream_t stream,
    hipGraphNode_t* attention_node,
    hipGraphNode_t* merge_node
){
    if (query_len == 0 || kv_cache_len == 0 || num_kv_heads == 0) return;


    if (num_heads % num_kv_heads != 0) {
        // Invalid grouped-query-attention configuration.
        return;
    }


    // Each Query head shares a kv head!
    // Compute the size of each query group
    const size_t query_group_size = num_heads / num_kv_heads;



    const size_t query_row_stride = q_stride ? q_stride : num_heads * head_dim;
    const size_t kv_row_stride = kv_stride ? kv_stride : num_kv_heads * head_dim;

    const size_t num_kv_chunks = (kv_cache_len + KV_CHUNK - 1) / KV_CHUNK;

    const float rsqrt_head_dim = static_cast<float>(rsqrtf(head_dim));

    // Each block will process 1 kv chunk = [kv_chunk, head_dim]
    dim3 grid(
        num_heads,                                        // query head
        num_kv_chunks,                                    // kv chunk
        1
    );

    constexpr dim3 block_dim(BLOCK_SIZE);

    causal_attention_decode<<<grid, block_dim, 0, stream>>>(
        d_q,
        d_k_cache,
        d_v_cache,
        d_partial_out,
        d_partial_max,
        d_partial_weight_sum,
        query_group_size,
        head_dim,
        kv_cache_len,
        kv_row_stride,
        rsqrt_head_dim,
        num_kv_chunks
    );

    if(attention_node){
        *attention_node = last_captured_node(stream);
    }

    // One block per query head
    grid = dim3(num_heads, 1, 1);

    merge_attention_decode<<<grid, block_dim, 0, stream>>>(d_partial_out, d_partial_max, d_partial_weight_sum, d_out, num_kv_chunks, head_dim);


    if(merge_node){
        *merge_node = last_captured_node(stream);
    }
}

void launch_causal_attention_decode_vectorized(
    const __half* d_q,
    const __half* d_k_cache,
    const __half* d_v_cache,
    __half* d_out,
    float* d_partial_out,
    float* d_partial_max,
    float* d_partial_weight_sum,
    size_t query_len,
    size_t kv_cache_len,
    size_t start_pos,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t q_stride,
    size_t kv_stride,
    hipStream_t stream,
    hipGraphNode_t* attention_node,
    hipGraphNode_t* merge_node
){
    (void)start_pos; // Decode attention uses the valid cache length directly.
    (void)q_stride; // Decode Q is one packed [num_heads, head_dim] vector.
    if (query_len == 0 || kv_cache_len == 0 || num_kv_heads == 0 || head_dim == 0) return;
    if (num_heads % num_kv_heads != 0) return;

    const size_t query_group_size = num_heads / num_kv_heads;
    const size_t kv_row_stride = kv_stride ? kv_stride : num_kv_heads * head_dim;
    const size_t num_kv_chunks = (kv_cache_len + KV_CHUNK - 1) / KV_CHUNK;
    const float rsqrt_head_dim = static_cast<float>(rsqrtf(head_dim));
    const dim3 grid(static_cast<uint>(num_heads), static_cast<uint>(num_kv_chunks), 1);
    constexpr dim3 block_dim(BLOCK_SIZE);

    causal_attention_decode_vectorized<<<grid, block_dim, 0, stream>>>(
        d_q, d_k_cache, d_v_cache, d_partial_out, d_partial_max,
        d_partial_weight_sum, static_cast<uint>(query_group_size),
        static_cast<uint>(head_dim), static_cast<uint>(kv_cache_len),
        static_cast<uint>(kv_row_stride), rsqrt_head_dim,
        static_cast<uint>(num_kv_chunks));

    if (attention_node)
        *attention_node = last_captured_node(stream);

    const dim3 merge_grid(static_cast<uint>(num_heads), 1, 1);
    merge_attention_decode<<<merge_grid, block_dim, 0, stream>>>(
        d_partial_out, d_partial_max, d_partial_weight_sum, d_out,
        static_cast<uint>(num_kv_chunks), static_cast<uint>(head_dim));

    if (merge_node)
        *merge_node = last_captured_node(stream);
}


void update_causal_attention_decode_node(
    hipGraphExec_t graph_exec,
    hipGraphNode_t node,
    size_t kv_cache_len
) {
    hipKernelNodeParams params{};
    HIP_CHECK(hipGraphKernelNodeGetParams(node, &params));

    uint cache_len = static_cast<uint>(kv_cache_len);
    uint chunks = static_cast<uint>(
        (kv_cache_len + KV_CHUNK - 1) / KV_CHUNK
    );

    // causal_attention_decode: 12 arguments.
    void* args[12];
    std::copy_n(params.kernelParams, 12, args);
    args[8] = &cache_len;
    args[11] = &chunks;

    params.kernelParams = args;
    params.extra = nullptr;
    params.gridDim.y = chunks;

    HIP_CHECK(hipGraphExecKernelNodeSetParams(
        graph_exec, node, &params
    ));
}

void update_merge_attention_decode_node(
    hipGraphExec_t graph_exec,
    hipGraphNode_t node,
    size_t kv_cache_len
) {
    hipKernelNodeParams params{};
    HIP_CHECK(hipGraphKernelNodeGetParams(node, &params));

    uint chunks = static_cast<uint>(
        (kv_cache_len + KV_CHUNK - 1) / KV_CHUNK
    );

    // merge_attention_decode: 6 arguments.
    void* args[6];
    std::copy_n(params.kernelParams, 6, args);
    args[4] = &chunks;

    params.kernelParams = args;
    params.extra = nullptr;

    HIP_CHECK(hipGraphExecKernelNodeSetParams(
        graph_exec, node, &params
    ));
}
