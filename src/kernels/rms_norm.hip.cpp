#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels/kernels.hpp"
#include <algorithm>
#include <cstdint>

static constexpr size_t BLOCK_SIZE = 512;



// Performs a block reduction
__device__ inline float block_reduction(float val){
    extern __shared__ float warp_sums[];

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
        float block_sum = lane_id < (blockDim.x / warpSize) ? warp_sums[lane_id]: 0.0f;
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

template <size_t VECTORS_PER_THREAD>
__global__ void rms_norm_kernel_vectorized(
    const __half* __restrict__ d_input,
    const __half* __restrict__ d_gamma,
    __half* __restrict__ d_out,
    const uint hidden_size,
    const float eps
){
    constexpr uint VECTOR_WIDTH = 8;
    const uint token_idx = blockIdx.x;
    const uint vector_count = hidden_size / VECTOR_WIDTH;
    const uint vector_start = threadIdx.x;
    const __half* __restrict__ token_input = d_input + token_idx * hidden_size;
    __half* __restrict__ out_token = d_out + token_idx * hidden_size;

    float x_cached[VECTORS_PER_THREAD * VECTOR_WIDTH];
    float local_sum_squared = 0.0f;

    #pragma unroll
    for (uint vector_i = 0; vector_i < VECTORS_PER_THREAD; ++vector_i) {
        const uint vector_id = vector_start + vector_i * blockDim.x;

        if (vector_id < vector_count) {
            const uint4 packed_x = reinterpret_cast<const uint4*>(token_input + vector_id * VECTOR_WIDTH)[0];
            const __half* x_values = reinterpret_cast<const __half*>(&packed_x);

            #pragma unroll
            for (uint j = 0; j < VECTOR_WIDTH; ++j) {
                const float x = __half2float(x_values[j]);
                x_cached[vector_i * VECTOR_WIDTH + j] = x;
                local_sum_squared += x * x;
            }
        }
        else {
            #pragma unroll
            for (uint j = 0; j < VECTOR_WIDTH; ++j) {
                x_cached[vector_i * VECTOR_WIDTH + j] = 0.0f;
            }
        }
    }

    const float mean_squared = block_reduction(local_sum_squared) / static_cast<float>(hidden_size);
    const float inv_rms = rsqrtf(mean_squared + eps);

    #pragma unroll
    for (uint vector_i = 0; vector_i < VECTORS_PER_THREAD; ++vector_i) {
        const uint vector_id = vector_start + vector_i * blockDim.x;

        if (vector_id < vector_count) {
            const uint4 packed_gamma = reinterpret_cast<const uint4*>(d_gamma + vector_id * VECTOR_WIDTH)[0];
            const __half* gamma_values = reinterpret_cast<const __half*>(&packed_gamma);
            uint4 packed_out{};
            __half* out_values = reinterpret_cast<__half*>(&packed_out);

            #pragma unroll
            for (uint j = 0; j < VECTOR_WIDTH; ++j) {
                const float x = x_cached[vector_i * VECTOR_WIDTH + j];
                const float gamma = __half2float(gamma_values[j]);
                out_values[j] = __float2half(x * inv_rms * gamma);
            }

            reinterpret_cast<uint4*>(out_token + vector_id * VECTOR_WIDTH)[0] = packed_out;
        }
    }
}

template <size_t ITEMS_PER_THREAD>
__global__ void rms_norm_kernel(const __half* __restrict__ d_input, const __half* __restrict__ d_gamma, __half* __restrict__ d_out, const uint hidden_size, const float eps, const uint total_elements)
{
    const uint global_id = blockIdx.x * hidden_size + threadIdx.x;

    const uint token_idx = blockIdx.x;
    const __half* token_input = d_input + token_idx * hidden_size;
    __half* out_token = d_out + token_idx * hidden_size;

    float x_cached[ITEMS_PER_THREAD];
    float local_sum_squared = 0.f;

    // Compute the local sum of squares
    for(uint i = 0; i < ITEMS_PER_THREAD; i++){
        const uint index = (i * BLOCK_SIZE) + threadIdx.x;
        x_cached[i] = index < hidden_size ? __half2float(token_input[index]): 0.f;
        local_sum_squared += x_cached[i] * x_cached[i];
    }

    // First compute the squared mean
    const float mean_squared = block_reduction(local_sum_squared) / static_cast<float>(hidden_size);
    const float inv_rms = rsqrtf(mean_squared + eps);

    // Compute and save to VRAM
    #pragma unroll
    for(uint i = 0; i < ITEMS_PER_THREAD; i++){
        const uint index = (i * BLOCK_SIZE) + threadIdx.x;
        if(index >= hidden_size) return;

        const float x = x_cached[i];
        // d_gamma [hidden_size]
        const float gamma = __half2float(d_gamma[index]);
        out_token[index] =  __float2half(x * inv_rms * gamma);
    }
}

void launch_rms_norm(
    const __half* d_input,
    __half* d_out,
    const __half* d_gamma,
    const size_t seq_len,
    const size_t hidden_size,
    const float eps,
    const uint warp_size,
    hipStream_t stream
){

    if (seq_len == 0 || hidden_size == 0) {
        return;
    }

    // 1 block = 1 token
    const uint total_elements = seq_len * hidden_size;
    const dim3 grid(seq_len);

    const size_t vector_count = hidden_size / 8;
    const bool vector_aligned =
        (hidden_size % 8 == 0) &&
        ((reinterpret_cast<uintptr_t>(d_input) & 15u) == 0) &&
        ((reinterpret_cast<uintptr_t>(d_gamma) & 15u) == 0) &&
        ((reinterpret_cast<uintptr_t>(d_out) & 15u) == 0);

    if (vector_aligned) {
        const size_t block_threads = std::min<size_t>(
            BLOCK_SIZE,
            std::max<size_t>(warp_size, ((vector_count + warp_size - 1) / warp_size) * warp_size));
        const size_t vectors_per_thread = (vector_count + block_threads - 1) / block_threads;
        const dim3 block_dim(block_threads);
        const uint num_warps = (block_threads + warp_size - 1) / warp_size;
        const size_t shared_bytes = num_warps * sizeof(float);

        if (vectors_per_thread == 1) {
            rms_norm_kernel_vectorized<1><<<grid, block_dim, shared_bytes, stream>>>(
                d_input, d_gamma, d_out, static_cast<uint>(hidden_size), eps);
            return;
        }

        if (vectors_per_thread == 2) {
            rms_norm_kernel_vectorized<2><<<grid, block_dim, shared_bytes, stream>>>(
                d_input, d_gamma, d_out, static_cast<uint>(hidden_size), eps);
            return;
        }

        if (vectors_per_thread == 4) {
            rms_norm_kernel_vectorized<4><<<grid, block_dim, shared_bytes, stream>>>(
                d_input, d_gamma, d_out, static_cast<uint>(hidden_size), eps);
            return;
        }

        if (vectors_per_thread == 8) {
            rms_norm_kernel_vectorized<8><<<grid, block_dim, shared_bytes, stream>>>(
                d_input, d_gamma, d_out, static_cast<uint>(hidden_size), eps);
            return;
        }
    }

}
