#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"

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
        x_cached[i] = index < hidden_size ? __half2float(d_input[index]): 0.f;
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

// This kernel is for when hidden size is different than the ones in the switch statement
__global__ void rms_norm_kernel_dynamic(const __half* __restrict__ d_input, const __half* __restrict__ d_gamma, __half* __restrict__ d_out, const uint hidden_size, const float eps, const uint total_elements) 
{
    const uint global_id = blockIdx.x * hidden_size + threadIdx.x; 

    const uint token_idx = blockIdx.x;
    const __half* token_input = d_input + token_idx * hidden_size;
    __half* out_token = d_out + token_idx * hidden_size; 

    float local_sum_squared = 0.f; 
    for(uint i = threadIdx.x; i < hidden_size; i+=BLOCK_SIZE){
        const float x = __half2float(token_input[i]);
        local_sum_squared += x * x;
    }
    
    // First compute the squared mean
    const float mean_squared = block_reduction(local_sum_squared) / static_cast<float>(hidden_size);
    const float inv_rms = rsqrtf(mean_squared + eps);

    // Compute and save to VRAM
    for(uint i = threadIdx.x; i < hidden_size; i+=BLOCK_SIZE){
        const float x = __half2float(token_input[i]);

        // d_gamma [hidden_size]
        const float gamma = __half2float(d_gamma[i]);
        out_token[i] =  __float2half(x * inv_rms * gamma);
    }
}

void launch_rms_norm(
    const __half* d_input, 
    __half* d_out, 
    const __half* d_gamma,
    const size_t seq_len, 
    const size_t hidden_size,
    const float eps,
    const uint warp_size
){

    // 1 block = 1 token 
    const uint total_elements = seq_len * hidden_size;
    const dim3 grid(seq_len);
    constexpr dim3 block_dim(BLOCK_SIZE);

    const uint num_warps = (BLOCK_SIZE + warp_size - 1) / warp_size;
    const size_t shared_bytes = num_warps * sizeof(float);

    switch (hidden_size) {
        case 2048:
            // 2048 / BLOCK_SIZE = 4
            rms_norm_kernel<2048/BLOCK_SIZE><<<grid, block_dim, shared_bytes>>>(d_input, d_gamma, d_out, hidden_size, eps, total_elements);
        case 4096:
            rms_norm_kernel<8><<<grid, block_dim, shared_bytes>>>(d_input, d_gamma, d_out, hidden_size, eps, total_elements);
        case 8192:
            rms_norm_kernel<16><<<grid, block_dim, shared_bytes>>>(d_input, d_gamma, d_out, hidden_size, eps, total_elements);
        default:
            rms_norm_kernel_dynamic<<<grid, block_dim, shared_bytes>>>(d_input, d_gamma, d_out, hidden_size, eps, total_elements);
    }

    HIP_CHECK(hipDeviceSynchronize());
}