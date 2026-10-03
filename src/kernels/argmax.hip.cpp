#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels/kernels.hpp"

static constexpr size_t BLOCK_SIZE = 1024;


__global__ void argmax_kernel(const __half* __restrict__ d_logits, int32_t* __restrict__ d_best_token_id, uint32_t vocab_size) 
{
    // 512 threads / 32 warpSize = 16 warps. 32 slots is plenty.
    __shared__ float   s_max_vals[32];
    __shared__ int32_t s_best_indices[32];

    const uint32_t warp_id = threadIdx.x / warpSize;
    const uint32_t lane_id = threadIdx.x % warpSize;

    // Step A: Each thread strides across the vocabulary
    float local_max = -1e9f;
    int32_t local_idx = -1;

    for (uint32_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = __half2float(d_logits[i]);
        if (val > local_max) {
            local_max = val;
            local_idx = static_cast<int32_t>(i);
        }
    }

    // Step B: In-warp pair reduction using shuffles
    for (uint32_t offset = warpSize >> 1; offset > 0; offset >>= 1) {
        float other_val   = __shfl_down(local_max, offset);
        int32_t other_idx = __shfl_down(local_idx, offset);

        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Step C: Warp leaders write their best candidate to shared memory
    if (lane_id == 0) {
        s_max_vals[warp_id]     = local_max;
        s_best_indices[warp_id] = local_idx;
    }
    __syncthreads();

    // Step D: Warp 0 reduces the warp winners
    if (warp_id == 0) {
        const uint32_t num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float block_max    = (lane_id < num_warps) ? s_max_vals[lane_id] : -1e9f;
        int32_t block_best = (lane_id < num_warps) ? s_best_indices[lane_id] : -1;

        for (uint32_t offset = warpSize >> 1; offset > 0; offset >>= 1) {
            float other_val   = __shfl_down(block_max, offset);
            int32_t other_idx = __shfl_down(block_best, offset);

            if (other_val > block_max) {
                block_max  = other_val;
                block_best = other_idx;
            }
        }

        // Thread 0 writes the single winning token ID to global VRAM
        if (lane_id == 0) {
            *d_best_token_id = block_best;
        }
    }
}

void launch_argmax(
    const __half *d_logits, int* d_best_token_id, size_t vocab_size
){

    // 1 block covers the whole vocabulary
    const dim3 grid(1);
    constexpr dim3 block_dim(BLOCK_SIZE);


    argmax_kernel<<<grid, block_dim>>>(
        d_logits,
        d_best_token_id,
        static_cast<uint32_t>(vocab_size)
    );
}