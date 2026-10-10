#include "kernels/kernels.hpp"

#define BLOCK_SIZE 128
#define WARP_SIZE 32


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

__global__ void matrix_vec_mul(
    const __half* __restrict__ d_weight,
    const __half* __restrict__ d_input,
    __half* __restrict__ d_output,
    size_t out_dim,
    size_t in_dim,
    float beta
){

    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE]; 
    
    const uint row = blockIdx.x;

    float sum = 0.f;

    
    for(uint i = threadIdx.x; i < in_dim; i+=blockDim.x){
        sum += __half2float(d_weight[row * in_dim + i]) * __half2float(d_input[i]);
    }

    // Perform the reduction
    sum = block_reduction(sum, warp_sums);

    if(threadIdx.x != 0) return;
    
    if(beta != 0.f){
        sum += beta * __half2float(d_output[row]);
    }

    d_output[row] = __float2half(sum);
}



// W: row-major [out_dim, in_dim]
// x: [in_dim]
// y: [out_dim]
// y = W * x + beta * y
void launch_matrix_vector_mul(
    const __half* d_weight,
    const __half* d_input,
    __half* d_output,
    size_t out_dim,
    size_t in_dim,
    float beta,
    hipStream_t stream
){

    if(out_dim == 0) return; 
    
    // 1 block for each output.
    const dim3 grid(out_dim);
    const dim3 blocks(BLOCK_SIZE);
    
    matrix_vec_mul<<<grid, blocks, 0, stream>>>(d_weight, d_input, d_output, out_dim, in_dim, beta);

}
