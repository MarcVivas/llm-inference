#include "kernels/kernels.hpp"

#define BLOCK_SIZE 256
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

__global__ void matrix_vec_mul_vectorized(
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

    // 8 __half elements = 16 bytes (128 bits / 4 words)
    constexpr int VEC_SIZE = 8;
    const uint vec_in_dim = in_dim / VEC_SIZE;

    // Cast pointers to float4 (16 bytes)
    const float4* vec_weight = reinterpret_cast<const float4*>(d_weight + row * in_dim);
    const float4* vec_input  = reinterpret_cast<const float4*>(d_input);

    // 1. Vectorized main loop (4 words per thread)
    for (uint i = threadIdx.x; i < vec_in_dim; i += blockDim.x) {
        float4 w_raw = vec_weight[i];
        float4 x_raw = vec_input[i];

        const __half* w_h = reinterpret_cast<const __half*>(&w_raw);
        const __half* x_h = reinterpret_cast<const __half*>(&x_raw);

        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k) {
            sum += __half2float(w_h[k]) * __half2float(x_h[k]);
        }
    }

    // 2. Scalar tail loop for remaining elements (if in_dim % 8 != 0)
    for (uint i = vec_in_dim * VEC_SIZE + threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(d_weight[row * in_dim + i]) * __half2float(d_input[i]);
    }

    // 3. Block Reduction
    sum = block_reduction(sum, warp_sums);

    if (threadIdx.x == 0) {
        if (beta != 0.f) {
            sum += beta * __half2float(d_output[row]);
        }
        d_output[row] = __float2half(sum);
    }
}

// One block computes one output row. Vector loads are used only when the
// row and input are aligned; partial vectors and unaligned rows use scalars.
__global__ __launch_bounds__(BLOCK_SIZE) void matrix_vec_mul_tuned(
    const __half* __restrict__ d_weight,
    const __half* __restrict__ d_input,
    __half* __restrict__ d_output,
    size_t out_dim,
    size_t in_dim,
    float beta
) {
    constexpr size_t VECTOR_ELEMENTS = sizeof(float4) / sizeof(__half);
    const size_t row = blockIdx.x;

    if (row >= out_dim) return;  // Uniform across the block, before any barrier.

    const __half* __restrict__ weights = d_weight + row * in_dim;

    float sum = 0.0f;

    const uint vectors = in_dim / VECTOR_ELEMENTS;
    float accumulators[VECTOR_ELEMENTS] = {};

    for (uint i = threadIdx.x; i < vectors; i += blockDim.x) {

        float4 w_raw = reinterpret_cast<const float4*>(weights)[i];
        float4 x_raw = reinterpret_cast<const float4*>(d_input)[i];

        const __half* w = reinterpret_cast<const __half*>(&w_raw);
        const __half* x = reinterpret_cast<const __half*>(&x_raw);

        #pragma unroll
        for (uint j = 0; j < VECTOR_ELEMENTS; ++j){
            accumulators[j] += __half2float(w[j]) * __half2float(x[j]);
        }
    }
    #pragma unroll
    for (uint j = 0; j < VECTOR_ELEMENTS; ++j) sum += accumulators[j];
    for (uint col = vectors * VECTOR_ELEMENTS + threadIdx.x; col < in_dim; col += blockDim.x){
            sum += __half2float(weights[col]) * __half2float(d_input[col]);
    }

    for(uint offset = warpSize >> 1; offset; offset >>= 1){
        sum += __shfl_down(sum, offset);
    }

    // WARP_SIZE is the minimum supported wave size, so this also fits wave64.
    __shared__ float wave_sums[BLOCK_SIZE / WARP_SIZE];
    const uint lane = threadIdx.x % warpSize;
    if (lane == 0) wave_sums[threadIdx.x / warpSize] = sum;
    __syncthreads();

    // Only the first wave needs the completed result, so no second barrier.
    if (threadIdx.x < warpSize) {
        sum = lane < blockDim.x / warpSize ? wave_sums[lane] : 0.0f;

        for (uint offset = warpSize / 2; offset; offset >>= 1){
            sum += __shfl_down(sum, offset);
        }

        if (lane == 0) {
            if (beta != 0.0f) sum += beta * __half2float(d_output[row]);
            d_output[row] = __float2half(sum);
        }
    }
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
    
    matrix_vec_mul_tuned<<<grid, blocks, 0, stream>>>(
        d_weight, d_input, d_output, out_dim, in_dim, beta
    );
}
