#include "kernels/kernels.hpp"
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 128

__device__ inline float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void swiglu(
    const __half* __restrict__ d_gate_up,  // [seq_len, 2 * intermediate_size]
    __half* __restrict__ d_out,  // [seq_len, intermediate_size]
    const uint total_elements,// seq_len * intermediate_size
    const uint intermediate_size
){
    const uint global_id = blockIdx.x * blockDim.x + threadIdx.x; 
    if(global_id >= total_elements) return;

    // Get the ids
    // Since we are launching a grid with seq_len * intermediate_size we divide by intermediate size
    const uint token_id = global_id / intermediate_size;
    const uint col = global_id % intermediate_size; 

    const uint stride = 2 * intermediate_size; 
    const __half* __restrict__ gate = d_gate_up + (token_id * stride + col);  
    const __half* __restrict__ up = gate + intermediate_size; 
    
    // get the data
    const float gate_val = __half2float(gate[0]); 
    const float up_val = __half2float(up[0]);

    d_out[global_id] = __float2half(silu(gate_val) * up_val);
}

void launch_swiglu(
    const __half* d_gate_up,   // [seq_len, 2 * intermediate_size]
    __half* d_out,          // [seq_len, intermediate_size]
    size_t seq_len, 
    size_t total_elements,   // seq_len * intermediate_size,
    size_t intermediate_size
){
    const dim3 grid((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    const dim3 block(BLOCK_SIZE);
    swiglu<<<grid, block>>>(d_gate_up, d_out, total_elements, intermediate_size);
}