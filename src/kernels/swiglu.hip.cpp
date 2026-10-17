#include "kernels/kernels.hpp"
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256
constexpr uint VECTOR_SIZE = sizeof(float4) / sizeof(__half);

__device__ inline float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ __launch_bounds__(BLOCK_SIZE) void swiglu_vectorized(
    const __half* __restrict__ d_gate_up,  // [seq_len, 2 * intermediate_size]
    __half* __restrict__ d_out,  // [seq_len, intermediate_size]
    const uint total_elements,// seq_len * intermediate_size
    const uint intermediate_size
){
    
    const uint global_id = blockIdx.x * blockDim.x + threadIdx.x; 
    if(global_id >= total_elements) return;


    const float4* __restrict__ gate_up_vec_raw = reinterpret_cast<const float4*>(d_gate_up);
    float4* __restrict__ out_vec_raw = reinterpret_cast<float4*>(d_out);


    // Get the ids
    // Since we are launching a grid with seq_len * intermediate_size we divide by intermediate size
    const uint token_id = global_id / intermediate_size;
    const uint col = global_id % intermediate_size; 

    const uint stride = 2 * intermediate_size; 
    const uint gate_idx = token_id * stride + col;  
    const uint up_idx = gate_idx + intermediate_size; 
    
    // get the data
    const float4 gate_val = gate_up_vec_raw[gate_idx]; 
    const float4 up_val = gate_up_vec_raw[up_idx];

    // Treat them as __half
    const __half* gate = reinterpret_cast<const __half*>(&gate_val);
    const __half* up = reinterpret_cast<const __half*>(&up_val);

    
    float4 out;
    __half* out_half = reinterpret_cast<__half*>(&out);
    
    #pragma unroll
    for(uint i = 0; i < 8; i++){
        const float g = __half2float(gate[i]);
        const float u = __half2float(up[i]);
        out_half[i] = __float2half(silu(g) * u);
    }

    // Write to VRAM (float4)
    out_vec_raw[global_id] = out; 
}


__global__ __launch_bounds__(BLOCK_SIZE) void swiglu(
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
    size_t intermediate_size,
    hipStream_t stream
){  
    const dim3 grid((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    const dim3 block(BLOCK_SIZE);
    
    swiglu<<<grid, block, 0, stream>>>(
        d_gate_up, d_out, static_cast<uint>(total_elements),
        static_cast<uint>(intermediate_size));
}


void launch_swiglu_vectorized(
    const __half* d_gate_up,   // [seq_len, 2 * intermediate_size]
    __half* d_out,          // [seq_len, intermediate_size]
    size_t seq_len, 
    size_t total_elements,   // seq_len * intermediate_size,
    size_t intermediate_size,
    hipStream_t stream
){
    const size_t total_vec_elements = total_elements / VECTOR_SIZE; 
    const size_t intermediate_size_vec = intermediate_size / VECTOR_SIZE;
    
    const dim3 grid((total_vec_elements + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    const dim3 block(BLOCK_SIZE);
    
    swiglu_vectorized<<<grid, block, 0, stream>>>(
        d_gate_up, d_out, static_cast<uint>(total_vec_elements),
        static_cast<uint>(intermediate_size_vec));
}
