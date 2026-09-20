#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "kernels/kernels.hpp"

__global__ void embedding_lookup_kernel(const int32_t* __restrict__ input_tokens, const __half* __restrict__ embed_table, __half* __restrict__ output_embeddings, const uint hidden_size, const uint total_threads) 
{
    const uint global_id = blockIdx.x * blockDim.x + threadIdx.x; 

    // Check out of bounds
    if(global_id >= total_threads) return;
    
    const uint token_idx = global_id / hidden_size;

    // Read the token id
    const int32_t token = input_tokens[token_idx];
    const uint offset = global_id % hidden_size;
    const uint index = token * hidden_size + offset;
    
    // Store the value to VRAM
    output_embeddings[global_id] = embed_table[index]; 
}

void launch_embedding_lookup(
    const int32_t* d_input_tokens, 
    const __half* d_embed_table, 
    __half* d_output_embeddings, 
    const size_t seq_len, 
    const size_t hidden_size
){

    constexpr size_t BLOCK_SIZE = 128;
    const size_t total_threads = seq_len * hidden_size;
    const dim3 grid((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
    constexpr dim3 block_dim(BLOCK_SIZE);
    
    embedding_lookup_kernel<<<grid, block_dim>>>(d_input_tokens, d_embed_table, d_output_embeddings, hidden_size, total_threads);

}