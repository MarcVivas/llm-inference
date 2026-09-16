#include <hip/hip_runtime.h>
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"

__global__ void embedding_lookup_kernel(const int32_t* __restrict__ input_tokens, const __half* __restrict__ embed_table, __half* __restrict__ output_embeddings, size_t hidden_size) 
{
    
}

void launch_embedding_lookup(
    const int32_t* d_input_tokens, 
    const __half* d_embed_table, 
    __half* d_output_embeddings, 
    size_t seq_len, 
    size_t hidden_size
){
    
}