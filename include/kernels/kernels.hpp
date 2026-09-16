#pragma once
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstddef>

void launch_embedding_lookup(
    const int32_t* d_input_tokens, 
    const __half* d_embed_table, 
    __half* d_output_embeddings, 
    size_t seq_len, 
    size_t hidden_size
);