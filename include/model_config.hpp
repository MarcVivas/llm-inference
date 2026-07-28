#pragma once
#include <cstddef>

struct ModelConfig {
    size_t dim = 2048;            // Hidden dimension size
    size_t hidden_dim = 8192;     // MLP / SwiGLU intermediate size
    size_t num_layers = 16;       // Number of Transformer blocks
    size_t num_heads = 32;        // Number of Query heads
    size_t num_kv_heads = 8;      // Number of Key/Value heads (GQA)
    size_t vocab_size = 49152;    // Vocabulary size
    size_t max_seq_len = 2048;    // Maximum context length
};
