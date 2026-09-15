#pragma once

#include "hip/driver_types.h"
#include "hip/hip_runtime.h"
#include "model_config.hpp"
#include <hip/hip_fp16.h>
#include <stdexcept>
#include <vector>

struct TransformerBlock {
    // RMSNorm [hidden_size]
    const __half *input_normalization = nullptr;
    // Query weights matrix [num_heads * head_dim, hidden_size]. What each token is looking for
    const __half *q_proj = nullptr;
    // Key weights matrix [num_kv_heads * head_dim, hidden_size]. What each token contains (a key)
    const __half *k_proj = nullptr;
    // Value weights matrix [num_kv_heads * head_dim, hidden_size]. Actual content to extract if the query matches the key. 
    const __half *v_proj = nullptr;

    // Scale weights for RMSNorm applied per head to Queries before dot products.
    const __half *q_norm = nullptr;
    // Scale weights for RMSNorm applied per head to Keys before dot products.
    const __half *k_norm = nullptr;

    // Output projection matrix [hidden_size, num_heads * head_dim]. Combines head results and projects them back into rhe main dimension. 
    const __half *o_proj = nullptr;

    // Post RMSNorm scale weights [hidden_size] 
    const __half* post_attn_norm = nullptr;
    // Projects input from hidden_size to intermediate_size.
    const __half* gate_proj = nullptr;
    // Projects input from hidden_size to intermediate_size
    const __half* up_proj = nullptr;
    // Projects intermediate size to hidden_size
    const __half* down_proj = nullptr;
};


struct DeviceWeights{
    // Store all the weights in a single device buffer
    __half* raw_device_buffer = nullptr;

    // Total bytes of the buffer
    size_t total_bytes = 0;

    // From token id to embedding vector [vocab_size, hidden_size]
    const __half* embedding_tokens = nullptr;

    // Num layers transformer blocks
    std::vector<TransformerBlock> transformer_blocks;

    // Final RMSNorm after the last transformer block [hidden size]
    const __half* final_norm = nullptr;

    // Unembedding layer [vocab_size, hidden_size]
    const __half* lm_head = nullptr;

    DeviceWeights(ModelConfig &config, const __half* host_weights, const size_t total_bytes){
        this->total_bytes = total_bytes;
        // Allocate 1 monolithic VRAM block
        auto err = hipMalloc(&this->raw_device_buffer, total_bytes);
        if(err != hipSuccess) {
            throw std::runtime_error(std::string("Failed to allocate GPU memory: ") + hipGetErrorString(err));
        }

        // Transfer the weights to the GPU buffer
        err = hipMemcpy(this->raw_device_buffer, host_weights, total_bytes, hipMemcpyHostToDevice);
        if(err != hipSuccess){
            throw std::runtime_error(std::string("Failed to copy weights to GPU: ") + hipGetErrorString(err));
        }

        const __half* cursor = this->raw_device_buffer;

        // Helper function to claim elements and advance the pointer. 
        auto claim = [&cursor](size_t num_elements) -> const __half* {
            const __half* ptr = cursor;
            cursor += num_elements;
            return ptr; 
        };

        // Embedding layer
        this->embedding_tokens = claim(config.vocab_size * config.hidden_size);

        // Transformer blocks
        this->transformer_blocks.resize(config.num_layers);

        const size_t q_dim = config.num_heads * config.head_dim;
        const size_t kv_dim = config.num_kv_heads * config.head_dim;

        for(size_t i = 0; i < config.num_layers; i++){
            TransformerBlock &block = this->transformer_blocks[i]; 

            block.input_normalization = claim(config.hidden_size);
            block.q_proj = claim(config.hidden_size * q_dim);
            block.k_proj = claim(config.hidden_size * kv_dim);
            block.v_proj = claim(config.hidden_size * kv_dim);

            block.o_proj = claim(q_dim * config.hidden_size);
            block.post_attn_norm = claim(config.hidden_size);

            block.gate_proj = claim(config.hidden_size * config.intermediate_size);
            block.up_proj = claim(config.hidden_size * config.intermediate_size);
            block.down_proj = claim(config.intermediate_size * config.hidden_size);
        }

        // Final normalization 
        this->final_norm = claim(config.hidden_size);

        // LM head
        this->lm_head = claim(config.vocab_size * config.hidden_size);
    }
    
    ~DeviceWeights() {
        if(raw_device_buffer){
            auto hip_error = hipFree(raw_device_buffer);
            raw_device_buffer = nullptr;
        }
    }

    // Move-only semantics so we don't accidentally double-free VRAM
    DeviceWeights() = default;
    DeviceWeights(const DeviceWeights&) = delete;
    // Move assignment operator
    DeviceWeights& operator=(const DeviceWeights&) = delete;
    DeviceWeights(DeviceWeights&& o) noexcept 
        : raw_device_buffer(o.raw_device_buffer), total_bytes(o.total_bytes),
          embedding_tokens(o.embedding_tokens), transformer_blocks(std::move(o.transformer_blocks)),
          final_norm(o.final_norm), lm_head(o.lm_head) {
        o.raw_device_buffer = nullptr;
    }

    // Move constructor
    DeviceWeights& operator=(DeviceWeights&& o) noexcept {
        if (this != &o) {
            // Free any memory this object currently holds
            if (raw_device_buffer) {
                auto hip_error = hipFree(raw_device_buffer);
            }

            // Transfer ownership from 'o'
            raw_device_buffer  = o.raw_device_buffer;
            total_bytes        = o.total_bytes;
            embedding_tokens   = o.embedding_tokens;
            transformer_blocks = std::move(o.transformer_blocks);
            final_norm         = o.final_norm;
            lm_head            = o.lm_head;

            // Reset the source object so its destructor doesn't free the memory
            o.raw_device_buffer = nullptr;
            o.total_bytes = 0;
        }
        return *this;
    }
    
};