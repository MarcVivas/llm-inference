#pragma once
#include "device_weights.hpp"
#include "model_config.hpp"
#include "memory_mapped_file.hpp"
#include "ping_pong_buffer.hpp"
#include <span>
#include<hip/hip_fp16.h>

struct Model {
    ModelConfig config;
    DeviceWeights device_weights;
    PingPongBuffer<__half> activations;
    // KVCache kv_cache; 
    
    Model(const MemoryMappedFile &file){

        // Get bytes of the file
        const std::span<const std::byte> bytes = file.bytes();

        // Get model config from the bytes
        this->config = ModelConfig(bytes);

        // Get the model weights from the bytes
        const size_t weights_bytes = bytes.size() - sizeof(ModelConfig);
        const __half* host_weights = reinterpret_cast<const __half*>(bytes.data() + sizeof(ModelConfig));

        // Transfer the model weights to GPU memory
        this->device_weights = DeviceWeights(config, host_weights, weights_bytes);
        this->activations = PingPongBuffer<__half>(Model::calculate_activation_size(config, config.max_seq_len));
    }

    void run_inference(const __half* prompt_embeddings, const size_t seq_len){
        
    }

    private:

        // Helper to calculate the maximum buffer size needed per layer
        static size_t calculate_activation_size(const ModelConfig &config, const size_t max_seq_len){
            const size_t max_dim = std::max(config.hidden_size, config.intermediate_size);
            return config.max_seq_len * max_dim;
        }
        
    
};