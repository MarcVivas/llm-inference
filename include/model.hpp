#pragma once
#include "device_weights.hpp"
#include "model_config.hpp"
#include "memory_mapped_file.hpp"
#include "ping_pong_buffer.hpp"
#include <span>
#include<hip/hip_fp16.h>
#include "tokenizers_cpp.h"

struct Model {
    ModelConfig config;
    DeviceWeights device_weights;
    PingPongBuffer<__half> activations;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;

    // KVCache kv_cache; 
    
    Model(const MemoryMappedFile &file){

        // Get bytes of the file
        const std::span<const std::byte> bytes = file.bytes();

        // Get model config from the bytes
        this->config = parse_model_config(bytes);
        
        size_t weights_offset = 0;
        this->tokenizer = parse_and_init_tokenizer(bytes, weights_offset);
        
        this->device_weights = upload_model_weights(bytes, weights_offset, this->config);
        this->activations = PingPongBuffer<__half>(Model::calculate_activation_size(config, config.max_seq_len));
    }

    void run_inference(const __half* prompt_embeddings, const size_t seq_len){
        
    }

    private:
        // Reads and validates the 44-byte configuration header
        static ModelConfig parse_model_config(std::span<const std::byte> bytes) {
            return ModelConfig(bytes);
        }
    
        // Extracts tokenizer JSON from mmap and instantiates the Tokenizer
        static std::unique_ptr<tokenizers::Tokenizer> parse_and_init_tokenizer(
            std::span<const std::byte> bytes, 
            size_t& out_weights_offset
        ) {
            auto after_header = bytes.subspan(sizeof(ModelConfig));
            if (after_header.size() < sizeof(uint32_t)) {
                throw std::runtime_error("Corrupted model file: Missing tokenizer length field");
            }
    
            uint32_t tok_len = *reinterpret_cast<const uint32_t*>(after_header.data());
            auto tok_data_span = after_header.subspan(sizeof(uint32_t));
    
            if (tok_data_span.size() < tok_len) {
                throw std::runtime_error("Corrupted model file: Tokenizer payload truncated");
            }
    
            std::string_view tok_json_view(
                reinterpret_cast<const char*>(tok_data_span.data()),
                tok_len
            );
    
            // Compute 64-byte aligned boundary where weights begin
            out_weights_offset = compute_aligned_weights_offset(tok_len);
    
            return tokenizers::Tokenizer::FromBlobJSON(std::string(tok_json_view));
        }
    
        // Calculates the 64-byte aligned memory boundary for GPU VRAM
        static constexpr size_t compute_aligned_weights_offset(uint32_t tok_len) noexcept {
            size_t raw_offset = sizeof(ModelConfig) + sizeof(uint32_t) + tok_len;
            return (raw_offset + 63) & ~size_t(63);
        }
    
        // Slices host weights span and uploads them to the GPU
        static DeviceWeights upload_model_weights(
            std::span<const std::byte> bytes, 
            size_t weights_offset, 
            const ModelConfig& config
        ) {
            if (bytes.size() <= weights_offset) {
                throw std::runtime_error("Corrupted model file: Missing weight payload");
            }
    
            auto weights_span = bytes.subspan(weights_offset);
            const auto* host_weights = reinterpret_cast<const __half*>(weights_span.data());
            size_t total_weight_bytes = weights_span.size();
    
            return DeviceWeights(config, host_weights, total_weight_bytes);
        }
        
        // Helper to calculate the maximum buffer size needed per layer
        static size_t calculate_activation_size(const ModelConfig &config, const size_t max_seq_len){
            const size_t max_dim = std::max(config.hidden_size, config.intermediate_size);
            return config.max_seq_len * max_dim;
        }

        
        
    
};