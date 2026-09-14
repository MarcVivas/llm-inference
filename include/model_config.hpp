#pragma once
#include <cstdint>
#include <cstring>
#include <print>
#include <span>
#include <cstddef>
#include <stdexcept>

#pragma pack(push, 1)
struct ModelConfig {
    
    public:
        char magic[4];
        uint32_t vocab_size = 49152;    // Vocabulary size
        uint32_t hidden_size;
        uint32_t num_layers = 16;       // Number of Transformer blocks
        uint32_t num_heads = 32;        // Number of Query heads
        uint32_t num_kv_heads = 8;      // Number of Key/Value heads (GQA)
        uint32_t head_dim;
        uint32_t intermediate_size;
        
        float rms_norm_eps = 1e-6;
        float rope_theta = 1.0;

        ModelConfig() = default;

        
        ModelConfig(const std::span<const std::byte> bytes){
            this->import_model_config(bytes);
        }

        void import_model_config(const std::span<const std::byte> bytes){

            std::memcpy(this, bytes.data(), sizeof(ModelConfig));

            if(std::string_view(this->magic, 4) != "QWEN") {
                throw std::runtime_error("Invalid binary file: Magic number is not 'QWEN'");
            }
            
        }
        
        void print_model_config(){
            std::println("Architecture Hyperparameters:");
            std::println("Layers: {}, Hidden Size: {}, Vocab: {}", this->num_layers, this->hidden_size, this->vocab_size);
            std::println("Heads (Q/KV): {}/{}, Head Dim: {}, ", this->num_heads, this->num_kv_heads, this->head_dim);
            std::println("Intermediate Size (SwiGLU): {}", this->intermediate_size);
            std::println("Rope Theta: {}, RMSNorm Epsilon: {}", this->rope_theta, this->rms_norm_eps);
        }
};
#pragma pack(pop)

static_assert(sizeof(ModelConfig) == 40);
