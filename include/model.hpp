#pragma once

#include "model_config.hpp"
#include "memory_mapped_file.hpp"
#include <filesystem>
#include <span>

struct Model {
    ModelConfig config;
    std::span<const float> weights; 

    Model(const MemoryMappedFile &file){
        const std::span<const std::byte> bytes = file.bytes();
        this->config = ModelConfig(bytes);
        
    }
};