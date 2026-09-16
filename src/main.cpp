#include <cstdio>
#include <iostream>
#include <vector>
#include "gpu_utils.hpp"
#include "memory_mapped_file.hpp"
#include "model_config.hpp"
#include "model.hpp"
#include "kernels/kernels.hpp"
#include <print>
#include <filesystem>

void print_gpu_properties(){
    int device_id = 0;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    std::println("[GPU] Device Name: {}", props.name);
    std::println("[GPU] Total VRAM: {:.1f} GB", static_cast<double>(props.totalGlobalMem) / (1024 * 1024 * 1024));
}

Model load_model_from_file(const std::string& filepath){
    std::println("=== Loading Model from {} ===", filepath);
    MemoryMappedFile file(filepath);
    Model model(file);

    std::println("\n[GPU] Weights uploaded successfully ({} MB)", 
                 model.device_weights.total_bytes / (1024 * 1024));
    model.config.print_model_config();

    return model;
    
}


std::string format_chat_prompt(const std::string& user_prompt) {
    return "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

std::vector<int> tokenize_and_print(tokenizers::Tokenizer& tokenizer, const std::string& text) {
    std::vector<int> tokens = tokenizer.Encode(text);

    std::println("\n=== Tokenizer Verification ===");
    std::println("Token Count: {}", tokens.size());
    std::print("Token IDs: [");
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::print("{}{}", tokens[i], (i + 1 < tokens.size()) ? ", " : "");
    }
    std::println("]");

    return tokens;
}

// Inspect individual decoded token pieces
void print_token_breakdown(tokenizers::Tokenizer& tokenizer, const std::vector<int>& tokens) {
    std::println("\n=== Decoding Breakdown ===");
    for (int id : tokens) {
        std::string piece = tokenizer.Decode({id});
        std::println("Token {:>6} -> \"{}\"", id, piece);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::println(stderr, "Usage: {} <path_to_model.bin>", argv[0]);
        return 1;
    }

    std::filesystem::path model_path = argv[1];
    
    std::println("=== Welcome to ROCm LLM Inference Engine ===");

    // Print AMD GPU Properties
    print_gpu_properties();

    // Load Model Configuration
    Model model = load_model_from_file(model_path);
    std::string user_prompt = "Hello! Tell me something about ROCm.";
    std::string formatted = format_chat_prompt(user_prompt);

    std::vector<int> input_ids = tokenize_and_print(*model.tokenizer, formatted);
    print_token_breakdown(*model.tokenizer, input_ids);

    return 0;
}
