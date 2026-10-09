#include "chat.hpp"
#include "gpu_utils.hpp"
#include "memory_mapped_file.hpp"
#include "model_config.hpp"
#include "model.hpp"
#include "model_inference.hpp"
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

    model.config.print_model_config();

    std::println("\n=== Initializing Inference Engine ===");
    ModelInference engine(model);
    Chat chat(model, engine);

    chat.start_interactive_session();
    
    return 0;
}
