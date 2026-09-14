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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::println(stderr, "Usage: {} <path_to_model.bin>", argv[0]);
        return 1;
    }

    std::filesystem::path model_path = argv[1];
    
    std::println("=== Welcome to ROCm LLM Inference Engine ===");

    // Query AMD GPU Properties
    int device_id = 0;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    std::println("[GPU] Device Name: {}", props.name);
    std::println("[GPU] Total VRAM: {:.1f} GB", static_cast<double>(props.totalGlobalMem) / (1024 * 1024 * 1024));

    // Load Model Configuration
    MemoryMappedFile file(model_path);

    Model model = Model(file);
    model.config.print_model_config();
    
    // 3. Test GPU Memory Allocation & Kernel Execution
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N, 0.0f);

    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

    // Launch HIP Kernel
    launch_vector_add(d_a, d_b, d_c, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

    std::cout << "[Test Kernel] Execution complete. Sample output [0]: " << h_c[0] << " (Expected: 3.0)" << std::endl;

    // Clean up GPU VRAM
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    return 0;
}
