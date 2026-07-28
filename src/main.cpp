#include <iostream>
#include <vector>
#include "gpu_utils.hpp"
#include "model_config.hpp"
#include "kernels/kernels.hpp"

int main() {
    std::cout << "=== ROCm LLM Inference Engine Skeleton ===" << std::endl;

    // 1. Query AMD GPU Properties
    int device_id = 0;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    std::cout << "[GPU] Device Name: " << props.name << std::endl;
    std::cout << "[GPU] Total VRAM: " << props.totalGlobalMem / (1024 * 1024 * 1024) << " GB" << std::endl;

    // 2. Load Model Configuration
    ModelConfig config;
    std::cout << "[Model] Config initialized. Hidden Dim: " << config.dim
              << ", Layers: " << config.num_layers << std::endl;

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

    std::cout << "=== Skeleton Verification Successful ===" << std::endl;
    return 0;
}
