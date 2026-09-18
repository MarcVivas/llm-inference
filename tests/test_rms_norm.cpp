#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>

// Forward declaration to access shared model
const Model& get_test_model();

// Helper: allocate and copy to GPU
static __half* allocate_and_upload(std::span<const __half> data) {
    __half* d_data = nullptr;
    HIP_CHECK(hipMalloc(&d_data, data.size_bytes()));
    HIP_CHECK(hipMemcpy(d_data, data.data(), data.size_bytes(), hipMemcpyHostToDevice));
    return d_data;
}

// Helper: copy GPU result back to host
static std::vector<__half> download_gpu_tensor(const __half* d_tensor, size_t num_elements) {
    std::vector<__half> host_data(num_elements);
    HIP_CHECK(hipMemcpy(host_data.data(), d_tensor, num_elements * sizeof(__half), hipMemcpyDeviceToHost));
    return host_data;
}

TEST_CASE("Kernel: RMSnorm") {
    const Model& model = get_test_model();

    // Load ground truth fixtures
    auto input_embeddings = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_input_embeddings.bin");
    auto expected_out = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_rms_norm_out.bin");

    const size_t total_elements = input_embeddings.size();
    const size_t hidden_size = model.config.hidden_size;
    const size_t seq_len = total_elements / hidden_size;
    const float eps = model.config.rms_norm_eps;

    // Prepare GPU input and output buffers
    __half* d_in = allocate_and_upload(input_embeddings);
    __half* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, total_elements * sizeof(__half)));


    hipDeviceProp_t props{};
    int device = 0;
    
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    
    int warp_size = props.warpSize;

    // Launch the Kernel
    launch_rms_norm(
        d_in, 
        d_out, 
        model.device_weights.transformer_blocks[0].input_normalization, // Gamma
        seq_len, 
        hidden_size,
        eps,
        warp_size
    );

    // Download result and verify
    std::vector<__half> actual_out = download_gpu_tensor(d_out, total_elements);
    test_utils::check_tensor_close(actual_out, expected_out);

    // Cleanup
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
}