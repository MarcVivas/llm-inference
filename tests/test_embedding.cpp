#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>

// Forward declaration to access shared model
const Model& get_test_model();

// Helper: allocate and copy tokens to GPU
static int32_t* allocate_and_upload_tokens(std::span<const int32_t> tokens) {
    int32_t* d_tokens = nullptr;
    HIP_CHECK(hipMalloc(&d_tokens, tokens.size_bytes()));
    HIP_CHECK(hipMemcpy(d_tokens, tokens.data(), tokens.size_bytes(), hipMemcpyHostToDevice));
    return d_tokens;
}

// Helper: copy GPU result back to host
static std::vector<__half> download_gpu_tensor(const __half* d_tensor, size_t num_elements) {
    std::vector<__half> host_data(num_elements);
    HIP_CHECK(hipMemcpy(host_data.data(), d_tensor, num_elements * sizeof(__half), hipMemcpyDeviceToHost));
    return host_data;
}

TEST_CASE("Kernel: Embedding Lookup") {
    const Model& model = get_test_model();

    // Load ground truth fixtures
    auto input_tokens = test_utils::load_binary_file<int32_t>("../../../../tests/reference/input_tokens.bin");
    auto expected_out = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_input_embeddings.bin");

    size_t seq_len = input_tokens.size();
    size_t hidden_size = model.config.hidden_size;
    size_t total_elements = seq_len * hidden_size;

    // Prepare GPU input and output buffers
    int32_t* d_tokens = allocate_and_upload_tokens(input_tokens);
    __half* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, total_elements * sizeof(__half)));

    // Launch the Kernel
    launch_embedding_lookup(
        d_tokens, 
        model.device_weights.embedding_tokens, 
        d_out, 
        seq_len, 
        hidden_size
    );

    // Download result and verify
    std::vector<__half> actual_out = download_gpu_tensor(d_out, total_elements);
    test_utils::check_tensor_close(actual_out, expected_out);

    // Cleanup
    HIP_CHECK(hipFree(d_tokens));
    HIP_CHECK(hipFree(d_out));
}