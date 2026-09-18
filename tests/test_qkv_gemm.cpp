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

TEST_CASE("Kernel: Fused QKV GEMM") {
    const Model& model = get_test_model();

    // Load ground truth fixtures
    auto input_norm = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_rms_norm_out.bin");
    auto exp_q = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_q_proj_out.bin");
    auto exp_k = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_k_proj_out.bin");
    auto exp_v = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_v_proj_out.bin");

    const size_t total_elements = input_norm.size();
    const size_t hidden_size = model.config.hidden_size;
    const size_t seq_len = total_elements / hidden_size;

    const size_t q_dim = model.config.num_heads * model.config.head_dim;
    const size_t kv_dim = model.config.num_kv_heads * model.config.head_dim; 
    const size_t total_qkv_dim = q_dim + 2 * kv_dim;

    // Prepare GPU input and output buffers
    __half* d_input_norm = allocate_and_upload(input_norm);
    __half* d_qkv_out = nullptr;
    HIP_CHECK(hipMalloc(&d_qkv_out, seq_len * total_qkv_dim * sizeof(__half)));

    rocblas_handle handle = get_rocblas_handle();

    // Launch the Kernel
    launch_qkv_gemm(
        d_input_norm, 
        d_qkv_out, 
        model.device_weights.transformer_blocks[0].q_proj, // Gamma
        seq_len, 
        hidden_size,
        total_qkv_dim,
        handle
    );

    // Download result and verify
    std::vector<__half> actual_out = download_gpu_tensor(d_qkv_out, seq_len * total_qkv_dim);

    // Separate the output in 3 matrices. 
    std::vector<__half> act_q(seq_len * q_dim);
    std::vector<__half> act_k(seq_len * kv_dim);
    std::vector<__half> act_v(seq_len * kv_dim);

    for (size_t s = 0; s < seq_len; ++s) {
        const __half* row = actual_out.data() + (s * total_qkv_dim);
        std::memcpy(act_q.data() + (s * q_dim),  row,                       q_dim * sizeof(__half));
        std::memcpy(act_k.data() + (s * kv_dim), row + q_dim,              kv_dim * sizeof(__half));
        std::memcpy(act_v.data() + (s * kv_dim), row + q_dim + kv_dim,     kv_dim * sizeof(__half));
    }

    INFO("Verifying Q projection slice...");
    test_utils::check_tensor_close(act_q, exp_q);

    INFO("Verifying K projection slice...");
    test_utils::check_tensor_close(act_k, exp_k);
    
    INFO("Verifying V projection slice...");
    test_utils::check_tensor_close(act_v, exp_v);

    // Cleanup
    HIP_CHECK(hipFree(d_input_norm));
    HIP_CHECK(hipFree(d_qkv_out));
}