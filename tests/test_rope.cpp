#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "rope_cache.hpp"
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>
#include <vector>

const Model& get_test_model();

// Helper: allocate and upload tensor to GPU
static __half* upload_to_device(std::span<const __half> host_data) {
    __half* d_ptr = nullptr;
    HIP_CHECK(hipMalloc(&d_ptr, host_data.size_bytes()));
    HIP_CHECK(hipMemcpy(d_ptr, host_data.data(), host_data.size_bytes(), hipMemcpyHostToDevice));
    return d_ptr;
}

// Helper: download tensor from GPU to host
static std::vector<__half> download_from_device(const __half* d_ptr, size_t num_elements) {
    std::vector<__half> host_data(num_elements);
    HIP_CHECK(hipMemcpy(host_data.data(), d_ptr, num_elements * sizeof(__half), hipMemcpyDeviceToHost));
    return host_data;
}

TEST_CASE("Kernel: RoPE (Rotary Position Embeddings)") {
    const Model& model = get_test_model();

    // Load fixtures
    // Input: unrotated Q and K from GEMM
    auto in_q  = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_q_proj_out.bin");
    auto in_k  = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_k_proj_out.bin");
    
    // Expected: rotated Q and K from PyTorch
    auto exp_q = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_q_rope.bin");
    auto exp_k = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_k_rope.bin");

    const size_t head_dim     = model.config.head_dim;
    const size_t num_heads    = model.config.num_heads;
    const size_t num_kv_heads = model.config.num_kv_heads;
    const size_t seq_len      = in_q.size() / (num_heads * head_dim);

    // Prepare RoPE Precomputed Cache
    RopeCache rope_cache = create_rope_cache(
        model.config.max_seq_len, 
        head_dim, 
        model.config.rope_theta
    );

    // Prepare GPU buffers (RoPE is applied in-place!)
    __half* d_q = upload_to_device(in_q);
    __half* d_k = upload_to_device(in_k);

    // Launch RoPE Kernel (Under Test)
    launch_rope(
        d_q,
        d_k,
        rope_cache.d_cos,
        rope_cache.d_sin,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        /*start_pos=*/0
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Download and verify
    std::vector<__half> actual_q = download_from_device(d_q, in_q.size());
    std::vector<__half> actual_k = download_from_device(d_k, in_k.size());

    INFO("Verifying Rotated Q...");
    test_utils::check_tensor_close(actual_q, exp_q, /*atol=*/2e-2f, /*rtol=*/2e-2f);

    INFO("Verifying Rotated K...");
    test_utils::check_tensor_close(actual_k, exp_k, /*atol=*/2e-2f, /*rtol=*/2e-2f);

    // Cleanup
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_k));
}