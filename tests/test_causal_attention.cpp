#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "kv_cache.hpp"
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>
#include <vector>

const Model& get_test_model();

TEST_CASE("Kernel: Causal Self-Attention") {
    const Model& model = get_test_model();

    // Load fixtures
    auto in_q = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_q_rope.bin");
    auto in_k = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_k_rope.bin");
    auto in_v = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_v_proj_out.bin");
    auto exp_attn = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_attn_out.bin");

    const size_t head_dim = model.config.head_dim;       
    const size_t num_heads = model.config.num_heads;      
    const size_t num_kv_heads = model.config.num_kv_heads; 
    const size_t seq_len = in_q.size() / (num_heads * head_dim);
    const size_t q_dim = num_heads * head_dim;
    const size_t kv_dim = num_kv_heads * head_dim;

    // Cache capacity exceeds the valid sequence length; attention must only
    // consume the populated rows. Use a small cache for this fixture test.
    ModelConfig cache_config = model.config;
    cache_config.num_layers = 1;
    cache_config.max_seq_len = seq_len + 2;
    KVCache kv_cache(cache_config);

    // Prepare GPU buffers
    __half* d_q = nullptr;
    __half* d_out = nullptr;

    HIP_CHECK(hipMalloc(&d_q, in_q.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_out, exp_attn.size() * sizeof(__half)));

    HIP_CHECK(hipMemcpy(d_q, in_q.data(), in_q.size() * sizeof(__half), hipMemcpyHostToDevice));
    const std::vector<__half> unused_rows(2 * kv_dim, __float2half(123.0f));
    HIP_CHECK(hipMemcpy(kv_cache.get_k_cache(0), in_k.data(), in_k.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(kv_cache.get_v_cache(0), in_v.data(), in_v.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(kv_cache.get_k_cache(0) + seq_len * kv_dim, unused_rows.data(), unused_rows.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(kv_cache.get_v_cache(0) + seq_len * kv_dim, unused_rows.data(), unused_rows.size() * sizeof(__half), hipMemcpyHostToDevice));

    // Launch Attention Kernel (Under Test)
    launch_causal_attention(
        d_q,
        kv_cache.get_k_cache(0),
        kv_cache.get_v_cache(0),
        d_out,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        q_dim,
        kv_dim
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Download and verify
    std::vector<__half> actual_attn(exp_attn.size());
    HIP_CHECK(hipMemcpy(actual_attn.data(), d_out, exp_attn.size() * sizeof(__half), hipMemcpyDeviceToHost));

    INFO("Verifying Attention Output...");
    test_utils::check_tensor_close(actual_attn, exp_attn, /*atol=*/2e-2f, /*rtol=*/2e-2f);

    // Cleanup
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_out));
}
