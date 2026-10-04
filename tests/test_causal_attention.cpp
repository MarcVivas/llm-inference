#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "kv_cache.hpp"
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
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
    launch_causal_attention_prefill(
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

TEST_CASE("Kernel: Decode attention vs reference prefill rows") {
    const Model& model = get_test_model();
    const auto q = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_q_rope.bin");
    const auto k = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_k_rope.bin");
    const auto v = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_v_proj_out.bin");
    const auto expected = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_attn_out.bin");
    const size_t q_dim = model.config.num_heads * model.config.head_dim;
    const size_t kv_dim = model.config.num_kv_heads * model.config.head_dim;
    const size_t seq_len = q.size() / q_dim;
    REQUIRE(seq_len > 4);
    REQUIRE(q.size() == seq_len * q_dim);
    REQUIRE(k.size() == seq_len * kv_dim);
    REQUIRE(v.size() == k.size());
    REQUIRE(expected.size() == q.size());

    ModelConfig cache_config = model.config;
    cache_config.num_layers = 1;
    cache_config.max_seq_len = seq_len + 2;
    KVCache cache(cache_config);
    // Future/unused cache rows are populated: incorrect bounds or masking must
    // change the output rather than accidentally reading convenient zeros.
    std::vector<__half> cache_k(cache_config.max_seq_len * kv_dim, __float2half(123.f));
    auto cache_v = cache_k;
    std::copy(k.begin(), k.end(), cache_k.begin());
    std::copy(v.begin(), v.end(), cache_v.begin());
    HIP_CHECK(hipMemcpy(cache.get_k_cache(0), cache_k.data(), cache_k.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(cache.get_v_cache(0), cache_v.data(), cache_v.size() * sizeof(__half), hipMemcpyHostToDevice));

    for (bool interleaved : {false, true}) {
        for (size_t position = 0; position < seq_len; ++position) {
            // Check each single-token query, then a query chunk spanning several
            // tiles to exercise Q stride, local output indexes, and offset masks.
            for (bool chunk : {false, true}) {
                if (chunk && position != 2) continue;
                const size_t query_len = chunk ? seq_len - position : 1;
                const size_t stride = interleaved ? q_dim + 2 * kv_dim : q_dim;
                std::vector<__half> input(query_len * stride, __float2half(-123.f));
                for (size_t row = 0; row < query_len; ++row)
                    std::copy_n(q.data() + (position + row) * q_dim, q_dim, input.data() + row * stride);
                __half* d_q = nullptr;
                __half* d_output = nullptr;
                HIP_CHECK(hipMalloc(&d_q, input.size() * sizeof(__half)));
                HIP_CHECK(hipMemcpy(d_q, input.data(), input.size() * sizeof(__half), hipMemcpyHostToDevice));
                // Guard output on both sides to detect indexing by absolute
                // query position or writes to padded rows of the query tile.
                std::vector<__half> output((query_len + 2) * q_dim, __float2half(-123.f));
                HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(__half)));
                const auto guards = output;

                for (bool full_cache : {false, true}) {
                    const size_t kv_len = full_cache ? seq_len : position + query_len;
                    CAPTURE(interleaved);
                    CAPTURE(position);
                    CAPTURE(query_len);
                    CAPTURE(kv_len);
                    HIP_CHECK(hipMemcpy(d_output, guards.data(), guards.size() * sizeof(__half), hipMemcpyHostToDevice));
                    launch_causal_attention_decode(d_q, cache.get_k_cache(0), cache.get_v_cache(0),
                        d_output + q_dim, query_len, kv_len, position,
                        model.config.num_heads, model.config.num_kv_heads, model.config.head_dim,
                        interleaved ? stride : 0, /*packed KV stride=*/0);
                    HIP_CHECK(hipGetLastError());
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(output.data(), d_output, output.size() * sizeof(__half), hipMemcpyDeviceToHost));
                    const auto actual = std::span<const __half>(output).subspan(q_dim, query_len * q_dim);
                    CHECK(std::all_of(actual.begin(), actual.end(),
                        [](__half x) { return std::isfinite(__half2float(x)); }));
                    test_utils::check_tensor_close(actual,
                        std::span<const __half>(expected).subspan(position * q_dim, query_len * q_dim), 2e-2f, 2e-2f);
                    CHECK(std::memcmp(output.data(), guards.data(), q_dim * sizeof(__half)) == 0);
                    CHECK(std::memcmp(output.data() + (query_len + 1) * q_dim,
                                      guards.data() + (query_len + 1) * q_dim, q_dim * sizeof(__half)) == 0);
                }
                HIP_CHECK(hipFree(d_q));
                HIP_CHECK(hipFree(d_output));
            }
        }
    }
}
