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
            const size_t stride = interleaved ? q_dim + 2 * kv_dim : q_dim;
            std::vector<__half> input(stride, __float2half(-123.f));
            std::copy_n(q.data() + position * q_dim, q_dim, input.data());
            __half *d_q = nullptr, *d_output = nullptr;
            float *d_partial = nullptr, *d_max = nullptr, *d_sum = nullptr;
            const size_t chunks = (position + 1 + KV_CHUNK - 1) / KV_CHUNK;
            const size_t states = model.config.num_heads * chunks;
            HIP_CHECK(hipMalloc(&d_q, input.size() * sizeof(__half)));
            HIP_CHECK(hipMalloc(&d_output, 3 * q_dim * sizeof(__half)));
            HIP_CHECK(hipMalloc(&d_partial, states * model.config.head_dim * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_max, states * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_sum, states * sizeof(float)));
            HIP_CHECK(hipMemcpy(d_q, input.data(), input.size() * sizeof(__half), hipMemcpyHostToDevice));
            std::vector<__half> output(3 * q_dim, __float2half(-123.f));
            const auto guards = output;
            HIP_CHECK(hipMemcpy(d_output, output.data(), output.size() * sizeof(__half), hipMemcpyHostToDevice));
            CAPTURE(interleaved);
            CAPTURE(position);
            launch_causal_attention_decode(d_q, cache.get_k_cache(0), cache.get_v_cache(0),
                d_output + q_dim, d_partial, d_max, d_sum,
                1, position + 1, position,
                model.config.num_heads, model.config.num_kv_heads, model.config.head_dim,
                interleaved ? stride : 0, 0);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(output.data(), d_output, output.size() * sizeof(__half), hipMemcpyDeviceToHost));
            const auto actual = std::span<const __half>(output).subspan(q_dim, q_dim);
            CHECK(std::all_of(actual.begin(), actual.end(),
                [](__half x) { return std::isfinite(__half2float(x)); }));
            test_utils::check_tensor_close(actual,
                std::span<const __half>(expected).subspan(position * q_dim, q_dim), 2e-2f, 2e-2f);
            CHECK(std::memcmp(output.data(), guards.data(), q_dim * sizeof(__half)) == 0);
            CHECK(std::memcmp(output.data() + 2 * q_dim, guards.data() + 2 * q_dim, q_dim * sizeof(__half)) == 0);
            HIP_CHECK(hipFree(d_q));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_partial));
            HIP_CHECK(hipFree(d_max));
            HIP_CHECK(hipFree(d_sum));
        }
    }
}

TEST_CASE("Kernel: Flash decode vs independent CPU softmax") {
    // Boundary lengths exercise both full chunks and a partial final chunk.
    const size_t lengths[] = {1, KV_CHUNK, KV_CHUNK + 1, 3 * KV_CHUNK + 1, 257};
    for (size_t head_dim : {size_t(32), size_t(128)}) {
        for (size_t kv_heads : {size_t(1), size_t(2), size_t(4)}) {
            const size_t heads = 4, q_dim = heads * head_dim, kv_dim = kv_heads * head_dim;
            for (size_t len : lengths) {
                for (bool padded : {false, true}) {
                    CAPTURE(head_dim);
                    CAPTURE(kv_heads);
                    CAPTURE(len);
                    CAPTURE(padded);
                    const size_t stride = kv_dim + (padded ? 17 : 0);
                    std::vector<__half> q(q_dim), k((len + 2) * stride, __float2half(123.f));
                    auto v = k;
                    for (size_t i = 0; i < q.size(); ++i)
                        q[i] = __float2half(2.f * std::sin(float(i + 1) * .37f));
                    for (size_t t = 0; t < len; ++t)
                        for (size_t i = 0; i < kv_dim; ++i) {
                            // Different chunks have different maxima: merging
                            // unscaled chunk outputs must fail this reference.
                            k[t * stride + i] = __float2half(3.f * std::cos(float(t * kv_dim + i + 1) * .13f));
                            v[t * stride + i] = __float2half(std::sin(float(t * kv_dim + i + 3) * .19f));
                        }
                    std::vector<__half> expected(q_dim);
                    for (size_t h = 0; h < heads; ++h) {
                        const size_t kh = h / (heads / kv_heads);
                        std::vector<double> scores(len);
                        double maximum = -INFINITY, denominator = 0.;
                        for (size_t t = 0; t < len; ++t) {
                            double dot = 0.;
                            for (size_t c = 0; c < head_dim; ++c)
                                dot += double(__half2float(q[h * head_dim + c])) *
                                    __half2float(k[t * stride + kh * head_dim + c]);
                            scores[t] = dot / std::sqrt(double(head_dim));
                            maximum = std::max(maximum, scores[t]);
                        }
                        for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
                        for (size_t c = 0; c < head_dim; ++c) {
                            double numerator = 0.;
                            for (size_t t = 0; t < len; ++t)
                                numerator += scores[t] * __half2float(v[t * stride + kh * head_dim + c]);
                            expected[h * head_dim + c] = __float2half(float(numerator / denominator));
                        }
                    }
                    __half *dq = nullptr, *dk = nullptr, *dv = nullptr, *out = nullptr;
                    float *partial = nullptr, *maximum = nullptr, *sum = nullptr;
                    const size_t states = heads * ((len + KV_CHUNK - 1) / KV_CHUNK);
                    HIP_CHECK(hipMalloc(&dq, q.size() * sizeof(__half)));
                    HIP_CHECK(hipMalloc(&dk, k.size() * sizeof(__half)));
                    HIP_CHECK(hipMalloc(&dv, v.size() * sizeof(__half)));
                    HIP_CHECK(hipMalloc(&out, (q_dim + 2) * sizeof(__half)));
                    HIP_CHECK(hipMalloc(&partial, states * head_dim * sizeof(float)));
                    HIP_CHECK(hipMalloc(&maximum, states * sizeof(float)));
                    HIP_CHECK(hipMalloc(&sum, states * sizeof(float)));
                    HIP_CHECK(hipMemcpy(dq, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemcpy(dk, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemcpy(dv, v.data(), v.size() * sizeof(__half), hipMemcpyHostToDevice));
                    std::vector<__half> actual(q_dim + 2, __float2half(-123.f));
                    HIP_CHECK(hipMemcpy(out, actual.data(), actual.size() * sizeof(__half), hipMemcpyHostToDevice));
                    launch_causal_attention_decode(dq, dk, dv, out + 1, partial, maximum, sum,
                        1, len, len - 1, heads, kv_heads, head_dim, 0, padded ? stride : 0);
                    HIP_CHECK(hipGetLastError());
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(actual.data(), out, actual.size() * sizeof(__half), hipMemcpyDeviceToHost));
                    CHECK(__half2float(actual.front()) == -123.f);
                    CHECK(__half2float(actual.back()) == -123.f);
                    auto result = std::span<const __half>(actual).subspan(1, q_dim);
                    CHECK(std::all_of(result.begin(), result.end(), [](__half x) { return std::isfinite(__half2float(x)); }));
                    test_utils::check_tensor_close(result, expected, 2e-3f, 2e-3f);
                    for (auto ptr : {dq, dk, dv, out}) HIP_CHECK(hipFree(ptr));
                    for (auto ptr : {partial, maximum, sum}) HIP_CHECK(hipFree(ptr));
                }
            }
        }
    }
}
