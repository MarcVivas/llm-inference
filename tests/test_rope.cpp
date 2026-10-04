#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "rope_cache.hpp"
#include "kv_cache.hpp"
#include <algorithm>
#include <cstring>
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

    const size_t q_stride = num_heads * head_dim;    
    const size_t k_stride = num_kv_heads * head_dim; 
    
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
        q_stride,
        k_stride,
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

TEST_CASE("Kernel: RoPE and KV cache prefill") {
    const Model& model = get_test_model();
    const auto in_q = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_q_proj_out.bin");
    const auto in_k = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_k_proj_out.bin");
    const auto in_v = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_v_proj_out.bin");
    const auto rotated_q = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_q_rope.bin");
    const auto rotated_k = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_k_rope.bin");
    const size_t q_dim = model.config.num_heads * model.config.head_dim;
    const size_t kv_dim = model.config.num_kv_heads * model.config.head_dim;
    const size_t total_dim = q_dim + 2 * kv_dim;
    const size_t seq_len = in_q.size() / q_dim;
    REQUIRE(seq_len > 2);
    REQUIRE(in_k.size() == seq_len * kv_dim);
    REQUIRE(in_v.size() == in_k.size());

    // Small real KVCache with a second layer and trailing rows as write guards.
    ModelConfig config = model.config;
    config.num_layers = 2;
    config.max_seq_len = seq_len + 2;
    KVCache cache(config);
    const size_t cache_elements = config.max_seq_len * kv_dim;
    const std::vector<__half> sentinel(cache_elements, __float2half(-123.0f));

    for (bool interleaved : {false, true}) {
        for (bool rotate : {false, true}) {
            for (size_t start_pos : {size_t(0), size_t(2)}) {
                CAPTURE(interleaved);
                CAPTURE(rotate);
                CAPTURE(start_pos);
                // A nonzero start uses fixture tokens at their original positions,
                // so PyTorch's rotated fixture remains the expected result.
                const size_t count = seq_len - start_pos;
                std::vector<__half> source;
                size_t q_offset = 0, k_offset, v_offset, q_stride, kv_stride;
                if (interleaved) {
                    source.resize(count * total_dim);
                    k_offset = q_dim; v_offset = q_dim + kv_dim;
                    q_stride = kv_stride = total_dim;
                    for (size_t t = 0; t < count; ++t) {
                        std::copy_n(in_q.data() + (start_pos + t) * q_dim, q_dim, source.data() + t * total_dim);
                        std::copy_n(in_k.data() + (start_pos + t) * kv_dim, kv_dim, source.data() + t * total_dim + k_offset);
                        std::copy_n(in_v.data() + (start_pos + t) * kv_dim, kv_dim, source.data() + t * total_dim + v_offset);
                    }
                } else {
                    source.insert(source.end(), in_q.begin() + start_pos * q_dim, in_q.end());
                    k_offset = source.size();
                    source.insert(source.end(), in_k.begin() + start_pos * kv_dim, in_k.end());
                    v_offset = source.size();
                    source.insert(source.end(), in_v.begin() + start_pos * kv_dim, in_v.end());
                    q_stride = q_dim; kv_stride = kv_dim;
                }
                const auto original_source = source;
                __half* d_source = upload_to_device(source);
                for (size_t layer = 0; layer < config.num_layers; ++layer) {
                    HIP_CHECK(hipMemcpy(cache.get_k_cache(layer), sentinel.data(), sentinel.size() * sizeof(__half), hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemcpy(cache.get_v_cache(layer), sentinel.data(), sentinel.size() * sizeof(__half), hipMemcpyHostToDevice));
                }
                launch_rope_and_store_kv(d_source + q_offset, d_source + k_offset, d_source + v_offset,
                    cache.get_k_cache(1), cache.get_v_cache(1), model.rope_cache.d_cos, model.rope_cache.d_sin,
                    count, config.num_heads, config.num_kv_heads, config.head_dim,
                    q_stride, kv_stride, start_pos, rotate);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipDeviceSynchronize());

                auto expected_k = sentinel;
                auto expected_v = sentinel;
                const auto& k_reference = rotate ? rotated_k : in_k;
                std::copy(k_reference.begin() + start_pos * kv_dim, k_reference.end(), expected_k.begin() + start_pos * kv_dim);
                std::copy(in_v.begin() + start_pos * kv_dim, in_v.end(), expected_v.begin() + start_pos * kv_dim);
                const auto actual_k = download_from_device(cache.get_k_cache(1), cache_elements);
                const auto actual_v = download_from_device(cache.get_v_cache(1), cache_elements);
                CHECK(std::all_of(actual_k.begin(), actual_k.end(),
                    [](__half value) { return std::isfinite(__half2float(value)); }));
                test_utils::check_tensor_close(actual_k, expected_k,
                                               rotate ? 2e-2f : 0.0f, rotate ? 2e-2f : 0.0f);
                // V and all untouched cache rows must be bit-for-bit identical.
                CHECK(std::memcmp(actual_v.data(), expected_v.data(), actual_v.size() * sizeof(__half)) == 0);
                CHECK(std::memcmp(actual_k.data(), sentinel.data(), start_pos * kv_dim * sizeof(__half)) == 0);
                CHECK(std::memcmp(actual_k.data() + seq_len * kv_dim, sentinel.data() + seq_len * kv_dim,
                                  (cache_elements - seq_len * kv_dim) * sizeof(__half)) == 0);
                for (const __half* ptr : {cache.get_k_cache(0), cache.get_v_cache(0)}) {
                    const auto untouched = download_from_device(ptr, cache_elements);
                    CHECK(std::memcmp(untouched.data(), sentinel.data(), untouched.size() * sizeof(__half)) == 0);
                }

                const auto actual_source = download_from_device(d_source, source.size());
                std::vector<__half> actual_q(count * q_dim);
                for (size_t t = 0; t < count; ++t) {
                    std::copy_n(actual_source.data() + t * q_stride, q_dim, actual_q.data() + t * q_dim);
                    CHECK(std::memcmp(actual_source.data() + k_offset + t * kv_stride,
                                      original_source.data() + k_offset + t * kv_stride, kv_dim * sizeof(__half)) == 0);
                    CHECK(std::memcmp(actual_source.data() + v_offset + t * kv_stride,
                                      original_source.data() + v_offset + t * kv_stride, kv_dim * sizeof(__half)) == 0);
                }
                const auto& q_reference = rotate ? rotated_q : in_q;
                CHECK(std::all_of(actual_q.begin(), actual_q.end(),
                    [](__half value) { return std::isfinite(__half2float(value)); }));
                test_utils::check_tensor_close(actual_q, std::span<const __half>(q_reference).subspan(start_pos * q_dim),
                                               rotate ? 2e-2f : 0.0f, rotate ? 2e-2f : 0.0f);
                HIP_CHECK(hipFree(d_source));
            }
        }
    }
}
