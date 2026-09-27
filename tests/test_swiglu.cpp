#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>
#include <vector>

const Model& get_test_model();

TEST_CASE("Kernel: SwiGLU Activation") {
    const Model& model = get_test_model();

    // Load fixtures
    auto in_norm    = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_post_attn_norm_out.bin");
    auto exp_swiglu = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_swiglu_intermediate.bin");

    const size_t intermediate_size = model.config.intermediate_size;
    const size_t seq_len = exp_swiglu.size() / intermediate_size;
    const size_t total_elements = seq_len * intermediate_size;

    // Allocate GPU buffers
    __half* d_in_norm     = nullptr;
    __half* d_gate_up_out = nullptr; // [seq_len, 2 * intermediate_size]
    __half* d_swiglu_out  = nullptr; // [seq_len, intermediate_size]
   
    HIP_CHECK(hipMalloc(&d_in_norm, in_norm.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_gate_up_out, seq_len * 2 * intermediate_size * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_swiglu_out, seq_len * intermediate_size * sizeof(__half)));
   
    HIP_CHECK(hipMemcpy(d_in_norm, in_norm.data(), in_norm.size() * sizeof(__half), hipMemcpyHostToDevice));

    const __half* d_w_gate_up = model.device_weights.transformer_blocks[0].gate_proj;
    launch_gate_up_gemm(
        d_in_norm,
        d_gate_up_out,
        d_w_gate_up,
        seq_len,
        model.config.hidden_size,
        intermediate_size
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Launch SwiGLU Kernel (Under Test)
    launch_swiglu(
        d_gate_up_out,
        d_swiglu_out,
        seq_len,
        total_elements,
        intermediate_size
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Download and verify
    std::vector<__half> actual_swiglu(total_elements);
    HIP_CHECK(hipMemcpy(actual_swiglu.data(), d_swiglu_out, total_elements * sizeof(__half), hipMemcpyDeviceToHost));

    INFO("Verifying SwiGLU intermediate output...");
    test_utils::check_tensor_close(actual_swiglu, exp_swiglu, /*atol=*/2e-2f, /*rtol=*/2e-2f);

    // Cleanup
    HIP_CHECK(hipFree(d_in_norm));
    HIP_CHECK(hipFree(d_gate_up_out));
    HIP_CHECK(hipFree(d_swiglu_out));
}