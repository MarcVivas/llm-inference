#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>
#include <vector>

const Model& get_test_model();

TEST_CASE("Kernel: Down projection") {
    const Model& model = get_test_model();

    // Load fixtures
    auto in_swiglu = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_swiglu_intermediate.bin");
    auto exp_mlp   = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_mlp_out.bin");

    const size_t intermediate_size = model.config.intermediate_size;
    const size_t hidden_size = model.config.hidden_size;
    const size_t seq_len = in_swiglu.size() / intermediate_size;

    // Prepare GPU buffers
    __half* d_in  = nullptr;
    __half* d_out = nullptr;

    HIP_CHECK(hipMalloc(&d_in,  in_swiglu.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_out, exp_mlp.size() * sizeof(__half)));

    HIP_CHECK(hipMemcpy(d_in, in_swiglu.data(), in_swiglu.size() * sizeof(__half), hipMemcpyHostToDevice));

    // Launch Down Projection GEMM
    const __half* d_w_down = model.device_weights.transformer_blocks[0].down_proj;

    launch_down_proj_gemm(
        d_in,
        d_out,
        d_w_down,
        seq_len,
        hidden_size,
        intermediate_size
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Download and verify
    std::vector<__half> actual_mlp(exp_mlp.size());
    HIP_CHECK(hipMemcpy(actual_mlp.data(), d_out, exp_mlp.size() * sizeof(__half), hipMemcpyDeviceToHost));

    INFO("Verifying Down Projection Output...");
    test_utils::check_tensor_close(actual_mlp, exp_mlp, /*atol=*/2e-2f, /*rtol=*/2e-2f);

    // Cleanup
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
}