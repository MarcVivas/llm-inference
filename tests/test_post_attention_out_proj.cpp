
#include <doctest/doctest.h>
#include "gpu_utils.hpp"
#include "hip/driver_types.h"
#include "kernels/kernels.hpp"
#include "model.hpp"
#include "test_utils.hpp"

const Model& get_test_model();

TEST_CASE("Kernel: Post attention output projection"){
    const Model &model = get_test_model();
   
    // Load fixtures
    std::vector<__half> input_attention = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_attn_out.bin");
    std::vector<__half> expected_projection = test_utils::load_binary_file<__half>("../../../../tests/reference/layer0_o_proj_out.bin");

    // Prepare GPU buffers
    __half* d_in = nullptr;
    __half* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, input_attention.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_out, expected_projection.size() * sizeof(__half)));

    // Copy the input to the device
    HIP_CHECK(hipMemcpy(d_in, input_attention.data(), input_attention.size() * sizeof(__half), hipMemcpyHostToDevice));

    // Launch the kernel 
    const __half* d_o_proj_weights = model.device_weights.transformer_blocks[0].o_proj; 

    const size_t seq_len = input_attention.size() / (model.config.num_heads * model.config.head_dim); 
    
    launch_attention_out_projection(d_in, d_out, d_o_proj_weights, seq_len, model.config.hidden_size, model.config.num_heads, model.config.head_dim);
    HIP_CHECK(hipDeviceSynchronize());

    // Check results
    std::vector<__half> actual_proj(expected_projection.size());
    HIP_CHECK(hipMemcpy(actual_proj.data(), d_out, expected_projection.size() * sizeof(__half), hipMemcpyDeviceToHost));

    INFO("Verifying o_proj output...");
    test_utils::check_tensor_close(actual_proj, expected_projection, 2e-2f, 2e-2f);

    // Cleanup
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
}