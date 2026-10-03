#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>
#include <vector>

const Model& get_test_model();

TEST_CASE("Kernel: GPU Argmax Reduction") {
    const Model& model = get_test_model();

    // Load ground truth final logits
    auto exp_logits = test_utils::load_binary_file<__half>("../../../../tests/reference/final_logits.bin");

    const size_t vocab_size = model.config.vocab_size;
    const size_t seq_len    = exp_logits.size() / vocab_size;

    // Point to the last token's logits
    const __half* last_token_logits = exp_logits.data() + ((seq_len - 1) * vocab_size);

    // Compute CPU expected winner
    float cpu_max = -1e9f;
    int32_t exp_token_id = -1;
    for (size_t i = 0; i < vocab_size; ++i) {
        float val = __half2float(last_token_logits[i]);
        if (val > cpu_max) {
            cpu_max = val;
            exp_token_id = static_cast<int32_t>(i);
        }
    }

    // Prepare GPU buffers
    __half*  d_logits = nullptr;
    int32_t* d_winner = nullptr;

    HIP_CHECK(hipMalloc(&d_logits, vocab_size * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_winner, sizeof(int32_t)));

    HIP_CHECK(hipMemcpy(d_logits, last_token_logits, vocab_size * sizeof(__half), hipMemcpyHostToDevice));

    // Launch GPU Argmax Kernel
    launch_argmax(d_logits, d_winner, vocab_size);
    HIP_CHECK(hipDeviceSynchronize());

    // Download only 4 bytes!
    int32_t actual_token_id = -1;
    HIP_CHECK(hipMemcpy(&actual_token_id, d_winner, sizeof(int32_t), hipMemcpyDeviceToHost));

    // Verify
    CHECK(actual_token_id == exp_token_id);

    // Cleanup
    HIP_CHECK(hipFree(d_logits));
    HIP_CHECK(hipFree(d_winner));
}
