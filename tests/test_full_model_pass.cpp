#include <doctest/doctest.h>
#include "test_utils.hpp"
#include "model.hpp"
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <iterator>
#include <vector>

const Model& get_test_model();

TEST_CASE("End-to-End: Full 36-Layer Model Pass vs Final Logits") {
    const Model& model = get_test_model();

    hipDeviceProp_t props{};
    int device = 0;
    
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    
    int warp_size = props.warpSize;
    
    // Load fixtures
    auto input_tokens = test_utils::load_binary_file<int32_t>("../../../../tests/reference/input_tokens.bin");
    auto exp_logits   = test_utils::load_binary_file<__half>("../../../../tests/reference/final_logits.bin");

    const size_t seq_len           = input_tokens.size();
    const size_t hidden_size       = model.config.hidden_size;
    const size_t num_layers        = model.config.num_layers;
    const size_t num_heads         = model.config.num_heads;
    const size_t num_kv_heads      = model.config.num_kv_heads;
    const size_t head_dim          = model.config.head_dim;
    const size_t intermediate_size = model.config.intermediate_size;
    const size_t vocab_size        = model.config.vocab_size;
    const float  rms_eps           = model.config.rms_norm_eps;

    const size_t q_dim             = num_heads * head_dim;
    const size_t kv_dim            = num_kv_heads * head_dim;
    const size_t total_qkv_dim     = q_dim + 2 * kv_dim;

    // Allocate Static Working VRAM Buffers (Reused for all 36 layers!)
    int32_t* d_tokens        = nullptr;
    __half*  d_x             = nullptr; // The main residual highway buffer
    __half*  d_norm_out      = nullptr; // Scratchpad for RMSNorm
    __half*  d_qkv_out       = nullptr; // Scratchpad for QKV
    __half*  d_attn_out      = nullptr; // Scratchpad for Flash Attention
    __half*  d_gate_up_out   = nullptr; // Scratchpad for Gate/Up
    __half*  d_swiglu_out    = nullptr; // Scratchpad for SwiGLU
    __half*  d_logits        = nullptr; // Final logits output

    HIP_CHECK(hipMalloc(&d_tokens,      seq_len * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_x,           seq_len * hidden_size * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_norm_out,    seq_len * hidden_size * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_qkv_out,     seq_len * total_qkv_dim * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_attn_out,    seq_len * hidden_size * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_gate_up_out, seq_len * 2 * intermediate_size * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_swiglu_out,  seq_len * intermediate_size * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_logits,      seq_len * vocab_size * sizeof(__half)));

    HIP_CHECK(hipMemcpy(d_tokens, input_tokens.data(), seq_len * sizeof(int32_t), hipMemcpyHostToDevice));

    RopeCache rope_cache = create_rope_cache(model.config.max_seq_len, head_dim, model.config.rope_theta);

    // Initial Embedding Lookup -> sets up the highway buffer d_x
    launch_embedding_lookup(d_tokens, model.device_weights.embedding_tokens, d_x, seq_len, hidden_size);

    // THE 36-LAYER TRANSFORMER LOOP
    for (size_t l = 0; l < num_layers; ++l) {
        const auto& block = model.device_weights.transformer_blocks[l];

        // --- Attention Sub-block ---
        launch_rms_norm(d_x, d_norm_out, block.input_normalization, seq_len, hidden_size, rms_eps, warp_size);

        launch_qkv_gemm(d_norm_out, d_qkv_out, block.q_proj, seq_len, hidden_size, total_qkv_dim);

        __half* d_q = d_qkv_out;
        // GEMM stores [Q_token, K_token, V_token] for each token.
        __half* d_k = d_qkv_out + q_dim;
        __half* d_v = d_qkv_out + q_dim + kv_dim;

        // RoPE: SmolLM3 skips RoPE on every 4th layer (layers 3, 7, 11...)
        if ((l + 1) % 4 != 0) {
            const size_t q_stride = total_qkv_dim;
            const size_t k_stride = total_qkv_dim;
            launch_rope(d_q, d_k, rope_cache.d_cos, rope_cache.d_sin, seq_len, num_heads, num_kv_heads, head_dim, q_stride, k_stride, /*start_pos=*/0);
        }
        launch_causal_attention(d_q, d_k, d_v, d_attn_out, seq_len, num_heads, num_kv_heads, head_dim, total_qkv_dim, total_qkv_dim);

        // Fused Residual Connection #1: d_x = d_x + o_proj(d_attn_out)
        launch_attention_out_projection(d_attn_out, d_x, block.o_proj, seq_len, hidden_size, num_heads, head_dim, /*beta=*/1.0f);

        // --- MLP Sub-block ---
        launch_rms_norm(d_x, d_norm_out, block.post_attn_norm, seq_len, hidden_size, rms_eps, warp_size);

        launch_gate_up_gemm(d_norm_out, d_gate_up_out, block.gate_proj, seq_len, hidden_size, intermediate_size);

        launch_swiglu(d_gate_up_out, d_swiglu_out, seq_len, seq_len * intermediate_size, intermediate_size);

        // Fused Residual Connection #2: d_x = d_x + down_proj(d_swiglu_out)
        launch_down_proj_gemm(d_swiglu_out, d_x, block.down_proj, seq_len, hidden_size, intermediate_size, /*beta=*/1.0f);
    }

    // Final RMSNorm
    launch_rms_norm(d_x, d_norm_out, model.device_weights.final_norm, seq_len, hidden_size, rms_eps, warp_size);

    // LM Head (Unembedding)
    launch_lm_head_gemm(d_norm_out, d_logits, model.device_weights.lm_head, seq_len, hidden_size, vocab_size);

    // Download and verify against PyTorch final logits
    std::vector<__half> actual_logits(seq_len * vocab_size);
    HIP_CHECK(hipMemcpy(actual_logits.data(), d_logits, actual_logits.size() * sizeof(__half), hipMemcpyDeviceToHost));

    INFO("Verifying End-to-End Final Logits...");
    test_utils::check_tensor_close(actual_logits, exp_logits, /*atol=*/5e-1f, /*rtol=*/5e-1f);

    // Argmax check: Pick the last token's prediction!
    const __half* last_token_logits = actual_logits.data() + ((seq_len - 1) * vocab_size);
    float max_val = -1e9f;
    int best_token_id = -1;

    for (size_t v = 0; v < vocab_size; ++v) {
        float val = __half2float(last_token_logits[v]);
        if (val > max_val) {
            max_val = val;
            best_token_id = static_cast<int>(v);
        }
    }

    std::string decoded_token = model.tokenizer->Decode({best_token_id});
    std::println("\n🎉 PREDICTED NEXT TOKEN: ID={} ('{}')", best_token_id, decoded_token);

    const auto expected_last = exp_logits.begin() + (seq_len - 1) * vocab_size;
    const auto expected_best = std::max_element(expected_last, exp_logits.end(),
        [](__half a, __half b) { return __half2float(a) < __half2float(b); });
    CHECK(best_token_id == std::distance(expected_last, expected_best));

    // Cleanup
    HIP_CHECK(hipFree(d_tokens));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_norm_out));
    HIP_CHECK(hipFree(d_qkv_out));
    HIP_CHECK(hipFree(d_attn_out));
    HIP_CHECK(hipFree(d_gate_up_out));
    HIP_CHECK(hipFree(d_swiglu_out));
    HIP_CHECK(hipFree(d_logits));
}
