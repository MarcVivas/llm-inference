#pragma once
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>
#include "model.hpp"
#include <algorithm>
#include <stdexcept>
#include "kernels/kernels.hpp"

class ModelInference{
    public:

        explicit ModelInference(const Model& model)
        : model_(model){
            hipDeviceProp_t props{};
            int device = 0;

            HIP_CHECK(hipGetDevice(&device));
            HIP_CHECK(hipGetDeviceProperties(&props, device));

            warp_size = props.warpSize;
            allocate_vram_buffers();
        }

        ~ModelInference(){
            free_vram_buffers();
        }

        // Move-only semantics (RAII)
        ModelInference(const ModelInference&) = delete;
        ModelInference& operator=(const ModelInference&) = delete;
        ModelInference(ModelInference&&) = delete;
        ModelInference& operator=(ModelInference&&) = delete;

        int prefill(std::span<const int> prompt_tokens) const{
            const size_t seq_len = prompt_tokens.size();

            if(seq_len == 0 || seq_len > model_.config.max_seq_len){
                throw std::runtime_error("Invalid prompt length");
            }

            // Copy the tokens to the GPU
            HIP_CHECK(hipMemcpy(d_tokens, prompt_tokens.data(), prompt_tokens.size_bytes(), hipMemcpyHostToDevice));

            launch_embedding_lookup(d_tokens, model_.device_weights.embedding_tokens, d_x, seq_len, model_.config.hidden_size);

            for (size_t l = 0; l < model_.config.num_layers; ++l) {
                forward_transformer_layer(l, seq_len, /*start_pos=*/0);
            }

            compute_last_token_logits(seq_len);


            launch_argmax(d_logits, d_best_token, model_.config.vocab_size);
            int32_t best_token_id = -1;
            HIP_CHECK(hipMemcpy(&best_token_id, d_best_token, sizeof(int32_t), hipMemcpyDeviceToHost));
            return best_token_id;
        }

        int decode(std::span<const int> tokens){
            return prefill(tokens);
        }

        int sample_greedy() const {
            const size_t vocab_size = model_.config.vocab_size;
            std::vector<__half> host_logits(vocab_size);

            // Copy ONLY the single token's logits back to CPU (takes 0.05 ms)
            HIP_CHECK(hipMemcpy(host_logits.data(), d_logits, vocab_size * sizeof(__half), hipMemcpyDeviceToHost));

            float max_val = -1e9f;
            int32_t best_id = 0;

            for (size_t i = 0; i < vocab_size; ++i) {
                float val = __half2float(host_logits[i]);
                if (val > max_val) {
                    max_val = val;
                    best_id = static_cast<int32_t>(i);
                }
            }

            return best_id;
        }


    private:
        const Model& model_;
        int warp_size;

        __half* d_x = nullptr;           // The Highway:   [seq_len, hidden_size]
        __half* d_scratch_a = nullptr;   // Large Scratch: [seq_len, 2 * intermediate_size]
        __half* d_scratch_b = nullptr;   // Med Scratch:   [seq_len, intermediate_size]

        // Tokens and output logits
        int* d_tokens = nullptr;    // [max_seq_len]
        __half* d_logits = nullptr; // [vocab_size]
        int* d_best_token = nullptr; // 4 bytes! (Direct output from GPU Argmax)

        void allocate_vram_buffers(){
            const auto& config = model_.config;

            const uint32_t highway_bytes = config.max_seq_len * config.hidden_size * sizeof(__half);
            const uint32_t scratch_a_bytes = config.max_seq_len * (2 * config.intermediate_size) * sizeof(__half);
            const uint32_t scratch_b_bytes = config.max_seq_len * std::max(config.intermediate_size, config.num_heads * config.head_dim) * sizeof(__half);

            HIP_CHECK(hipMalloc(&d_x, highway_bytes));
            HIP_CHECK(hipMalloc(&d_scratch_a, scratch_a_bytes));
            HIP_CHECK(hipMalloc(&d_scratch_b, scratch_b_bytes));

            HIP_CHECK(hipMalloc(&d_tokens, config.max_seq_len * sizeof(int)));
            HIP_CHECK(hipMalloc(&d_logits, config.vocab_size * sizeof(__half)));
            HIP_CHECK(hipMalloc(&d_best_token, sizeof(int32_t)));

        }

        void free_vram_buffers() noexcept {
            if (d_x) auto a = hipFree(d_x);
            if (d_scratch_a) auto a = hipFree(d_scratch_a);
            if (d_scratch_b) auto a = hipFree(d_scratch_b);
            if (d_tokens) auto a = hipFree(d_tokens);
            if (d_logits) auto a = hipFree(d_logits);
            if (d_logits) auto a = hipFree(d_best_token);
        }
        
        // Executes one Transformer Block (Layer l) using Ping-Pong buffers
        void forward_transformer_layer(size_t layer_idx, size_t seq_len, size_t start_pos) const {
            const auto& block = model_.device_weights.transformer_blocks[layer_idx];
            const auto& cfg   = model_.config;

            const size_t q_dim          = cfg.num_heads * cfg.head_dim;
            const size_t kv_dim         = cfg.num_kv_heads * cfg.head_dim;
            const size_t total_qkv_dim  = q_dim + (2 * kv_dim);

            // Attention Sub-Block
            // RMSNorm: d_x -> d_scratch_b
            launch_rms_norm(d_x, d_scratch_b, block.input_normalization, seq_len, cfg.hidden_size, cfg.rms_norm_eps, warp_size);

            // Fused QKV GEMM: d_scratch_b -> d_scratch_a
            launch_qkv_gemm(d_scratch_b, d_scratch_a, block.q_proj, seq_len, cfg.hidden_size, total_qkv_dim);

            // Strided RoPE: In-place on d_scratch_a (SmolLM3 skips RoPE every 4th layer)
            if ((layer_idx + 1) % 4 != 0) {
                launch_rope(
                    d_scratch_a, d_scratch_a + q_dim,
                    model_.rope_cache.d_cos, model_.rope_cache.d_sin,
                    seq_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                    total_qkv_dim, total_qkv_dim, start_pos
                );
            }

            // Flash Attention: reads d_scratch_a (Q, K, V) -> writes d_scratch_b
            launch_causal_attention(
                d_scratch_a, d_scratch_a + q_dim, d_scratch_a + q_dim + kv_dim,
                d_scratch_b, seq_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                total_qkv_dim, total_qkv_dim
            );

            // o_proj with beta = 1.0f: d_scratch_b + d_x -> d_x (Highway update #1)
            launch_attention_out_projection(
                d_scratch_b, d_x, block.o_proj,
                seq_len, cfg.hidden_size, cfg.num_heads, cfg.head_dim, /*beta=*/1.0f
            );

            // SwiGLU MLP Sub-Block
            // Post-Attention RMSNorm: d_x -> d_scratch_b
            launch_rms_norm(d_x, d_scratch_b, block.post_attn_norm, seq_len, cfg.hidden_size, cfg.rms_norm_eps, warp_size);

            // Fused Gate & Up GEMM: d_scratch_b -> d_scratch_a
            launch_gate_up_gemm(d_scratch_b, d_scratch_a, block.gate_proj, seq_len, cfg.hidden_size, cfg.intermediate_size);

            // SwiGLU Activation: d_scratch_a -> d_scratch_b
            launch_swiglu(d_scratch_a, d_scratch_b, seq_len, seq_len * cfg.intermediate_size, cfg.intermediate_size);

            // down_proj with beta = 1.0f: d_scratch_b + d_x -> d_x (Highway update #2)
            launch_down_proj_gemm(
                d_scratch_b, d_x, block.down_proj,
                seq_len, cfg.hidden_size, cfg.intermediate_size, /*beta=*/1.0f
            );
        }

        // Projects the last token through Final Norm and LM Head
        void compute_last_token_logits(size_t seq_len) const {
            const auto& cfg = model_.config;

            // Pointer to the last token on the highway
            const __half* d_last_token = d_x + ((seq_len - 1) * cfg.hidden_size);

            // Final RMSNorm: d_last_token -> d_scratch_b
            launch_rms_norm(d_last_token, d_scratch_b, model_.device_weights.final_norm, 1, cfg.hidden_size, cfg.rms_norm_eps, warp_size);

            // LM Head: d_scratch_b -> d_logits_ (vocab_size logits for only the last token!)
            launch_lm_head_gemm(d_scratch_b, d_logits, model_.device_weights.lm_head, 1, cfg.hidden_size, cfg.vocab_size);
        }


};
