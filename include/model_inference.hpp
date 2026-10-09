#pragma once
#include "gpu_utils.hpp"
#include "profiling.hpp"
#include <format>
#include <hip/hip_runtime.h>
#include "kv_cache.hpp"
#include "model.hpp"
#include <algorithm>
#include <stdexcept>
#include "kernels/kernels.hpp"

enum class ForwardMode { Prefill, Decode };

class ModelInference{
    public:

        explicit ModelInference(const Model& model)
        : model_(model), kv_cache(KVCache(model.config)){
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

        int prefill(std::span<const int> prompt_tokens){
            profiling::Range range("Prefill");
            const size_t seq_len = prompt_tokens.size();

            if(seq_len == 0 || seq_len > model_.config.max_seq_len){
                throw std::runtime_error("Invalid prompt length");
            }

            // Copy the tokens to the GPU
            HIP_CHECK(hipMemcpy(d_tokens, prompt_tokens.data(), prompt_tokens.size_bytes(), hipMemcpyHostToDevice));

            kv_cache.reset_cache();

            {
                profiling::Range range("Embedding");
                launch_embedding_lookup(d_tokens, model_.device_weights.embedding_tokens, d_x, seq_len, model_.config.hidden_size);
            }

            for (size_t l = 0; l < model_.config.num_layers; ++l) {
                forward_transformer_layer(l, seq_len, /*start_pos=*/0, ForwardMode::Prefill);
            }

            compute_last_token_logits(seq_len);

            kv_cache.set_cached_len(seq_len);

            return sample_greedy();
        }

        int decode(int token){
            profiling::Range range("Decode");
            const size_t start_pos = kv_cache.get_cached_len();

            if(start_pos == 0){
                throw std::runtime_error("Call prefill before decode");
            }
            if(start_pos >= model_.config.max_seq_len){
                throw std::runtime_error("KV cache is full");
            }

            if(token < 0 || size_t(token) >= model_.config.vocab_size){
                throw std::runtime_error("Invalid token ID");
            }

            // Copy the token to the GPU
            HIP_CHECK(hipMemcpy(d_tokens, &token, sizeof(token), hipMemcpyHostToDevice));

            {
                profiling::Range range("Embedding");
                launch_embedding_lookup(d_tokens, model_.device_weights.embedding_tokens, d_x, 1 /*seq_len*/, model_.config.hidden_size);
            }

            for (size_t l = 0; l < model_.config.num_layers; ++l) {
                forward_transformer_layer(l, 1 /*seq_len*/, /*start_pos=*/start_pos, ForwardMode::Decode);
            }

            compute_last_token_logits(1 /*seq_len*/);

            const int next_token = sample_greedy();

            kv_cache.set_cached_len(start_pos + 1);

            return next_token;
        }

        int sample_greedy() const {
            profiling::Range range("SampleGreedy");
            launch_argmax(d_logits, d_best_token, model_.config.vocab_size);
            int best_token_id = -1;
            HIP_CHECK(hipMemcpy(&best_token_id, d_best_token, sizeof(int32_t), hipMemcpyDeviceToHost));
            return best_token_id;
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

        // Decode attention workspace
        float* d_partial_output = nullptr;
        float* d_partial_max = nullptr;
        float* d_partial_weight_sum = nullptr;
        size_t decode_chunk_capacity = 0;
        // KV cache
        KVCache kv_cache;

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

            decode_chunk_capacity = (config.max_seq_len + KV_CHUNK - 1) / KV_CHUNK;

            const size_t partial_count = static_cast<size_t>(config.num_heads) * decode_chunk_capacity;

            HIP_CHECK(hipMalloc(&d_partial_output, partial_count * config.head_dim * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_partial_max, partial_count * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_partial_weight_sum, partial_count * sizeof(float)));
        }

        void free_vram_buffers() noexcept {
            if (d_x) (void)hipFree(d_x);
            if (d_scratch_a) (void)hipFree(d_scratch_a);
            if (d_scratch_b) (void)hipFree(d_scratch_b);
            if (d_tokens) (void)hipFree(d_tokens);
            if (d_logits) (void)hipFree(d_logits);
            if (d_best_token) (void)hipFree(d_best_token);
            if (d_partial_output) (void)hipFree(d_partial_output);
            if (d_partial_max) (void)hipFree(d_partial_max);
            if (d_partial_weight_sum) (void)hipFree(d_partial_weight_sum);
        }

        // Executes one Transformer Block (Layer l) using Ping-Pong buffers
        void forward_transformer_layer(size_t layer_idx, size_t seq_len, size_t start_pos, ForwardMode forward_mode) {
            const auto layer_name = std::format("Layer {}", layer_idx);
            profiling::Range layer_range(layer_name.c_str());
            const auto& block = model_.device_weights.transformer_blocks[layer_idx];
            const auto& cfg   = model_.config;

            const size_t q_dim          = cfg.num_heads * cfg.head_dim;
            const size_t kv_dim         = cfg.num_kv_heads * cfg.head_dim;
            const size_t total_qkv_dim  = q_dim + (2 * kv_dim);

            // Attention Sub-Block
            // RMSNorm: d_x -> d_scratch_b
            {
                profiling::Range range("InputNorm");
                launch_rms_norm(d_x, d_scratch_b, block.input_normalization, seq_len, cfg.hidden_size, cfg.rms_norm_eps, warp_size);
            }

            // Fused QKV GEMM: d_scratch_b -> d_scratch_a
            {
                profiling::Range range("QKV");
                launch_qkv_gemm(d_scratch_b, d_scratch_a, block.q_proj, seq_len, cfg.hidden_size, total_qkv_dim);
            }

            // Strided RoPE: In-place on d_scratch_a (SmolLM3 skips RoPE every 4th layer)
            const bool apply_rotation = (layer_idx + 1) % 4 != 0;
            {
                profiling::Range range("RoPE+StoreKV");
                launch_rope_and_store_kv(
                    d_scratch_a, d_scratch_a + q_dim, d_scratch_a + q_dim + kv_dim, kv_cache.get_k_cache(layer_idx), kv_cache.get_v_cache(layer_idx),
                    model_.rope_cache.d_cos, model_.rope_cache.d_sin,
                    seq_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                    total_qkv_dim, total_qkv_dim, start_pos, apply_rotation
                );
            }



            if(forward_mode == ForwardMode::Prefill){
                // Q remains in scratch; rotated K and unchanged V are in the cache.
                {
                    profiling::Range range("AttentionPrefill");
                    launch_causal_attention_prefill(
                        d_scratch_a, kv_cache.get_k_cache(layer_idx), kv_cache.get_v_cache(layer_idx),
                        d_scratch_b, seq_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                        total_qkv_dim, kv_dim
                    );
                }
            }
            else {
                {
                    profiling::Range range("FlashDecode");
                    launch_causal_attention_decode(
                        d_scratch_a,
                        kv_cache.get_k_cache(layer_idx),
                        kv_cache.get_v_cache(layer_idx),
                        d_scratch_b,
                        d_partial_output,
                        d_partial_max,
                        d_partial_weight_sum,
                        seq_len,
                        start_pos + seq_len,
                        start_pos,
                        cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                        total_qkv_dim, kv_dim
                    );
                }
            }



            // o_proj with beta = 1.0f: d_scratch_b + d_x -> d_x (Highway update #1)
            {
                profiling::Range range("AttentionOutput");
                launch_attention_out_projection(
                    d_scratch_b, d_x, block.o_proj,
                    seq_len, cfg.hidden_size, cfg.num_heads, cfg.head_dim, /*beta=*/1.0f
                );
            }

            // SwiGLU MLP Sub-Block
            // Post-Attention RMSNorm: d_x -> d_scratch_b
            {
                profiling::Range range("PostAttentionNorm");
                launch_rms_norm(d_x, d_scratch_b, block.post_attn_norm, seq_len, cfg.hidden_size, cfg.rms_norm_eps, warp_size);
            }

            // Fused Gate & Up GEMM: d_scratch_b -> d_scratch_a
            {
                profiling::Range range("GateUp");
                launch_gate_up_gemm(d_scratch_b, d_scratch_a, block.gate_proj, seq_len, cfg.hidden_size, cfg.intermediate_size);
            }

            // SwiGLU Activation: d_scratch_a -> d_scratch_b
            {
                profiling::Range range("SwiGLU");
                launch_swiglu(d_scratch_a, d_scratch_b, seq_len, seq_len * cfg.intermediate_size, cfg.intermediate_size);
            }

            // down_proj with beta = 1.0f: d_scratch_b + d_x -> d_x (Highway update #2)
            {
                profiling::Range range("Down");
                launch_down_proj_gemm(
                    d_scratch_b, d_x, block.down_proj,
                    seq_len, cfg.hidden_size, cfg.intermediate_size, /*beta=*/1.0f
                );
            }
        }

        // Projects the last token through Final Norm and LM Head
        void compute_last_token_logits(size_t seq_len) const {
            const auto& cfg = model_.config;

            // Pointer to the last token on the highway
            const __half* d_last_token = d_x + ((seq_len - 1) * cfg.hidden_size);

            // Final RMSNorm: d_last_token -> d_scratch_b
            {
                profiling::Range range("FinalNorm");
                launch_rms_norm(d_last_token, d_scratch_b, model_.device_weights.final_norm, 1, cfg.hidden_size, cfg.rms_norm_eps, warp_size);
            }

            // LM Head: d_scratch_b -> d_logits_ (vocab_size logits for only the last token!)
            {
                profiling::Range range("LMHead");
                launch_lm_head_gemm(d_scratch_b, d_logits, model_.device_weights.lm_head, 1, cfg.hidden_size, cfg.vocab_size);
            }
        }


};
