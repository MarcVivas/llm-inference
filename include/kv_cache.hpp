#pragma once
#include "gpu_utils.hpp"
#include "model_config.hpp"
#include <hip/hip_runtime.h>
#include<hip/hip_fp16.h>
#include <format>

class KVCache{
  __half *d_kv_buffer = nullptr;
  const ModelConfig& config;
  size_t current_cache_len;

  public:
      explicit KVCache(const ModelConfig& config): config(config), current_cache_len(0){
          HIP_CHECK(hipMalloc(&d_kv_buffer, compute_total_bytes(config)));
      }

       

      // Key pointer for Layer l: base + (l * layer_stride)
      [[nodiscard]] __half* get_k_cache(size_t layer_idx) noexcept {
          return d_kv_buffer + (layer_idx * layer_stride());
      }
      [[nodiscard]] const __half* get_k_cache(size_t layer_idx) const noexcept {
          return d_kv_buffer + (layer_idx * layer_stride());
      }

      // Value pointer for Layer l: base + (l * layer_stride) + (max_seq_len * kv_dim)
      [[nodiscard]] __half* get_v_cache(size_t layer_idx) noexcept {
          return d_kv_buffer + (layer_idx * layer_stride()) + (config.max_seq_len * kv_dim());
      }
      [[nodiscard]] const __half* get_v_cache(size_t layer_idx) const noexcept {
          return d_kv_buffer + (layer_idx * layer_stride()) + (config.max_seq_len * kv_dim());
      }

      size_t get_cached_len() const noexcept{
          return this->current_cache_len;
      }

      void set_cached_len(size_t cached_len){
          if (cached_len > config.max_seq_len)
              throw std::out_of_range(std::format("KV cache capacity exceeded: requested {} tokens, capacity {}",
                                                  cached_len, config.max_seq_len));
          this->current_cache_len = cached_len;
      }

      void reset_cache(){
          this->current_cache_len = 0;
      }

      ~KVCache(){
          cleanup();
          d_kv_buffer = nullptr;
      }

      KVCache(const KVCache&) = delete;
      KVCache& operator=(const KVCache&) = delete;
      KVCache(KVCache&& o) noexcept
          : d_kv_buffer(o.d_kv_buffer), config(o.config), current_cache_len(o.current_cache_len)
      {
          o.d_kv_buffer = nullptr;
          o.current_cache_len = 0; 
      }
  
      KVCache& operator=(KVCache&& o) = delete; 
  private:
      void cleanup(){
          if(d_kv_buffer) auto a = hipFree(d_kv_buffer);
      }

      size_t compute_total_bytes(const ModelConfig& config){
          return 2 * config.num_kv_heads * config.head_dim * config.max_seq_len * sizeof(__half) * config.num_layers;
      }


      size_t kv_dim() const noexcept{
          return config.num_kv_heads * config.head_dim;
      }
      
      size_t layer_stride() const noexcept{
          return 2 * config.num_kv_heads * config.head_dim * config.max_seq_len; 
      }
};
