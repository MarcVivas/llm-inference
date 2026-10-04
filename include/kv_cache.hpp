#pragma once
#include "gpu_utils.hpp"
#include "model_config.hpp"
#include <hip/hip_runtime.h>
#include<hip/hip_fp16.h>

class KVCache{
  __half *d_kv_buffer = nullptr;
  const ModelConfig& config;

  public:
      explicit KVCache(const ModelConfig& config): config(config){
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



      ~KVCache(){
          cleanup();
          d_kv_buffer = nullptr;
      }

      KVCache(const KVCache&) = delete;
      KVCache& operator=(const KVCache&) = delete;
      KVCache(KVCache&& o) noexcept
          : config(o.config), d_kv_buffer(o.d_kv_buffer) 
      {
          o.d_kv_buffer = nullptr;
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
