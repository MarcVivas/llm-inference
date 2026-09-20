#pragma once

#include "gpu_utils.hpp"
#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>

struct RopeCache {
    __half *d_cos = nullptr;
    __half *d_sin = nullptr;
    
    ~RopeCache() {
        if (d_cos)
        HIP_CHECK(hipFree(d_cos));
        if (d_sin)
        HIP_CHECK(hipFree(d_sin));
    }
    
    // Move-only semantics (RAII safety)
    RopeCache() = default;
    RopeCache(const RopeCache &) = delete;
    RopeCache &operator=(const RopeCache &) = delete;
    
    RopeCache(RopeCache &&o) noexcept
        : d_cos(o.d_cos), d_sin(o.d_sin)
    
    {
        o.d_cos = nullptr;
        o.d_sin = nullptr;
    }
    
    RopeCache &operator=(RopeCache &&o) noexcept {
        if (this != &o) {
        if (d_cos)
            HIP_CHECK(hipFree(d_cos));
        if (d_sin)
            HIP_CHECK(hipFree(d_sin));
        d_cos = o.d_cos;
        d_sin = o.d_sin;
        o.d_cos = nullptr;
        o.d_sin = nullptr;
        }
        return *this;
    }
};

// Compute theta frequencies for each of the 64 pairs
inline std::vector<float> compute_inverse_frequencies(size_t head_dim,
                                                      float rope_theta) {
  size_t half_dim = head_dim / 2;
  std::vector<float> inv_freq(half_dim);

  for (size_t i = 0; i < half_dim; ++i) {
    // theta_i = 1.0 / (rope_theta ^ (2i / head_dim))
    float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
    inv_freq[i] = 1.0f / std::pow(rope_theta, exponent);
  }
  return inv_freq;
}

// Precompute cos & sin tables on host CPU
inline void fill_host_tables(std::vector<__half> &host_cos,
                             std::vector<__half> &host_sin,
                             const std::vector<float> &inv_freq,
                             size_t max_seq_len, size_t half_dim) {
  for (size_t pos = 0; pos < max_seq_len; ++pos) {
    for (size_t i = 0; i < half_dim; ++i) {
      float angle = static_cast<float>(pos) * inv_freq[i];
      size_t idx = pos * half_dim + i;

      host_cos[idx] = __float2half(std::cos(angle));
      host_sin[idx] = __float2half(std::sin(angle));
    }
  }
}

// Factory function: orchestrates math and uploads to VRAM
inline RopeCache create_rope_cache(size_t max_seq_len, size_t head_dim,
                                   float rope_theta) {
  RopeCache cache;
  const size_t half_dim = head_dim / 2;

  const size_t total_elements = max_seq_len * half_dim;
  const size_t total_bytes = total_elements * sizeof(__half);

  // Compute on CPU
  std::vector<float> inv_freq =
      compute_inverse_frequencies(head_dim, rope_theta);
  std::vector<__half> host_cos(total_elements);
  std::vector<__half> host_sin(total_elements);
  fill_host_tables(host_cos, host_sin, inv_freq, max_seq_len, half_dim);

  // Allocate on GPU
  HIP_CHECK(hipMalloc(&cache.d_cos, total_bytes));
  HIP_CHECK(hipMalloc(&cache.d_sin, total_bytes));

  // Copy to GPU VRAM
  HIP_CHECK(hipMemcpy(cache.d_cos, host_cos.data(), total_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(cache.d_sin, host_sin.data(), total_bytes,
                      hipMemcpyHostToDevice));

  return cache;
}
