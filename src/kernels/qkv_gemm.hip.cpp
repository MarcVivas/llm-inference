#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"

static constexpr size_t BLOCK_SIZE = 512;


void launch_qkv_gemm(
    const __half* d_input_norm,
    __half* d_qkv_out,
    const __half* d_qkv_weights,
    const size_t seq_len,
    const size_t hidden_size, 
    const size_t total_qkv_dim,
    rocblas_handle rocblas_handle
){

    HIP_CHECK(hipDeviceSynchronize());
}