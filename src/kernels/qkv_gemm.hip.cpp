#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/internal/rocblas-types.h>
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"

static constexpr size_t BLOCK_SIZE = 512;


void launch_qkv_gemm(
    const __half* d_input_norm,
    __half* d_qkv_out,
    const __half* d_qkv_weights,
    const size_t seq_len,
    const size_t hidden_size,
    const size_t total_qkv_dim
) {
    rocblas_handle handle = get_rocblas_handle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Fused QKV projection:
    //
    //   QKV = Input * W_qkv^T
    //
    // Row-major tensor shapes:
    //
    //   Input: [seq_len, hidden_size]
    //
    //   W_qkv: [total_qkv_dim, hidden_size]
    //          = Q weights followed by K weights followed by V weights
    //
    //   QKV:   [seq_len, total_qkv_dim]
    //
    // rocBLAS uses column-major matrices. The same memory is therefore
    // interpreted by rocBLAS as:
    //
    //   W_qkv: [hidden_size, total_qkv_dim]
    //   Input: [hidden_size, seq_len]
    //   QKV:   [total_qkv_dim, seq_len]
    //
    // rocBLAS computes:
    //
    //   W_qkv^T * Input
    //
    //   [total_qkv_dim, hidden_size]
    //              *
    //   [hidden_size, seq_len]
    //
    //              =
    //
    //   [total_qkv_dim, seq_len]
    //
    // whose memory is the desired row-major
    // [seq_len, total_qkv_dim] output.

    const rocblas_int output_dim = static_cast<rocblas_int>(total_qkv_dim);
    const rocblas_int num_tokens = static_cast<rocblas_int>(seq_len);
    const rocblas_int input_dim  = static_cast<rocblas_int>(hidden_size);

    const rocblas_int weights_stride = input_dim;
    const rocblas_int input_stride   = input_dim;
    const rocblas_int output_stride  = output_dim;

    rocblas_status status = rocblas_gemm_ex(
        handle,

        rocblas_operation_transpose,
        rocblas_operation_none,

        output_dim,
        num_tokens,
        input_dim,

        &alpha,

        d_qkv_weights,
        rocblas_datatype_f16_r,
        weights_stride,

        d_input_norm,
        rocblas_datatype_f16_r,
        input_stride,

        &beta,

        d_qkv_out,
        rocblas_datatype_f16_r,
        output_stride,

        d_qkv_out,
        rocblas_datatype_f16_r,
        output_stride,

        rocblas_datatype_f32_r,

        rocblas_gemm_algo_standard,
        0,
        0
    );

    if (status != rocblas_status_success) {
        throw std::runtime_error(
            "rocblas_gemm_ex failed for QKV projection"
        );
    }
}

void launch_qkv_decode(
    const __half* d_input_norm,
    __half* d_qkv_out,
    const __half* d_qkv_weights,
    const size_t hidden_size,
    const size_t total_qkv_dim
){
    launch_matrix_vector_mul(d_qkv_weights, d_input_norm, d_qkv_out, total_qkv_dim, hidden_size);
}