#include "gpu_utils.hpp"
#include "kernels/kernels.hpp"
#include <rocblas/internal/rocblas-types.h>
#include <stdexcept>



void launch_attention_out_projection(
    const __half* d_in, //[seq_len, num_heads, head_dim]
    __half* d_out,  //[seq_len, hidden_size] During inference put here the pre attention matrix
    const __half* d_o_proj_weights, // [hidden_size, num_heads * head_dim] (Pytorch [out, in])
    size_t seq_len,
    size_t hidden_size,
    size_t num_heads,
    size_t head_dim,
    float beta   // 0 for unit test, 1 for fused residual add
){
    rocblas_handle handle = get_rocblas_handle();

    const float alpha = 1.f;

    // X_attention[seq_len x attention_dim] x W_T[attention_dim x hidden_size] = OUT[seq_len x hidden_size] 

    // Output dimension (hidden_size from W_T)
    const rocblas_int m = static_cast<rocblas_int>(hidden_size);

    // Batch dimension (seq_len from X_attention)
    const rocblas_int n = static_cast<rocblas_int>(seq_len);

    // Reduction/common dimension (attention_dim = num_heads * head_dim)
    // This dim is not in the output
    const rocblas_int k = static_cast<rocblas_int>(num_heads * head_dim);

    // Leading dimensions
    // Strides in memory: In a flat 1D buffer, how many elements do you have to skip to jump to the next row. 
    const rocblas_int lda = hidden_size;
    const rocblas_int ldb = num_heads * head_dim;
    const rocblas_int ldc = hidden_size;

    rocblas_status status = rocblas_gemm_ex(
        handle, 
        rocblas_operation_transpose, // Transpose W: [k, m] -> [m, k]
        rocblas_operation_none, // Keep X: [k, n]
        m, n, k,
        &alpha, 
        d_o_proj_weights, rocblas_datatype_f16_r, lda, 
        d_in, rocblas_datatype_f16_r, ldb, 
        &beta, // Residual add if beta == 1.f
        d_out, rocblas_datatype_f16_r, ldc, // Add residual/shortcut connection. Remember: alpha*(A*B) + beta*C 
        d_out, rocblas_datatype_f16_r, ldc, // Output
        rocblas_datatype_f32_r, 
        rocblas_gemm_algo_standard,
        0, 0
    );

    if(status != rocblas_status_success){
        throw std::runtime_error("rocblas_gemm_ex failed for o_proj");
    }
}