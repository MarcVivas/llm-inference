#include "gpu_utils.hpp"
#include "kernels/kernels.hpp"
#include <rocblas/internal/rocblas-types.h>
#include <stdexcept>



void launch_lm_head_gemm(
    const __half* d_final_norm_out, // [seq_len, hidden_size]
    __half*       d_logits_out, // [seq_len, vocab_size]
    const __half* d_weights_lm_head, // Pytorch: [vocab_size, hidden_size] Actual: [hidden_size, vocab_size]
    size_t seq_len,
    size_t hidden_size,
    size_t vocab_size
){
    rocblas_handle handle = get_rocblas_handle();

    const float alpha = 1.f;
    const float beta = 0.f; 
    
    // d_final_norm_out[seq_len x hidden_size] x d_weights_lm_head[hidden_size x vocab_size] = d_logits_out[seq_len x vocab_size] 

    // Output dimension (vocab_size from d_weights_lm_head)
    const rocblas_int m = static_cast<rocblas_int>(vocab_size);

    // Batch dimension (seq_len from d_final_norm_out)
    const rocblas_int n = static_cast<rocblas_int>(seq_len);

    // Reduction/common dimension (hidden_size)
    // This dim is not in the output
    const rocblas_int k = static_cast<rocblas_int>(hidden_size);

    // Leading dimensions
    // Strides in memory: In a flat 1D buffer, how many elements do you have to skip to jump to the next row. 
    const rocblas_int lda = hidden_size;
    const rocblas_int ldb = hidden_size;
    const rocblas_int ldc = vocab_size;

    rocblas_status status = rocblas_gemm_ex(
        handle, 
        rocblas_operation_transpose, // Transpose W: [k, m] -> [m, k]
        rocblas_operation_none, // Keep X: [k, n]
        m, n, k,
        &alpha, 
        d_weights_lm_head, rocblas_datatype_f16_r, lda, 
        d_final_norm_out, rocblas_datatype_f16_r, ldb, 
        &beta, // Residual add if beta == 1.f
        d_logits_out, rocblas_datatype_f16_r, ldc, // Add residual/shortcut connection. Remember: alpha*(A*B) + beta*C 
        d_logits_out, rocblas_datatype_f16_r, ldc, // Output
        rocblas_datatype_f32_r, 
        rocblas_gemm_algo_standard,
        0, 0
    );

    if(status != rocblas_status_success){
        throw std::runtime_error("rocblas_gemm_ex failed for lm head gemm");
    }
}

void launch_lm_head_decode(
    const __half* d_final_norm_out, // [seq_len, hidden_size]
    __half*       d_logits_out, // [seq_len, vocab_size]
    const __half* d_weights_lm_head, // Pytorch: [vocab_size, hidden_size] Actual: [hidden_size, vocab_size]
    size_t hidden_size,
    size_t vocab_size
){
    launch_matrix_vector_mul(d_weights_lm_head, d_final_norm_out, d_logits_out, vocab_size, hidden_size);
}