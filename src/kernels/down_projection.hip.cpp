#include "gpu_utils.hpp"
#include "kernels/kernels.hpp"
#include <rocblas/internal/rocblas-types.h>
#include <stdexcept>



void launch_down_proj_gemm(
    const __half* d_in, // [seq_len, intermediate_size]
    __half* d_out,  // [seq_len, hidden_size]
    const __half* d_weights_down, // [hidden_size, intermediate_size] in pytorch is transposed. This is the actual: [intermediate_size, hidden_size]
    size_t seq_len,
    size_t hidden_size,
    size_t intermediate_size,
    float beta    
){
    rocblas_handle handle = get_rocblas_handle();

    const float alpha = 1.f;

    // d_in[seq_len x intermediate_size] x d_weights_down[intermediate_size x hidden_size] = d_out[seq_len x hidden_size] 

    // Output dimension (hidden_size from d_weights_down)
    const rocblas_int m = static_cast<rocblas_int>(hidden_size);

    // Batch dimension (seq_len from d_in)
    const rocblas_int n = static_cast<rocblas_int>(seq_len);

    // Reduction/common dimension (intermediate_size)
    // This dim is not in the output
    const rocblas_int k = static_cast<rocblas_int>(intermediate_size);

    // Leading dimensions
    // Strides in memory: In a flat 1D buffer, how many elements do you have to skip to jump to the next row. 
    const rocblas_int lda = intermediate_size;
    const rocblas_int ldb = intermediate_size;
    const rocblas_int ldc = hidden_size;

    rocblas_status status = rocblas_gemm_ex(
        handle, 
        rocblas_operation_transpose, // Transpose W: [k, m] -> [m, k]
        rocblas_operation_none, // Keep X: [k, n]
        m, n, k,
        &alpha, 
        d_weights_down, rocblas_datatype_f16_r, lda, 
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