#include "gpu_utils.hpp"
#include "kernels/kernels.hpp"
#include <rocblas/internal/rocblas-types.h>
#include <stdexcept>


// Compute gate and up matrices
void launch_gate_up_gemm(
    const __half* d_input_norm,      // Input:  [seq_len, hidden_size]
    __half* d_gate_up_out,     // Output: [seq_len, 2 * intermediate_size]
    const __half* d_weight_gate_up,       // Weight: [2 * intermediate_size, hidden_size]
    size_t seq_len,
    size_t hidden_size,
    size_t intermediate_size
){
    rocblas_handle handle = get_rocblas_handle();

    const float alpha = 1.f;
    const float beta = 0.f;

    // X_input_norm[seq_len x hidden_size] x W_T[hidden_size x 2 * intermediate_size] = OUT[seq_len x 2 * intermediate_size] 
    // Since rocblas uses col-major we want Y_T:
    // (X*W_T)T = Y_T 
    // W_T_T X_T = Y_T
    // W[2*intermediate_size x hidden_size] x X_T[hidden_size x seq_len] = Y_T[2*intermediate_size, seq_len]
    
    
    // Output dimension
    const rocblas_int m = static_cast<rocblas_int>(2 * intermediate_size);

    // Batch dimension 
    const rocblas_int n = static_cast<rocblas_int>(seq_len);

    // Reduction/common dimension
    // This dim is not in the output
    const rocblas_int k = static_cast<rocblas_int>(hidden_size);

    // Leading dimensions
    // Strides in memory: In a flat 1D buffer, how many elements do you have to skip to jump to the next row. 
    const rocblas_int lda = static_cast<rocblas_int>(hidden_size);
    const rocblas_int ldb = static_cast<rocblas_int>(hidden_size);
    const rocblas_int ldc = static_cast<rocblas_int>(2 * intermediate_size);

    rocblas_status status = rocblas_gemm_ex(
        handle, 
        rocblas_operation_transpose, // Transpose W: [k, m] -> [m, k]
        rocblas_operation_none, // Keep X: [k, n]
        m, n, k,
        &alpha, 
        d_weight_gate_up, rocblas_datatype_f16_r, lda, 
        d_input_norm, rocblas_datatype_f16_r, ldb, 
        &beta, 
        d_gate_up_out, rocblas_datatype_f16_r, ldc,
        d_gate_up_out, rocblas_datatype_f16_r, ldc, // Output
        rocblas_datatype_f32_r, 
        rocblas_gemm_algo_standard,
        0, 0
    );

    if(status != rocblas_status_success){
        throw std::runtime_error("rocblas_gemm_ex failed for gate_up");
    }
}