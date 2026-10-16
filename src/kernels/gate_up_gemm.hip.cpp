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

void launch_gate_up_decode(
    const __half* d_input_norm,      // Input:  [seq_len, hidden_size]
    __half* d_gate_up_out,     // Output: [seq_len, 2 * intermediate_size]
    const __half* d_weight_gate_up,       // Weight: [2 * intermediate_size, hidden_size]
    size_t hidden_size,
    size_t intermediate_size,
    hipStream_t stream
){
    launch_matrix_vector_mul(d_weight_gate_up, d_input_norm, d_gate_up_out, 2 * intermediate_size, hidden_size, 0.0f, stream);
}

// ---------------------------------------------------------------------------
// Fused Gate & Up GEMV with SwiGLU Activation (Decode)
// 1 block computes gate row i and up row (i + intermediate_size), evaluates
// silu(gate) * up in registers, and directly writes the intermediate_size output.
// ---------------------------------------------------------------------------

#define BLOCK_SIZE_SWIGLU 256
#define WARP_SIZE_SWIGLU 32

__device__ inline float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ inline void block_reduction_2pair(
    float val0, float val1,
    float* out0, float* out1,
    float* warp_sums0, float* warp_sums1
) {
    const uint warp_id = threadIdx.x / warpSize;
    const uint lane_id = threadIdx.x % warpSize;

    #pragma unroll
    for (uint offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val0 += __shfl_down(val0, offset);
        val1 += __shfl_down(val1, offset);
    }
    if (lane_id == 0) {
        warp_sums0[warp_id] = val0;
        warp_sums1[warp_id] = val1;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b0 = (lane_id < (blockDim.x / warpSize)) ? warp_sums0[lane_id] : 0.0f;
        float b1 = (lane_id < (blockDim.x / warpSize)) ? warp_sums1[lane_id] : 0.0f;
        #pragma unroll
        for (uint offset = warpSize >> 1; offset > 0; offset >>= 1) {
            b0 += __shfl_down(b0, offset);
            b1 += __shfl_down(b1, offset);
        }
        if (lane_id == 0) {
            *out0 = b0;
            *out1 = b1;
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE_SWIGLU) void fused_gate_up_swiglu_decode_kernel(
    const __half* __restrict__ d_weight,
    const __half* __restrict__ d_input,
    __half* __restrict__ d_output,
    uint intermediate_size,
    uint hidden_size
) {
    constexpr uint VECTOR_ELEMENTS = sizeof(float4) / sizeof(__half);
    const uint row_gate = blockIdx.x;
    if (row_gate >= intermediate_size) return;

    const uint row_up = row_gate + intermediate_size;
    const __half* __restrict__ wg = d_weight + static_cast<size_t>(row_gate) * hidden_size;
    const __half* __restrict__ wu = d_weight + static_cast<size_t>(row_up) * hidden_size;

    const uint vectors = hidden_size / VECTOR_ELEMENTS;
    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    for (uint i = threadIdx.x; i < vectors; i += blockDim.x) {
        float4 x_raw = reinterpret_cast<const float4*>(d_input)[i];
        float4 wg_raw = reinterpret_cast<const float4*>(wg)[i];
        float4 wu_raw = reinterpret_cast<const float4*>(wu)[i];

        const __half* x = reinterpret_cast<const __half*>(&x_raw);
        const __half* w_g = reinterpret_cast<const __half*>(&wg_raw);
        const __half* w_u = reinterpret_cast<const __half*>(&wu_raw);

        #pragma unroll
        for (uint j = 0; j < VECTOR_ELEMENTS; ++j) {
            float xf = __half2float(x[j]);
            sum_gate += __half2float(w_g[j]) * xf;
            sum_up   += __half2float(w_u[j]) * xf;
        }
    }

    for (uint col = vectors * VECTOR_ELEMENTS + threadIdx.x; col < hidden_size; col += blockDim.x) {
        float xf = __half2float(d_input[col]);
        sum_gate += __half2float(wg[col]) * xf;
        sum_up   += __half2float(wu[col]) * xf;
    }

    __shared__ float warp_sums_g[BLOCK_SIZE_SWIGLU / WARP_SIZE_SWIGLU];
    __shared__ float warp_sums_u[BLOCK_SIZE_SWIGLU / WARP_SIZE_SWIGLU];
    __shared__ float final_gate, final_up;

    block_reduction_2pair(sum_gate, sum_up, &final_gate, &final_up, warp_sums_g, warp_sums_u);

    if (threadIdx.x == 0) {
        d_output[row_gate] = __float2half(silu_f(final_gate) * final_up);
    }
}

void launch_gate_up_swiglu_decode(
    const __half* d_input_norm,      // Input:  [hidden_size]
    __half* d_swiglu_out,            // Output: [intermediate_size]
    const __half* d_weight_gate_up,  // Weight: [2 * intermediate_size, hidden_size]
    size_t hidden_size,
    size_t intermediate_size,
    hipStream_t stream
) {
    if (intermediate_size == 0 || hidden_size == 0) return;

    const dim3 grid(static_cast<uint>(intermediate_size));
    const dim3 block(BLOCK_SIZE_SWIGLU);

    fused_gate_up_swiglu_decode_kernel<<<grid, block, 0, stream>>>(
        d_weight_gate_up, d_input_norm, d_swiglu_out,
        static_cast<uint>(intermediate_size),
        static_cast<uint>(hidden_size)
    );
}
