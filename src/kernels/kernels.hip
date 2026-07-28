#include "kernels.hpp"
#include "gpu_utils.hpp"

// CUDA/HIP Kernel executing on GPU threads
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// C++ Host launcher function
void launch_vector_add(const float* d_a, const float* d_b, float* d_c, int n) {
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(vector_add_kernel, dim3(blocks), dim3(threads_per_block), 0, 0, d_a, d_b, d_c, n);
    HIP_CHECK(hipGetLastError());
}
