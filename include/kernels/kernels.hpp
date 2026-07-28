#pragma once
#include <hip/hip_runtime.h>

// Vector Addition Kernel (Dummy test kernel to verify ROCm execution)
void launch_vector_add(const float* d_a, const float* d_b, float* d_c, int n);
