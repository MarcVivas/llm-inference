#include <doctest/doctest.h>
#include "kernels/kernels.hpp"
#include "gpu_utils.hpp"
#include <cmath>
#include <limits>
#include <vector>

TEST_CASE("Kernel: Matrix vector multiplication vs CPU reference") {
    struct Shape { size_t out_dim; size_t in_dim; };
    // Short reductions, partial blocks, multiple outputs, and real GateUp shape.
    const Shape shapes[] = {{1, 1}, {7, 31}, {129, 129}, {37, 2048},
                            {19, 11008}, {22016, 2048}};
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    for (const auto shape : shapes) {
        const auto [out_dim, in_dim] = shape;
        std::vector<__half> weights(out_dim * in_dim), input(in_dim);
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = __float2half(static_cast<float>(static_cast<int>(i % 29) - 14) / 64.0f);
        for (size_t i = 0; i < in_dim; ++i)
            input[i] = __float2half(static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f);

        __half *d_weights, *d_input, *d_output;
        HIP_CHECK(hipMalloc(&d_weights, weights.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(__half)));
        // Guards detect writes outside the output, including an excessive grid.
        HIP_CHECK(hipMalloc(&d_output, (out_dim + 2) * sizeof(__half)));
        HIP_CHECK(hipMemcpy(d_weights, weights.data(), weights.size() * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(__half), hipMemcpyHostToDevice));

        for (float beta : {0.0f, 1.0f, -0.5f}) {
            INFO("out_dim=" << out_dim << ", in_dim=" << in_dim << ", beta=" << beta);
            std::vector<__half> output(out_dim + 2, __float2half(123.0f));
            for (size_t row = 0; row < out_dim; ++row)
                output[row + 1] = __float2half(beta == 0.0f
                    ? std::numeric_limits<float>::quiet_NaN()
                    : static_cast<float>(static_cast<int>(row % 11) - 5) / 8.0f);
            const auto original = output;
            HIP_CHECK(hipMemcpyAsync(d_output, output.data(), output.size() * sizeof(__half), hipMemcpyHostToDevice, stream));
            launch_matrix_vector_mul(d_weights, d_input, d_output + 1, out_dim, in_dim, beta, stream);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipMemcpyAsync(output.data(), d_output, output.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            CHECK(__half2float(output.front()) == 123.0f);
            CHECK(__half2float(output.back()) == 123.0f);
            size_t mismatches = 0;
            for (size_t row = 0; row < out_dim; ++row) {
                double expected = 0.0;
                for (size_t col = 0; col < in_dim; ++col)
                    expected += static_cast<double>(__half2float(weights[row * in_dim + col]))
                              * __half2float(input[col]);
                if (beta != 0.0f) expected += beta * __half2float(original[row + 1]);
                const float actual = __half2float(output[row + 1]);
                if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-3 + 1e-3 * std::abs(expected))
                    ++mismatches;
            }
            CHECK(mismatches == 0);
        }
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_weights));
    }
    // Empty output must not launch a zero-sized grid or touch null pointers.
    launch_matrix_vector_mul(nullptr, nullptr, nullptr, 0, 2048);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamDestroy(stream));
}
