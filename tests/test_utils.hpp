#pragma once

#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <span>
#include <cmath>
#include <print>
#include <hip/hip_fp16.h>

namespace test_utils {

// Load flat binary fixtures dumped by reference.py
template <typename T>
std::vector<T> load_binary_file(const std::filesystem::path& filepath) {
    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("Fixture not found: " + filepath.string());
    }

    size_t file_bytes = std::filesystem::file_size(filepath);
    std::vector<T> buffer(file_bytes / sizeof(T));

    std::ifstream file(filepath, std::ios::binary);
    file.read(reinterpret_cast<char*>(buffer.data()), file_bytes);
    return buffer;
}

// Compare two FP16 spans using doctest assertions
inline void check_tensor_close(
    std::span<const __half> actual, 
    std::span<const __half> expected, 
    float atol = 1e-2f, 
    float rtol = 1e-2f
) {
    REQUIRE(actual.size() == expected.size());

    size_t mismatches = 0;
    float max_diff = 0.0f;

    for (size_t i = 0; i < actual.size(); ++i) {
        float a = __half2float(actual[i]);
        float e = __half2float(expected[i]);
        float diff = std::fabs(a - e);

        if (diff > max_diff) {
            max_diff = diff;
        }

        float allowed = atol + rtol * std::fabs(e);
        if (diff > allowed) {
            if (mismatches < 3) { // Print first 3 errors
                std::println("Mismatch at [{}]: actual={:.5f}, expected={:.5f}, diff={:.5f}", i, a, e, diff);
            }
            mismatches++;
        }
    }

    INFO("Max difference: ", max_diff);
    INFO("Mismatched elements: ", mismatches);
    CHECK(mismatches == 0);
}

} // namespace test_utils