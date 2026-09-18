#pragma once
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <rocblas/rocblas.h>
#include <print>
// Macro to wrap all HIP API calls for automatic error checking
#define HIP_CHECK(command)                                                      \
    do {                                                                        \
        hipError_t status = (command);                                          \
        if (status != hipSuccess) {                                             \
            std::println(stderr, "[HIP Error] {} at {}:{}", hipGetErrorString(status), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)


inline rocblas_handle get_rocblas_handle() {
    struct RocblasContext {
        rocblas_handle handle = nullptr;

        RocblasContext() {
            rocblas_status status = rocblas_create_handle(&handle);
            if (status != rocblas_status_success) {
                throw std::runtime_error("Failed to initialize rocBLAS handle");
            }
        }

        ~RocblasContext() {
            if (handle) {
                rocblas_destroy_handle(handle);
            }
        }

        // Prevent copying / moving
        RocblasContext(const RocblasContext&) = delete;
        RocblasContext& operator=(const RocblasContext&) = delete;
    };

    // Initialized once on first call
    static RocblasContext instance;
    return instance.handle;
}