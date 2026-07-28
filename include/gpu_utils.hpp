#pragma once
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

// Macro to wrap all HIP API calls for automatic error checking
#define HIP_CHECK(command)                                                      \
    do {                                                                        \
        hipError_t status = (command);                                          \
        if (status != hipSuccess) {                                             \
            std::cerr << "[HIP Error] " << hipGetErrorString(status)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
