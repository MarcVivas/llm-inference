#pragma once
#include "gpu_utils.hpp"
#include <hip/hip_runtime.h>

class GpuTimer {
    hipEvent_t start_ = nullptr; 
    hipEvent_t end_ = nullptr; 

    public:
        GpuTimer() {
            HIP_CHECK(hipEventCreate(&start_));
            HIP_CHECK(hipEventCreate(&end_));
        }

        ~GpuTimer(){
            if(start_) HIP_CHECK(hipEventDestroy(start_));
            if(end_) HIP_CHECK(hipEventDestroy(end_));
        }

        void start(){
            HIP_CHECK(hipEventRecord(start_, nullptr));
        }

        // Records stop, waits the GPU to finish, and returns the elapsed time in milliseconds
        float stop_and_sync(){
            HIP_CHECK(hipEventRecord(end_, nullptr));
            HIP_CHECK(hipEventSynchronize(end_));

            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, start_, end_));
            return ms;
        }
        
};