#pragma once

#include "hip/hip_runtime.h"
#include <cstddef>
#include <stdexcept>
#include "gpu_utils.hpp"

template <typename T>
class PingPongBuffer{
    private:
        T* buffer_a = nullptr;
        T* buffer_b = nullptr;
        bool flipped = false; 
        size_t capacity_ = 0;
        
    public:
        
        explicit PingPongBuffer(const size_t num_elements){

            const size_t total_bytes = sizeof(T) * num_elements;

            hipError_t err = hipMalloc(&this->buffer_a, total_bytes);
            if(err != hipSuccess){
                throw std::runtime_error("Failed to allocate Buffer A: " + std::string(hipGetErrorString(err)));
            }

            err = hipMalloc(&this->buffer_b, total_bytes);
            if(err != hipSuccess){
                throw std::runtime_error("Failed to allocate Buffer B: " + std::string(hipGetErrorString(err)));
            }

            this->capacity_ = num_elements;
        }

        ~PingPongBuffer(){
            if(this->buffer_a){
                auto err = hipFree(this->buffer_a);
            }
            if(this->buffer_b){
                auto err = hipFree(this->buffer_b);
            }
        }
        PingPongBuffer() = default;


        // Swap roles: src becomes dst and dst becomes src
        void swap(){
            this->flipped = !this->flipped;
        }

        // The current input buffer
        const T* src(){
            return !flipped ? this->buffer_a : this->buffer_b; 
        }

        // The current output buffer
        const T* dst(){
            return flipped ? this->buffer_a : this->buffer_b; 
        }

        // Reset to initial state
        void reset() noexcept {
            this->flipped = false;
        }

        size_t capacity() {
            return this->capacity_;
        }

        size_t total_bytes(){
            return sizeof(T) * this->capacity_;
        }

        void fill_buffer(const T* dst, const T* src, const size_t total_bytes, hipMemcpyKind kind){
            auto err = hipMemcpy(dst, src, total_bytes, kind);
            if(err != hipSuccess){
                throw std::runtime_error("Failed to copy to Ping pong Buffer: " + std::string(hipGetErrorString(err)));
            }
        }

        // RAII

        // Disable copying
        // Prevents the object from being copied (e.g., PingPongBuffer b2 = b1; or b2 = b1; will cause a compile-time error).
        PingPongBuffer(const PingPongBuffer&) = delete;
        PingPongBuffer& operator=(const PingPongBuffer&) = delete;

        // Move constructor
        PingPongBuffer(PingPongBuffer&& o) noexcept
            : buffer_a(o.buffer_a), buffer_b(o.buffer_b), capacity_(o.capacity_), flipped(o.flipped) 
        {
            o.buffer_a = nullptr;
            o.buffer_b = nullptr;
            o.capacity_ = 0;
            o.flipped = false; 
        }

        // Move assignment operator
        PingPongBuffer& operator=(PingPongBuffer&& o) noexcept {
            if (this != &o) {
                if (buffer_a) HIP_CHECK(hipFree(buffer_a));
                if (buffer_b) HIP_CHECK(hipFree(buffer_b));
                buffer_a = o.buffer_a;
                buffer_b = o.buffer_b;
                capacity_ = o.capacity_;
                flipped = o.flipped;
                o.buffer_a = nullptr;
                o.buffer_b = nullptr;
                o.capacity_ = 0;
                o.flipped = false;
            }
            return *this;
        }

        
        
};