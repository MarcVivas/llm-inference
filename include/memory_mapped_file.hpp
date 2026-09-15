#pragma once
#include "model_config.hpp"
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>

class MemoryMappedFile {
    public: 
        explicit MemoryMappedFile(const std::filesystem::path &path){

            // Check if path exists
            if(!std::filesystem::exists(path)){
                throw std::runtime_error("File does not exist: " + path.string());
            }

            // Get file size
            this->file_size = std::filesystem::file_size(path);
            if(this->file_size < sizeof(ModelConfig)){
                throw std::runtime_error("File is smaller than model header");
            }

            this->fd = ::open(path.c_str(), O_RDONLY);
            if(this->fd == -1){
                throw std::system_error(errno, std::generic_category(), "Failed to open file");
            }

            void* address = ::mmap(nullptr, this->file_size, PROT_READ, MAP_SHARED, this->fd, 0);
            if(address == MAP_FAILED){
                ::close(this->fd);
                throw std::system_error(errno, std::generic_category(), "Failed to mmap file");
            }

            ::madvise(address, this->file_size, MADV_WILLNEED);

            this->data = static_cast<const std::byte*>(address);
        }

        // Destructor
        ~MemoryMappedFile(){
            this->reset();
        }

        // RAII 
        
        MemoryMappedFile(const MemoryMappedFile&) = delete;
        MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
        
        MemoryMappedFile(MemoryMappedFile &&other) noexcept : data(other.data), file_size(other.file_size), fd(other.fd) {
            other.data = nullptr;
            other.file_size = 0;
            other.fd = -1;
        }

        MemoryMappedFile& operator=(MemoryMappedFile &&other) noexcept {
            if(this != &other){
                reset();
                this->data = other.data;
                this->fd = other.fd; 
                this->file_size = other.file_size; 

                other.data = nullptr;
                other.fd = -1;
                other.file_size = 0;
            }
            return *this;
        }

        std::span<const std::byte> bytes() const noexcept {
            return {this->data, this->file_size};
        }

        size_t size() const noexcept { return this->file_size;}
        
    private:
        size_t file_size = 0;
        const std::byte* data = nullptr;
        int fd = -1; 

        void reset() noexcept {
            if(this->data && this->data != MAP_FAILED){
                ::munmap(const_cast<std::byte*>(this->data), this->file_size);
                this->data = nullptr;
            }
            if(this->fd != -1){
                ::close(this->fd);
                this->fd = -1; 
            }
        }
        
};