#pragma once

#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace Gpu::Memory
{
    uint32_t static const DefaultAlign{8}; // 64-bit aligned

    template<typename T>
    T * mallocStd(uint32_t size)
    {
        void * memory = std::malloc(size);
        assert(memory != nullptr);
        return reinterpret_cast<T *>(memory);
    }

    template<typename T>
    T * mallocHost(uint32_t size)
    {
        void * memory = nullptr;
        cudaError_t status = cudaMallocHost(&memory, size);
        assert(status == cudaSuccess);
        assert(memory != nullptr);
        return reinterpret_cast<T *>(memory);
    }

    template<typename T>
    T * mallocDevice(uint32_t size)
    {
        void * memory = nullptr;
        cudaError_t status = cudaMalloc(&memory, size);
        assert(status == cudaSuccess);
        assert(memory != nullptr);
        return reinterpret_cast<T *>(memory);
    }

    template<typename T>
    T * mallocManaged(uint32_t size)
    {
        void * memory = nullptr;
        cudaError_t status = cudaMallocManaged(&memory, size);
        assert(status == cudaSuccess);
        assert(memory != nullptr);
        return reinterpret_cast<T *>(memory);
    }
}