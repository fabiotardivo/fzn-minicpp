#pragma once

#include <cstdint>
#include "Memory.cuh"

namespace Gpu
{
    class LinearAllocator
    {
        private:
            uintptr_t const begin;
            uintptr_t current;
            uintptr_t const end;
        public:
            __host__ __device__ inline LinearAllocator(void * memory, uint32_t size);
            template<typename T>
            __host__ __device__ inline T * allocate(uint32_t size);
            __host__ __device__ inline void clear() {current = begin;};
            __host__ __device__ inline void * getMemory() const {return reinterpret_cast<void *>(begin);};
            __host__ __device__ inline void * getFreeMemory() const {return reinterpret_cast<void *>(current);};
            __host__ __device__ inline uint32_t getFreeMemorySize() const {return static_cast<uint32_t>(end - current);};
            __host__ __device__ inline uint32_t getUsedMemorySize() const {return static_cast<uint32_t>(current - begin);};
            __host__ __device__ inline uint32_t getTotalMemorySize() const {return static_cast<uint32_t>(end - begin);};
    };

    __host__ __device__
    LinearAllocator::LinearAllocator(void * memory, uint32_t size) :
            begin(reinterpret_cast<uintptr_t>(memory)), current(begin), end(begin + static_cast<uintptr_t>(size))
    {
        assert(begin < end);
    }

    template<typename T>
    __host__ __device__
    T * LinearAllocator::allocate(uint32_t size)
    {
        uintptr_t memory = current;
        uint32_t offset = memory % Memory::DefaultAlign;
        if (offset != 0)
        {
            memory += Memory::DefaultAlign - offset;
        }
        current = memory + size;
        assert(current < end);
        return reinterpret_cast<T *>(memory);
    }
}
