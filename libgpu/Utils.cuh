#pragma once

#include <cstdint>

namespace Gpu::Utils
{
    namespace Parallel
    {
        __device__ inline
        void getBeginEnd(uint32_t * begin, uint32_t * end, uint32_t index, uint32_t workers, uint32_t jobs)
        {
            uint32_t const jobsPerWorker = (jobs + workers - 1) / workers; // Fast ceil integer division
            *begin = jobsPerWorker * index;
            *end = min(jobs, *begin + jobsPerWorker);
        }

        __host__ inline
        uint32_t getBlocksCount(uint32_t block_size, uint32_t jobs)
        {
            double blocks_count = static_cast<double>(jobs) / static_cast<double>(block_size);
            blocks_count = ceil(blocks_count);
            return static_cast<uint32_t>(blocks_count);
        }
    }
}