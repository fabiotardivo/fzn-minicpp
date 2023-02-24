#pragma once

#include <cstdint>

namespace Gpu::Utils
{
    namespace Parallel
    {
        __device__
        void getBeginEnd(uint32_t * begin, uint32_t * end, uint32_t index, uint32_t workers, uint32_t jobs)
        {
            uint32_t const jobsPerWorker = (jobs + workers - 1) / workers; // Fast ceil integer division
            *begin = jobsPerWorker * index;
            *end = min(jobs, *begin + jobsPerWorker);
        }
    }
}