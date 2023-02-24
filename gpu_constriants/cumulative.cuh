#pragma once

#include <tuple>
#include <libfca/Types.hpp>
#include <libgpu/Memory.cuh>
#include <libgpu/LinearAllocator.cuh>
#include <libminicpp/varitf.hpp>
#include <libminicpp/constraint.hpp>
#include "global_constraints/cumulative.hpp"

#define MAX_INTERVALS_PER_ACTIVITY_PAIR 12
#define CUMULATIVE_BLOCK_SIZE 128
#define MAX_ACTIVITIES 1024
#define INPUT_OUTPUT_MEMORY 128 * 1024 // 128 KB

// References:
// - Constraint-Based Scheduling (ISBN: 978-1-4615-1479-4)
// - A New Characterization of Relevant Intervals for Energetic Reasoning (DOI: 10.1007/978-3-319-10428-7_22)

class CumulativeGPU : public Cumulative
{
   private:
        Fca::i32 * p_d; // Processing time
        Fca::i32 * h_d; // Height
        Fca::i32 * nIntervals_d;
        Interval * i_d;
        bool * isConsistent_h;
        StartInterval * si_h;
        bool * isConsistent_d;
        StartInterval * si_d;
        Gpu::LinearAllocator * allocator_h;
        Gpu::LinearAllocator * allocator_d;

        // CUDA
        Fca::u32 sm_count;
        cudaStream_t cu_stream;
        cudaGraph_t cu_graph;
        cudaGraphExec_t propagate_low_latency;

    public:
        CumulativeGPU(std::vector<var<int>::Ptr> & s, std::vector<int> const & p, std::vector<int> const & h, int c);
        void post() override;
        void propagate() override;

    private:
        void initPropagateLowLatency();
        void propagateBase();
};

__global__ void resetIntervalsKernel(Fca::i32 * nIntervals_d);
__global__ void calcIntervalsKernel(Fca::i32 nActivities, Cumulative::StartInterval const * si_d, Fca::i32 const * p_d, Fca::i32 * nIntervals_d, Cumulative::Interval * i_d);
__global__ void resetConsistencyKernel(bool * isConsistent_d);
__global__ void updateBoundsKernel(Fca::i32 nActivities, Fca::i32 const * h_d, Fca::i32 const * p_d, Fca::i32 c, Fca::i32 * nIntervals_d, Cumulative::Interval const * i_d, Cumulative::StartInterval * si_d, bool * isConsistent_d);
