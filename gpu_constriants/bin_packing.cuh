#pragma once

#include <libgpu/LinearAllocator.cuh>
#include "global_constraints/bin_packing.hpp"

#define BP_INPUT_OUTPUT_MEMORY (128 * 1024) // 128 KB
#define BP_BLOCK_SIZE 128

class BinPackingLoadGPU : public BinPackingLoad
{
    public:
        struct InstanceData
        {
            Fca::i32 lowerbound;
            Fca::i32 nBins;
            // Alternative Reductions
            Fca::i32 deltaReductions[BinPackingLoad::REDUCTIONS_COUNT];
            // Base Reduction
            Fca::i32 nWeights;
            Fca::i32 capacityBaseReduction;
            Fca::i32 * weightsBaseReduction_h;
            Fca::i32 * weightsBaseReduction_d;
        };

    private:
        // Allocators
        Gpu::LinearAllocator * allocator_h;
        Gpu::LinearAllocator * allocator_d;
        // CPU-GPU mirror data
        InstanceData * id_d;
        InstanceData * id_h;
        // Shared memory
        Fca::u32 shared_mem_size_lowerbounds;
        // CUDA
        cudaStream_t cu_streams[REDUCTIONS_COUNT * DFFS_COUNT];

    public:
        BinPackingLoadGPU(std::vector<var<int>::Ptr> const & load, std::vector<var<int>::Ptr> const & bin, std::vector<int> const & w);
        void post() override;
        void propagate() override;
    private:
        void checkLowerbounds();
};

// Dual-Feasible Functions
inline __device__ Fca::i32 fCCM1_d(Fca::i32 w, Fca::i32 l, Fca::i32 c);
inline __device__ Fca::i32 fMT_d(Fca::i32 w, Fca::i32 l, Fca::i32 c);
inline __device__ Fca::i32 fBJ1_d(Fca::i32 w, Fca::i32 l, Fca::i32 c);
inline __device__ Fca::i32 fVB2Base_d(Fca::i32 w, Fca::i32 l, Fca::i32 c);
inline __device__ Fca::i32 fVB2_d(Fca::i32 w, Fca::i32 l, Fca::i32 c);
inline __device__ Fca::i32 fFS1_d(Fca::i32 w, Fca::i32 l, Fca::i32 c);
inline __device__ Fca::i32 dfRAD2Base_d(Fca::i32 w, Fca::i32 l, Fca::i32 c);
inline __device__ Fca::i32 fRAD2_d(Fca::i32 w, Fca::i32 l, Fca::i32 c);

template<Fca::i32 f(Fca::i32, Fca::i32, Fca::i32)>
__device__ void calcDffLowerboundSingleLambda(Fca::i32 * lowerbound, Fca::i32 const * const weights, Fca::i32 nWeights, Fca::i32 capacity, Fca::i32 lambda)
{
#ifdef __NVCC__
    using namespace Fca;
    using namespace Fca::Utils::Math;
    using namespace Gpu;
    using namespace Gpu::Utils;

    // Transform and sum the weights
    i32 sumTransformedWeights = 0;
    for (i32 wIdx = 0; wIdx < nWeights; wIdx += 1)
    {
        sumTransformedWeights += f(weights[wIdx], lambda, capacity);
    }

    // Transform the capacity
    i32 const transformedCapacity = f(capacity, lambda, capacity);

    // Lowerbound
    *lowerbound = ceilDivPosInt(sumTransformedWeights, transformedCapacity);
#endif
}

template<Fca::i32 f(Fca::i32, Fca::i32, Fca::i32)>
__global__ void calcDffLowerboundKernel(BinPackingLoadGPU::InstanceData * id, Fca::i32 deltaReduction,  Fca::i32 minLambda,  Fca::i32 maxLambda)
{
#ifdef __NVCC__
    using namespace Fca;
    using namespace Fca::Utils::Math;
    using namespace Gpu;
    using namespace Gpu::Utils;

    extern __shared__ i32 weights_s[];
    __shared__ i32 lowerbound_s;

    // Prepare reduction
    i32 const capacity_r = id->capacityBaseReduction + deltaReduction;
    i32 const nWeights_r = id->nWeights;

    if (id->lowerbound <= id->nBins and capacity_r > 0)
    {

        // Calculate reduction in shared memory
        lowerbound_s = 0;
        for (u32 wIdx = threadIdx.x; wIdx < nWeights_r; wIdx += blockDim.x)
        {
            weights_s[wIdx] = id->weightsBaseReduction_d[wIdx] + (wIdx < id->nBins ? deltaReduction : 0);
        }
        __syncthreads();

        // Calculate lowerbounds
        i32 tIndex = blockIdx.x * blockDim.x + threadIdx.x;
        i32 lowerbound_r = 0;
        i32 const lambda = minLambda + tIndex;
        if (lambda <= maxLambda)
        {
            calcDffLowerboundSingleLambda<f>(&lowerbound_r, weights_s, nWeights_r, capacity_r, lambda);
        }

        // Calculate the biggest lowerbound for the current warp
        u32 const warpMask = __activemask();
        i32 const lowerbound_w = __reduce_max_sync(warpMask, lowerbound_r);

        // Update lowerbound in shared memory
        if (threadIdx.x % 32 == 0)
        {
            atomicMax_block(&lowerbound_s, lowerbound_w);
        }
    }
    __syncthreads();

    // Update the lowerbound in global memory
    if (threadIdx.x == 0)
    {
        atomicMax(&id->lowerbound, lowerbound_s);
    }
#endif
}