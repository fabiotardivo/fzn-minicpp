#include <libgpu/LinearAllocator.cuh>
#include <libgpu/Utils.cuh>
#include <libfca/Timer.hpp>
#include "bin_packing.cuh"

BinPackingLoadGPU::BinPackingLoadGPU(std::vector<var<int>::Ptr> const & load, std::vector<var<int>::Ptr> const & bin, std::vector<int> const & w) :
        BinPackingLoad(load, bin, w)
{
    using namespace Fca;
    using namespace Gpu::Memory;

    setPriority(CLOW);

    // Allocators
    allocator_h = new Gpu::LinearAllocator(mallocHost<void>(BP_INPUT_OUTPUT_MEMORY), BP_INPUT_OUTPUT_MEMORY);
    allocator_d = new Gpu::LinearAllocator(mallocDevice<void>(BP_INPUT_OUTPUT_MEMORY), BP_INPUT_OUTPUT_MEMORY);

    // CPU-GPU mirror data
    id_h = allocator_h->allocate<InstanceData>(sizeof(InstanceData));
    id_d = allocator_d->allocate<InstanceData>(sizeof(InstanceData));
    id_h->weightsBaseReduction_h = allocator_h->allocate<i32>(sizeof(i32) * (nItems + nBins));
    id_h->weightsBaseReduction_d = allocator_d->allocate<i32>(sizeof(i32) * (nItems + nBins));

    // Shared Memory
    shared_mem_size_lowerbounds = (sizeof(i32) * (nItems + nBins)) + DefaultAlign; // weights_s

    // CUDA
    for (int sIdx = 0; sIdx < REDUCTIONS_COUNT * DFFS_COUNT; sIdx += 1)
    {
        cudaStreamCreate(&cu_streams[sIdx]);
    }
}

void BinPackingLoadGPU::post()
{
    using namespace Fca;

    for (auto const & l : _load)
    {
        l->propagateOnBoundChange(this);
    }

    for (auto const & b : _bin)
    {
        b->propagateOnDomainChange(this);
    }

    propagate();
}

void BinPackingLoadGPU::propagate()
{

    using namespace Fca;
    using namespace Gpu::Utils::Parallel;

    calculateNumBins();
    saveBinLoadSizes();
    calculateRequiredPossibleCandidates();
    propagateLoadCoherence();
    calculateSumLoads();
    propagateBasicLoadTightening();
    propagateBasicItemEliminationCommitment();
    if (not checkBinLoadSizeChanges())
    {
        checkPackableBins();
        propagateKnapsackLoadTightening();
        propagateKnapsackItemEliminationCommitment();

        if (not checkBinLoadSizeChanges())
        {
            prepareReductions();
            checkLowerbounds();
        }
    }
}

__device__
Fca::i32 fMT_d(Fca::i32 w, Fca::i32 l, Fca::i32 c)
{
    using namespace Fca;

    // Branch-less f0 transformation

    // Conditions
    i32 const c0 = w > c - l;
    i32 const c1 = l <= w and w <= c - l;
    i32 const c2 = w < l;

    // Values
    i32 const v0 = c;
    i32 const v1 = w;
    i32 const v2 = 0;

    return (c0 * v0) + (c1 * v1) + (c2 * v2);
}

__device__
Fca::i32 fCCM1_d(Fca::i32 w, Fca::i32 l, Fca::i32 c)
{
    using namespace Fca;

    // Branch-less fCMM transformation

    // Conditions
    i32 const c0 = 2 * w > c;  // x > c / 2
    i32 const c1 = 2 * w == c; // x == c / 2
    i32 const c2 = 2 * w < c;  // x < c / 2

    // Values
    i32 const v0 = 2 * (c / l - (c - w) / l);
    i32 const v1 = c / l;
    i32 const v2 = 2 * (w / l);

    return (c0 * v0) + (c1 * v1) + (c2 * v2);
}

__device__
Fca::i32 fFS1_d(Fca::i32 w, Fca::i32 l, Fca::i32 c)
{
    using namespace Fca;

    // Branch-less fFS1 transformation

    // Conditions
    i32 const c0 = (w * (l + 1)) % c == 0;
    i32 const c1 = (w * (l + 1)) % c != 0;

    // Values
    i32 const v0 = w * l;
    i32 const v1 = ((w * (l + 1)) / c) * c;

    return (c0 * v0) + (c1 * v1);
}

__device__
Fca::i32 fBJ1_d(Fca::i32 w, Fca::i32 l, Fca::i32 c)
{
    using namespace Fca;

    // Branch-less fBJ1 transformation

    // Auxiliary values
    i32 const p = l - (c % l);

    // Conditions
    i32 const c0 = w % l <= c % l;
    i32 const c1 = w % l > c % l;

    // Values
    i32 const v0 = (w / l) * p;
    i32 const v1 = (w / l) * p + (w % l) - (c % l);

    return (c0 * v0) + (c1 * v1);
}

__device__
Fca::i32 fVB2Base_d(Fca::i32 w, Fca::i32 l, Fca::i32 c)
{
    using namespace Fca::Utils::Math;

    Fca::i32 v = ceilDivPosInt(l * w, c);
    return v > 0 ? v - 1 : 0;
}

__device__
Fca::i32 fVB2_d(Fca::i32 w, Fca::i32 l, Fca::i32 c)
{
    using namespace Fca::Utils::Math;

    Fca::i32 const c0 = 2 * w > c;
    Fca::i32 const c1 = 2 * w == c;
    Fca::i32 const c2 = 2 * w < c;

    Fca::i32 const t0 = fVB2Base_d(c, l, c);
    Fca::i32 const t1 = fVB2Base_d(w, l, c);
    Fca::i32 const t2 = fVB2Base_d(c - w, l, c);

    Fca::i32 const v0 = 2 * t0 - 2 * t2;
    Fca::i32 const v1 = t0;
    Fca::i32 const v2 = 2 * t1;

    return (c0 * v0) + (c1 * v1) + (c2 * v2);
}

__device__
Fca::i32 dfRAD2Base_d(Fca::i32 w, Fca::i32 l, Fca::i32 c)
{
    // Conditions
    Fca::i32 const c0 = w < l;
    Fca::i32 const c1 = l <= w and w <= c - 2 * l;
    Fca::i32 const c2 = c - 2 * l < w and w < 2 * l;

    // Values
    Fca::i32 const v0 = 0;
    Fca::i32 const v1 = c / 3;
    Fca::i32 const v2 = c / 2;

    return (c0 * v0) + (c1 * v1) + (c2 * v2);
}

__device__
Fca::i32 fRAD2_d(Fca::i32 w, Fca::i32 l, Fca::i32 c)
{
    // Conditions
    Fca::i32 const c0 = w < 2 * l;
    Fca::i32 const c1 = 2 * l <= w;

    assert(c0 + c1 == 1);

    // Values
    Fca::i32 const v0 = dfRAD2Base_d(w, l, c);
    Fca::i32 const v1 = c - dfRAD2Base_d(c - w, l, c);

    return (c0 * v0) + (c1 * v1);
}


void BinPackingLoadGPU::checkLowerbounds()
{
    using namespace Fca;
    using namespace Gpu::Utils::Parallel;

    using namespace Fca;

    // Prepare data to copy to GPU
    int nWeights = static_cast<i32>(weightsBaseReduction.size());
    id_h->lowerbound = 0;
    id_h->nBins = nBins;
    id_h->deltaReductions[0] = deltaReductions[0];
    id_h->deltaReductions[1] = deltaReductions[1];
    id_h->deltaReductions[2] = deltaReductions[2];
    id_h->nWeights = nWeights;
    id_h->capacityBaseReduction = capacityBaseReduction;
    std::copy(weightsBaseReduction.begin(), weightsBaseReduction.end(), id_h->weightsBaseReduction_h);

    // Copy data to GPU
    cudaMemcpy(allocator_d->getMemory(), allocator_h->getMemory(), allocator_d->getUsedMemorySize(), cudaMemcpyHostToDevice);

    // Launch kernels
    for (int rIdx = 0; rIdx < REDUCTIONS_COUNT; rIdx += 1)
    {
        // Calculate reduction parameters
        int const & delta = deltaReductions[rIdx];
        int const capacity = capacityBaseReduction + delta;

        // Analyze reduction
        int notZeroCount = 0;
        int maxWeight = 0;
        for (int wIdx = 0; wIdx < nWeights; wIdx += 1)
        {
            int weight = weightsBaseReduction[wIdx] + (wIdx < nBins ? delta : 0);
            notZeroCount += weight != 0;
            maxWeight = std::max(maxWeight, weight);
        }

        // fCCM1
        auto lRange = lCCM1(capacity);
        u32 nBlocks = std::max(1u, getBlocksCount(BP_BLOCK_SIZE, lRange.max - lRange.min + 1));
        calcDffLowerboundKernel<fCCM1_d><<<nBlocks, BP_BLOCK_SIZE, shared_mem_size_lowerbounds, cu_streams[rIdx * DFFS_COUNT + 0]>>>(id_d, delta, lRange.min, lRange.max);

        // fMT
        lRange = lMT(capacity);
        nBlocks = std::max(1u, getBlocksCount(BP_BLOCK_SIZE, lRange.max - lRange.min + 1));
        calcDffLowerboundKernel<fMT_d><<<nBlocks, BP_BLOCK_SIZE, shared_mem_size_lowerbounds, cu_streams[rIdx * DFFS_COUNT + 1]>>>(id_d, delta, lRange.min, lRange.max);

        // fBJ1
        lRange = lBJ1(capacity);
        nBlocks = std::max(1u, getBlocksCount(BP_BLOCK_SIZE, lRange.max - lRange.min + 1));
        calcDffLowerboundKernel<fBJ1_d><<<nBlocks, BP_BLOCK_SIZE, shared_mem_size_lowerbounds, cu_streams[rIdx * DFFS_COUNT + 2]>>>(id_d, delta, lRange.min, lRange.max);

        // fVB2
        lRange = sanitizeLambdaRange(lVB2(capacity), notZeroCount, maxWeight);
        nBlocks = std::max(1u, getBlocksCount(BP_BLOCK_SIZE, lRange.max - lRange.min + 1));
        calcDffLowerboundKernel<fVB2_d><<<nBlocks, BP_BLOCK_SIZE, shared_mem_size_lowerbounds, cu_streams[rIdx * DFFS_COUNT + 3]>>>(id_d, delta, lRange.min, lRange.max);

        // fFS1
        lRange = sanitizeLambdaRange(lFS1(capacity), notZeroCount, maxWeight);
        nBlocks = std::max(1u, getBlocksCount(BP_BLOCK_SIZE, lRange.max - lRange.min + 1));
        calcDffLowerboundKernel<fFS1_d><<<nBlocks, BP_BLOCK_SIZE, shared_mem_size_lowerbounds, cu_streams[rIdx * DFFS_COUNT + 4]>>>(id_d, delta, lRange.min, lRange.max);

        // fRAD2
        lRange = lRAD2(capacity);
        nBlocks = std::max(1u, getBlocksCount(BP_BLOCK_SIZE, lRange.max - lRange.min + 1));
        calcDffLowerboundKernel<fRAD2_d><<<nBlocks, BP_BLOCK_SIZE, shared_mem_size_lowerbounds, cu_streams[rIdx * DFFS_COUNT + 5]>>>(id_d, delta, lRange.min, lRange.max);
    }

    // Copy data from GPU
    cudaMemcpy(&id_h->lowerbound, &id_d->lowerbound, sizeof(Fca::i32), cudaMemcpyDefault);

    if (id_h->lowerbound > nBins)
    {
        failNow();
    }
}