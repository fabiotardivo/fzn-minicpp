#include <libfca/Array.hpp>
#include <libfca/Utils.hpp>
#include <libgpu/Utils.cuh>
#include "gpu_constriants/cumulative.cuh"

CumulativeGPU::CumulativeGPU(std::vector<var<int>::Ptr> & s, std::vector<int> const & p, std::vector<int> const & h, int c) :
        Cumulative(s,p,h,c)
{
    using namespace std;
    using namespace Fca;
    using namespace Gpu::Memory;

    //Constraints
    setPriority(CLOW);

    // Memory allocation
    p_d = mallocDevice<i32>(Array<i32>::getDataSize(nActivities));
    h_d = mallocDevice<i32>(Array<i32>::getDataSize(nActivities));
    nIntervals_d = mallocDevice<i32>(sizeof(i32));
    i_d = mallocDevice<Interval>(Array<Interval>::getDataSize(MAX_INTERVALS_PER_ACTIVITY_PAIR * nActivities * nActivities));
    allocator_h = new Gpu::LinearAllocator(mallocHost<void>(INPUT_OUTPUT_MEMORY), INPUT_OUTPUT_MEMORY);
    allocator_d = new Gpu::LinearAllocator(mallocDevice<void>(INPUT_OUTPUT_MEMORY), INPUT_OUTPUT_MEMORY);
    isConsistent_h = allocator_h->allocate<bool>(sizeof(bool));
    si_h = allocator_h->allocate<StartInterval>(Array<StartInterval>::getDataSize(nActivities));
    isConsistent_d = allocator_d->allocate<bool>(sizeof(bool));
    si_d = allocator_d->allocate<StartInterval>(Array<StartInterval>::getDataSize(nActivities));

    // CUDA initialization
    cudaDeviceProp cu_prop;
    cudaGetDeviceProperties(&cu_prop, 0);
    sm_count = cu_prop.multiProcessorCount;
    cudaStreamCreate(&cu_stream);
    initPropagateLowLatency();
}

void CumulativeGPU::post()
{

    using namespace std;
    using namespace Fca;

    for (auto const & v : s)
    {
        v->propagateOnBoundChange(this);
    }

    // Copy constants data on GPU
    cudaMemcpyAsync(p_d, p.data(), Array<i32>::getDataSize(nActivities), cudaMemcpyDefault, cu_stream);
    cudaMemcpyAsync(h_d, h.data(), Array<i32>::getDataSize(nActivities), cudaMemcpyDefault, cu_stream);
}

void CumulativeGPU::propagate()
{
    // Initialization
    initStartIntervals(nActivities, s.data(), si_h);

    // Propagation
    //propagateBase();
    cudaGraphLaunch(propagate_low_latency, cu_stream);
    cudaStreamSynchronize(cu_stream);

    // Filtering
    if (*isConsistent_h)
    {
        for (auto i = 0; i < nActivities; i += 1)
        {
           s.at(i)->removeBelow(si_h[i].min);
           s.at(i)->removeAbove(si_h[i].max);
        }
    }
    else
    {
        failNow();
    }
}

void CumulativeGPU::propagateBase()
{
    using namespace Fca;
    cudaMemcpyAsync(si_d, si_h, Array<StartInterval>::getDataSize(s.size()), cudaMemcpyDefault, cu_stream);
    resetIntervalsKernel<<<1,1,0,cu_stream>>>(nIntervals_d);
    calcIntervalsKernel<<<sm_count, CUMULATIVE_BLOCK_SIZE, 0, cu_stream>>>(nActivities, si_d, p_d, nIntervals_d, i_d);
    resetConsistencyKernel<<<1,1,0,cu_stream>>>(isConsistent_d);
    updateBoundsKernel<<<sm_count, CUMULATIVE_BLOCK_SIZE, 0, cu_stream>>>(nActivities, h_d, p_d, c, nIntervals_d, i_d, si_d, isConsistent_d);
    cudaMemcpyAsync(allocator_h->getMemory(), allocator_d->getMemory(), allocator_h->getUsedMemorySize(), cudaMemcpyDefault, cu_stream);
}

void CumulativeGPU::initPropagateLowLatency()
{
    cudaStreamBeginCapture(cu_stream, cudaStreamCaptureModeGlobal);
    propagateBase();
    cudaStreamEndCapture(cu_stream, &cu_graph);
    cudaGraphInstantiate(&propagate_low_latency, cu_graph, nullptr, nullptr, 0);
}

__global__
void resetIntervalsKernel(Fca::i32 * nIntervals_d)
{
    *nIntervals_d = 0;
}

__global__
void calcIntervalsKernel(Fca::i32 nActivities, Cumulative::StartInterval const * si_d, Fca::i32 const * p_d, Fca::i32 * nIntervals_d, Cumulative::Interval * i_d)
{
    using namespace Fca;
    using namespace Gpu::Utils::Parallel;

    u32 ijBegin, ijEnd;
    getBeginEnd(&ijBegin, &ijEnd, blockIdx.x, gridDim.x, nActivities * nActivities);
    for (u32 ijIdx = ijBegin + threadIdx.x; ijIdx < ijEnd; ijIdx += blockDim.x)
    {
        u32 const i = ijIdx / nActivities;
        u32 const j = ijIdx % nActivities;
        u32 nGoodIntervals = 0;
        Cumulative::Interval intervalsToTest[MAX_INTERVALS_PER_ACTIVITY_PAIR];
        //if (s_d[i].changed or s_d[j].changed)
        {
            i32 const pi = p_d[i];
            i32 const siMin = si_d[i].min;
            i32 const siMax = si_d[i].max;
            i32 const eiMax = siMax + pi;

            i32 const pj = p_d[j];
            i32 const sjMin = si_d[j].min;
            i32 const sjMax = si_d[j].max;
            i32 const ejMin = sjMin + pj;
            i32 const ejMax = sjMax + pj;

            // Case 1
            intervalsToTest[0] = {siMin, ejMin};
            intervalsToTest[1] = {siMin, ejMax};
            intervalsToTest[2] = {siMax, ejMin};
            intervalsToTest[3] = {siMax, ejMax};

            // Case 2
            intervalsToTest[4] = {siMin, sjMin + ejMax - siMin};
            intervalsToTest[5] = {siMin, sjMin + ejMax - siMax};
            intervalsToTest[6] = {siMax, sjMin + ejMax - siMin};
            intervalsToTest[7] = {siMax, sjMin + ejMax - siMax};

            // Case 3
            intervalsToTest[8] = {siMin + eiMax - ejMin, ejMin};
            intervalsToTest[9] = {siMin + eiMax - ejMin, ejMax};
            intervalsToTest[10] = {siMin + eiMax - ejMax, ejMin};
            intervalsToTest[11] = {siMin + eiMax - ejMax, ejMax};

            for (u32 k = 0; k < MAX_INTERVALS_PER_ACTIVITY_PAIR; k += 1)
            {
                if (intervalsToTest[k].t1 < intervalsToTest[k].t2)
                {
                    intervalsToTest[nGoodIntervals] = intervalsToTest[k];
                    nGoodIntervals += 1;
                }
            }
        }

        if (nGoodIntervals > 0)
        {
            u32 const freeIntervalIdx = atomicAdd(nIntervals_d, nGoodIntervals);
            for (u32 k = 0; k < nGoodIntervals; k += 1)
            {
                i_d[freeIntervalIdx + k] = intervalsToTest[k];
            }
        }
    }
}

__global__
void resetConsistencyKernel(bool * isConsistent_d)
{
    *isConsistent_d = true;
}

__global__
void updateBoundsKernel(Fca::i32 nActivities, Fca::i32 const * h_d, Fca::i32 const * p_d, Fca::i32 c, Fca::i32 * nIntervals_d, Cumulative::Interval const * i_d, Cumulative::StartInterval * si_d, bool * isConsistent_d)
{
    using namespace Fca;
    using namespace Gpu::Memory;
    using namespace Gpu::Utils::Parallel;

    __shared__ Cumulative::StartInterval si_s[MAX_ACTIVITIES];

    if (*isConsistent_d)
    {
        for (auto a = threadIdx.x; a < nActivities; a += blockDim.x)
        {
            si_s[a] = si_d[a];
        }
        __syncthreads();

        u32 iBegin, iEnd;
        getBeginEnd(&iBegin, &iEnd, blockIdx.x, gridDim.x, *nIntervals_d);
        for (auto i = iBegin + threadIdx.x; i < iEnd; i += blockDim.x)
        {
            i32 const t1 = i_d[i].t1;
            i32 const t2 = i_d[i].t2;

            i32 w = 0;
            for (i32 a = 0; a < nActivities; a += 1)
            {
                i32 const ha = h_d[a];
                i32 const saMin = si_s[a].min;
                i32 const saMax = si_s[a].max;
                i32 const pa = p_d[a];
                i32 const eaMin = saMin + pa;
                i32 const eaMax = saMax + pa;
                i32 const ls = max(0, min(eaMin, t2) - max(saMin, t1));
                i32 const rs = max(0, min(eaMax, t2) - max(saMax, t1));
                i32 const mi = min(ls, rs);
                w += ha * mi;
            }

            if (w <= c * (t2 - t1))
            {
                for (auto a = 0; a < nActivities; a += 1)
                {
                    i32 const ha = h_d[a];
                    i32 const saMin = si_s[a].min;
                    i32 const saMax = si_s[a].max;
                    i32 const pa = p_d[a];
                    i32 const eaMin = saMin + pa;
                    i32 const eaMax = saMax + pa;
                    i32 const ls = max(0, min(eaMin, t2) - max(saMin, t1));
                    i32 const rs = max(0, min(eaMax, t2) - max(saMax, t1));
                    i32 const mi = min(ls, rs);
                    i32 const avail = c * (t2 - t1) - w + ha * mi;
                    if (avail < ha * ls)
                    {
                        atomicMax_block(&si_s[a].min, t2 - (avail / ha));
                    }
                    if (avail < ha * rs)
                    {
                        atomicMin_block(&si_s[a].max, t1 + (avail / ha) - pa);
                    }
                }
            }
            else
            {
                *isConsistent_d = false;
                break;
            }
        }
        __syncthreads();

        if (*isConsistent_d)
        {
            for (auto a = threadIdx.x; a < nActivities; a += blockDim.x)
            {
                atomicMax(&si_d[a].min, si_s[a].min);
                atomicMin(&si_d[a].max, si_s[a].max);
            }
        }
    }
}