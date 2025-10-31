/*
 * fzn-minicpp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * fzn-minicpp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c) 2022. by Fabio Tardivo
 */

#include <new>
#include <algorithm>
#include <chrono>
#include <gpu_constraints/all_different.cuh>
#include <libfca/SqrBitMatrix.cuh>
#include <libgpu/Memory.cuh>

using namespace std::chrono;

AllDifferentGPU::AllDifferentGPU(std::vector<var<int>::Ptr> const & vars) :
        AllDifferentAC(vars)
{
    using namespace Fca;
    using namespace Gpu::Memory;

    setPriority(CLOW);
    AllDifferentAC::init();
    updateBounds();
    iteration = 0;

    //Initialize memory
    u32 * mem = mallocHost<u32>(BitMatrix::getDataSize(_nNodes, _nNodes));
    graph_h = new BitMatrix(_nNodes, _nNodes, mem);

    matrix1_d = mallocDevice<u32>(graph_h->getDataSize());
    matrix2_d = mallocDevice<u32>(graph_h->getDataSize());
    matrix3_d = mallocDevice<u32>(graph_h->getDataSize());

    mem = mallocHost<u32>(Array<u32>::getDataSize(_nNodes));
    scc_h = new Array<u32>(_nNodes, mem);
    scc_d = mallocDevice<u32>(Array<u32>::getDataSize(_nNodes));

    cudaStreamCreate(&cuStream);
    initGraph2sccLowLatency();
}

void AllDifferentGPU::updateBounds()
{
    using namespace Fca;
    using namespace Fca::Utils::Math;

    _minVal = INT32_MAX;
    _maxVal = INT32_MIN;
    for(u32 i = 0; i < _nVar; i += 1)
    {
        _minVal = std::min(_minVal, _x[i]->min());
        _maxVal = std::max(_maxVal, _x[i]->max());
    }
    _minVal = roundDown(_minVal, 32);
    _maxVal = _minVal + ceilDivPosInt(_maxVal - _minVal + 1, 32) * 32 - 1;
    _nVal = _maxVal - _minVal + 1;
    _nNodes = _nVal + _nVar + 1;
    if (_nNodes <= 64)
    {
        _nNodes = 64;
    }
    else if (_nNodes <= 128)
    {
        _nNodes = 128;
    }
    else
    {
        _nNodes = ceilDivPosInt(_nNodes, 128) * 128;
    }
}

void AllDifferentGPU::post()
{
    for (int i = 0; i < _nVar; i += 1)
    {
        _x[i]->propagateOnDomainChange(this);
    }
    propagate();
}
void AllDifferentGPU::initGraph2sccLowLatency()
{
    cudaStreamBeginCapture(cuStream, cudaStreamCaptureModeGlobal);
    graph2scc();
    cudaStreamEndCapture(cuStream, &graph);
    cudaGraphInstantiate(&graph2sccLowLatency, graph, nullptr, nullptr, 0);
}

void AllDifferentGPU::propagate()
{
    //Timer::begin("AllDifferentGPU");
    if (AllDifferentAC::calcMM() < _nVar)
    {
        //Timer::end("AllDifferentGPU");
        failNow();
    }

    iteration += 1;
    nEdges = 0;
    for (int i  = 0; i < _nVar; i += 1)
    {
        nEdges += _x[i]->size();
    }

    {
       // Timer::begin("SCCsGPU");
        domains2graph(_match);
        //graph2scc();
        cudaGraphLaunch(graph2sccLowLatency, cuStream);
        cudaStreamSynchronize(cuStream);
        //Timer::end("SCCsGPU");

        for (int var = 0; var < _nVar; var += 1)
        {
            int const minVal = _x[var]->min();
            int const maxVal = _x[var]->max();
            for (int val = minVal; val <= maxVal; val += 1)
            {
                int const varNode = _nVal + var;
                int const valNode = val - _minVal;
                if (_match[var] != val and *scc_h->at(varNode) != *scc_h->at(valNode))
                {
                    _x[var]->remove(val);
                }
            }
        }
    }
    //Timer::end("AllDifferentGPU");
}

void AllDifferentGPU::domains2graph(int * match)
{
    graph_h->clear();

    // Edges variables -> values
    for(int var = 0; var < _nVar; var += 1)
    {
        int const varNode = _nVal + var;
        _x[var]->dump(_minVal, _maxVal, graph_h->getRow(varNode));
    }

    // Edges sink -> values
    int const sinkNode = _nVal + _nVar;
    for (int val = _minVal; val <= _maxVal; val += 1)
    {
        int const valNode = val - _minVal;
        graph_h->set(valNode, sinkNode, true);
    }

    // Match edges
    for (int var = 0; var < _nVar; var += 1)
    {
        int const valNode = match[var] - _minVal;
        int const varNode = _nVal + var;

        // Edges variables <-> values
        graph_h->set(varNode, valNode, false);
        graph_h->set(valNode, varNode, true);

        // Edges sink <-> values
        graph_h->set(valNode, sinkNode, false);
        graph_h->set(sinkNode, valNode, true);
    }
}

void AllDifferentGPU::graph2scc()
{
    using namespace Fca;

    u32 * reach = matrix1_d;
    u32 * reach_ = matrix2_d;
    u32 * reach_t = matrix3_d;

    if (_nNodes == 64)
    {
        cudaMemcpyAsync(reach, graph_h->getData(), graph_h->getDataSize(), cudaMemcpyHostToDevice, cuStream);
        scc64<<<1,64,0, cuStream>>>(reach, scc_d);
        cudaMemcpyAsync(scc_h->getData(), scc_d, scc_h->getDataSize(), cudaMemcpyDeviceToHost, cuStream);
        cudaStreamSynchronize(cuStream);
    }
    else if (_nNodes == 128)
    {
        cudaMemcpyAsync(reach, graph_h->getData(), graph_h->getDataSize(), cudaMemcpyHostToDevice, cuStream);
        scc128<<<1,128,0, cuStream>>>(reach, scc_d);
        cudaMemcpyAsync(scc_h->getData(), scc_d, scc_h->getDataSize(), cudaMemcpyDeviceToHost, cuStream);
        cudaStreamSynchronize(cuStream);
    }
    else
    {
        u32 const blocks = _nNodes / 128;
        dim3 const dimBlock2 = dim3(blocks, 2);
        dim3 const dimBlockBlock = dim3(blocks, blocks);

        cudaMemcpyAsync(reach, graph_h->getData(), graph_h->getDataSize(), cudaMemcpyHostToDevice, cuStream);
        arrangeKernel<<<dimBlockBlock, 128, 0, cuStream>>>(_nNodes, reach, reach_);

        for (u32 i = 0; i < _nNodes; i += 128)
        {
            reachabilityArrangedKernel1<<<1, 128, 0, cuStream>>>(i, _nNodes, reach_);
            reachabilityArrangedKernel2<<<dimBlock2, 128, 0, cuStream>>>(i, _nNodes, reach_);
            reachabilityArrangedKernel3<<<dimBlockBlock, 128, 0, cuStream>>>(i, _nNodes, reach_);
        }

        transposeArrangedKernel<<<dimBlockBlock, 128, 0, cuStream>>>(_nNodes, reach_, reach_t);
        bitwiseAndKernel<<<dimBlockBlock, 128, 0, cuStream>>>(_nNodes, reach_, reach_t, reach);
        initSccKernel<<<blocks, 128, 0, cuStream>>>(_nNodes, scc_d);
        findSccArrangedKernel<<<blocks, 128, 0, cuStream>>>(_nNodes, reach, scc_d);

        cudaMemcpyAsync(scc_h->getData(), scc_d, scc_h->getDataSize(), cudaMemcpyDeviceToHost, cuStream);
    }
}