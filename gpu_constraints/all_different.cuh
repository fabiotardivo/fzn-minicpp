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

#pragma once

#include <vector>
#include <intvar.hpp>
#include <constraint.hpp>
#include <libfca/Array.hpp>
#include <libfca/BitMatrix.cuh>
#include <libfca/Types.hpp>

class AllDifferentGPU : public AllDifferentAC
{
    private:
        Fca::BitMatrix * graph_h;
        Fca::u32* matrix1_d;
        Fca::u32* matrix2_d;
        Fca::u32* matrix3_d;
        Fca::Array<Fca::u32> * scc_h;
        Fca::u32* scc_d;
        cudaStream_t cuStream;
        Fca::u32 nEdges;
        cudaGraph_t graph;
        cudaGraphExec_t graph2sccLowLatency;
        Fca::u32 iteration;

    public:
        AllDifferentGPU(std::vector<var<int>::Ptr> const & vars);
        void updateBounds();
        void post() override;
        void propagate() override;
    private:
        void initGraph2sccLowLatency();
        void domains2graph(int * match);
        void graph2scc();
};