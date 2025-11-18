#pragma once

#include "global_constraints/table.hpp"
#include <libfca/Types.hpp>
#include <libgpu/Memory.cuh>
#include <libgpu/LinearAllocator.cuh>
#include <varitf.hpp>
#include <constraint.hpp>
using namespace std;
using namespace Fca;
using namespace Gpu::Memory;

class TableGPU : public Table{

    private:

        unsigned int *_supports_dev; //array of arrays linearized
        int * _currTable_size_dev; //just a pointer to a single element
        int * _supportSize_dev; //just a pointer to a single element
        int * _supportOffsetJmp_dev; //array
        int * _variablesOffsets_dev; //array
        int * _vars_dev; //array (matrix) (the domains)
        int currTableSize;
        int noVars;
        int *_noVars_dev;
        int *_vars_host;
        unsigned int * _vars_to_remove_host;
        unsigned int * _tmpMasks;

        unsigned int* _CT_mask_svs_host;
        unsigned int* _CT_mask_svs_dev;
        

        int noBlocksFilter;
        const int noStreams=1; //hardcoded, don't touch IN THIS BRANCH

        cudaStream_t* streams;

        int *doms_doms_before_dev;
        int *doms_doms_before_host;

        unsigned int* buffer; //just for the new dump
        int buffSize;
        unsigned int* _supportsT_host;
        unsigned int* _supportsT_dev;
        int* noTuples_dev; 
        
    public:
        TableGPU(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples);
        void post() override;
        void propagate() override;
    private:
        void dumpDomainsGPU2();

        
};

__global__ void  filterDomainsGPU(unsigned int * _CT_mask_svs_dev, int* _currTable_dev_size, int* _vars_dev, int *_supportOffsetJmp_dev, unsigned int* _supports_dev , int * supportSize_dev);
__global__ void updateTableGPU(unsigned int* _supports_dev,unsigned int * _svSize_off_sval_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* _currTable_dev_size, int* _vars_dev, int* offsetsAndLimits, unsigned int* _tmpMasks,int * noTuples);
__global__ void reduce(unsigned int* _CT_mask_svs_dev,unsigned int* _tmpMasks,int* _currTable_size_dev);

void transposeSupports(unsigned int* supports, unsigned int* supportsTransposed, int supportSize, int currTableSize, int noVars , int* _supportOffsetJmp_dev, vector<var<int>::Ptr> & vars);
int bitsFromRight(int n);
int bitsFromLeft(int n);
