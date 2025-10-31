#include <cassert>
#include <libfca/Array.hpp>
#include <libfca/BitMatrix.cuh>
#include <libfca/SqrBitMatrix.cuh>
#include <libfca/u64.cuh>
#include <libfca/Utils.hpp>

namespace Fca
{
    __host__ __device__
    void cpyMatrixToBlock(u32 row, u32 col, u32 size, u32 const *matrix, u128 *block)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);
#endif
        assert(col % 128 == 0);
        assert(row % 128 == 0);


        u32 const wordIdx = col / 128;
        u32 const wordsPerRow = size / 128;
        u128 const *const _m = reinterpret_cast<u128 const *>(matrix) + (row * wordsPerRow + wordIdx);
#ifdef __CUDA_ARCH__
        u32 const bRow = threadIdx.x;
#else
        for (u32 bRow = 0; bRow < 128; bRow += 1)
#endif
        {
            block[bRow] = _m[bRow * wordsPerRow];
        }
    }

    __host__ __device__
    void cpyArrangedToBlock(u32 row, u32 col, u32 size, u32 const *arranged, u128 *block)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);
#endif
        assert(col % 128 == 0);
        assert(row % 128 == 0);

        u32 const xBlock = col / 128;
        u32 const yBlock = row / 128;
        u32 const blocksPerRow = size / 128;
        u128 const *const _a = reinterpret_cast<u128 const *>(arranged) + ((yBlock * blocksPerRow + xBlock) * 128);
#ifdef __CUDA_ARCH__
        u32 const bWord = threadIdx.x;
#else
        for (u32 bWord = 0; bWord < 128; bWord += 1)
#endif
        {
            block[bWord] = _a[bWord];
        }
    }

    __host__ __device__
    void cpyBlockToMatrix(u32 row, u32 col, u32 size, u128 const *block, u32 *matrix)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);
#endif
        assert(col % 128 == 0);
        assert(row % 128 == 0);

        u32 const wordIdx = col / 128;
        u32 const wordsPerRow = size / 128;
        u128 *const _m = reinterpret_cast<u128 *>(matrix) + (row * wordsPerRow + wordIdx);
#ifdef __CUDA_ARCH__
        u32 const bRow = threadIdx.x;
#else
        for (u32 bRow = 0; bRow < 128; bRow += 1)
#endif
        {
            _m[bRow * wordsPerRow] = block[bRow];
        }
    }

    __host__ __device__
    void cpyBlockToArranged(u32 row, u32 col, u32 size, u128 const *block, u32 *arranged)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);
#endif
        assert(col % 128 == 0);
        assert(row % 128 == 0);

        u32 const xBlock = col / 128;
        u32 const yBlock = row / 128;
        u32 const blocksPerRow = size / 128;
        u128 *const _a = reinterpret_cast<u128 *>(arranged) + ((yBlock * blocksPerRow + xBlock) * 128);
#ifdef __CUDA_ARCH__
        u32 const bWord = threadIdx.x;
#else
        for (u32 bWord = 0; bWord < 128; bWord += 1)
#endif
        {
            _a[bWord] = block[bWord];
        }
    }

    __host__ __device__
    void transposeBlock(u64 const *block, u64 *transposed)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == 64);
#endif

#ifdef __CUDA_ARCH__
        u32 const tRow = threadIdx.x;
#else
        for (u32 tRow = 0; tRow < 64; tRow += 1)
#endif
        {
            u64 tRowBuffer = 0;
            for (u32 bRow = 0; bRow < 64; bRow += 1) {
                bool const b = getBit(block[bRow], 63 - tRow);
                orBit(tRowBuffer, 63 - bRow, b);
            }
            transposed[tRow] = tRowBuffer;
        }
    }

    __host__ __device__
    void transposeBlock(u128 const *block, u128 *transposed)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);
#endif

#ifdef __CUDA_ARCH__
        u32 const tRow = threadIdx.x;
#else
        for (u32 tRow = 0; tRow < 128; tRow += 1)
#endif
        {
            u128 tRowBuffer = {0, 0, 0, 0};
            for (u32 bRow = 0; bRow < 128; bRow += 1) {
                bool const b = getBit(block[bRow], 127 - tRow);
                orBit(tRowBuffer, 127 - bRow, b);
            }
            transposed[tRow] = tRowBuffer;
        }
    }

    __host__ __device__
    void bitwiseAndBlock(u128 const *aBlock, u128 const *bBlock, u128 *cBlock)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);

        u32 const bRow = threadIdx.x;
#else
        for (u32 bRow = 0; bRow < 128; bRow += 1)
#endif
        {
            cBlock[bRow] = bitwiseAnd(aBlock[bRow], bBlock[bRow]);
        }
    }

    __host__ __device__
    void findSccBlock(u32 row, u32 col, u128 const *block, u32 *scc)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);

        u32 const bRow = threadIdx.x;
#else
        for (u32 bRow = 0; bRow < 128; bRow += 1)
#endif
        {
            u32 const bitIdx = firstBitSet(block[bRow]);
            u32 const _scc = bitIdx != UINT32_MAX ? col + bitIdx : UINT32_MAX;
            scc[bRow] = min(scc[bRow], _scc);
        }
    }

    __host__ __device__
    void reachabilityBlock(u128 *ijBlock, u128 const *iBlock, u128 const *kBlock)
    {
        /* Secret speedup ;)
    #ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);

        u32 const i = threadIdx.x;
    #else
        for (u32 i = 0; i < 128; i += 1)
    #endif
        {
            u128 prevRow = iBlock[i];
            u128 currRow = prevRow;
            u128 diffRow = prevRow;

            while (not isZero(diffRow))
            {
                for (u32 k = 0; k < 128; k += 1)
                {
                    if (getBit(diffRow, 127 - k))
                    {
                        bitwiseOr(currRow, kBlock[k]);
                    }
                }

                diffRow = currRow;
                bitwiseXor(diffRow, prevRow);
                prevRow = currRow;
            }
            bitwiseOr(ijBlock[i], currRow);
        }
         */


#ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);
#endif
        for (u32 k = 0; k < 128; k += 1) {
#ifdef __CUDA_ARCH__
            u32 const i = threadIdx.x;
#else
            for (u32 i = 0; i < 128; i += 1)
#endif
            {
                u128 kRow = {0, 0, 0, 0};
                bool const ik = getBit(iBlock[i], 127 - k);
                kRow = ik ? kBlock[k] : kRow;
#ifdef __CUDA_ARCH__
                __syncthreads();
#endif
                bitwiseOr(ijBlock[i], kRow);
#ifdef __CUDA_ARCH__
                __syncthreads();
#endif
            }
        }
    }

    __host__ __device__
    void reachabilityBlock3(u128 *ijBlock, u128 const *iBlock, u128 const *kBlock)
    {
#ifdef __CUDA_ARCH__
        assert(blockDim.x == 128);
#endif

#ifdef __CUDA_ARCH__
        u32 const i = threadIdx.x;
#else
        for (u32 i = 0; i < 128; i += 1)
#endif
        {
            u128 ijRow = ijBlock[i];
            u128 const iRow = iBlock[i];
            for (u32 k = 0; k < 128; k += 1) {
                u128 kRow = {0, 0, 0, 0};
                bool const ik = getBit(iRow, 127 - k);
                kRow = ik ? kBlock[k] : kRow;
                bitwiseOr(ijRow, kRow);
            }
            ijBlock[i] = ijRow;
        }
    }

    void arrange(u32 size, const u32 *matrix, u32 *arranged)
    {
        u128 block[128];
        for (u32 row = 0; row < size; row += 128) {
            for (u32 col = 0; col < size; col += 128) {
                cpyMatrixToBlock(row, col, size, matrix, block);
                cpyBlockToArranged(row, col, size, block, arranged);
            }
        }
    }

    __global__
    void arrangeKernel(u32 size, const u32 *matrix, u32 *arranged)
    {
        assert(gridDim.x * 128 == size);
        assert(gridDim.y * 128 == size);
        assert(blockDim.x == 128);

        __shared__ u128 block[128];
        u32 const row = blockIdx.y * 128;
        u32 const col = blockIdx.x * 128;
        cpyMatrixToBlock(row, col, size, matrix, block);
        cpyBlockToArranged(row, col, size, block, arranged);
    }

    void dearrange(u32 size, u32 const *arranged, u32 *matrix)
    {
        u128 block[128];
        for (u32 row = 0; row < size; row += 128) {
            for (u32 col = 0; col < size; col += 128) {
                cpyArrangedToBlock(row, col, size, arranged, block);
                cpyBlockToMatrix(row, col, size, block, matrix);
            }
        }
    }

    __global__
    void dearrangeKernel(u32 size, u32 const *arranged, u32 *matrix)
    {
        assert(gridDim.x * 128 == size);
        assert(gridDim.y * 128 == size);
        assert(blockDim.x == 128);

        __shared__ u128 block[128];
        u32 const row = blockIdx.y * 128;
        u32 const col = blockIdx.x * 128;
        cpyArrangedToBlock(row, col, size, arranged, block);
        cpyBlockToMatrix(row, col, size, block, matrix);
    }

    void transposeArranged(u32 size, const u32 *arranged, u32 *transposed)
    {
        u128 aBlock[128];
        u128 tBlock[128];
        for (u32 row = 0; row < size; row += 128) {
            for (u32 col = 0; col < size; col += 128) {
                cpyArrangedToBlock(row, col, size, arranged, aBlock);
                transposeBlock(aBlock, tBlock);
                cpyBlockToArranged(col, row, size, tBlock, transposed);
            }
        }
    }

    __global__
    void transposeArrangedKernel(u32 size, const u32 *arranged, u32 *transposed)
    {
        assert(gridDim.x * 128 == size);
        assert(gridDim.y * 128 == size);
        assert(blockDim.x == 128);

        __shared__ u128 arrangedBlock[128];
        __shared__ u128 transposedBlock[128];
        u32 const row = blockIdx.y * 128;
        u32 const col = blockIdx.x * 128;
        cpyArrangedToBlock(row, col, size, arranged, arrangedBlock);
        __syncthreads();
        transposeBlock(arrangedBlock, transposedBlock);
        __syncthreads();
        cpyBlockToArranged(col, row, size, transposedBlock, transposed);
    }

    void reachabilityArranged(u32 size, u32 *reach)
    {
        u128 block1[128];
        u128 block2[128];
        u128 block3[128];
        for (u32 i = 0; i < size; i += 128) {
            //Phase 1
            cpyArrangedToBlock(i, i, size, reach, block1);
            reachabilityBlock(block1, block1, block1);
            cpyBlockToArranged(i, i, size, block1, reach);

            //Phase 2
            for (u32 y = 0; y < 2; y += 1) {
                for (u32 x = 0; x < size; x += 128) {
                    if (x != i) {
                        if (y == 0) // Horizontal
                        {
                            cpyArrangedToBlock(i, x, size, reach, block2);
                            reachabilityBlock(block2, block1, block2);
                            cpyBlockToArranged(i, x, size, block2, reach);
                        } else // Vertical
                        {
                            cpyArrangedToBlock(x, i, size, reach, block2);
                            reachabilityBlock(block2, block2, block1);
                            cpyBlockToArranged(x, i, size, block2, reach);
                        }
                    }
                }
            }

            // Phase 3
            for (u32 col = 0; col < size; col += 128) {
                if (col != i) {
                    cpyArrangedToBlock(i, col, size, reach, block3);
                    for (u32 row = 0; row < size; row += 128) {
                        if (row != i) {
                            cpyArrangedToBlock(row, i, size, reach, block2);
                            cpyArrangedToBlock(row, col, size, reach, block1);
                            reachabilityBlock3(block1, block2, block3);
                            cpyBlockToArranged(row, col, size, block1, reach);
                        }
                    }
                }
            }
        }
    }

    __global__
    void scc64(u32 const *graph, u32 *scc)
    {
        assert(gridDim.x == 1);
        assert(blockDim.x == 64);

        __shared__ u64 block[64];

        // Copy matrix to shared
        u32 const i = threadIdx.x;
        u64 const *const _graph = reinterpret_cast<u64 const *>(graph);
        block[i] = _graph[i];
        __syncthreads();

        // Reachability
        for (u32 k = 0; k < 64; k += 1) {
            u64 kRow = 0;
            bool const ik = getBit(block[i], 63 - k);
            kRow = ik ? block[k] : kRow;
            __syncthreads();
            block[i] |= kRow;
            __syncthreads();
        }
        /* Secret speedup
        u64 prevRow = block[i];
        u64 currRow = prevRow;
        u64 diffRow = prevRow;
        while (diffRow != 0)
        {
            for (u32 k = 0; k < 128; k += 1)
            {
                if (getBit(diffRow, 127 - k))
                {
                    currRow |= block[k];
                }
            }
            diffRow = currRow;
            diffRow ^= prevRow;
            prevRow = currRow;
        }
        block[i] = currRow;
        */

        // SCC
        u32 _scc = i;
        u64 const iRow = block[i];
        for (u32 j = 0; j < 64; j += 1) {
            bool ij = getBit(iRow, 63 - j);
            bool ji = getBit(block[j], 63 - i);
            _scc = ij and ji and (j < _scc) ? j : _scc;
        }
        scc[i] = _scc;
    }


    __global__
    void scc128(u32 const *adj, u32 *scc)
    {
        assert(gridDim.x == 1);
        assert(blockDim.x == 128);

        __shared__ u128 block[128];

        // Copy matrix to shared
        u32 const i = threadIdx.x;
        u128 const *const _graph = reinterpret_cast<u128 const *>(adj);
        block[i] = _graph[i];
        __syncthreads();

        // Reachability
        for (u32 k = 0; k < 128; k += 1) {
            u128 kRow = {0, 0, 0, 0};
            bool const ik = getBit(block[i], 127 - k);
            kRow = ik ? block[k] : kRow;
            __syncthreads();
            bitwiseOr(block[i], kRow);
            __syncthreads();
        }
        /* Secret speedup
        u128 prevRow = block[i];
        u128 currRow = prevRow;
        u128 diffRow = prevRow;
        while (not isZero(diffRow))
        {
            for (u32 k = 0; k < 128; k += 1)
            {
                if (getBit(diffRow, 127 - k))
                {
                    bitwiseOr(currRow,block[k]);
                }
            }
            diffRow = currRow;
            bitwiseXor(diffRow,prevRow);
            prevRow = currRow;
        }
        block[i] = currRow;
         */

        // SCC
        u32 _scc = i;
        u128 const iRow = block[i];
        for (u32 j = 0; j < 128; j += 1) {
            bool ij = getBit(iRow, 127 - j);
            bool ji = getBit(block[j], 127 - i);
            _scc = ij and ji and j < _scc ? j : _scc;
        }
        scc[i] = _scc;
    }


    __global__
    void reachabilityArrangedKernel1(u32 i, u32 size, u32 *reach)
    {
        assert(gridDim.x == 1);
        assert(blockDim.x == 128);

        __shared__ u128 block1[128];
        __shared__ u128 block2[128];
        __shared__ u128 block3[128];
        cpyArrangedToBlock(i, i, size, reach, block1);
        __syncthreads();
        reachabilityBlock(block1, block1, block1);
        __syncthreads();
        cpyBlockToArranged(i, i, size, block1, reach);
    }

    __global__
    void reachabilityArrangedKernel2(u32 i, u32 size, u32 *reach)
    {
        assert(gridDim.x == size / 128);
        assert(gridDim.y == 2);
        assert(blockDim.x == 128);

        __shared__ u128 block1[128];
        __shared__ u128 block2[128];
        u32 const x = blockIdx.x * 128;
        u32 const y = blockIdx.y;

        if (x != i) {
            if (y == 0) // Horizontal
            {
                cpyArrangedToBlock(i, i, size, reach, block1);
                cpyArrangedToBlock(i, x, size, reach, block2);
                __syncthreads();
                reachabilityBlock(block2, block1, block2);
                __syncthreads();
                cpyBlockToArranged(i, x, size, block2, reach);
            } else // Vertical
            {
                cpyArrangedToBlock(i, i, size, reach, block1);
                cpyArrangedToBlock(x, i, size, reach, block2);
                __syncthreads();
                reachabilityBlock(block2, block2, block1);
                __syncthreads();
                cpyBlockToArranged(x, i, size, block2, reach);
            }
        }
    }

    __global__
    void reachabilityArrangedKernel3(u32 i, u32 size, u32 *reach)
    {
        assert(gridDim.x == size / 128);
        assert(gridDim.y == size / 128);
        assert(blockDim.x == 128);

        __shared__ u128 block1[128];
        __shared__ u128 block2[128];
        __shared__ u128 block3[128];

        u32 const row = blockIdx.y * 128;
        u32 const col = blockIdx.x * 128;

        if (col != i and row != i) {
            cpyArrangedToBlock(i, col, size, reach, block3);
            cpyArrangedToBlock(row, i, size, reach, block2);
            cpyArrangedToBlock(row, col, size, reach, block1);
            __syncthreads();
            reachabilityBlock3(block1, block2, block3);
            __syncthreads();
            cpyBlockToArranged(row, col, size, block1, reach);
        }
    }

    void bitwiseAnd(u32 size, const u32 *a, const u32 *b, u32 *c)
    {
        u128 aBlock[128];
        u128 bBlock[128];
        u128 cBlock[128];
        for (u32 row = 0; row < size; row += 128) {
            for (u32 col = 0; col < size; col += 128) {
                cpyArrangedToBlock(row, col, size, a, aBlock);
                cpyArrangedToBlock(row, col, size, b, bBlock);
                bitwiseAndBlock(aBlock, bBlock, cBlock);
                cpyBlockToArranged(row, col, size, cBlock, c);
            }
        }
    }

    __global__
    void bitwiseAndKernel(u32 size, const u32 *a, const u32 *b, u32 *c)
    {
        assert(gridDim.x * 128 == size);
        assert(gridDim.y * 128 == size);
        assert(blockDim.x == 128);

        __shared__ u128 aBlock[128];
        __shared__ u128 bBlock[128];
        __shared__ u128 cBlock[128];
        u32 const row = blockIdx.y * 128;
        u32 const col = blockIdx.x * 128;
        cpyArrangedToBlock(row, col, size, a, aBlock);
        cpyArrangedToBlock(row, col, size, b, bBlock);
        __syncthreads();
        bitwiseAndBlock(aBlock, bBlock, cBlock);
        __syncthreads();
        cpyBlockToArranged(row, col, size, cBlock, c);
    }

    void initScc(u32 size, u32 *scc)
    {
        for (u32 i = 0; i < size; i += 1) {
            scc[i] = i;
        }
    }

    __global__
    void initSccKernel(u32 size, u32 *scc)
    {
        assert(gridDim.x * 128 == size);
        assert(blockDim.x == 128);

        for (u32 i = threadIdx.x; i < size; i += blockDim.x) {
            scc[i] = i;
        }
    }

    void findSccArranged(u32 size, u32 const *a, u32 *scc)
    {
        u128 block[128];
        u32 _scc[128];
        for (u32 row = 0; row < size; row += 128) {
            for (u32 bRow = 0; bRow < 128; bRow += 1) {
                _scc[bRow] = scc[row + bRow];
            }

            for (u32 col = 0; col < size; col += 128) {
                cpyArrangedToBlock(row, col, size, a, block);
                findSccBlock(row, col, block, _scc);
            }

            for (u32 bRow = 0; bRow < 128; bRow += 1) {
                scc[row + bRow] = _scc[bRow];
            }
        }
    }

    __global__
    void findSccArrangedKernel(u32 size, const u32 *a, u32 *scc)
    {
        assert(gridDim.x * 128 == size);
        assert(blockDim.x == 128);

        __shared__ u128 block[128];
        __shared__ u32 _results[128];
        u32 const row = blockIdx.x * 128;
        u32 const bRow = threadIdx.x;

        _results[bRow] = scc[row + bRow];
        for (u32 col = 0; col < size; col += 128) {
            cpyArrangedToBlock(row, col, size, a, block);
            __syncthreads();
            findSccBlock(row, col, block, _results);
            __syncthreads();
        }

        scc[row + bRow] = _results[bRow];
    }

    void debugWorkflow()
    {
        /*
        u32 const size = 256;
        BitMatrix * m0_h = MALLOC_HOST(BitMatrix, BitMatrix::memory_size(size, size));
        BitMatrix * m1_h = MALLOC_HOST(BitMatrix, BitMatrix::memory_size(size, size));
        BitMatrix * m2_h = MALLOC_HOST(BitMatrix, BitMatrix::memory_size(size, size));
        new (m0_h) BitMatrix(size, size);
        new (m1_h) BitMatrix(size, size);
        new (m2_h) BitMatrix(size, size);

        u32 * m0_d = MALLOC_DEVICE(u32, m0_h->data_memory_size());
        u32 * m1_d = MALLOC_DEVICE(u32, m0_h->data_memory_size());
        u32 * m2_d = MALLOC_DEVICE(u32, m0_h->data_memory_size());

        Array<u32> * a_h = MALLOC_HOST(Array<u32>, Array<u32>::memory_size(size));
        new (a_h) Array<u32>(size);
        u32 * a_d = MALLOC_DEVICE(u32, a_h->data_memory_size());

        u32 const blocks = size / 128;
        dim3 const dimBlockBlock = dim3(blocks, blocks);

        // Matrix
        m0_h->clear();
        for(u32 r = 0; r < size; r += 1)
        {
            m0_h->set(r, (r + 1) % size, true);
        }
        printf("Matrix:\n");
        m0_h->print();

        //Arrange
        arrange(size, m0_h->data(), m1_h->data());
        printf("Arranged CPU:\n");
        m1_h->print();

        cudaMemcpy(m0_d, m0_h->data(), m0_h->data_memory_size(), cudaMemcpyHostToDevice);
        arrangeKernel<<<dimBlockBlock, 128>>>(size, m0_d, m1_d);
        cudaMemcpy(m1_h->data(), m1_d, m0_h->data_memory_size(), cudaMemcpyDeviceToHost);
        printf("Arranged GPU:\n");
        m1_h->print();

        //Dearrange
        dearrange(size, m1_h->data(), m0_h->data());
        printf("Dearranged CPU:\n");
        m0_h->print();

        dearrangeKernel<<<dimBlockBlock, 128>>>(size, m1_d, m0_d);
        cudaMemcpy(m0_h->data(), m0_d, m0_h->data_memory_size(), cudaMemcpyDeviceToHost);
        printf("Dearranged GPU:\n");
        m0_h->print();

        // Transpose
        arrange(size, m0_h->data(), m1_h->data());
        arrangeKernel<<<dimBlockBlock, 128>>>(size, m0_d, m1_d);

        transposeArranged(size, m1_h->data(), m0_h->data());
        printf("Transposed (arranged) CPU:\n");
        m0_h->print();

        transposeArrangedKernel<<<dimBlockBlock, 128>>>(size, m1_d, m0_d);
        cudaMemcpy(m0_h->data(), m0_d, m0_h->data_memory_size(), cudaMemcpyDeviceToHost);
        printf("Transposed (arranged) GPU:\n");
        m0_h->print();

        // Bitwise And
        m0_h->clear();
        for(u32 r = 0; r < size; r += 1)
        {
            for(u32 c = 0; c < size; c += 1)
            {
                m0_h->set(r, c, c - r < 32);
            }
        }

        m1_h->clear();
        for(u32 r = 0; r < size; r += 1)
        {
            for(u32 c = 0; c < size; c += 1)
            {
                m1_h->set(r, c, r - c < 32);
            }
        }

        bitwiseAnd(size, m0_h->data(), m1_h->data(), m2_h->data());
        printf("And CPU:\n");
        m2_h->print();

        cudaMemcpy(m0_d, m0_h->data(), m0_h->data_memory_size(), cudaMemcpyHostToDevice);
        cudaMemcpy(m1_d, m1_h->data(), m0_h->data_memory_size(), cudaMemcpyHostToDevice);
        bitwiseAndKernel<<<dimBlockBlock, 128>>>(size, m0_d, m1_d, m2_d);
        cudaMemcpy(m2_h->data(), m2_d, m0_h->data_memory_size(), cudaMemcpyDeviceToHost);
        printf("And GPU:\n");
        m2_h->print();

        // SCC
        m0_h->clear();
        for(u32 r = 0; r < size; r += 1)
        {
            u32 c;
            if (r < size / 4)
            {
                c = 4;
            }
            else if(r < size / 2)
            {
                c = r - 1;
            }
            else if (r < size * 3 / 4)
            {
                c = r + 1;
            }
            else
            {
                c = size - 4;
            }
            m0_h->set(r, c, true);
        }
        arrange(size, m0_h->data(), m1_h->data());

        initScc(size, a_h->data());
        findSccArranged(size, m1_h->data(), a_h->data());
        printf("SCC CPU:\n");
        a_h->print();

        cudaMemcpy(m1_d, m1_h->data(), m0_h->data_memory_size(), cudaMemcpyHostToDevice);
        initSccKernel<<<blocks, 128>>>(size, a_d);
        findSccArrangedKernel<<<blocks, 128>>>(size, m1_d, a_d);
        cudaMemcpy(a_h->data(), a_d, a_h->data_memory_size(), cudaMemcpyDeviceToHost);
        printf("SCC GPU:\n");
        a_h->print();
         */
    }

    void debugReachability(u32 size, u32 *matrix)
    {
        /*
        BitMatrix * m0_h = MALLOC_HOST(BitMatrix, BitMatrix::memory_size(size, size));
        BitMatrix * m1_h = MALLOC_HOST(BitMatrix, BitMatrix::memory_size(size, size));
        BitMatrix * m2_h = MALLOC_HOST(BitMatrix, BitMatrix::memory_size(size, size));
        new (m0_h) BitMatrix(size, size);
        new (m1_h) BitMatrix(size, size);
        new (m2_h) BitMatrix(size, size);

        u32 * m0_d = MALLOC_DEVICE(u32, m0_h->data_memory_size());
        u32 * m1_d = MALLOC_DEVICE(u32, m0_h->data_memory_size());

        u32 const blocks = size / 128;
        dim3 const dimBlock2 = dim3(blocks, 2);
        dim3 const dimBlockBlock = dim3(blocks, blocks);

        //CPU
        arrange(size, matrix, m0_h->data());
        reachabilityArranged(size, m0_h->data());
        dearrange(size, m0_h->data(), m2_h->data());
        printf("Reachability CPU:\n");
        m2_h->print();

        //GPU
        cudaMemcpy(m0_d, matrix, m0_h->data_memory_size(), cudaMemcpyHostToDevice);
        arrangeKernel<<<dimBlockBlock, 128>>>(size, m0_d, m1_d);
        for(u32 i = 0; i < size; i += 128)
        {
            reachabilityArrangedKernel1<<<1,128>>>(i, size, m1_d);
            reachabilityArrangedKernel2<<<dimBlock2,128>>>(i, size, m1_d);
            reachabilityArrangedKernel3<<<dimBlockBlock,128>>>(i, size, m1_d);
        }
        dearrangeKernel<<<dimBlockBlock, 128>>>(size, m1_d, m0_d);
        cudaMemcpy(m0_h->data(), m0_d, m0_h->data_memory_size(), cudaMemcpyDeviceToHost);
        printf("Reachability GPU:\n");
        m0_h->print();
         */
    }

    void profileReachability()
    {
        /*
        using namespace Memory;
        u32 maxSize = 16384;

        // Random host matrix
        u32 matrixDataSize = BitMatrix::getDataSize(maxSize, maxSize);
        u32 * m_h = mallocHost<u32>(matrixDataSize);
        for(u32 i = 0; i < (matrixDataSize / sizeof(u32)); i +=1 )
        {
            m_h[i] = rand();
        }
        // Copy random matrix to device
        u32 * m_d = mallocDevice<u32>(matrixDataSize);
        cudaMemcpy(m_h, m_d, matrixDataSize, cudaMemcpyDeviceToHost);

        // Cuda instrumentation
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (u32 nNodes = 128; nNodes <= 16384; nNodes *= 2)
        {
            u32 const blocks = nNodes / 128;
            dim3 const dimBlock2 = dim3(blocks, 2);
            dim3 const dimBlockBlock = dim3(blocks, blocks);
            cudaEventRecord(start);
            for (u32 i = 0; i < nNodes; i += 128)
            {
                reachabilityArrangedKernel1<<<1, 128, 0,0>>>(i, nNodes, m_d);
                reachabilityArrangedKernel2<<<dimBlock2, 128, 0, 0>>>(i, nNodes, m_d);
                reachabilityArrangedKernel3<<<dimBlockBlock, 128, 0, 0>>>(i, nNodes, m_d);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds,start,stop);
            printf("%dx%d -> %.3f us\n", nNodes, nNodes, milliseconds * 1000);
        }

        BitMatrix * m0_h = MALLOC_HOST(BitMatrix, BitMatrix::memory_size(size, size));
        BitMatrix * m1_h = MALLOC_HOST(BitMatrix, BitMatrix::memory_size(size, size));
        BitMatrix * m2_h = MALLOC_HOST(BitMatrix, BitMatrix::memory_size(size, size));
        new (m0_h) BitMatrix(size, size);
        new (m1_h) BitMatrix(size, size);
        new (m2_h) BitMatrix(size, size);

        u32 * m0_d = MALLOC_DEVICE(u32, m0_h->data_memory_size());
        u32 * m1_d = MALLOC_DEVICE(u32, m0_h->data_memory_size());

        u32 const blocks = size / 128;
        dim3 const dimBlock2 = dim3(blocks, 2);
        dim3 const dimBlockBlock = dim3(blocks, blocks);

        //CPU
        arrange(size, matrix, m0_h->data());
        reachabilityArranged(size, m0_h->data());
        dearrange(size, m0_h->data(), m2_h->data());
        printf("Reachability CPU:\n");
        m2_h->print();

        //GPU
        cudaMemcpy(m0_d, matrix, m0_h->data_memory_size(), cudaMemcpyHostToDevice);
        arrangeKernel<<<dimBlockBlock, 128>>>(size, m0_d, m1_d);
        for(u32 i = 0; i < size; i += 128)
        {
            reachabilityArrangedKernel1<<<1,128>>>(i, size, m1_d);
            reachabilityArrangedKernel2<<<dimBlock2,128>>>(i, size, m1_d);
            reachabilityArrangedKernel3<<<dimBlockBlock,128>>>(i, size, m1_d);
        }
        dearrangeKernel<<<dimBlockBlock, 128>>>(size, m1_d, m0_d);
        cudaMemcpy(m0_h->data(), m0_d, m0_h->data_memory_size(), cudaMemcpyDeviceToHost);
        printf("Reachability GPU:\n");
        m0_h->print();
         */
    }
}
