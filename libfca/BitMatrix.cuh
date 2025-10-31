#pragma once

#include <libfca/Matrix.cuh>

namespace Fca
{
    class BitMatrix : public Array<u32>
    {
        // Members
        protected:
            u32 _rows;
            u32 _cols;

        // Functions
        public:
            BitMatrix(u32 rows, u32 cols, u32 * data);
            bool get(u32 row, u32 col) const;
            u32 * getRow(u32 row) const;
            u32 * getData() const;
            u32 getDataSize() const;
            void set(u32 row, u32 col, bool value);
            void clear();
            void print() const;
            static u32 getDataSize(u32 rows, u32 cols);
            __host__ __device__ static bool get(u32 row, u32 col, u32 rows, u32 cols, u32 const * data);
            __host__ __device__ static void print(u32 rows, u32 cols, u32 const * data);

    };

    inline
    bool BitMatrix::get(u32 row, u32 col) const
    {
        return get(row, col, _rows, _cols, Array<u32>::_data);
    }

    __host__ __device__ inline
    bool BitMatrix::get(u32 row, u32 col, u32 rows, u32 cols, u32 const * data )
    {
        assert(row < rows);
        assert(col < cols);

        u32 const bitIdx = row * cols + col;
        u32 const bitWord = bitIdx / 32;
        u32 const bitOffset = bitIdx % 32;
        u32 const mask = 1 << (31 - bitOffset);
        return (data[bitWord] & mask) != 0;
    }

    inline
    void BitMatrix::set(u32 row, u32 col, bool value)
    {
        assert(row < _rows);

        u32 const bitIdx = row * _cols + col;
        u32 const bitWord = bitIdx / 32;
        u32 const bitOffset = bitIdx % 32;
        u32 const mask = 1 << (31 - bitOffset);
        if(value)
        {
            *Array<u32>::at(bitWord) |= mask;
        }
        else
        {
            *Array<u32>::at(bitWord) &= ~mask;
        }
    }

    inline
    void BitMatrix::print() const
    {
        print(_rows, _cols, Array<u32>::_data);
    }

    inline
    u32 BitMatrix::getDataSize(u32 rows, u32 cols)
    {
        return  (rows * cols * sizeof(u32)) / 32;
    }

    inline
    u32 * BitMatrix::getRow(u32 row) const
    {
        assert(row < _rows);

        u32 const bitIdx = row * _cols;
        u32 const bitWord = bitIdx / 32;
        return Array<u32>::at(bitWord);
    }

    inline
    u32 BitMatrix::getDataSize() const
    {
        return getDataSize(_rows, _cols);
    }

    inline
    u32 * BitMatrix::getData() const
    {
        return Array<u32>::_data;
    }
}