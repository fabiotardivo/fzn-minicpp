#pragma once

#include <type_traits>
#include "Array.hpp"

namespace Fca
{
    template<typename T>
    class Matrix : protected Array<T>
    {
        // Members
        protected:
            u32 _rows;
            u32 _cols;

        // Functions
        public:
            __host__ __device__ inline Matrix(u32 rows, u32 cols, T * data);
            __host__ __device__ inline u32 rows() const;
            __host__ __device__ inline u32 cols() const;
            __host__ __device__ inline T * at(u32 row, u32 col) const;
            __host__ __device__ inline T * getRow(u32 row) const;
            __host__ __device__ inline void reshape(u32 rows, u32 cols);
            __host__ __device__ inline u32 getDataSize() const;
            __host__ __device__ inline static u32 getDataSize(u32 rows, u32 cols);
            __host__ __device__ inline void print() const;

        private:
            __host__ __device__ static void print(u32 rows, u32 cols, T const * data);
    };

    template<typename T>
    __host__ __device__
    Matrix<T>::Matrix(u32 rows, u32 cols, T * data) :
        Array<T>(rows * cols, data),
        _rows(rows),
        _cols(cols)
    {}

    template<typename T>
    __host__ __device__ inline
    T * Matrix<T>::at(u32 row, u32 col) const
    {
        assert(row < _rows);
        assert(col < _cols);
        u32 const index = (row * _cols) + col;
        return Array<T>::at(index);
    }

    template<typename T>
    __host__ __device__ inline
    T * Matrix<T>::getRow(Fca::u32 row) const
    {
        u32 const index = (row * _cols);
        return Array<T>::at(index);
    }

    template<typename T>
    __host__ __device__
    u32 Matrix<T>::rows() const
    {
        return _rows;
    }

    template<typename T>
    __host__ __device__
    u32 Matrix<T>::cols() const
    {
        return _cols;
    }

    template<typename T>
    __host__ __device__
    void Matrix<T>::print() const
    {
        print(_rows, _cols, Array<T>::_data);
    }

    template<typename T>
    __host__ __device__
    void Matrix<T>::print(u32 rows, u32 cols, T const * data)
    {
        static_assert(std::is_integral<T>::value);
        for (u32 r = 0; r < rows; r += 1)
        {
            for (u32 c = 0; c < cols; c += 1)
            {
                printf("%d ", data[(r * cols) + c]);
            }
            printf("\n");
        }
    }

    template<typename T>
    __host__ __device__
    u32 Matrix<T>::getDataSize(u32 rows, u32 cols)
    {
        return Array<T>::getDataSize(rows * cols);
    }


    template<typename T>
    __host__ __device__
    void Matrix<T>::reshape(u32 rows, u32 cols)
    {
        assert(rows * cols <= Array<T>::_size);
        _rows = rows;
        _cols = cols;
    }

    template<typename T>
    __host__ __device__
    u32 Matrix<T>::getDataSize() const
    {
        return getDataSize(_rows, _cols);
    }
}
