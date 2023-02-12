#pragma once

#include <cstdio>
#include <cassert>
#include "Types.hpp"

// Compile with C++ only compiler
#ifndef __NVCC__
#define __host__
#define __device__
#endif

namespace Fca
{
    template<typename T>
    class Slice
    {
        // Members
        protected:
            u32 _capacity;
            T * _data;

        // Functions
        public:
            __host__ __device__ inline Slice(u32 size, T * data);
            __host__ __device__ inline T * at(u32 index) const;
            __host__ __device__ inline T * begin() const;
            __host__ __device__ inline T * end() const;
            __host__ __device__ inline T * first() const;
            __host__ __device__ inline T * last() const;
            __host__ __device__ inline u32 size() const;
            __host__ __device__ Slice<T> & operator=(Slice<T> & other) = delete;
            __host__ __device__ Slice<T> & operator=(Slice<T> const & other) = delete;
            __host__ __device__ inline Slice<T> & operator=(Slice<T> && other);
            __host__ __device__ void print() const;
        protected:
            __host__ __device__ static void print(T const * begin, T const * end);
    };

    template<typename T>
    __host__ __device__
    Slice<T>::Slice(u32 size, T * data) :
            _capacity(size),_data(data)
    {}

    template<typename T>
    __host__ __device__
    T * Slice<T>::at(u32 index) const
    {
        assert(_capacity > 0);
        assert(index < _capacity);
        return _data + index;
    }

    template<typename T>
    __host__ __device__
    T * Slice<T>::begin() const
    {
        return _data;
    }

    template<typename T>
    __host__ __device__
    T * Slice<T>::first() const
    {
        return _data;
    }

    template<typename T>
    __host__ __device__
    T * Slice<T>::last() const
    {
        return _data + (_capacity - 1);
    }

    template<typename T>
    __host__ __device__
    T * Slice<T>::end() const
    {
        return _data + _capacity;
    }

    template<typename T>
    __host__ __device__
    u32 Slice<T>::size() const
    {
        return _capacity;
    }

    template<typename T>
    __host__ __device__
    Slice<T> & Slice<T>::operator=(Slice<T> && other)
    {
        if (this != &other)
        {
            _capacity = other._capacity;
            _data = other._data;
            other._capacity = 0;
            other._data = nullptr;
        }
        return *this;
    }

    template<typename T>
    __host__ __device__
    void Slice<T>::print() const
    {
        print(begin(), end());
    }

    template<typename T>
    __host__ __device__
    void Slice<T>::print(T const * begin, T const * end)
    {
        static_assert(std::is_integral<T>::value);
        T const * t = begin;
        printf("%d", *t);
        for(t += 1; t != end; t += 1)
        {
            printf(",%d", *t);
        }
        printf("\n");
    }
}