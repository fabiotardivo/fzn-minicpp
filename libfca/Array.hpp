#pragma once

#include "Slice.hpp"

namespace Fca
{
    template<typename T>
    class Array : public Slice<T>
    {
        // Functions
        public:
            __host__ __device__ inline Array(u32 capacity, T * data);
            __host__ __device__ inline T * at(u32 index) const;
            __host__ __device__ inline T * begin() const;
            __host__ __device__ inline T * end() const;
            __host__ __device__ inline T * first() const;
            __host__ __device__ inline T * last() const;
            __host__ __device__ inline u32 capacity() const;
            __host__ __device__ inline T * getData() const;
            __host__ __device__ inline u32 getDataSize() const;
            __host__ __device__ inline static u32 getDataSize(u32 capacity);
            __host__ __device__ Array<T> & operator=(Array<T> & other) = delete;
            __host__ __device__ Array<T> & operator=(Array<T> const & other) = delete;
            __host__ __device__ inline Array<T> & operator=(Array<T> && other);
            __host__ __device__ void print() const;
    };

    template<typename T>
    __host__ __device__
    Array<T>::Array(u32 capacity, T * data) :
           Slice<T>(capacity, data)
    {}

    template<typename T>
    __host__ __device__
    T * Array<T>::at(u32 index) const
    {
        return Slice<T>::at(index);
    }

    template<typename T>
    __host__ __device__
    T * Array<T>::begin() const
    {
        return Slice<T>::begin();
    }

    template<typename T>
    __host__ __device__
    T * Array<T>::first() const
    {
        return Slice<T>::first();
    }

    template<typename T>
    __host__ __device__
    T * Array<T>::last() const
    {
        return Slice<T>::last();
    }

    template<typename T>
    __host__ __device__
    T * Array<T>::getData() const
    {
        return this->_data;
    }

    template<typename T>
    __host__ __device__
    T * Array<T>::end() const
    {
        return Slice<T>::end();
    }

    template<typename T>
    __host__ __device__
    u32 Array<T>::capacity() const
    {
        return this->_capacity;
    }

    template<typename T>
    __host__ __device__
    Array<T> & Array<T>::operator=(Array<T> && other)
    {
        if (this != &other)
        {
            Slice<T>::operator=(other);
        }
        return *this;
    }

    template<typename T>
    __host__ __device__
    void Array<T>::print() const
    {
        Slice<T>::print(begin(), end());
    }

    template<typename T>
    __host__ __device__
    u32 Array<T>::getDataSize() const
    {
       return getDataSize(this->_capacity);
    }

    template<typename T>
    __host__ __device__
    u32 Array<T>::getDataSize(u32 capacity)
    {
        return sizeof(T) * capacity;
    }
}