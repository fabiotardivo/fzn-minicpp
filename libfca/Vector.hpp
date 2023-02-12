#pragma once

#include "Array.hpp"

namespace Fca
{
    template<typename T>
    class Vector : public Array<T>
    {
        // Members
        private:
            u32 _size;

        // Functions
        public:
            __host__ __device__ inline Vector(u32 capacity, T * data);
            __host__ __device__ inline T * at(u32 index) const;
            __host__ __device__ inline T * begin() const;
            __host__ __device__ inline T * end() const;
            __host__ __device__ inline u32 size() const;
            __host__ __device__ inline u32 capacity() const;
            __host__ __device__ inline void resize(u32 size);
            __host__ __device__ inline void clear();
            __host__ __device__ inline T * getData() const;
            __host__ __device__ inline u32 getDataSize() const;
            __host__ __device__ inline static u32 getDataSize(u32 capacity);
            __host__ __device__ Vector<T> & operator=(Vector<T> & other) = delete;
            __host__ __device__ Vector<T> & operator=(Vector<T> const & other) = delete;
            __host__ __device__ inline Vector<T> & operator=(Vector<T> && other);
            __host__ __device__ inline void push_back(T t);
            __host__ __device__ void print() const;
    };

    template<typename T>
    __host__ __device__
    Vector<T>::Vector(u32 capacity, T * data) :
            Array<T>(capacity, data), _size(0)
    {}

    template<typename T>
    __host__ __device__
    T * Vector<T>::at(u32 index) const
    {
        return Array<T>::at(index);
    }

    template<typename T>
    __host__ __device__
    T * Vector<T>::begin() const
    {
        return Array<T>::begin();
    }

    template<typename T>
    __host__ __device__
    T * Vector<T>::end() const
    {
        return Array<T>::begin() + _size;
    }

    template<typename T>
    __host__ __device__
    u32 Vector<T>::size() const
    {
        return _size;
    }

    template<typename T>
    __host__ __device__
    u32 Vector<T>::capacity() const
    {
        return Array<T>::capacity();
    }

    template<typename T>
    __host__ __device__
    Vector<T> & Vector<T>::operator=(Vector<T> && other)
    {
        if (this != &other)
        {
            Array<T>::operator=(other);
            other._size = 0;
        }
        return *this;
    }

    template<typename T>
    __host__ __device__ void
    Vector<T>::print() const
    {
        Array<T>::print(begin(), end());
    }

      template<typename T>
    __host__ __device__
    void Vector<T>::push_back(T t)
    {
        assert(_size < this->_size);
        _size += 1;
        *at(_size - 1) = t;
    }

    template<typename T>
    __host__ __device__
    void Vector<T>::clear()
    {
        resize(0);
    }

    template<typename T>
    __host__ __device__
    void Vector<T>::resize(u32 size)
    {
        assert(size <= Array<T>::capacity());
        _size = size;
    }

    template<typename T>
    __host__ __device__
    T * Vector<T>::getData() const
    {
        return Array<T>::getData();
    }

    template<typename T>
    __host__ __device__
    u32 Vector<T>::getDataSize() const
    {
        return Array<T>::getDataSize(Array<T>::capacity());
    }

    template<typename T>
    __host__ __device__
    u32 Vector<T>::getDataSize(u32 capacity)
    {
        return Array<T>::getDataSize(capacity);
    }
}
