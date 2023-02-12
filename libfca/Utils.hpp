#pragma once

#include <cmath>
#include <type_traits>

// Compile with C++ only compiler
#ifndef __NVCC__
#define __host__
#define __device__
#endif

namespace Fca::Utils
{
    namespace Math
    {
        template<typename T>
        __host__ __device__ inline
        double log(T base, T number)
        {
            static_assert(std::is_arithmetic_v<T>);
            return log2(static_cast<double>(number)) / log2(static_cast<double>(base));
        }

        template<typename T>
        __host__ __device__ inline
        double pow(T base, T exponent)
        {
            static_assert(std::is_arithmetic_v<T>);
            return pow(static_cast<double>(base), static_cast<double>(exponent));
        }

        template<typename T>
        __host__ __device__ inline
        double division(T a, T b)
        {
            static_assert(std::is_arithmetic_v<T>);
            return static_cast<double>(a) / static_cast<double>(b);
        }
    }
}