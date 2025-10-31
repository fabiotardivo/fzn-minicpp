#pragma once

#include <utility>
#include "Utils.hpp"
#include "Types.hpp"
#include <cassert>

#ifndef __NVCC__
struct alignas(16) uint4 { u32 x, y, z, w; };
#define __host__
#define __device__
#endif

namespace Fca
{
    using u128 = uint4;

    __host__ __device__ inline
    bool getBit(u128 const &n, u32 b)
    {
        assert(b < 128);
        u32 const *const tmp = reinterpret_cast<u32 const *>(&n);
        u32 const wordIdx = 3 - (b / 32);
        u32 const bitIdx = b % 32;
        u32 const mask = 1 << bitIdx;
        return (tmp[wordIdx] & mask) != 0;
    }

    __host__ __device__
    inline bool isZero(u128 const &n)
    {
        u64 const *_n = reinterpret_cast<u64 const *>(&n);
        return (_n[0] | _n[1]) == 0;
    }

    __host__ __device__
    inline void orBit(u128 &n, u32 b, bool value)
    {
        assert(b < 128);

        u32 *const tmp = reinterpret_cast<u32 *>(&n);
        u32 const wordIdx = 3 - (b >> 5);
        u32 const bitIdx = b & 31;
        u32 const mask = static_cast<u32>(value) << bitIdx;
        tmp[wordIdx] |= mask;
    }

#pragma GCC diagnostic ignored "-Wuninitialized"

    __host__ __device__
    inline u128 bitwiseAnd(u128 const &a, u128 const &b)
    {
        u128 result = a;
        u64 *const _result = reinterpret_cast<u64 *>(&result);
        u64 const *const _b = reinterpret_cast<u64 const *>(&b);
        _result[0] &= _b[0];
        _result[1] &= _b[1];
        return result;
    }

    __host__ __device__
    inline void bitwiseOr(u128 &a, u128 const &b)
    {
        u64 *const _a = reinterpret_cast<u64 *>(&a);
        u64 const *const _b = reinterpret_cast<u64 const *>(&b);
        _a[0] |= _b[0];
        _a[1] |= _b[1];
    }

    __host__ __device__
    inline void bitwiseXor(u128 &a, u128 const &b)
    {
        u64 *const _a = reinterpret_cast<u64 *>(&a);
        u64 const *const _b = reinterpret_cast<u64 const *>(&b);
        _a[0] ^= _b[0];
        _a[1] ^= _b[1];
    }

#pragma GCC diagnostic ignored "-Wuninitialized"

    __host__ __device__
    inline u32 firstBitSet(u128 const &n)
    {
        using namespace Utils::Bits;
        u128 const _n = {n.y, n.x, n.w, n.z};
        u64 const *__n = reinterpret_cast<u64 const *>(&_n);
#ifdef __CUDA_ARCH__
        u32 const tmp0 = __n[0] != 0 ?      getLeftmostOneIndex64(__n[0]) : UINT32_MAX;
        u32 const tmp1 = __n[1] != 0 ? 64 + getLeftmostOneIndex64(__n[1]) : UINT32_MAX;
#else
        u32 const tmp0 = __n[0] != 0 ? getLeftmostOneIndex64(__n[0]) : UINT32_MAX;
        u32 const tmp1 = __n[1] != 0 ? 64 + getLeftmostOneIndex64(__n[1]) : UINT32_MAX;
#endif
        return min(tmp0, tmp1);
    }
}