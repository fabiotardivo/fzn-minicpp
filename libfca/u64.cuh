#pragma once

#include "Types.hpp"
#include <cassert>

namespace Fca
{
    __host__ __device__
    inline bool getBit(u64 const &n, u32 b)
    {
        assert(b < 64);
        u32 const *tmp = reinterpret_cast<u32 const *>(&n);
        u32 const wordIdx = 1 - (b >> 5);
        u32 const bitIdx = b % 32;
        u32 const mask = 1 << bitIdx;
        return (tmp[wordIdx] & mask) != 0;
    }

    __host__ __device__
    inline void orBit(u64 &n, u32 b, bool value)
    {
        assert(b < 64);

        u32 *const tmp = reinterpret_cast<u32 *>(&n);
        u32 const wordIdx = 1 - (b >> 5);
        u32 const bitIdx = b & 31;
        u32 const mask = static_cast<u32>(value) << bitIdx;
        tmp[wordIdx] |= mask;
    }
}