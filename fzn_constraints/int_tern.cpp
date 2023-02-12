#include <cmath>
#include <limits>

#include "int_tern.hpp"
#include <libfca/Utils.hpp>

void calMulBounds(int aMin, int aMax, int bMin, int bMax, int & min, int & max)
{
    int bounds[4];
    bounds[0] = aMin * bMin;
    bounds[1] = aMax * bMin;
    bounds[2] = aMin * bMax;
    bounds[3] = aMax * bMax;

    min = std::numeric_limits<int>::max();
    max = std::numeric_limits<int>::min();
    for(int i = 0; i < 4; i += 1)
    {
        min = std::min(min, bounds[i]);
        max = std::max(max, bounds[i]);
    }
}

void calDivBounds(int aMin, int aMax, int bMin, int bMax, int & min, int & max)
{
    assert(bMin != 0 or bMax != 0);

    auto _aMin = static_cast<double>(aMin);
    auto _aMax = static_cast<double>(aMax);
    auto _bMin = static_cast<double>(bMin != 0 ? bMin : 1);
    auto _bMax = static_cast<double>(bMax != 0 ? bMax : -1);

    double bounds[4];
    bounds[0] = _aMin / _bMin;
    bounds[1] = _aMax / _bMin;
    bounds[0] = _aMin / _bMax;
    bounds[1] = _aMax / _bMax;

    min = std::numeric_limits<int>::max();
    max = std::numeric_limits<int>::min();
    for (int i = 0; i < 4; i += 1)
    {
        min = std::min(min, static_cast<int>(ceil(bounds[i])));
        max = std::max(max, static_cast<int>(floor(bounds[i])));
    }
}

void calPowBounds(int aMin, int aMax, int bMin, int bMax, int & min, int & max)
{
    using namespace Fca::Utils::Math;

    min = std::numeric_limits<int>::max();
    max = std::numeric_limits<int>::min();

    if (aMax > 0 and bMax > 0)
    {
        int aMinPos = std::max(1, aMin);
        int bMinPos = std::max(1, bMin);
        min = std::min(min, static_cast<int>(ceil(pow(aMinPos, bMinPos))));
        max = std::max(max, static_cast<int>(floor(pow(aMax, bMax))));
    }
    if (aMax > 0 and bMin < 0)
    {
        int aMinPos = std::max(1, aMin);
        int bMaxNeg = std::min(-1, bMax);
        min = std::min(min, static_cast<int>(ceil(pow(aMax, bMin))));
        max = std::max(max, static_cast<int>(floor(pow(aMinPos, bMaxNeg))));
    }
    if (aMin < 0 and bMax > 0)
    {
        min = std::min(min, static_cast<int>(ceil(pow(aMin, bMax))));
        min = std::min(min, static_cast<int>(ceil(pow(aMin, bMax-1))));
        max = std::max(max, static_cast<int>(floor(pow(aMin, bMax))));
        max = std::max(max, static_cast<int>(floor(pow(aMin, bMax-1))));
    }
    if (aMin < 0 and bMin < 0)
    {
        int aMaxNeg = std::min(-1, aMax);
        int bMaxNeg = std::min(-1, bMax);
        min = std::min(min, static_cast<int>(ceil(pow(aMaxNeg, bMaxNeg))));
        min = std::min(min, static_cast<int>(ceil(pow(aMaxNeg, bMaxNeg-1))));
        max = std::max(max, static_cast<int>(floor(pow(aMaxNeg, bMaxNeg))));
        max = std::max(max, static_cast<int>(floor(pow(aMaxNeg, bMaxNeg-1))));
    }
}

void calPowBounds(int aMin, int aMax, double bVal, int & min, int & max)
{
    min = std::numeric_limits<int>::max();
    max = std::numeric_limits<int>::min();

    if (aMax > 0 and bVal > 0)
    {
        int aMinPos = std::max(1, aMin);
        min = std::min(min, static_cast<int>(ceil(pow(aMinPos, bVal))));
        max = std::max(max, static_cast<int>(floor(pow(aMax, bVal))));
    }
    if (aMax > 0 and bVal < 0)
    {
        int aMinPos = std::max(1, aMin);
        min = std::min(min, static_cast<int>(ceil(pow(aMax, bVal))));
        max = std::max(max, static_cast<int>(floor(pow(aMinPos, bVal))));
    }
    if (aMin < 0 and bVal > 0)
    {
        min = std::min(min, static_cast<int>(ceil(pow(aMin, bVal))));
        max = std::max(max, static_cast<int>(floor(pow(aMin, bVal))));
    }
    if (aMin < 0 and bVal < 0)
    {
        int aMaxNeg = std::min(-1, aMax);
        min = std::min(min, static_cast<int>(ceil(pow(aMaxNeg, bVal))));
        max = std::max(max, static_cast<int>(floor(pow(aMaxNeg, bVal))));
    }
}

int_tern::int_tern(var<int>::Ptr const a, var<int>::Ptr const b, var<int>::Ptr const c) :
    Constraint(a->getSolver()), _a(a), _b(b),_c(c)
{}

void int_tern::post()
{
    _a->propagateOnBoundChange(this);
    _b->propagateOnBoundChange(this);
    _c->propagateOnBoundChange(this);
}

void int_div::propagate()
{
    // Semantic: a / b = c
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();
    int cMin = _c->min();
    int cMax = _c->max();
    int boundsMin;
    int boundsMax;

    // b != 0
    if (_b->contains(0))
    {
        _b->remove(0);
    }

    //Propagation: a / b -> c
    if(bMin != 0 or bMax != 0)
    {
        calDivBounds(aMin, aMax, bMin, bMax, boundsMin, boundsMax);
        _c->updateBounds(boundsMin, boundsMax);
    }

    //Propagation: a / b <- c
    calMulBounds(cMin, cMax, bMin, bMax, boundsMin, boundsMax);
    _a->updateBounds(boundsMin, boundsMax);
    if (cMin != 0 or cMax != 0)
    {
        calDivBounds(aMin, aMax, cMin, cMax, boundsMin, boundsMax);
        _b->updateBounds(boundsMin, boundsMax);
    }
}

void int_max::propagate()
{
    //Semantic: max(a,b) = c
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();
    int cMax = _c->max();

    //Propagation: max(a,b) -> c
    _c->updateBounds(std::max(aMin, bMin), std::max(aMax, bMax));

    //Propagation: max(a,b) <- c
    _a->removeAbove(cMax);
    _b->removeAbove(cMax);
}

void int_min::propagate()
{
    //Semantic: min(a,b) = c
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();
    int cMin = _c->min();

    //Propagation: min(a,b) -> c
    _c->updateBounds(std::min(aMin, bMin), std::min(aMax, bMax));

    //Propagation: min(a,b) <- c
    _a->removeBelow(cMin);
    _b->removeBelow(cMin);
}

void int_mod::propagate()
{
    //Semantic: a % b = c (Reminder of integer division)
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    //Propagation a % b -> c
    if(aMin > 0 and bMin > 0)
    {
        _c->removeAbove(std::min(aMax, bMax-1));
        if(aMax < bMin)
        {
            _c->removeBelow(aMin);
        }
    }
    else if(aMax < 0 and bMax < 0)
    {
        _c->removeBelow(std::max(aMin, bMin+1));
        if(aMax > bMax)
        {
            _c->removeAbove(aMax);
        }
    }
    else if (aMin > 0 and bMax < 0)
    {
        _c->removeAbove(std::min(aMax, -bMin-1));
        if(aMax < -bMax)
        {
            _c->removeBelow(aMin);
        }
    }
    else if(aMax < 0 and bMin > 0)
    {
        _c->removeBelow(std::max(aMin, -bMax+1));
        if(aMax > -bMin)
        {
            _c->removeAbove(aMax);
        }
    }
    else if((aMin == aMax and aMin == 0) or (bMin == bMax and bMin == 1))
    {
        _c->assign(0);
    }
    else if(aMin == aMax and bMin == bMax)
    {
        _c->assign(aMin % bMin);
    }

    //Propagation a % b <- c
    int cMin  = _c->min();
    int cMax  = _c->max();
    if(aMin > 0 and bMin > 0)
    {
        _a->removeBelow(cMin);
        if(aMax > cMax)
        {
            _b->removeAbove(cMax+1);
        }
    }
    else if(aMax < 0 and bMax < 0)
    {
        _a->removeAbove(cMax);
        if(aMin < cMin)
        {
            _b->removeBelow(cMin-1);
        }
    }
    else if (aMin > 0 and bMax < 0)
    {
        _a->removeBelow(cMin);
        if(aMax > cMin)
        {
            _b->removeBelow(-cMax-1);
        }
    }
    else if(aMax < 0 and bMin > 0)
    {
        _a->removeAbove(cMax);
        if(aMin < cMin)
        {
            _b->removeAbove(-cMin+1);
        }
    }
}

void int_pow::propagate()
{
    using namespace Fca::Utils::Math;

    //Semantic: a ^ b = c
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();
    int boundsMin;
    int boundsMax;

    //Propagation: a ^ b -> c
    calPowBounds(aMin, aMax, bMin, bMax, boundsMin, boundsMax);
    _c->updateBounds(boundsMin, boundsMax);
    if(bMin == bMax and bMin == 0 and (not _a->contains(0)))
    {
        _c->assign(1);
    }
    if(aMin == aMax and aMin == 0 and bMin > 0)
    {
        _c->assign(0);
    }

    //Propagation: a ^ b <- c
    int cMin = _c->min();
    int cMax = _c->max();
    if(aMin > 1 and cMin > 1)
    {
        boundsMin = static_cast<int>(ceil(log(aMax, cMin)));
        boundsMax = static_cast<int>(floor(log(aMin, cMax)));
        _b->updateBounds(boundsMin, boundsMax);
    }
    if(bMin == bMax and bMin != 0)
    {
        calPowBounds(cMin, cMax, static_cast<double>(1) / static_cast<double>(bMin), boundsMin, boundsMax);
        _a->updateBounds(boundsMin, boundsMax);
        if (cMin == cMax and cMin == 0)
        {
            _a->assign(0);
        }
    }
}

void int_times::propagate()
{
    //Semantic: a * b = c
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();
    int cMin = _c->min();
    int cMax = _c->max();
    int boundsMin;
    int boundsMax;

    //Propagation: a * b -> c
    calMulBounds(aMin, aMax, bMin, bMax, boundsMin, boundsMax);
    _c->updateBounds(boundsMin, boundsMax);

    //Propagation: a * b <- c
    if (bMin != 0 or bMax != 0)
    {
        calDivBounds(cMin, cMax, bMin, bMax, boundsMin, boundsMax);
        _a->updateBounds(boundsMin, boundsMax);
    }
    if (aMin != 0 or aMax != 0)
    {
        calDivBounds(cMin, cMax, aMin, aMax, boundsMin, boundsMax);
        _b->updateBounds(boundsMin, boundsMax);
    }
}