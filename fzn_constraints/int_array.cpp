#include <limits>

#include "int_array.hpp"

array_int::array_int(var<int>::Ptr const m, std::vector<var<int>::Ptr> const & x) :
        Constraint(m->getSolver()), _m(m), _x(x)
{}

void array_int::post()
{
    _m->propagateOnBoundChange(this);
    for (auto x : _x)
    {
        x->propagateOnBoundChange(this);
    }
}

void array_int_maximum::propagate()
{
    //Semantic: m = max(x1,...,xn)
    int mMin = std::numeric_limits<int>::min();
    int mMax = std::numeric_limits<int>::min();

    //Propagation: m <- max(x1,...,xn)
    for (auto x : _x)
    {
        mMin = std::max(mMin, x->min());
        mMax = std::max(mMax, x->max());
    }
    _m->updateBounds(mMin, mMax);

    //Propagation: m -> max(x1,...,xn)
    mMax = _m->max();
    for (auto x : _x)
    {
        x->removeAbove(mMax);
    }
}

void array_int_minimum::propagate()
{
    //Semantic: m = min(x1,...,xn)
    int mMin = std::numeric_limits<int>::max();
    int mMax = std::numeric_limits<int>::max();

    //Propagation: m <- min(x1,...,xn)
    for(auto x : _x)
    {
        mMin = std::min(mMin, x->min());
        mMax = std::min(mMax, x->max());
    }
    _m->updateBounds(mMin, mMax);

    //Propagation: m -> min(x1,...,xn)
    mMin = _m->min();
    for (auto x : _x)
    {
        x->removeBelow(mMin);
    }
}