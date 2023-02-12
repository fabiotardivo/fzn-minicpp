#include "int_bin.hpp"

int_bin::int_bin(var<int>::Ptr a, var<int>::Ptr b) :
    Constraint(a->getSolver()), _a(a), _b(b)
{}

void int_bin::post()
{
    _a->propagateOnBoundChange(this);
    _b->propagateOnBoundChange(this);
}

int_bin_imp::int_bin_imp(var<int>::Ptr a, var<int>::Ptr b, var<bool>::Ptr r, Constraint::Ptr pfi, std::function<void()> & bi) :
        int_bin(a,b), _r(r), _pfi(pfi), _bi(std::move(bi))
{}

void int_bin_imp::post()
{
    int_bin::post();
    _r->propagateOnBind(this);
}

void int_bin_imp::propagate()
{
    // r -> a ? b
    if (_r->isTrue())
    {
        _pfi->propagate();
    }

    // r <- a ? b
    _bi();
}

int_bin_reif::int_bin_reif(var<int>::Ptr a, var<int>::Ptr b, var<bool>::Ptr r, Constraint::Ptr pfi, Constraint::Ptr nfi, std::function<void()> & bi) :
        int_bin_imp(a,b,r,pfi,bi), _nfi(nfi)
{}

void int_bin_reif::propagate()
{
    // r -> a ? b
    if (_r->isTrue())
    {
        _pfi->propagate();
    }
    else if (_r->isFalse())
    {
        _nfi->propagate();
    }

    // r <- a ? b
    _bi();
}

void int_abs::propagate()
{
    //Semantic: b = |a|
    int aMin = _a->min();
    int aMax = _a->max();
    int bMin = _b->min();
    int bMax = _b->max();

    // b >= 0
    if (bMin < 0)
    {
        _b->removeBelow(0);
    }

    //Propagation: b <- |a|
    if(aMin >= 0)
    {
        bMin = std::max(bMin, aMin);
        bMax = std::min(bMax, aMax);
    }
    else if (aMax <= 0)
    {
        bMin = std::max(bMin, -aMax);
        bMax = std::min(bMax, -aMin);
    }
    else
    {
        bMin = std::max(bMin, 0);
        bMax = std::min(bMax, std::max(-aMin, aMax));
    }
    _b->updateBounds(bMin,bMax);

    //Propagation: b -> |a|
    if(aMin >= 0)
    {
        aMin = std::max(aMin, bMin);
        aMax = std::min(aMax, bMax);
    }
    else if (aMax <= 0)
    {
        aMin = std::max(aMin, -bMax);
        aMax = std::min(aMax, -bMin);
    }
    else
    {
        aMin = std::max(aMin, -bMax);
        aMax = std::min(aMax, bMax);
    }
    _a->updateBounds(aMin,aMax);
}