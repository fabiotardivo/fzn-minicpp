#include "bool_bin.hpp"

bool_bin::bool_bin(var<bool>::Ptr a, var<bool>::Ptr b) :
    Constraint(a->getSolver()), _a(a), _b(b)
{}

void bool_bin::post()
{
    _a->propagateOnBind(this);
    _b->propagateOnBind(this);
}

bool_bin_imp::bool_bin_imp(var<bool>::Ptr a, var<bool>::Ptr b, var<bool>::Ptr r, Constraint::Ptr pfi, std::function<void()> bi) :
        bool_bin(a,b), _r(r), _pfi(pfi), _bi(bi)
{}

void bool_bin_imp::post()
{
    bool_bin::post();
    _r->propagateOnBind(this);
}

void bool_bin_imp::propagate()
{
    // r -> a ? b
    if (_r->isTrue())
    {
        _pfi->propagate();
    }

    // r <- a ? b
    _bi();
}

bool_bin_reif::bool_bin_reif(var<bool>::Ptr a, var<bool>::Ptr b, var<bool>::Ptr r, Constraint::Ptr pfi, Constraint::Ptr nfi, std::function<void()> bi) :
        bool_bin_imp(a,b,r,pfi,bi), _nfi(nfi)
{}

void bool_bin_reif::propagate()
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

void bool_and::propagate()
{
    //Semantic: a /\ b
    _a->assign(true);
    _b->assign(true);
}

void bool_nand::propagate()
{
    //Semantic: -(a /\ b)
    if (_a->isTrue())
    {
        _b->assign(false);
    }
    else if (_b->isTrue())
    {
        _a->assign(false);
    }
}

void bool_or::propagate()
{
    //Semantic: a \/ b
    if (_a->isFalse())
    {
        _b->assign(true);
    }
    else if (_b->isFalse())
    {
        _a->assign(true);
    }
}

void bool_nor::propagate()
{
    //Semantic: -(a \/ b)
    _a->assign(false);
    _b->assign(false);
}

void bool_xor::propagate()
{
    //Semantic: a + b
    if (_a->isTrue())
    {
        _b->assign(false);
    }
    else if (_a->isFalse())
    {
        _b->assign(true);
    }
    else if (_b->isTrue())
    {
        _a->assign(false);
    }
    else if (_b->isFalse())
    {
        _a->assign(true);
    }
}

void bool_nxor::propagate()
{
    //Semantic: -(a + b)
    if (_a->isTrue())
    {
        _b->assign(true);
    }
    else if (_a->isFalse())
    {
        _b->assign(false);
    }
    else if (_b->isTrue())
    {
        _a->assign(true);
    }
    else if (_b->isFalse())
    {
        _a->assign(false);
    }
}
