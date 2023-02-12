#include <cmath>
#include "int_lin.hpp"

int_lin_imp::int_lin_imp(std::vector<var<int>::Ptr> const & bs, var<bool>::Ptr r, Constraint::Ptr pfi, std::function<void()> & bi) :
        Constraint(bs.at(0)->getSolver()), _bs(bs), _r(r), _pfi(pfi), _bi(std::move(bi))
{}

void int_lin_imp::post()
{
    for (auto const var : _bs)
    {
        var->propagateOnBoundChange(this);
    }
    _r->propagateOnBind(this);
}

void int_lin_imp::propagate()
{
    // r -> as[0] * bs[0] + ... + as[n] * bs[n] ? c
    if (_r->isTrue())
    {
        _pfi->propagate();
    }

    // r <- as[0] * bs[0] + ... + as[n] * bs[n] ? c
    _bi();
}

int_lin_reif::int_lin_reif(std::vector<var<int>::Ptr> const & bs, var<bool>::Ptr r, Constraint::Ptr pfi, Constraint::Ptr nfi, std::function<void()> & bi) :
        int_lin_imp(bs,r,pfi,bi), _nfi(nfi)
{}

void int_lin_reif::propagate()
{
    // r -> as[0] * bs[0] + ... + as[n] * bs[n] ? c
    if (_r->isTrue())
    {
        _pfi->propagate();
    }
    else if (_r->isFalse())
    {
        _nfi->propagate();
    }

    // r <- as[0] * bs[0] + ... + as[n] * bs[n] ? c
    _bi();
}