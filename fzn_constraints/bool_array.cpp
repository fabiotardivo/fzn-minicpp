#include "bool_array.hpp"

array_bool::array_bool(std::vector<var<bool>::Ptr> const & as) :
    Constraint(as.at(0)->getSolver()), _as(as)
{}

void array_bool::post()
{
    for(auto x : _as)
    {
        x->propagateOnBind(this);
    }
}

array_bool_imp::array_bool_imp(std::vector<var<bool>::Ptr> const & as, var<bool>::Ptr r, Constraint::Ptr pfi, std::function<void()> bi) :
        array_bool(as), _r(r), _pfi(pfi), _bi(bi)
{}

void array_bool_imp::post()
{
    array_bool::post();
    _r->propagateOnBind(this);
}

void array_bool_imp::propagate()
{
    // r -> as[0] ? ... ? as[n]
    if (_r->isTrue())
    {
        _pfi->propagate();
    }

    // r <- as[0] ? ... ? as[n]
    _bi();
}

array_bool_reif::array_bool_reif(std::vector<var<bool>::Ptr> const & as, var<bool>::Ptr r, Constraint::Ptr pfi, Constraint::Ptr nfi, std::function<void()> bi) :
        array_bool_imp(as,r,pfi,bi), _nfi(nfi)
{}

void array_bool_reif::propagate()
{
    // r -> as[0] ? ... ? as[n]
    if (_r->isTrue())
    {
        _pfi->propagate();
    }
    else if (_r->isFalse())
    {
        _nfi->propagate();
    }

    // r <- as[0] ? ... ? as[n]
    _bi();
}

void array_bool_and::propagate()
{
    // Semantic: as1 /\ ... /\ asn
    for (auto x: _as)
    {
        x->assign(true);
    }
}

void array_bool_nand::propagate()
{
    // Semantic: -(as1 /\ ... /\ asn)
    int notBoundCount = 0;
    var<bool>::Ptr asNotBound = nullptr;
    for (auto x: _as)
    {
        if (not x->isBound())
        {
            notBoundCount += 1;
            asNotBound = x;
        }
        else if (x->isFalse())
        {
            setActive(false);
            return;
        }
    }
    if (notBoundCount == 0)
    {
        failNow();
    }
    else if (notBoundCount == 1)
    {
        asNotBound->assign(false);
    }
}

void array_bool_or::propagate()
{
    // Semantic: as1 \/ ... \/ asn
    int notBoundCount = 0;
    var<bool>::Ptr asNotBound = nullptr;
    for (auto x : _as)
    {
        if(not x->isBound())
        {
            notBoundCount += 1;
            asNotBound = x;
        }
        else if (x->isTrue())
        {
            setActive(true);
            return;
        }
    }
    if (notBoundCount == 0)
    {
        failNow();
    }
    else if (notBoundCount == 1)
    {
        asNotBound->assign(true);
    }
}

void array_bool_nor::propagate()
{
    // Semantic: -(as1 \/ ... \/ asn)
    for (auto x: _as)
    {
        x->assign(false);
    }
}

void array_bool_xor::propagate()
{
    //Semantic: as1 + ... + asn (The number of true variables is odd)
    int trueCount = 0;
    int notBoundCount = 0;
    var<bool>::Ptr asNotBound = nullptr;
    for (auto x : _as)
    {
        if(not x->isBound())
        {
            notBoundCount += 1;
            asNotBound = x;
        }
        else if (x->isTrue())
        {
            trueCount += 1;
        }
    }

    if (notBoundCount == 1)
    {
        asNotBound->assign(trueCount % 2 == 0);
    }
}