#include "bool_misc.hpp"

bool_clause::bool_clause(const std::vector<var<bool>::Ptr> & as, const std::vector<var<bool>::Ptr> & bs) :
        Constraint(as.at(0)->getSolver()), _as(as), _bs(bs)
{}

void bool_clause::post()
{
    for (auto const a : _as)
    {
        a->propagateOnBind(this);
    }
    for (auto const b : _bs)
    {
        b->propagateOnBind(this);
    }
}

void bool_clause::propagate()
{
    //Semantic: (as[0] \/ ... \/ as[n]) \/ (-bs[0] \/ ... \/ -bs[m])
    int asUnboundCount = 0;
    int bsUnboundCount = 0;
    var<bool>::Ptr asNotBound;
    var<bool>::Ptr bsNotBound;
    for (auto x : _as)
    {
        if (not x->isBound())
        {
            asUnboundCount += 1;
            asNotBound = x;
        }
        else if (x->isTrue())
        {
            setActive(false);
            return;
        }
    }
    for (auto x : _bs)
    {
        if (not x->isBound())
        {
            bsUnboundCount += 1;
            bsNotBound = x;
        }
        else if (x->isFalse())
        {
            setActive(false);
            return;
        }
    }

    if (asUnboundCount == 0 and bsUnboundCount == 0)
    {
        failNow();
    }
    else if (asUnboundCount == 1 and bsUnboundCount == 0)
    {
        asNotBound->assign(true);
    }
    else if (asUnboundCount == 0 and bsUnboundCount == 1)
    {
        bsNotBound->assign(false);
    }
}
