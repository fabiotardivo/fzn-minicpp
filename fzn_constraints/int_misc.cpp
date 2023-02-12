#include "int_misc.hpp"

set_in::set_in(var<int>::Ptr x, std::vector<int> const & s) :
   Constraint(x->getSolver()), _x(x), _s(s.begin(), s.end())
{}

void set_in::post()
{
    _x->propagateOnDomainChange(this);
}

void set_in::propagate()
{
    int const xMax = _x->max();
    for(auto val = _x->min(); val <= xMax; val += 1)
    {
        if (_x->contains(val))
        {
            if (_s.count(val) == 0)
            {
                _x->remove(val);
            }
        }
    }
    setActive(false);
}
