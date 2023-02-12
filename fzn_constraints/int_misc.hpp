#pragma once

#include <set>
#include <libminicpp/intvar.hpp>

class set_in : public Constraint
{
    protected:
        var<int>::Ptr _x;
        std::set<int> _s;

    public:
        set_in(var<int>::Ptr x, std::vector<int> const & s);
        void post() override;
        void propagate() override;
};
