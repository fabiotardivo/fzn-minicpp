#pragma once

#include <libminicpp/intvar.hpp>

class array_int : public Constraint
{
    protected:
        var<int>::Ptr _m;
        std::vector<var<int>::Ptr> _x;

    public:
        array_int(var<int>::Ptr const m, std::vector<var<int>::Ptr> const & x);
        void post() override;
};

class array_int_maximum : public array_int
{
    public:
        array_int_maximum(var<int>::Ptr const m, std::vector<var<int>::Ptr> const & x) : array_int(m,x) {};
        void propagate() override;
};

class array_int_minimum : public array_int
{
    public:
        array_int_minimum(var<int>::Ptr const m, std::vector<var<int>::Ptr> const & x) : array_int(m,x) {};
        void propagate() override;
};
