#pragma once

#include <libminicpp/varitf.hpp>

class Extern : public Constraint
{
    // Constraint private data structures
    protected:
        std::vector<var<int>::Ptr> & _x;
        std::vector<int> _y;
        std::string const _l;

    public:
        Extern(std::vector<var<int>::Ptr> & x, std::vector<int> & y);
        void post() override;
        void propagate() override;
};

