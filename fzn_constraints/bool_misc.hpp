#pragma once

#include <libminicpp/intvar.hpp>

class bool_clause : public Constraint
{
    protected:
        std::vector<var<bool>::Ptr> _as;
        std::vector<var<bool>::Ptr> _bs;

    public:
        bool_clause(std::vector<var<bool>::Ptr> const & as, std::vector<var<bool>::Ptr> const & bs);
        void post() override;
        void propagate() override;
};
