#pragma once

#include <libminicpp/varitf.hpp>


class SmartTable : public Constraint
{
    // Constraint private data structures
    protected:
        std::vector<var<int>::Ptr> & _x;
        std::vector<std::vector<int>> const & _t;
        std::vector<std::vector<int>> const & _sto;

        // Examples:
        // Backtrackable int vector
        //std::vector<trail<int>> biv;

    public:
        SmartTable(std::vector<var<int>::Ptr> & x,  std::vector<std::vector<int>> const & t, std::vector<std::vector<int>> const & sto);
        void post() override;
        void propagate() override;
};


