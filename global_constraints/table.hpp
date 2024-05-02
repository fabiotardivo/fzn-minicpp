#pragma once

#include <libminicpp/varitf.hpp>


class Table : public Constraint
{
    // Constraint private data structures
    protected:
        std::vector<var<int>::Ptr> & _x;
        std::vector<std::vector<int>> const & _t;

        // Examples:
        // Backtrackable int vector
        //std::vector<trail<int>> biv;

    public:
        Table(std::vector<var<int>::Ptr> & x,  std::vector<std::vector<int>> const & t);
        void post() override;
        void propagate() override;
};


