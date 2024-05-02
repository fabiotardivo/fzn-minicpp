#pragma once

#include <libminicpp/varitf.hpp>


class StableMatching : public Constraint
{
    // Constraint private data structures
    protected:
        std::vector<var<int>::Ptr> & _m;
        std::vector<var<int>::Ptr> & _w;
        std::vector<std::vector<int>> const & _pm;
        std::vector<std::vector<int>> const & _pw;

        // Examples:
        // Backtrackable int vector
        // std::vector<trail<int>> biv;

    public:
        StableMatching(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & pm, std::vector<std::vector<int>> const & pw);
        void post() override;
        void propagate() override;
};


