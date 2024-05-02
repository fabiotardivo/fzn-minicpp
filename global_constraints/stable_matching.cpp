#include "stable_matching.hpp"

StableMatching::StableMatching(std::vector<var<int>::Ptr> & m, std::vector<var<int>::Ptr> & w, std::vector<std::vector<int>> const & pm, std::vector<std::vector<int>> const & pw) :
    Constraint(m[0]->getSolver()), _m(m), _w(w), _pm(pm), _pw(pw)
{
    setPriority(CLOW);

    // Examples:
    // Initialization backtrackable int vector: [3,3,3,3,3,3,3,3,3,3]
    //for (int i = 0; i < 10; i  += 1)
    //{
    //    biv.push_back(trail<int>(x[0]->getSolver()->getStateManager(), 3));
    //}
}

void StableMatching::post()
{
    for (auto const & v : _m)
    {
        // v->propagateOnBoundChange(this);
        // v->whenBoundsChange([this, v] {v->removeAbove(0);});
    }

    for (auto const & v : _w)
    {
        // v->propagateOnBoundChange(this);
        //v->whenBoundsChange([this, v] {v->removeAbove(0);});
    }

    propagate();
}

void StableMatching::propagate()
{
    // Implement the propagation logic
    printf("%%%%%% Stable matching propagation called.\n");
}
