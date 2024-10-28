#include "extern.hpp"

Extern::Extern(std::vector<var<int>::Ptr> & x):
        Constraint(x[0]->getSolver()), _x(x)
{

    setPriority(CLOW);
    // Examples:
    //Initialization backtrackable int vector: [3,3,3,3,3,3,3,3,3,3]
    //for (int i = 0; i < 10; i  += 1)
    //{
    //    biv.push_back(trail<int>(x[0]->getSolver()->getStateManager(), 3));
    //}

}

void Extern::post()
{
    for (auto const & v : _x)
    {
        // v->propagateOnBoundChange(this);
        // v->whenBoundsChange([this, v] {v->removeAbove(0);});
    }
    propagate();
}

void Extern::propagate()
{
    // Implement the propagation logic
    printf("%%%%%% Extern propagation called.\n");
}
