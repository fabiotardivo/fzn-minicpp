#include "table.hpp"

Table::Table(std::vector<var<int>::Ptr> & x, const std::vector<std::vector<int>> & t) :
    Constraint(x[0]->getSolver()), _x(x), _t(t)
{
    setPriority(CLOW);

    // Examples:
    // Initialization backtrackable int vector: [3,3,3,3,3,3,3,3,3,3]
    //for (int i = 0; i < 10; i  += 1)
    //{
    //    biv.push_back(trail<int>(x[0]->getSolver()->getStateManager(), 3));
    //}
}

void Table::post()
{
    for (auto const & v : _x)
    {
        // v->propagateOnBoundChange(this);
        // v->whenBoundsChange([this, v] {v->removeAbove(0);});
    }
    propagate();
}

void Table::propagate()
{
    printf("%%%%%% Table propagation called.\n");
    // Implement the propagation logic
}
