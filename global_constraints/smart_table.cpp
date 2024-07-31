#include "smart_table.hpp"

SmartTable::SmartTable(std::vector<var<int>::Ptr> & x, const std::vector<std::vector<int>> & t, const std::vector<std::vector<int>> & sto) :
    Constraint(x[0]->getSolver()), _x(x), _t(t), _sto(sto)
{
    setPriority(CLOW);

    // Examples:
    // Initialization backtrackable int vector: [3,3,3,3,3,3,3,3,3,3]
    //for (int i = 0; i < 10; i  += 1)
    //{
    //    biv.push_back(trail<int>(x[0]->getSolver()->getStateManager(), 3));
    //}
}

void SmartTable::post()
{
    for (auto const & v : _x)
    {
        // v->propagateOnBoundChange(this);
        // v->whenBoundsChange([this, v] {v->removeAbove(0);});
    }
    propagate();
}

void SmartTable::propagate()
{
    printf("%%%%%% Smart Table propagation called.\n");
      // Implement the propagation logic
}
