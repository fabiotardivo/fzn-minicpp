#pragma once

#include <variant>
#include <functional>

#include <libfzn/Types.h>
#include "fzn_variables_helper.h"

class FznConstraintHelper
{
    using constraint_builder_t = std::function<Constraint::Ptr(std::vector<Fzn::constraint_arg_t> const & args, std::vector<Fzn::annotation_t> const & anns)>;

    private:
        CPSolver::Ptr solver;
        FznVariablesHelper & fvh;
        std::unordered_map<Fzn::identifier_t, constraint_builder_t> constriants_builders;

    public:
        FznConstraintHelper(CPSolver::Ptr solver, FznVariablesHelper & fvh);
        bool makeConstraints(Fzn::Model const & fzn_model);
    private:
        void addIntConstraintsBuilders();
        void addBoolConstraintsBuilders();
        void addGlobalConstraintsBuilders();
        template <typename Var>
        static var<int>::Ptr makeLinSum(std::vector<int> const & as, std::vector<Var> const & bs);
};

template <typename Var>
var<int>::Ptr FznConstraintHelper::makeLinSum(std::vector<int> const & as, std::vector<Var> const & bs)
{
    auto const & solver = bs.at(0)->getSolver();
    auto const sum_size = bs.size();

    int sumMin = 0;
    int sumMax = 0;
    std::vector<var<int>::Ptr> vars(sum_size);
    for (auto i = 0; i < sum_size; i += 1)
    {
        var<int>::Ptr var;
        if (as.at(i) > 0)
        {
            var = new (solver) IntVarViewMul(bs.at(i), as.at(i));
        }
        else
        {
            var = new (solver) IntVarViewMul(bs.at(i), -as.at(i));
            var = new (solver) IntVarViewOpposite(var);
        }
        sumMin += var->min();
        sumMax += var->max();
        vars.at(i) = var;
    }
    auto sumVar = Factory::makeIntVar(solver,sumMin,sumMax);
    auto sum = new (solver) Sum(vars, sumVar);
    sum->post();
    return sumVar;
}
