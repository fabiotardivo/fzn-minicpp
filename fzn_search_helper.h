#pragma once

#include <functional>

#include <cxxopts.hpp>

#include "Model.h"
#include "search.hpp"
#include "fzn_variables_helper.h"

class FznSearchHelper
{
    private:
        CPSolver::Ptr solver;
        FznVariablesHelper & fvh;

    public:
        FznSearchHelper(CPSolver::Ptr solver, FznVariablesHelper & fvh);
        std::function<Branches(void)> getSearchStrategy(Fzn::Model const & fzn_model);
        std::function<Branches(void)> getSampleStrategy(Fzn::Model const & fzn_model);
        template <typename EFun>
        std::function<Branches(void)> getMLSearchStrategy(Fzn::Model const & fzn_model, int nInitiallyAssigned, EFun eval_fun);
        std::vector<var<int>::Ptr> getIntDecisionalVars(Fzn::Model const & fzn_model);
        std::vector<var<int>::Ptr> getIntDecisionalVars(Fzn::var_expr_t var_expr);
        std::vector<var<bool>::Ptr> getBoolDecisionalVars(Fzn::var_expr_t vars_expr);
        static Limit makeSearchLimits(Fzn::Model const & fzn_model, cxxopts::ParseResult const & args);

    private:
        std::function<Branches(void)> makeBasicSearchStrategy(Fzn::basic_search_annotation_t const & basic_search_annotation);
        std::function<Branches(void)> makeBasicSampleStrategy(Fzn::basic_search_annotation_t const & basic_search_annotation);
        template<typename Vars, typename Var>
        static std::function<Var(Vars const &)> makeVariableSelection(Fzn::pred_identifier_t const & variable_selection);
        template <typename Var>
        static std::function<Branches(CPSolver::Ptr, Var)> makeValueSelection(Fzn::pred_identifier_t const & value_selection);
        static unsigned int getMaxSolutions(Fzn::Model const & fzn_model, cxxopts::ParseResult const & args);
        static unsigned int getMaxSearchTime(cxxopts::ParseResult const & args);

};

template<typename Vars, typename Var>
std::function<Var(Vars const &)> FznSearchHelper::makeVariableSelection(Fzn::pred_identifier_t const & variable_selection)
{
    using namespace std;

    if (variable_selection == "first_fail")
    {
        return [](Vars const & vars) -> Var { return first_fail<Vars,Var>(vars); };
    }
    else if (variable_selection == "input_order")
    {
        return [](Vars const &  vars) -> Var { return input_order<Vars,Var>(vars); };
    }
    else if (variable_selection == "smallest")
    {
        return [](Vars const &  vars) -> Var { return smallest<Vars,Var>(vars); };
    }
    else if (variable_selection == "largest")
    {
        return [](Vars const &  vars) -> Var { return largest<Vars,Var>(vars); };
    }
    else if (variable_selection == "random")
    {
        return [](Vars const &  vars) -> Var { return random<Vars,Var>(vars); };
    }
    else
    {
        stringstream msg;
        msg << "Unsupported variable selection : " << variable_selection;
        throw runtime_error(msg.str());
    }
}

template<typename Var>
std::function<Branches(CPSolver::Ptr, Var)> FznSearchHelper::makeValueSelection(Fzn::pred_identifier_t const & value_selection)
{
    using namespace std;

    if (value_selection == "indomain_min")
    {
        return [](CPSolver::Ptr s, Var var) -> Branches {return indomain_min<Var>(s, var);};
    }
    else if (value_selection == "indomain_max")
    {
        return [](CPSolver::Ptr s, Var var) -> Branches {return indomain_max<Var>(s, var);};
    }
    else if (value_selection == "indomain_split")
    {
        return [](CPSolver::Ptr s, Var var) -> Branches { return indomain_split<Var>(s, var); };
    }
    else if (value_selection == "indomain_random")
    {
        return [](CPSolver::Ptr s, Var var) -> Branches { return indomain_random<Var>(s, var); };
    }
    else
    {
        stringstream msg;
        msg << "Unsupported value selection : " << value_selection;
        throw runtime_error(msg.str());
    }
}

template<typename EFun>
std::function<Branches(void)> FznSearchHelper::getMLSearchStrategy(Fzn::Model const & fzn_model, int nInitiallyAssigned, EFun eval_fun)
{
    using namespace std;

    for (auto const & search_annotation: fzn_model.search_strategy)
    {
        if (holds_alternative<Fzn::basic_search_annotation_t>(search_annotation))
        {
            auto const & basic_search_annotation = get<Fzn::basic_search_annotation_t>(search_annotation);
            auto search_strategy = makeBasicSearchStrategy(basic_search_annotation);
            auto const & pred_identifier = get<0>(basic_search_annotation);
            auto const & var_expr = get<1>(basic_search_annotation);
            if (pred_identifier == "int_search")
            {
                using int_var_t = var<int>::Ptr;
                using array_int_var_t = vector<int_var_t>;

                // Decision variables
                array_int_var_t array_int_var = getIntDecisionalVars(var_expr);
                auto const nVars = static_cast<int>(array_int_var.size());

                auto ml_search_strategy = [=]()
                {
                    int nAssigned = 0;
                    for (auto const & var : array_int_var)
                    {
                        nAssigned += var->isBound();
                    }

                    if (true) //nAssigned - nInitiallyAssigned < 5)
                    {
                        int_var_t bestVar = nullptr;
                        auto bestVal = std::numeric_limits<int>::max();;
                        float bestScore = std::numeric_limits<float>::max();
                        float uncBest = 0;
                        for (auto varIdx = 0; varIdx < nVars; varIdx += 1) {
                            auto const &var = array_int_var[varIdx];
                            if (not var->isBound()) {
                                auto [score, unc, val] = eval_fun(varIdx, array_int_var);
                                bool const smallerDomain = bestVar != nullptr ? var->size() < bestVar->size(): true;
                                if (score < bestScore or score == bestScore and smallerDomain)
                                {
                                    bestScore = score;
                                    bestVal = val;
                                    bestVar = var;
                                    uncBest = unc;
                                    //printf("New best score %5.3f for var[%3d] = %3d\n", bestScore, varIdx, val);
                                }
                            }
                        }
                        printf("Using ML strategy (Bounded vars %d/%d)\n", nAssigned, nVars);
                        return indomain_fixed(array_int_var[0]->getSolver(), bestVar, bestVal);
                    //if (uncBest < 0.15 and bestVar != nullptr)
                    //{
                    //     printf("Using ML strategy (Uncertainty %.2f)\r", uncBest);
                    //     return indomain_fixed(array_int_var[0]->getSolver(), bestVar, bestVal);
                    }
                    else
                    {
                        return search_strategy();
                    }
                };

                return  ml_search_strategy;
            }
            else
            {
                throw std::runtime_error("Unsupported search annotation");
            }
        }
        else
        {
            throw std::runtime_error("Unsupported search annotation");
        }
    }

    return {};
}


