#pragma once

#include <functional>

#include <cxxopts.hpp>

#include "Model.h"
#include "search.hpp"
#include "fzn_variables_helper.h"
#include "ml/utils.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

class FznSearchHelper
{
    private:
        CPSolver::Ptr solver;
        FznVariablesHelper & fvh;

    public:
        FznSearchHelper(CPSolver::Ptr solver, FznVariablesHelper & fvh);
        std::function<Branches(void)> getSearchStrategy(Fzn::Model const & fzn_model);
        std::function<Branches(void)> getSampleStrategy(Fzn::Model const & fzn_model);
        template<typename Inference>
        std::function<Branches(void)> getMLSearchStrategy(Fzn::Model const & fzn_model, ML::RankType valRank, ML::RankType varRank, Inference & infer);
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

template<typename Inference>
std::function<Branches(void)> FznSearchHelper::getMLSearchStrategy(Fzn::Model const & fzn_model, ML::RankType valRank, ML::RankType varRank, Inference & infer)
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
            auto const & annotations = get<2>(basic_search_annotation);


            if (pred_identifier == "int_search")
            {
                // Decision variables
                auto const array_int_var = getIntDecisionalVars(var_expr);
                auto const nVars = static_cast<int>(array_int_var.size());

                auto const varSel = [=,&infer]()
                {
                    auto pa = ML::getPartialAssignmentMask(array_int_var);
                    auto scores = infer.scoreVariables(pa);

                    ML::filterScores(scores,pa);

                    auto stats = ML::getStats(scores);
                    auto const [min_idx, max_idx, min_score, max_score, mean, eom] = stats;

                    // fmt::print("PA = [{}]\n", fmt::join(pa, ", "));
                    // fmt::print("SC = [{}]\n", fmt::join(scores, ", "));
                    // fmt::print("ST = [{}]\n", fmt::join(stats, ", "));
                    // std::flush(std::cout);

                    int const varIdx = ML::evalVarsStat(stats, valRank);
                    assert(not array_int_var[varIdx]->isBound());
                    return varIdx;
                };

                auto const valSel= [=, &infer](int varIdx)
                {
                    auto pa = ML::getPartialAssignment(array_int_var);
                    auto [vals, scores] = infer.scoreAllValuesForVar(pa, varIdx, array_int_var[varIdx]);

                    auto stats = ML::getStats(scores);
                    auto const valIdx = evalValsStat(stats, valRank);
                    return vals[valIdx];
                };

                auto const valOrd = [=, &infer](int varIdx)
                {
                    auto pa = ML::getPartialAssignment(array_int_var);
                    auto [vals, scores] = infer.scoreAllValuesForVar(pa, varIdx, array_int_var[varIdx]);

                    assert(valRank == ML::RankType::BEST or valRank == ML::RankType::WORST);
                    auto constexpr bestCmp = [](float const & s1, float const & s2) {return s1 < s2;};
                    auto constexpr worstCmp = [](float const & s1, float const & s2) {return s1 > s2;};
                    ML::sortByKey(scores,vals, valRank == ML::RankType::BEST ? bestCmp : worstCmp);

                    return vals;
                };

                auto ml_search_strategy = [=]()
                {
                    auto const varIdx = varSel();
                    if (varIdx >= 0)
                    {
                        auto const & var = array_int_var[varIdx];
                        return indomain_list(array_int_var[0]->getSolver(), var,valOrd(varIdx));

                        //auto const val = valSel(varIdx);
                        // printf("Selected var[%d] = %d\n", varIdx, val);
                        // fflush(stdout);
                        //return indomain_fixed(array_int_var[0]->getSolver(), var, val);
                    }
                    else
                    {
                        return Branches({});
                    }
                };

               return ml_search_strategy;
                //return search_strategy();
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
