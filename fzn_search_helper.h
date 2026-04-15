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
        std::function<Branches(void)> getSampleStrategy(Fzn::Model const & fzn_model, double temperatore);
        template<typename Inference>
        std::function<Branches(void)> getMLSearchStrategy(Fzn::Model const & fzn_model, ML::RankType valRank, ML::RankType varRank, Inference & infer);
        std::vector<var<int>::Ptr> getIntDecisionalVars(Fzn::Model const & fzn_model);
        std::vector<var<int>::Ptr> getIntDecisionalVars(Fzn::var_expr_t var_expr);
        std::vector<var<bool>::Ptr> getBoolDecisionalVars(Fzn::var_expr_t vars_expr);
        static Limit makeSearchLimits(Fzn::Model const & fzn_model, cxxopts::ParseResult const & args);

    private:
        std::function<Branches(void)> makeBasicSearchStrategy(Fzn::basic_search_annotation_t const & basic_search_annotation);
        std::function<Branches(void)> makeBasicSampleStrategy(Fzn::basic_search_annotation_t const & basic_search_annotation, double temperature);
        template<typename Vars, typename Var>
        static std::function<Var(Vars const &)> makeVariableSelection(Fzn::pred_identifier_t const & variable_selection);
        template <typename Var>
        static std::function<Branches(CPSolver::Ptr, Var)> makeValueSelection(Fzn::pred_identifier_t const & value_selection);
        static unsigned int getMaxSolutions(Fzn::Model const & fzn_model, cxxopts::ParseResult const & args);
        static unsigned int getMaxSearchTime(cxxopts::ParseResult const & args);
        static unsigned long long getMaxFailures(cxxopts::ParseResult const & args);

};


template<typename Inference>
std::function<Branches(void)> FznSearchHelper::getMLSearchStrategy(Fzn::Model const & fzn_model, ML::RankType valRank, ML::RankType varRank, Inference & infer)
{

    using namespace std;
    using int_var_t = var<int>::Ptr;
    using array_int_var_t = vector<int_var_t>;

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
                auto array_int_var = getIntDecisionalVars(var_expr);
                auto const nVars = static_cast<int>(array_int_var.size());

                auto constexpr bestCmp = [](float const & s1, float const & s2) {return s1 < s2;};
                auto constexpr worstCmp = [](float const & s1, float const & s2) {return s1 > s2;};

                auto const varOrd = [=,&infer]()
                {
                    auto pa = ML::getPartialAssignment(array_int_var);
                    auto * variables = new std::vector<int>();
                    std::vector<float> scores;
                    for (int varIdx = 0; varIdx < nVars; ++varIdx)
                    {
                        if (not array_int_var[varIdx]->isBound())
                        {
                            auto [vals, valScores] = infer.scoreAllValuesForVar(pa, varIdx, array_int_var[varIdx]);
                            auto [min_idx, max_idx, min_score, max_score, mean, eom01] = ML::getStats(valScores);
                            variables->emplace_back(varIdx);
                            //scores.emplace_back(mean);
                            std::uniform_real_distribution<float> noise_dist(-1e-3f, 1e-3f);
                            float size_bonus = 1e-4f / static_cast<float>(array_int_var[varIdx]->size());
                            //float noise = noise_dist(getVarRng());  // uses seeded RNG
                            float min_bonus = 1e-4f  / (static_cast<float>(array_int_var[varIdx]->min()) + 1.0f);
                            float confidence = 1.0f - eom01;
                            scores.emplace_back( mean + size_bonus +  min_bonus);
                        }
                    }

                    assert(varRank == ML::RankType::BEST or varRank == ML::RankType::WORST);
                    ML::sortByKey(scores, *variables, varRank == ML::RankType::BEST ? bestCmp : worstCmp);
                    std::reverse(variables->begin(), variables->end());

                    if (not variables->empty())
                    {
                        auto const varIdx = variables->back();
                        variables->pop_back();
                        return varIdx;
                    }
                    return -1;
                };
                // auto const varOrd = [=,&infer]()
                // {
                //     auto pa = ML::getPartialAssignment(array_int_var);
                //     auto * variables = new std::vector<int>();
                //     std::vector<float> scores;
                //
                //     for (int varIdx = 0; varIdx < nVars; ++varIdx)
                //     {
                //         if (not array_int_var[varIdx]->isBound())
                //         {
                //             auto [vals, valScores] = infer.scoreAllValuesForVar(pa, varIdx, array_int_var[varIdx]);
                //             auto [min_idx, max_idx, min_score, max_score, mean, eom01] = ML::getStats(valScores);
                //             variables->emplace_back(varIdx);
                //             float composite = max_score / static_cast<float>(array_int_var[varIdx]->size());
                //             scores.emplace_back(composite);
                //         }
                //     }
                //
                //     if (variables->empty()) return -1;
                //
                //     // Softmax sampling with temperature
                //     constexpr float temperature = 0.1f;
                //     float max_s = *std::max_element(scores.begin(), scores.end());
                //     std::vector<float> probs;
                //     float sum = 0.0f;
                //     for (float s : scores) {
                //         float e = std::exp((s - max_s) / temperature);
                //         probs.push_back(e);
                //         sum += e;
                //     }
                //
                //     static std::mt19937 rng(std::random_device{}());
                //     std::uniform_real_distribution<float> dist(0.0f, sum);
                //     float r = dist(rng);
                //     float cumsum = 0.0f;
                //     for (int i = 0; i < (int)probs.size(); i++) {
                //         cumsum += probs[i];
                //         if (r <= cumsum)
                //             return (*variables)[i];
                //     }
                //     return variables->back();
                // };
                //
                // auto varOrd = [=, &infer]() mutable -> int
                // {
                //     auto pa = ML::getPartialAssignment(array_int_var);
                //
                //     // variable model: directly predicts which variable causes failure
                //     auto varProbs = infer.scoreVariables(pa);
                //
                //     int   bestVar   = -1;
                //     float bestScore = -std::numeric_limits<float>::infinity();
                //
                //     for (int i = 0; i < nVars; ++i)
                //     {
                //         if (array_int_var[i]->isBound()) continue;
                //
                //         if (bestVar == -1 or varProbs[i] > bestScore)
                //         {
                //             bestScore = varProbs[i];
                //             bestVar   = i;
                //         }
                //     }
                //     return bestVar;
                // };
                // auto varOrd = [=, &infer]() mutable -> int
                // {
                //     auto pa = ML::getPartialAssignment(array_int_var);
                //     auto varProbs = infer.scoreVariables(pa);
                //
                //     int   bestVar   = -1;
                //     float bestScore = -std::numeric_limits<float>::infinity();
                //
                //     for (int i = 0; i < nVars; ++i)
                //     {
                //         if (array_int_var[i]->isBound()) continue;
                //
                //         float composite = varProbs[i] + 1e-4f / static_cast<float>(array_int_var[i]->size());
                //
                //         if (bestVar == -1 or composite > bestScore)
                //         {
                //             bestScore = composite;
                //             bestVar   = i;
                //         }
                //     }
                //     return bestVar;
                // };
                //
                auto const valOrd = [=, &infer](int varIdx)
                {
                    // ML
                    auto pa = ML::getPartialAssignment(array_int_var);
                    auto [vals, scores] = infer.scoreAllValuesForVar(pa, varIdx, array_int_var[varIdx]);

                    // valOrd with small randomness:
                    //std::uniform_real_distribution<float> val_noise(-1e-3f, 1e-3f);
                    //for (auto& s : scores) s += val_noise(getValRng());

                    assert(valRank == ML::RankType::BEST or valRank == ML::RankType::WORST);
                    ML::sortByKey(scores,vals, valRank == ML::RankType::BEST ? bestCmp : worstCmp);

                    return vals;
                };

                //auto ml_search_strategy = [=]()
                auto ml_search_strategy = [this, varOrd, valOrd, array_int_var]() mutable -> Branches

                {
                    std::vector<std::function<void(void)>> branches;
                    auto const varIdx = varOrd();
                    if (varIdx >= 0)
                    {
                        auto & var = array_int_var[varIdx];
                        assert(not var->isBound());


                       // int val = valOrd(varIdx).front();
                       //  branches.emplace_back([this,var,val, varIdx, array_int_var]()
                       //  {
                       //      auto pa = ML::getPartialAssignment(array_int_var);
                       //      PARecord::printPA(pa,std::cout);
                       //      printf("\n");
                       //      std::cerr << "%% Choosing  x[" << varIdx <<  "] == " << val  << " (Size " << var->size() << ")" << std::endl << std::flush;
                       //      return solver->post(new (solver) EQc(var, val));
                       //  });
                       //  branches.emplace_back([this,var,val, varIdx, array_int_var]()
                       // {
                       //     auto pa = ML::getPartialAssignment(array_int_var);
                       //     PARecord::printPA(pa,std::cout);
                       //     printf("\n");
                       //     std::cerr << "%% Choosing  x[" << varIdx <<  "] != " << val  << " (Size " << var->size() << ")" << std::endl << std::flush;
                       //     return solver->post(new (solver) NEQc(var, val));
                       // });

                        for (int const & val : valOrd(varIdx))
                        {
                            branches.emplace_back([this,var,val, varIdx, array_int_var]()
                            {
                                auto pa = ML::getPartialAssignment(array_int_var);
                                PARecord::printPA(pa,std::cout);
                                printf("\n");
                                std::cerr << "%% Choosing  x[" << varIdx <<  "] == " << val  << " (Size " << var->size() << ")" << std::endl << std::flush;
                                return solver->post(new (solver) EQc(var, val));
                            });

                        }
                    }
                    return Branches(branches);
                };

               return ml_search_strategy;
               // return getSearchStrategy(fzn_model);
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


/*
template<typename Inference>
std::function<Branches(void)> FznSearchHelper::getMLSearchStrategy(
    Fzn::Model const & fzn_model,
    ML::RankType valRank,
    ML::RankType varRank,
    Inference & infer)
{
    using namespace std;
    using int_var_t = var<int>::Ptr;
    using array_int_var_t = vector<int_var_t>;

    for (auto const & search_annotation : fzn_model.search_strategy)
    {
        if (not holds_alternative<Fzn::basic_search_annotation_t>(search_annotation))
            throw std::runtime_error("Unsupported search annotation");

        auto const & basic_search_annotation = get<Fzn::basic_search_annotation_t>(search_annotation);
        auto const & pred_identifier = get<0>(basic_search_annotation);
        auto const & var_expr        = get<1>(basic_search_annotation);

        if (pred_identifier != "int_search")
            throw std::runtime_error("Unsupported search annotation");

        auto array_int_var = getIntDecisionalVars(var_expr);
        auto const nVars   = static_cast<int>(array_int_var.size());

        // varRank WORST = fail-first  = pick variable with HIGHEST min-value-failure-prob
        //                               i.e. even the safest value looks bad → branch early
        // varRank BEST  = safe-first  = pick variable with LOWEST  min-value-failure-prob
        //                               i.e. at least one value looks safe → branch last
        auto const varCmp = (varRank == ML::RankType::WORST)
            ? std::function<bool(float,float)>([](float a, float b){ return a > b; })
            : std::function<bool(float,float)>([](float a, float b){ return a < b; });

        // valRank BEST  = succeed-first = try value with LOWEST  failure probability first
        // valRank WORST = fail-first    = try value with HIGHEST failure probability first
        auto const valCmp = (valRank == ML::RankType::BEST)
            ? std::function<bool(float,float)>([](float a, float b){ return a < b; })
            : std::function<bool(float,float)>([](float a, float b){ return a > b; });

        // Shared cache: varOrd populates it, valOrd reads it.
        // No separate variable model needed — variable score derived from value scores.
        using ValCache = std::unordered_map<int, typename Inference::BatchResult>;
        auto cache = std::make_shared<ValCache>();

        // ── Variable selection ────────────────────────────────────────────
        // Calls the VALUE model once per unbound variable.
        // Variable score = min of value failure probabilities:
        //   "what is the best the model thinks I can do for this variable?"
        //   A high min means even the safest value looks risky → branch on it first (fail-first).
        // auto varOrd = [=, &infer, cache, varCmp]() mutable -> int
        // {
        //     auto pa = ML::getPartialAssignment(array_int_var);
        //
        //     // Variable model: one call scores all variables at once
        //     auto varProbs = infer.scoreVariables(pa);  // vector<float>[nVars]
        //
        //     cache->clear();
        //
        //     int   bestVar   = -1;
        //     float bestScore = -std::numeric_limits<float>::infinity();
        //
        //     for (int i = 0; i < nVars; ++i)
        //     {
        //         if (array_int_var[i]->isBound()) continue;
        //
        //         float varScore = varProbs[i];
        //
        //         if (bestVar == -1 or varCmp(varScore, bestScore))
        //         {
        //             bestScore = varScore;
        //             bestVar   = i;
        //         }
        //     }
        //
        //     // Now call the value model only for the chosen variable
        //     // (to populate the cache for valOrd)
        //     if (bestVar >= 0)
        //     {
        //         auto result = infer.scoreAllValuesForVar(pa, bestVar, array_int_var[bestVar]);
        //         (*cache)[bestVar] = std::move(result);
        //     }
        //
        //     return bestVar;
        // };
        auto varOrd = [=, &infer, cache, varCmp]() mutable -> int
        {
            auto pa = ML::getPartialAssignment(array_int_var);
            cache->clear();

            int   bestVar   = -1;
            float bestScore = -std::numeric_limits<float>::infinity();

            for (int i = 0; i < nVars; ++i)
            {
                if (array_int_var[i]->isBound()) continue;

                auto result      = infer.scoreAllValuesForVar(pa, i, array_int_var[i]);
                auto const & probs = result.probabilities;
                if (probs.empty()) continue;

                // --- pick ONE of these three ---
                // MIN: "best case" — how safe is the safest value?
                //float varScore = *std::min_element(probs.begin(), probs.end());

                // MEAN: "average danger"
                // float varScore = std::accumulate(probs.begin(), probs.end(), 0.0f) / probs.size();

                // MAX: "worst case" — how bad is the worst value?
                 float varScore = *std::max_element(probs.begin(), probs.end());

                (*cache)[i] = std::move(result);

                if (bestVar == -1 or varCmp(varScore, bestScore))
                {
                    bestScore = varScore;
                    bestVar   = i;
                }
            }

            return bestVar;
        };

        // ── Value ordering ────────────────────────────────────────────────
        // Reads from the cache filled by varOrd — no extra inference call.
        auto const valOrd = [cache, valCmp](int varIdx) -> std::vector<int32_t>
        {
            auto it = cache->find(varIdx);
            if (it == cache->end())
                throw std::runtime_error("valOrd: varIdx not in cache");

            auto vals   = it->second.values;
            auto scores = it->second.probabilities;

            ML::sortByKey(scores, vals, valCmp);

            return vals;
        };

        // ── Branch builder ────────────────────────────────────────────────
        return [this, varOrd, valOrd, array_int_var]() mutable -> Branches
        {
            std::vector<std::function<void(void)>> branches;

            int const varIdx = varOrd();
            if (varIdx < 0) return Branches(branches);  // all variables bound

            auto & var = array_int_var[varIdx];
            assert(not var->isBound());

            for (int32_t const val : valOrd(varIdx))
            {
                branches.emplace_back([this, var, val, varIdx, array_int_var]()
                {
                    auto pa = ML::getPartialAssignment(array_int_var);
                    PARecord::printPA(pa, std::cout);
                    printf("\n");
                    std::cerr << "%% Choosing  x[" << varIdx << "] == " << val
                              << " (Size " << var->size() << ")" << std::endl << std::flush;
                    solver->post(new (solver) EQc(var, val));
                });
            }
            return Branches(branches);
        };
    }

    return {};
}
*/

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
    if (value_selection == "indomain")
    {
        return [](CPSolver::Ptr s, Var var) -> Branches {return indomain<Var>(s, var);};
    }
    else if (value_selection == "indomain_min")
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

template<typename Vars, typename Var>
static std::function<Var(Vars const &)> makeVariableSelectionProb(
    Fzn::pred_identifier_t const & variable_selection, double temperature = 1.0)
{
    if (variable_selection == "first_fail")
        return [temperature](Vars const & vars) -> Var {
            return first_fail_prob<Vars,Var>(vars, temperature); };
    else if (variable_selection == "input_order")
        return [temperature](Vars const & vars) -> Var {
            return input_order_prob<Vars,Var>(vars, temperature); };
    else if (variable_selection == "smallest")
        return [temperature](Vars const & vars) -> Var {
            return smallest_prob<Vars,Var>(vars, temperature); };
    else if (variable_selection == "largest")
        return [temperature](Vars const & vars) -> Var {
            return largest_prob<Vars,Var>(vars, temperature); };
    else
        return [](Vars const & vars) -> Var {
            return random<Vars,Var>(vars); };
}

template <typename Var>
static std::function<Branches(CPSolver::Ptr, Var)> makeValueSelectionProb(
    Fzn::pred_identifier_t const & value_selection, double temperature = 1.0)
{
    if (value_selection == "indomain_min")
        return [temperature](CPSolver::Ptr s, Var var) -> Branches {
            return indomain_min_prob<Var>(s, var, temperature); };
    else if (value_selection == "indomain_max")
        return [temperature](CPSolver::Ptr s, Var var) -> Branches {
            return indomain_max_prob<Var>(s, var, temperature); };
    else
        return [](CPSolver::Ptr s, Var var) -> Branches {
            return indomain_random<Var>(s, var); };
}
