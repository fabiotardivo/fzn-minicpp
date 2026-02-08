#pragma once

#include <cmath>
#include <list>
#include <vector>
#include <stdexcept>
#include <ranges>

#include "Common.h"

namespace ML
{

    using IntVar = var<int>::Ptr;
    using IntVars = std::vector<IntVar>;
    using EvalResultType = std::tuple<float,float,int>;
    using StatsType = std::tuple<int,int,float,float,float,float>;
    using EvalFunctionType = std::function<EvalResultType(int, IntVars const &)>;
    using PAType = std::vector<float>;

    enum RankType
    {
        BEST,
        WORST,
        BEST_AVG,
        WORST_AVG
    };

    inline
    RankType rankFromString(std::string const & s)
    {
        static const std::unordered_map<std::string, RankType> map
                {
                        {"best", RankType::BEST},
                        {"worst", RankType::WORST},
                        {"bestAvg", RankType::BEST_AVG},
                        {"worstAvg", RankType::WORST_AVG}
                };

        auto it = map.find(s);
        if (it != map.end()) return it->second;
        throw std::invalid_argument("Invalid rank: " + s);
    }

    inline
    void printPartialAssignment(std::vector<float> const & vals)
    {
        auto const nVals = static_cast<int>(vals.size());
        for(int i = 0; i < nVals; i += 1)
        {
            printf("%3.0f%s",vals[i], i == nVals - 1 ? "\n" : ",");
        }
    }

    inline
    std::vector<int> getPartialAssignment(IntVars const & vars)
    {
        std::vector<int> result = {};
        for(auto const & var : vars)
        {
            result.push_back(var->isBound() ? var->min() : UNASSIGNED_VALUE);
        }
        return result;
    }

    inline
    std::list<std::vector<float>> getAllPartialAssignments(int varIdx, IntVars const & vars, std::vector<float> const & pa)
    {
        std::list<std::vector<float>> result = {};
        if (std::isnan(pa[varIdx]))
        {
            auto const & var = vars[varIdx];
            auto const minVal = var->min();
            auto const maxVal = var->max();
            for (auto val = minVal; val <= maxVal; val += 1)
            {
                if (var->contains(val))
                {
                    std::vector<float> tmp = pa;
                    tmp[varIdx] = static_cast<float>(val);
                    result.push_back(std::move(tmp));
                }
            }
        }
        else
        {
            throw std::runtime_error("Variable already assigned.");
        }
        return result;
    }

    inline
    std::list<std::vector<float>> getAllPartialAssignments(IntVars const & vars, std::list<std::vector<float>> const & pas)
    {
        int const nVars = static_cast<int>(vars.size());
        std::list<std::vector<float>> result = {};
        for (auto const & pa : pas)
        {
            for (int varIdx = 0; varIdx < nVars; varIdx +=1)
            {
                if (std::isnan(pa[varIdx]))
                {
                    auto tmp = getAllPartialAssignments(varIdx,vars,pa);
                    result.splice(result.end(),tmp);
                }
            }
        }
        return result;
    }

    inline
    std::list<std::vector<float>> getLookahead(int depth, IntVars const & vars, std::list<std::vector<float>> const & pas)
    {
        if (depth == 0)
        {
            return pas;
        }
        else
        {
            return getLookahead(depth-1,vars,getAllPartialAssignments(vars, pas));
        }
    }

    inline float bernoulli_entropy_01(float p)
    {
        // Returns in [0,1], where 1 = max uncertainty at p=0.5
        const float eps = 1e-12f;
        p = std::clamp(p, eps, 1.0f - eps);

        float h = -(p * std::log(p) + (1.0f - p) * std::log(1.0f - p)); // nats
        return h / std::log(2.0f); // normalize to [0,1]
    }

    inline std::tuple<int,int,float,float,float,float>
    getStats(std::vector<float> const& scores)
    {
        int min_idx = 0;
        int max_idx = 0;
        float min_score = scores[0];
        float max_score = scores[0];
        float mean = 0.0f;

        int n = 0;
        int i = 0;

        for (float x : scores)
        {
            ++n;
            mean += (x - mean) / n;  // incremental mean

            if (x < min_score) { min_score = x; min_idx = i; }
            if (x > max_score) { max_score = x; max_idx = i; }

            ++i;
        }

        float eom = bernoulli_entropy_01(mean);
        return {min_idx, max_idx, min_score, max_score, mean, eom};
    }

    inline
    int evalValsStat(StatsType const & stats, RankType valRank)
    {
        auto [min_idx, max_idx, min_score, max_score, mean, eom01] = stats;
        switch (valRank)
        {
        case BEST:
            return min_idx;
        case WORST:
            return  max_idx;
        default:
            throw std::runtime_error("Invalid value rank.");
        }
    }

    inline
    int evalVarsStat(StatsType const & stats, RankType varRank)
    {
        auto [min_idx, max_idx, min_score, max_score, mean, eom01] = stats;
        switch (varRank)
        {
        case BEST:
            return min_idx;
        case WORST:
            return max_idx;
        default:
            throw std::runtime_error("Unsupported variable rank.");
        }
    }
}
