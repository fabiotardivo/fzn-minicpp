#pragma once

#include "Common.h"
#include "fzn_constraints_helper.h"
#include "fzn_search_helper.h"
#include "fzn_variables_helper.h"

#include "Parser.h"
#include "libfca/Matrix.hpp"
#include "search.hpp"
#include "solver.hpp"

namespace ML
{
    inline
    float strToFloat(std::string_view const & str)
    {
        float result = NAN;
        if (not str.empty())
        {
            auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
            if (ec != std::errc())
            {
                throw std::runtime_error("Failed to parse: " + std::string(str));
            }
        }
        return result;
    }

    inline
    std::vector<float> strToRecord(std::string_view const & line)
    {
        std::vector<float> record;
        record.reserve(std::count(line.begin(), line.end(), ',') + 1);

        auto pos = 0;
        while (pos < line.size())
        {
            auto const nextComma = line.find(',', pos);
            auto const end = (nextComma == std::string_view::npos) ? line.size() : nextComma;
            record.push_back(strToFloat(line.substr(pos, end - pos)));
            pos = (nextComma == std::string_view::npos) ? line.size() : nextComma + 1;
        }
        return record;
    }

    inline
    void saveInRecord(std::span<float> const & pa, std::vector<float> const & us, std::vector<float> & record)
    {
        assert(record.size() == pa.size() + us.size());
        std::copy(pa.begin(), pa.end(), record.begin());
        std::copy(us.begin(), us.end(), record.begin() + pa.size());
    }

    template<typename Var>
    bool testConsistency(CPSolver::Ptr const & solver, std::vector<Var> const & vars, std::vector<float> const & ipa)
    {
        const auto sm = solver->getStateManager();
        bool isConsistent = true;
        sm->saveState();
        TRYFAIL
            for (auto vIdx = 0; vIdx < ipa.size(); vIdx += 1)
            {
                float const val = ipa[vIdx];
                if (not std::isnan(val))
                {
                    vars[vIdx]->assign(static_cast<int>(val));
                }
            }
        ONFAIL
            isConsistent = false;
        ENDFAIL
        sm->restoreState();

        return isConsistent;
    }

    inline
    std::vector<float> calcUS(CPSolver::Ptr const & solver, std::vector<var<int>::Ptr> const & vars, std::mt19937 & rng, int const nAttempts, std::span<float> const & ipa)
    {
            // Initialize US
            int const nVars = static_cast<int>(ipa.size());
            int bestSize = 0;
            std::vector<float> bestUS(nVars, 0.0);
            for (auto vIdx = 0; vIdx < nVars; vIdx += 1)
            {
                bool const isAssigned = std::isnan(ipa[vIdx]);
                bestUS.push_back(isAssigned);
                bestSize += isAssigned;
            }

            std::vector<float> candidateUS(nVars, 0.0);
            std::vector<int> evalOrder(nVars,0);
            std::iota(evalOrder.begin(), evalOrder.end(), 0);
            for (auto aIdx = 0; aIdx < nAttempts; aIdx += 1)
            {
                std::ranges::shuffle(evalOrder, rng);
                int candidateSize = 0;
                std::ranges::fill(candidateUS, 0.0);
                std::vector<float> candidatePA(ipa.begin(), ipa.end());
                for (auto const & vIdx : evalOrder)
                {
                    if (not std::isnan(candidatePA[vIdx]))
                    {
                        auto const val = candidatePA[vIdx];
                        candidatePA[vIdx] = NAN;
                        bool const isConsistent = testConsistency(solver, vars, candidatePA);
                        if (not isConsistent)
                        {
                            candidatePA[vIdx] = val;
                            candidateUS[vIdx] = 1.0;
                            candidateSize += 1;
                        }
                    }
                }
                if (candidateSize < bestSize)
                {
                    bestSize = candidateSize;
                    bestUS = candidateUS;
                }
            }
        return bestUS;
    }

    inline
    void USFinder(int fIdx, int const nAttempts, std::span<std::string_view const> pasLines, std::string const & fznPath, std::ostream & outFile, std::mutex & outMutex, bool & stop)
    {
        // First thread write the bounds
        if (fIdx == 0)
        {
            outMutex.lock();
        }

        // RNG
        thread_local std::mt19937 rng{static_cast<std::mt19937::result_type>(fIdx)};

        // Create Solver
        CPSolver::Ptr solver = Factory::makeSolver();

        // FlatZinc parsing
        Fzn::Parser fznParser;
        Fzn::Model const & fznModel = fznParser.parse(fznPath);

        // Variables
        FznVariablesHelper varsHelper(solver, fznModel);
        varsHelper.makeBoolVariables(fznModel.bool_vars);
        varsHelper.makeIntVariables(fznModel.int_vars);

        // Constraints
        FznConstraintHelper cstrHelper(solver, varsHelper);
        if (not cstrHelper.makeConstraints(fznModel))
        {
            throw std::runtime_error("The problem is inconsistent");
        }

        // Create Search
        FznSearchHelper searchHelper(solver, varsHelper);
        DFSearch search(solver, searchHelper.getSampleStrategy(fznModel));
        auto const intDecVars = searchHelper.getIntDecisionalVars(fznModel);
        auto const nIntDecVars = intDecVars.size();

        // First thread write the bounds
        int const recordSize = static_cast<int>(nIntDecVars) * 2; // PA + US
        if (fIdx == 0)
        {
            writeBounds(intDecVars, recordSize, outFile);
            outMutex.unlock();
        }

        // Buffering
        constexpr static int BufferSize = 100;
        RecordsBuffer buffer(BufferSize, recordSize);
        USRecord usRecord(recordSize, 0.0);
        for (auto lIdx = 0; lIdx < pasLines.size() and (not stop); lIdx += 1)
        {
            PARecord paRecord(strToRecord(pasLines[lIdx]));
            std::vector<float> us(nIntDecVars, 0.0);
            if (not paRecord.isConsistent())
            {
                us = calcUS(solver, intDecVars, rng, nAttempts,paRecord.getPA());
            }
            usRecord.from(paRecord.getPA(),us);
            buffer.safeAdd(usRecord, outMutex, outFile);
        }
        buffer.dump(outMutex,outFile);
    }
}
