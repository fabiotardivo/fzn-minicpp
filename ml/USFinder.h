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
    std::vector<int> strToRecord(std::string_view line)
    {
        std::vector<int> record;
        record.reserve(std::count(line.begin(), line.end(), ',') + 1);

        size_t pos = 0;
        while (pos < line.size())
        {
            size_t nextComma = line.find(',', pos);
            size_t end = (nextComma == std::string_view::npos) ? line.size() : nextComma;

            // Extract substring
            std::string_view token = line.substr(pos, end - pos);

            // Handle empty token (UNASSIGNED_VALUE variable)
            if (token.empty())
            {
                record.push_back(UNASSIGNED_VALUE);
            } else
            {
                // Parse integer
                int value;
                auto result = std::from_chars(token.data(), token.data() + token.size(), value);
                if (result.ec == std::errc()) {
                    record.push_back(value);
                } else {
                    throw std::invalid_argument("Invalid integer: " + std::string(token));
                }
            }

            pos = (nextComma == std::string_view::npos) ? line.size() : nextComma + 1;
        }
        return record;
    }

    template<typename Var>
    bool testConsistency(CPSolver::Ptr const & solver, std::vector<Var> const & vars, std::vector<int> const & ipa)
    {
        const auto sm = solver->getStateManager();
        bool isConsistent = true;
        sm->saveState();
        TRYFAIL
            for (auto vIdx = 0; vIdx < ipa.size(); vIdx += 1)
            {
                int const val = ipa[vIdx];
                if (val != UNASSIGNED_VALUE)
                {
                    vars[vIdx]->assign(val);
                }
            }
            solver->fixpoint();
        ONFAIL
            isConsistent = false;
        ENDFAIL
        sm->restoreState();

        return isConsistent;
    }

    inline
    std::vector<int> calcUS(CPSolver::Ptr const & solver, std::vector<var<int>::Ptr> const & vars, std::mt19937 & rng, int const nAttempts, std::span<int> const & ipa)
    {
            // Initialize US
            int const nVars = static_cast<int>(ipa.size());
            int bestSize = 0;
            std::vector<int> bestUS(nVars, UNASSIGNED_VALUE);
            for (auto vIdx = 0; vIdx < nVars; vIdx += 1)
            {
                bool const isAssigned = ipa[vIdx] != UNASSIGNED_VALUE;
                if (isAssigned)
                {
                    bestUS[vIdx] = 1;
                    bestSize += 1;
                }
            }

            std::vector<int> candidateUS(nVars, UNASSIGNED_VALUE);
            std::vector<int> evalOrder(nVars,0);
            std::iota(evalOrder.begin(), evalOrder.end(), 0);
            for (auto aIdx = 0; aIdx < nAttempts; aIdx += 1)
            {
                //std::cout << "ORIGINAL  ";
                //PARecord::printPA(ipa, std::cout);
                //std::cout << std::endl;

                std::vector<int> candidatePA(ipa.begin(), ipa.end());
                int candidateSize = 0;
                std::ranges::fill(candidateUS, UNASSIGNED_VALUE);
                std::ranges::shuffle(evalOrder, rng);
                for (auto const & vIdx : evalOrder)
                {
                    if (candidatePA[vIdx] != UNASSIGNED_VALUE)
                    {
                        auto const val = candidatePA[vIdx];
                        candidatePA[vIdx] = UNASSIGNED_VALUE;

                        //std::cout << "CANDIDATE ";
                        //PARecord::printPA(candidatePA, std::cout);
                        bool const isConsistent = testConsistency(solver, vars, candidatePA);
                        //std::cout << " | CONS = " << isConsistent << std::endl;

                        candidateUS[vIdx] = isConsistent;
                        if (isConsistent)
                        {
                            candidatePA[vIdx] = val;
                            candidateSize += 1;
                        }
                    }
                }
                assert(0 < candidateSize);
                if (candidateSize < bestSize)
                {
                    bestSize = candidateSize;
                    bestUS = candidateUS;

                    // std::cout << "BETTER ";
                    // PARecord::printPA(candidatePA, std::cout);
                    // std::cout << std::endl;
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
        auto const nIntDecVars = static_cast<int>(intDecVars.size());

        // First thread write the bounds
        int const recordSize = nIntDecVars * 2; // PA + US
        if (fIdx == 0)
        {
            writeBounds(intDecVars, recordSize, outFile);
            outMutex.unlock();
        }

        // Buffering
        constexpr static int BufferSize = 100;
        RecordsBuffer buffer(BufferSize, recordSize);
        USRecord usRecord(recordSize, 0.0);

        // Finding
        for (auto lIdx = 0; lIdx < pasLines.size() and (not stop); lIdx += 1)
        {
            PARecord paRecord(strToRecord(pasLines[lIdx]));

            std::vector<int> us(nIntDecVars, UNASSIGNED_VALUE);
            auto const pa = paRecord.getPA();
            for (auto vIdx = 0; vIdx < pa.size(); vIdx += 1)
            {
                us[vIdx] = pa[vIdx] == UNASSIGNED_VALUE ? UNASSIGNED_VALUE : 0;
            }
            if (not paRecord.isConsistent())
            {
                us = calcUS(solver, intDecVars, rng, nAttempts,paRecord.getPA());
            }
            usRecord.from(paRecord.getPA(),us);

            // paRecord.print(std::cout);
            // usRecord.print(std::cout);

            buffer.safeAdd(usRecord, outMutex, outFile);
        }
        buffer.dump(outMutex,outFile);
    }
}
