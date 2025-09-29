#pragma once

#include "fzn_constraints_helper.h"
#include "fzn_search_helper.h"
#include "fzn_variables_helper.h"

#include "Parser.h"
#include "libfca/Matrix.hpp"
#include "search.hpp"
#include "solver.hpp"

namespace ML
{
    template<typename Var>
    double getValue(Var const & var)
    {
        return var->isBound() ? static_cast<double>(var->min()) : static_cast<double>(NAN);
    };

    void flushSamples(Fca::Matrix<double> const & buffer, int & nSamples, int sampleSize, std::mutex & outMutex)
    {
        std::lock_guard<std::mutex> lock(outMutex);

        for (auto sIdx = 0; sIdx < nSamples; sIdx += 1)
        {
            auto  const sample = buffer.getRow(sIdx);
            std::cout << sample[0];
            for (auto idx = 1; idx < sampleSize; idx += 1)
            {
                std::cout << "," << sample[idx];
            }
            std::cout << std::endl;
        }
        std::flush(std::cout);

        nSamples = 0;
    }

    constexpr static int BufferSize = 100;
    static_assert(BufferSize % 2 == 0); // Samples come in pair: 1 positive, 1 negative

    inline
    void Sampler(int idx, std::string & fzn, std::mutex & outMutex, bool & stop)
    {
        // Create Solver
        CPSolver::Ptr solver = Factory::makeSolver();

        // FlatZinc parsing
        Fzn::Parser fznParser;
        Fzn::Model const & fznModel = fznParser.parse(fzn);

        // Variables
        FznVariablesHelper varsHelper(solver, fznModel);
        varsHelper.makeBoolVariables(fznModel.bool_vars);
        varsHelper.makeIntVariables(fznModel.int_vars);

        // Constraints
        FznConstraintHelper cstrHelper(solver, varsHelper);
        auto const isConsistent = cstrHelper.makeConstraints(fznModel);

        if (isConsistent)
        {
            // Create Search
            FznSearchHelper searchHelper(solver, varsHelper);
            auto const intDecVars = searchHelper.getIntDecisionalVars(fznModel);
            DFSearch search(solver, searchHelper.getSampleStrategy(fznModel));
            auto const nIntDecVars = intDecVars.size();
            std::vector<double> lastValidPA(nIntDecVars, NAN);
            std::vector<double> badPA(nIntDecVars, NAN);

            // Collect partial assignments
            auto const sampleSize = nIntDecVars + 1;
            auto nSamples = 0;
            auto const rawBuffer = new double[BufferSize * sampleSize];
            Fca::Matrix<double> buffer(BufferSize, sampleSize, rawBuffer);
            search.onBranch([&,intDecVars]()
            {
                for (auto vIdx = 0; vIdx < nIntDecVars; vIdx += 1)
                {
                    auto const value = getValue(intDecVars[vIdx]);
                    lastValidPA[vIdx] = value;
                }
            });
            search.onFailure([&]()
            {
                for (auto vIdx = 0; vIdx < nIntDecVars; vIdx += 1)
                {
                    auto const value = getValue(intDecVars[vIdx]);
                    badPA[vIdx] = value;
                }

                auto nFixedBad = 0;
                auto nFixedLast = 0;
                for (auto vIdx = 0; vIdx < nIntDecVars; vIdx += 1)
                {
                    nFixedBad  += not std::isnan(badPA[vIdx]);
                    nFixedLast += not std::isnan(lastValidPA[vIdx]);
                }

                if ( 0 < nFixedLast and nFixedLast < nFixedBad and nFixedBad < nIntDecVars)
                {
                    auto positiveSample = buffer.getRow(nSamples);
                    auto negativeSample = buffer.getRow(nSamples + 1);
                    nSamples += 2;

                    // Features
                    for (auto vIdx = 0; vIdx < nIntDecVars; vIdx += 1)
                    {
                        positiveSample[vIdx] = lastValidPA[vIdx];
                        negativeSample[vIdx] = badPA[vIdx];
                    }

                    // Labels
                    positiveSample[nIntDecVars] = 1;
                    negativeSample[nIntDecVars] = 0;
                    if (nSamples == BufferSize)
                    {
                        flushSamples(buffer,nSamples, sampleSize,outMutex);\
                        nSamples = 0;
                    }
                }
            });

            search.sample(stop);

            flushSamples(buffer,nSamples, sampleSize,outMutex);

            free(rawBuffer);
        }
    }
}
