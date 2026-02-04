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
    inline
    void flushSamples(Fca::Matrix<float> const & buffer, int & nSamples, int sampleSize, std::mutex & outMutex)
    {
        std::lock_guard<std::mutex> lock(outMutex);

        for (auto sIdx = 0; sIdx < nSamples; sIdx += 1)
        {
            auto const sample = buffer.getRow(sIdx);
            for (auto i = 0; i < sampleSize; i += 1)
            {
                printf(i == 0 ? "%.1f" : ",%.1f", sample[i]);
            }
            std::cout << std::endl;
        }
        std::flush(std::cout);

        nSamples = 0;
    }

    inline
    float token2float(std::string_view const & token)
    {
        float result = NAN;
        if (token != "nan")
        {
            auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), result);
            if (ec != std::errc())
            {
                throw std::runtime_error("Failed to parse token: " + std::string(token));
            }
        }
        return result;
    }

    inline
    std::vector<float> str2floats(std::string_view const & line)
    {
        std::vector<float> result;
        result.reserve(std::count(line.begin(), line.end(), ',') + 1);

        auto pos = 0;
        while (pos < line.size())
        {
            auto nextComma = line.find(',', pos);
            auto end = (nextComma == std::string_view::npos) ? line.size() : nextComma;

            result.push_back(token2float(line.substr(pos, end - pos)));

            pos = (nextComma == std::string_view::npos) ? line.size() : nextComma + 1;
        }

        return result;
    }

    inline
    bool testConsistency(CPSolver::Ptr const & solver, std::vector<var<int>::Ptr> const & vars, std::vector<float> const & pa)
    {
        const auto sm = solver->getStateManager();
        bool isConsistent = true;
        sm->saveState();
        TRYFAIL
            for (auto vIdx = 0; vIdx < pa.size(); vIdx += 1)
            {
                float const val = pa[vIdx];
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

    constexpr static int BufferSize = 100;
    static_assert(BufferSize % 2 == 0); // Samples come in pair: 1 positive, 1 negative

    inline
    void UsFinder(int fIdx, int const nAttempts, std::span<std::string_view const> pasLines, std::string & fzn, std::mutex & outMutex, bool & stop)
    {
        // RNG
        thread_local std::mt19937 rng{static_cast<std::mt19937::result_type>(fIdx)};

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
        bool isConsistent = cstrHelper.makeConstraints(fznModel);

        if (isConsistent)
        {
            // Create Search
            FznSearchHelper searchHelper(solver, varsHelper);
            auto const intDecVars = searchHelper.getIntDecisionalVars(fznModel);
            int const nIntDecVars = intDecVars.size();

            int const sampleSize = nIntDecVars * 2;
            int nSamples = 0;
            auto const rawBuffer = new float[BufferSize * sampleSize];
            Fca::Matrix<float> buffer(BufferSize,sampleSize, rawBuffer);

            for (auto lIdx = 0; lIdx < pasLines.size(); lIdx += 2)
            {
                std::vector<float> cpa = str2floats(pasLines[lIdx]);
                std::vector<float> ipa = str2floats(pasLines[lIdx+1]);

                // Initialize US
                int bestUSSize = 0;
                std::vector<float> bestUS;
                for (float const & val : ipa)
                {
                    bestUS.push_back(val);
                    bestUSSize += not std::isnan(val);
                }

                std::vector<float> candidateUS(nIntDecVars);
                std::vector<int> evalOrder(nIntDecVars);
                std::iota(evalOrder.begin(), evalOrder.end(), 0);
                for (auto aIdx = 0; aIdx < nAttempts; aIdx += 1)
                {
                    std::ranges::shuffle(evalOrder, rng);
                    std::ranges::fill(candidateUS, 0.0);
                    std::vector<float> paToTest(ipa);
                    int usSize = 0;
                    for (auto const & vIdx : evalOrder)
                    {
                        if (not std::isnan(paToTest[vIdx]))
                        {
                            auto const val = paToTest[vIdx];
                            paToTest[vIdx] = NAN;
                            isConsistent = testConsistency(solver, intDecVars, paToTest);
                            if (not isConsistent)
                            {
                                paToTest[vIdx] = val;
                                usSize += 1;
                                candidateUS[vIdx] = 1.0;
                            }
                        }
                    }
                    if (usSize < bestUSSize)
                    {
                        bestUSSize = usSize;
                        bestUS = candidateUS;
                    }
                }

                auto pSample = buffer.getRow(nSamples);
                auto nSample = buffer.getRow(nSamples + 1);
                nSamples += 2;

                // Partial assigment
                int sIdx = 0;
                for (auto vIdx = 0; vIdx < nIntDecVars; vIdx += 1)
                {
                    pSample[sIdx] = ipa[vIdx];
                    nSample[sIdx] = cpa[vIdx];
                    sIdx += 1;
                }
                // Unsatisfiable subset
                for (auto vIdx = 0; vIdx < nIntDecVars; vIdx += 1)
                {
                    pSample[sIdx] = bestUS[vIdx];
                    nSample[sIdx] = 0.0;
                    sIdx += 1;
                }

                if (nSamples == BufferSize)
                {
                    flushSamples(buffer,nSamples, sampleSize,outMutex);
                }
            }

            flushSamples(buffer,nSamples, sampleSize,outMutex);

            free(rawBuffer);
        }
    }
}
