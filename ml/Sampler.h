#pragma once

#include "fzn_constraints_helper.h"
#include "fzn_search_helper.h"
#include "fzn_variables_helper.h"

#include "Parser.h"
#include "search.hpp"
#include "solver.hpp"

#include "Common.h"

namespace ML
{
    template<typename Var>
    float varToFloat(Var const & var)
    {
        return var->isBound() ? var->min() : NAN;
    }

    template<typename Var>
    void saveInRecord(std::vector<Var> const & vars, bool const isConsistent, std::vector<float> & record)
    {
        assert(record.size() == vars.size() + 1);
        for (auto vIdx = 0; vIdx < vars.size(); vIdx += 1)
        {
            record[vIdx] = varToFloat(vars[vIdx]);
        }
        record[vars.size()] = static_cast<float>(isConsistent);
    }

    inline
    void Sampler(int sIdx, std::string const & fznPath, std::ostream & outFile, std::mutex & outMutex, bool & stop)
    {
        // First thread write the bounds
        if (sIdx == 0)
        {
            outMutex.lock();
        }

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
        int const recordSize = static_cast<int>(nIntDecVars) + 1; // PA + Flag
        if (sIdx == 0)
        {
            writeBounds(intDecVars, recordSize, outFile);
            outMutex.unlock();
        }

        // Buffering
        constexpr static int BufferSize = 100;
        RecordsBuffer buffer(BufferSize, recordSize);
        PARecord cpaRecord(recordSize, NAN);
        PARecord ipaRecord(recordSize, NAN);

        // Collect partial assignments
        search.onBranch([&]()
        {
            cpaRecord.from(intDecVars , true);
        });
        search.onFailure([&]()
        {
            ipaRecord.from(intDecVars, false);

            int const cpaSize = cpaRecord.countAssignedVars();
            int const ipaSize = ipaRecord.countAssignedVars();
            if ( 0 < cpaSize and cpaSize == ipaSize - 1 and ipaSize < nIntDecVars)
            {
                buffer.safeAdd(cpaRecord,outMutex,outFile);
                buffer.safeAdd(ipaRecord,outMutex,outFile);
            }
        });

        // Sampling
        search.sample(stop);

        // Flush the buffer
        buffer.dump(outMutex,outFile);
    }
}
