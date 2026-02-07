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
        auto const nIntDecVars = static_cast<int>(intDecVars.size());

        // First thread write the bounds
        int const recordSize = nIntDecVars + 1; // PA + Flag
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
            cpaRecord.from(intDecVars, false);

        });
        search.onFailure([&]()
        {
            ipaRecord.from(intDecVars, true);

            // cpaRecord.print(std::cout);
            // ipaRecord.print(std::cout);

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
