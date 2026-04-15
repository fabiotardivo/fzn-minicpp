#include <ml/BatchedOnnxInferDual.h>
#include <Parser.h>
#include <Printer.h>
#include <solver.hpp>
#include <search.hpp>
#include "fzn_constraints_helper.h"
#include "fzn_output_helper.h"
#include "fzn_search_helper.h"
#include "fzn_statistics_helper.h"
#include "fzn_variables_helper.h"
#include <libfca/Slice.hpp>

#include <ml/utils.h>
#include <ml/Common.h>
#include <sstream>
#include <fstream>

#include <fmt/format.h>
#include <fmt/ranges.h>

int main(int argc, char * argv[])
{
    using namespace std;

    // Parse options
    std::string fzn;
    std::string valModel;
    std::string varModel;
    std::string varRankStr = "worst";
    std::string valRankStr = "worst";
    std::string failureLog;
    bool gpuInference = false;
    int seed = -1;  // -1 = random

    cxxopts::Options optsParser("fzn-minicpp-ml", "A C++ MiniZinc solver based on MiniCP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("a", "Print all solutions", cxxopts::value<bool>())
        ("n", "Stop search after <n> solutions", cxxopts::value<unsigned int>())
        ("s", "Print search statistics", cxxopts::value<bool>())
        ("t", "Stop search after <t> ms", cxxopts::value<unsigned int>())
        ("val-model", "Value ML model in ONNX format", cxxopts::value<std::string>(valModel))
        ("var-model", "Variable ML model in ONNX format", cxxopts::value<std::string>(varModel))
        ("g,gpu", "Use GPU for inference", cxxopts::value<bool>(gpuInference))
        ("var-rank", "Criteria to rank variables (Default = worst)", cxxopts::value<std::string>(varRankStr))
        ("val-rank", "Criteria to rank values (Default = worst)", cxxopts::value<std::string>(valRankStr))
        ("failure-log", "Path to write CPA/IPA failure pairs (CSV)", cxxopts::value<std::string>(failureLog))
        ("seed", "Random seed for search (-1 = random)", cxxopts::value<int>(seed))
        ("fzn", "FlatZinc", cxxopts::value<std::string>(fzn))
        ("h,help", "Print usage");
    optsParser.parse_positional({"fzn"});

    auto args = optsParser.parse(argc, argv);

    if ((args.count("h") == 0) and (not fzn.empty()) and (not valModel.empty()) and (not varModel.empty()))
    {
        // Seed the RNGs
        if (seed >= 0)
        {
            getVarRng().seed(static_cast<unsigned int>(seed));
            getValRng().seed(static_cast<unsigned int>(seed));
        }

        // Create Statistics
        SearchStatistics stats;
        stats.setStartTime();

        // Create Solver
        CPSolver::Ptr solver = Factory::makeSolver();

        // FlatZinc parsing
        Fzn::Parser fznParser;
        Fzn::Model const & fznModel = fznParser.parse(fzn);

        // Variables and Printer
        FznVariablesHelper varsHelper(solver, fznModel);
        Fzn::Printer fznPrinter;
        varsHelper.makeBoolVariables(fznModel.bool_vars, &fznPrinter);
        varsHelper.makeIntVariables(fznModel.int_vars, &fznPrinter);

        // Constraints
        FznConstraintHelper constrsHelper(solver, varsHelper);
        auto const isConsistent = constrsHelper.makeConstraints(fznModel);

        // Create Search
        FznSearchHelper searchHelper(solver, varsHelper);
        auto const intDecVars = searchHelper.getIntDecisionalVars(fznModel);
        auto const nIntDecVars = static_cast<int>(intDecVars.size());

        // Load ML evaluator with ONNX
        ML::RankType varRank = ML::rankFromString(varRankStr);
        ML::RankType valRank = ML::rankFromString(valRankStr);
        BatchedOnnxInferDual infer(valModel, varModel, gpuInference);

        DFSearch search(solver, searchHelper.getMLSearchStrategy(fznModel, valRank, varRank, infer));
        FznStatisticsHelper::hookToSearch(stats, search);

        // Failure logging — same pattern as Sampler.h
        int const recordSize = nIntDecVars + 1; // PA + flag
        PARecord cpaRecord(recordSize, UNASSIGNED_VALUE);
        PARecord ipaRecord(recordSize, UNASSIGNED_VALUE);

        std::shared_ptr<RecordsBuffer> buffer;
        std::shared_ptr<std::ostream> outFile;
        std::mutex outMutex;

        if (!failureLog.empty())
        {
            // Open in append mode so multiple parallel instances can write to different files
            auto file = std::make_shared<std::ofstream>(failureLog, std::ios::trunc);
            if (!file->is_open())
                throw std::runtime_error("Cannot open failure log: " + failureLog);

            // Write bounds header (same format as sampler CSV)
            writeBounds(intDecVars, recordSize, *file);

            outFile = file;
            buffer = std::make_shared<RecordsBuffer>(100, recordSize);

            search.onBranch([&cpaRecord, &intDecVars]()
            {
                cpaRecord.from(intDecVars, false);
            });

            search.onFailure([&]()
            {
                //printf("Fail\n");
                ipaRecord.from(intDecVars, true);

                int const cpaSize = cpaRecord.countAssignedVars();
                int const ipaSize = ipaRecord.countAssignedVars();
                if (0 < cpaSize and cpaSize == ipaSize - 1 and ipaSize < nIntDecVars)
                {
                    buffer->safeAdd(cpaRecord, outMutex, *outFile);
                    buffer->safeAdd(ipaRecord, outMutex, *outFile);
                }
            });
        }

        // Search limits
        Limit searchLimits = FznSearchHelper::makeSearchLimits(fznModel, args);

        // Output
        FznOutputHelper outputHelper(fznPrinter, cout, fznModel, args);
        outputHelper.hookToSearch(search);

        // Launch Search
        stats.setSearchStartTime();
        if (isConsistent)
        {
            if (fznModel.solve_type == "satisfy")
                search.solve(stats, searchLimits);
            else if (fznModel.solve_type == "minimize")
            {
                Objective::Ptr obj = Factory::minimize(varsHelper.getObjectiveVar());
                obj->onFailure([&](){ stats.incrTighteningFail();});
                search.optimize(obj, stats, searchLimits);
            }
            else if (fznModel.solve_type == "maximize")
            {
                Objective::Ptr obj = Factory::maximize(varsHelper.getObjectiveVar());
                obj->onFailure([&](){ stats.incrTighteningFail();});
                search.optimize(obj, stats, searchLimits);
            }
            else
                throw std::runtime_error("Unknown problem type");
        }
        else
            stats.setCompleted();

        // Flush failure buffer
        if (buffer and outFile)
            buffer->dump(outMutex, *outFile);

        stats.setSearchEndTime();

        // Final output
        outputHelper.printFinalOutput(stats.getCompleted(), stats.getSolutions());

        if (args["s"].count() != 0)
            FznStatisticsHelper::printStatistics(stats, solver, fznModel, search, cout);

        // Exit code: 0 = solved, 1 = no solution found
        exit(stats.getSolutions() > 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    else
    {
        std::cout << optsParser.help();
        exit(EXIT_FAILURE);
    }
}