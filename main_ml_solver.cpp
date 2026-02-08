#include <ml/OnnxHandler.h>
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
#include <sstream>

#include <fmt/format.h>
#include <fmt/ranges.h>

int main(int argc, char * argv[])
{
    using namespace std;

    // Parse options
    std::string fzn;
    std::string model;
    std::string varRankStr = "worst";
    std::string valRankStr = "worst";
    bool gpuInference = false;

    cxxopts::Options optsParser("fzn-minicpp-ml", "A C++ MiniZinc solver based on MiniCP.");
    optsParser.custom_help("[Options]");
    optsParser.positional_help("<FlatZinc>");
    optsParser.add_options()
        ("a", "Print all solutions", cxxopts::value<bool>())
        ("n", "Stop search after <n> solutions", cxxopts::value<unsigned int>())
        ("s", "Print search statistics", cxxopts::value<bool>())
        ("t", "Stop search after <t> ms", cxxopts::value<unsigned int>())
        ("model", "Machine learning model in ONNX format", cxxopts::value<std::string>(model))
        ("g,gpu", "Use GPU for inference", cxxopts::value<bool>(gpuInference))
        ("var-rank", "Criteria to rank variables: best, worst, bestAvg, worstAvg (Default = worstAvg)", cxxopts::value<std::string>(varRankStr))
        ("val-rank", "Criteria to rank values: best, worst, bestAvg, worstAvg (Default = worst)", cxxopts::value<std::string>(valRankStr))
         ("fzn", "FlatZinc", cxxopts::value<std::string>(fzn))
        ("h,help", "Print usage");
    optsParser.parse_positional({"fzn"});

    auto args = optsParser.parse(argc, argv);

    if ((args.count("h") == 0) and (not fzn.empty()) and (not model.empty()))
    {
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

        // Load ML evaluator with ONNX
        ML::RankType varRank = ML::rankFromString(varRankStr);
        ML::RankType valRank = ML::rankFromString(valRankStr);
        BatchedOnnxInfer infer(model,gpuInference);
        std::vector<float> mlScores;
        std::vector<int> mlVals;
        std::vector<int> mlPa;
        int nInitiallyAssigned = 0;
        for (auto const & var : intDecVars)
        {
            nInitiallyAssigned += var->isBound();
        }

        // Batched ML evaluation function
        ML::EvalFunctionType eval_fun = [&](int varIdx, ML::IntVars const& vars)
        {
            mlPa = ML::getPartialAssignment(vars);
            auto [mlVals, mlScores] = infer.scoreAllValuesForVar(mlPa, varIdx, vars[varIdx]);
            auto stats = ML::getStats(mlScores);

            auto [min_idx, max_idx, min_score, max_score, mean, eom01] = stats;

            // fmt::print(
            //     "Var {} | Info {:.2f} {:.2f} {:.2f} {:.2f} | Scores = {:.2f}\n",
            //     varIdx, min_score, max_score, mean, eom01 fmt::join(mlScores, ", ")
            // );

            auto [score, unc, idx] = ML::getScoreVal(stats, varRank, valRank);
            return std::make_tuple(score, unc, mlVals[idx]);
        };



        DFSearch search(solver, searchHelper.getMLSearchStrategy(fznModel, nInitiallyAssigned, eval_fun));
        FznStatisticsHelper::hookToSearch(stats, search);

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
            {
                search.solve(stats, searchLimits);
            }
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
            {
                throw std::runtime_error("Unknown problem type");
            }
        }
        else
        {
            stats.setCompleted();
        }

        stats.setSearchEndTime();

        // Final output
        outputHelper.printFinalOutput(stats.getCompleted(), stats.getSolutions());

        // Statistics output
        if (args["s"].count() != 0)
        {
            FznStatisticsHelper::printStatistics(stats, solver, fznModel, search, cout);
        }

        exit(EXIT_SUCCESS);
    }
    else
    {
        std::cout << optsParser.help();
        exit(EXIT_FAILURE);
    }
}